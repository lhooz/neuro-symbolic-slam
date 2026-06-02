#!/usr/bin/env python3
"""
snn_slam_system.py — Neuromorphic SLAM Orchestrator (v3)

Integrates three biological modules into a closed-loop navigation system:

  1. VisionCSNN   (256 feature neurons) — event-based edge features
  2. PoseCANN     (1024 spatial + 64 heading) — dead-reckoning via IMU
  3. PlaceCellNetwork + Parallel Ring Memory — depth-aware spatial memory

================================================================
  PARALLEL RING MEMORY + DUAL-KEY GATING — INSECT BRAIN v3
================================================================

Problem: Pure vision loop closures are ambiguous (same wall = same memory).
Solution: Dual-Key gating — BOTH place cells (WHERE) AND ring cells (WHICH WAY)
must agree, plus heading plausibility check eliminates false positives.

Author: Ada 🦊
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle, FancyArrow
import time, sys
import collections

# ============================================================================
# WORKSPACE
# ============================================================================

# Dynamically find the project root (one folder up from where this script lives)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sparse_forest import (
    generate_fixed_room_dataset,
    N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H, FOV_DEG,
    VX_RANGE, VY_RANGE, OMEGA_RANGE,
)
from src.snn_vision_fusion import DualStreamVisionCortex
from src.snn_pose_cann import (
    PoseCANN,
    build_2d_cann_weights,
    build_1d_ring_weights,
    build_asymmetric_ring_weights,
    build_asymmetric_cann_weights_x,
    build_asymmetric_cann_weights_y,
    CANN_SIZE, RING_N,
    ring_readout,
)
from src.snn_place_cells import (
    PlaceCellNetwork,
    MAP_SIZE, N_PLACE,
)

# ============================================================================
#  🎛️  HYPERPARAMETERS
# ============================================================================

N_VISION        = 128     # VisionSTDP feature neurons

N_DEPTH_PER_RAY = 64
N_DEPTH         = N_DEPTH_PER_RAY * 3  # 192 Total Apical Dendrites

TOF_MIN         = 0.1    # meters
TOF_MAX         = 9.9    # meters
TOF_SIGMA       = 0.1    # 10cm precision

DRIFT_START     = 80     # (Offline Default) step at which drift kicks in
DRIFT_OMEGA     = 0.001  # rad/s artificial yaw drift per timestep

N_TRAJ_SHOW     = 1
SAVE_FIG        = True


# ============================================================================
#  🧠  TOF POPULATION CODER (Gaussian RBF)
# ============================================================================

class ToFPopulationCoder:
    """Convert 3-Ray ToF depth → Gaussian population code."""

    def __init__(self, n_depth_per_ray=N_DEPTH_PER_RAY, tof_min=TOF_MIN, tof_max=TOF_MAX, sigma=TOF_SIGMA):
        self.n_depth_per_ray = n_depth_per_ray
        self.sigma = sigma
        self.centers = jnp.linspace(tof_min, tof_max, n_depth_per_ray)

    def __call__(self, tof_array):
        B = tof_array.shape[0]
        diff = tof_array[:, :, None] - jnp.array(self.centers)[None, None, :]
        activations = jnp.exp(-(diff ** 2) / (2 * self.sigma ** 2))
        activations = activations / (activations.max(axis=2, keepdims=True) + 1e-8)
        return activations.reshape(B, -1)

# ============================================================================
# 🌿 V4: NEUROMORPHIC TOPOLOGICAL RELAXATION (SPRING-MASS-DAMPER PHYSICS)
# ============================================================================

class SpikingMapState(collections.namedtuple('SpikingMapState', ['v_mem', 'spikes'])):
    pass

class SpikingOccupancyGrid:
    """A 2D sheet of Leaky Integrate-and-Fire (LIF) neurons for spatial mapping."""
    def __init__(self, map_size_m=30.0, res=0.10, offset_m=10.0): # 🌟 UPDATED
        self.res = res
        self.offset_m = offset_m # 🌟 NEW
        self.grid_w = int(map_size_m / res)
        self.grid_h = int(map_size_m / res)
        
        # LIF Dynamics
        self.v_th = 1.0         # Spiking threshold
        self.v_reset = 0.0      # Post-spike reset
        self.v_rest = 0.0       # Resting potential
        self.beta = 0.9995        # Leak rate (0.98 = forgets unconfirmed hits slowly)
        self.w_exc = 0.35       # Excitatory weight (ToF hit)
        self.w_inh = -0.15      # Inhibitory weight (Free space)

    def init_state(self):
        return SpikingMapState(
            v_mem=jnp.full((self.grid_w, self.grid_h), self.v_rest, dtype=jnp.float32),
            spikes=jnp.zeros((self.grid_w, self.grid_h), dtype=jnp.float32)
        )

    @partial(jax.jit, static_argnames=['self'])
    def update(self, state: SpikingMapState, hit_idx, free_idx):
        # 1. Natural Leak
        v_next = state.v_mem * self.beta
        
        # 2. Inject Inhibitory Current (Free Space)
        # Using JAX at[].add() to accumulate current safely
        v_next = v_next.at[free_idx[:, 0], free_idx[:, 1]].add(self.w_inh)
        
        # 3. Inject Excitatory Current (ToF Hits)
        v_next = v_next.at[hit_idx[:, 0], hit_idx[:, 1]].add(self.w_exc)
        
        # Prevent voltage from dropping infinitely low
        v_next = jnp.maximum(v_next, -0.5)

        # 4. Spiking Mechanism
        spikes = jnp.where(v_next >= self.v_th, 1.0, 0.0)
        v_next = jnp.where(spikes > 0.5, self.v_reset, v_next)
        
        return SpikingMapState(v_mem=v_next, spikes=spikes)

def wrap_angle(theta):
    """Keeps angles bound between -pi and pi to prevent winding spring tension."""
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

@partial(jax.jit, static_argnames=['iterations'])
def relax_graph(poses, odom_edges, loop_closures, loop_mask, iterations=1000): # Increased iterations!
    """
    3DOF Force-directed graph relaxation (X, Y, Theta).
    Poses: (N, 3) | Odom: (N-1, 3) | Loops: (MAX_LOOPS, 2)
    """
    # Translational parameters
    k_odom_pos = 0.20
    k_loop_pos = 0.80
    
    # 🌟 NEW: Rotational parameters (Torsional springs)
    k_odom_th = 0.15 
    k_loop_th = 0.60 
    
    damping = 0.85 

    # 🌟 NEW: Velocities now track X, Y, AND Theta
    velocities = jnp.zeros((poses.shape[0], 3))

    def step_fn(i, state):
        p, v = state
        
        # --- 1. Odometry Springs ---
        p_A = p[:-1]
        p_B = p[1:]
        
        err_x = (p_B[:, 0] - p_A[:, 0]) - odom_edges[:, 0]
        err_y = (p_B[:, 1] - p_A[:, 1]) - odom_edges[:, 1]
        # 🌟 NEW: Calculate rotational error and wrap it!
        err_th = wrap_angle((p_B[:, 2] - p_A[:, 2]) - odom_edges[:, 2])

        f_x = err_x * k_odom_pos
        f_y = err_y * k_odom_pos
        f_th = err_th * k_odom_th

        dp_odom_x = jnp.pad(f_x, (0, 1)) + jnp.pad(-f_x, (1, 0))
        dp_odom_y = jnp.pad(f_y, (0, 1)) + jnp.pad(-f_y, (1, 0))
        dp_odom_th = jnp.pad(f_th, (0, 1)) + jnp.pad(-f_th, (1, 0))

        # --- 2. Loop Closure Springs ---
        lc_A = p[loop_closures[:, 0]]
        lc_B = p[loop_closures[:, 1]]
        
        lc_err_x = (lc_B[:, 0] - lc_A[:, 0]) * loop_mask
        lc_err_y = (lc_B[:, 1] - lc_A[:, 1]) * loop_mask
        # 🌟 NEW: Calculate rotational loop error
        lc_err_th = wrap_angle(lc_B[:, 2] - lc_A[:, 2]) * loop_mask

        lc_f_x = lc_err_x * k_loop_pos
        lc_f_y = lc_err_y * k_loop_pos
        lc_f_th = lc_err_th * k_loop_th

        # Accumulate forces
        dp_loop_x = jax.ops.segment_sum(lc_f_x, loop_closures[:, 0], num_segments=p.shape[0]) - \
                    jax.ops.segment_sum(lc_f_x, loop_closures[:, 1], num_segments=p.shape[0])
        dp_loop_y = jax.ops.segment_sum(lc_f_y, loop_closures[:, 0], num_segments=p.shape[0]) - \
                    jax.ops.segment_sum(lc_f_y, loop_closures[:, 1], num_segments=p.shape[0])
        dp_loop_th = jax.ops.segment_sum(lc_f_th, loop_closures[:, 0], num_segments=p.shape[0]) - \
                     jax.ops.segment_sum(lc_f_th, loop_closures[:, 1], num_segments=p.shape[0])

        # --- 3. Integrate Kinematics ---
        v_new_x = (v[:, 0] + dp_odom_x + dp_loop_x) * damping
        v_new_y = (v[:, 1] + dp_odom_y + dp_loop_y) * damping
        v_new_th = (v[:, 2] + dp_odom_th + dp_loop_th) * damping

        p_new_x = p[:, 0] + v_new_x
        p_new_y = p[:, 1] + v_new_y
        # 🌟 NEW: Update heading and wrap it to prevent numeric explosion
        p_new_th = wrap_angle(p[:, 2] + v_new_th)

        # Anchor Node 0 (Translation and Rotation)
        p_new_x = p_new_x.at[0].set(0.0)
        p_new_y = p_new_y.at[0].set(0.0)
        p_new_th = p_new_th.at[0].set(0.0) 
        
        v_new_x = v_new_x.at[0].set(0.0)
        v_new_y = v_new_y.at[0].set(0.0)
        v_new_th = v_new_th.at[0].set(0.0)

        p_new = jnp.stack([p_new_x, p_new_y, p_new_th], axis=1)
        v_new = jnp.stack([v_new_x, v_new_y, v_new_th], axis=1)

        return (p_new, v_new)

    final_p, final_v = jax.lax.fori_loop(0, iterations, step_fn, (poses, velocities))
    return final_p

# ============================================================================
#  ⚖️ SOTA GAUGE ALIGNMENT (UMEYAMA'S ALGORITHM)
# ============================================================================

def get_optimal_alignment_2d(P_est, P_gt):
    """
    Computes the optimal rotation (R) and translation (t) to align P_est to P_gt.
    Uses Umeyama's algorithm (SVD) to eliminate Gauge Freedom / unobservable drift.
    """
    if len(P_est) < 5:
        return np.eye(2), np.zeros(2)

    # 1. Find centroids
    mu_est = np.mean(P_est, axis=0)
    mu_gt = np.mean(P_gt, axis=0)

    # 2. Center the points
    P_est_c = P_est - mu_est
    P_gt_c = P_gt - mu_gt

    # 3. Calculate Covariance Matrix & SVD
    H = P_est_c.T @ P_gt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 4. Handle reflection (ensure it's a pure rotation)
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    # 5. Calculate final translation
    t = mu_gt - R @ mu_est
    return R, t


# ============================================================================
#  🌍 INFINITE LIVE ENVIRONMENT
# ============================================================================
class LiveEnvironment:
    def __init__(self, key, chunk_size=2000):
        self.key = key
        self.chunk_size = chunk_size
        self.obstacles = None
        self.generate_new_chunk()

    def generate_new_chunk(self):
        self.key, subkey = random.split(self.key)
        events, labels, tof_dists, positions, headings, obs, segments, intensities = \
            generate_fixed_room_dataset(subkey, 1, time_steps=self.chunk_size, obstacles=self.obstacles)
        
        self.obstacles = np.array(obs)
        self.ev = np.array(events[0])
        self.kin = np.array(jnp.stack([labels[0, :, 0] * abs(VX_RANGE[1]), 
                                       labels[0, :, 1] * abs(VY_RANGE[1]), 
                                       labels[0, :, 2] * abs(OMEGA_RANGE[1])], axis=1))
        self.tof = np.array(tof_dists[0])
        self.pos = np.array(positions[0, :, :2])
        self.th = np.array(headings[0])
        self.intensities = np.array(intensities[0])
        self.t = 0

    def step(self):
        if self.t >= self.chunk_size:
            print("\n 🔄 Robot reached end of planned trajectory. Generating next path chunk...")
            self.generate_new_chunk()
            
        frame = (self.ev[self.t], self.kin[self.t], self.tof[self.t], 
                 self.pos[self.t], self.th[self.t], self.intensities[self.t])
        self.t += 1
        return frame


# ============================================================================
#  🧠  MASTER SYSTEM CLASS
# ============================================================================

class SNNSLAMSystem:
    def __init__(self, key, n_depth=N_DEPTH):
        self.n_depth = n_depth

        W_cann        = build_2d_cann_weights()
        W_ring        = build_1d_ring_weights()
        W_ring_asym   = build_asymmetric_ring_weights()
        W_cann_asym_x = build_asymmetric_cann_weights_x()
        W_cann_asym_y = build_asymmetric_cann_weights_y()

        k_vision, key = random.split(key)
        self.vision = DualStreamVisionCortex(k_vision, n_pixels=N_PIXELS)

        self.tof_coder = ToFPopulationCoder(n_depth_per_ray=self.n_depth // 3)

        k_pose, key = random.split(key)
        self.pose = PoseCANN(k_pose, W_cann, W_ring,
                              W_cann_asym_x, W_cann_asym_y, W_ring_asym)

        self.place = PlaceCellNetwork(n_csnn=128, n_stdp=256, n_depth=self.n_depth)

        self.vision_state = None
        self.place_state = None
        self._initialized = False
        self._step = 0

    def reset(self, B):
        self.vision_state = self.vision.init_state(B)
        self.place_state = self.place.init_state(B)
        self.pose.reset(B)
        self._initialized = False
        self._step = 0

    def initialize_from_gt(self, gt_pos, gt_heading):
        self.pose.initialize_from_gt(gt_pos, gt_heading)
        pose_bump = self.pose.get_state_flat()
        ring_bump = self.pose.get_ring_activity()
        self.place_state = self.place.initialize_from_pose(self.place_state, pose_bump, ring_bump=ring_bump)
        self._initialized = True

    def phase_perception(self, events_t, tof_t):
        # Vision now returns a tuple: (vis_csnn, vis_stdp)
        self.vision_state, dual_vis_features = self.vision(self.vision_state, events_t, tof_t[:, 1], learn=True)
        tof_pop = self.tof_coder(tof_t)
        return dual_vis_features, tof_pop

    def phase_inference(self, dual_vis_features, tof_features, pose_xy, current_heading_rads):
        vis_csnn, vis_stdp = dual_vis_features
        # 🌟 V4: Place Cells now output boolean confidence and an integer Memory ID
        self.place_state, is_confident, peak_idx_place, debug_gates = self.place.compute_confidence_with_gates(
            self.place_state, vis_csnn, vis_stdp, tof_features, pose_xy, current_heading_rads
        )
        return is_confident, peak_idx_place, debug_gates

    def phase_odometry(self, kin_t, inject_drift=False):
        if inject_drift:
            omega_drift = kin_t[:, 2] + DRIFT_OMEGA
            kin_injected = jnp.stack([kin_t[:, 0], kin_t[:, 1], omega_drift], axis=1)
        else:
            kin_injected = kin_t

        # 🌟 V4: CANN is now a pure, isolated odometry tracker. No map corrections!
        pose_est = self.pose(kin_injected)
        self.pose.update_cerebellum(kin_injected, pose_est[:, :2], pose_est[:, 2])

        pose_bump = self.pose.get_state_flat()
        ring_bump = self.pose.get_ring_activity()
        return pose_est, pose_bump, ring_bump

    def phase_mapping(self, dual_vis_features, tof_features, pose_bump, ring_bump):
        vis_csnn, vis_stdp = dual_vis_features
        self.place_state, _ = self.place.forward_mapping(
            self.place_state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump=ring_bump, learn=True)
        r_place = self.place.get_place_activity_flat(self.place_state)
        r_ring  = self.place.get_ring_activity_flat(self.place_state)
        return r_place, r_ring

    def forward_step(self, events_t, kin_t, tof_t, inject_drift=False):
        dual_vis_features, tof_features = self.phase_perception(events_t, tof_t)
        pose_xy = self.pose.estimate_position()
        current_heading_rads = self.pose.estimate_heading()

        is_confident, peak_idx_place, debug_gates = self.phase_inference(
            dual_vis_features, tof_features, pose_xy, current_heading_rads
        )

        pose_est, pose_bump, ring_bump = self.phase_odometry(kin_t, inject_drift)
        r_place, r_ring = self.phase_mapping(dual_vis_features, tof_features, pose_bump, ring_bump)

        self._step += 1
        return pose_est, r_place, r_ring, is_confident, peak_idx_place, debug_gates

    def forward_step_open_loop(self, events_t, kin_t, tof_t, inject_drift=False):
        dual_vis_features, tof_features = self.phase_perception(events_t, tof_t)
        
        # 🌟 V4: CANN is isolated! No dummy zeros or loop_conf variables needed.
        pose_est, pose_bump, ring_bump = self.phase_odometry(kin_t, inject_drift=inject_drift)

        r_place, r_ring = self.phase_mapping(dual_vis_features, tof_features, pose_bump, ring_bump)

        self._step += 1
        return pose_est, r_place, r_ring

# ============================================================================
#  🚀 LIVE SLAM ORCHESTRATOR
# ============================================================================

def get_ray_indices(cx, cy, cth, tof_dists, tof_angles, res=0.10, grid_size=300, offset_m=10.0): # 🌟 UPDATED
    hit_idx, free_idx = [], []
    
    # Define a strict cutoff for what counts as a real wall vs. open space
    MAX_VALID_RANGE = 7.4 
    
    for i in range(3):
        d = tof_dists[i]
        
        # 1. Carve out FREE SPACE regardless of whether we hit a wall or maxed out
        trace_dist = min(d, MAX_VALID_RANGE)
        for s in range(1, int(trace_dist / res)):
            fx = cx + (s * res) * np.cos(cth + tof_angles[i])
            fy = cy + (s * res) * np.sin(cth + tof_angles[i])
            # 🌟 NEW: Add offset_m to safely handle negative coordinates
            fix, fiy = int((fx + offset_m) / res), int((fy + offset_m) / res)
            if 0 <= fix < grid_size and 0 <= fiy < grid_size:
                free_idx.append([fix, fiy])

        # 2. ONLY register a HIT if it's actually closer than the max range
        if d < MAX_VALID_RANGE:
            hx = cx + d * np.cos(cth + tof_angles[i])
            hy = cy + d * np.sin(cth + tof_angles[i])
            # 🌟 NEW: Add offset_m here too
            ix, iy = int((hx + offset_m) / res), int((hy + offset_m) / res)
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                hit_idx.append([ix, iy])

    if not hit_idx: hit_idx.append([0, 0])
    if not free_idx: free_idx.append([0, 0])
    return np.array(hit_idx, dtype=np.int32), np.array(free_idx, dtype=np.int32)

def run_live_slam(key):
    env = LiveEnvironment(key, chunk_size=30000)
    
    print(f"\n 🧠 Initializing Twin SNN SLAM Systems v3 (Vis={N_VISION}, ToF={N_DEPTH})...")
    system_ol = SNNSLAMSystem(random.PRNGKey(42), n_depth=N_DEPTH)
    system_cl = SNNSLAMSystem(random.PRNGKey(43), n_depth=N_DEPTH)
    system_ol.reset(1); system_cl.reset(1)

    _, _, _, pos0, th0, _ = env.step()
    system_ol.initialize_from_gt(jnp.array([pos0]), jnp.array([th0]))
    system_cl.initialize_from_gt(jnp.array([pos0]), jnp.array([th0]))

    history = collections.defaultdict(list)
    x_imu, y_imu, th_imu = pos0[0], pos0[1], th0
    
    live_drift_start = 1000
    
    # 🌟 V4.1: Graph Data Structures & Keyframing
    graph_poses = []
    graph_odom_edges = []
    node_tof_hits = []       # Replaces history_map_pts!
    loop_closures = []
    place_to_node = {}
    
    MAX_LOOPS = 200
    KEYFRAME_DIST = 0.15     # Add node every 15cm
    KEYFRAME_ANG = 0.20      # Or every 0.20 radians (~11 degrees)
    last_kf_cann = None      # Tracks the CANN state at the last keyframe
    
    # ---------------------------------------------------------
    # 🎨 SETUP LIVE PLOTTING
    # ---------------------------------------------------------
    plt.ion()
    fig = plt.figure(figsize=(18, 6))
    ax_map = fig.add_subplot(131); ax_map.set_title("Phase 3: Real-Time Map")
    ax_brain = fig.add_subplot(132); ax_brain.set_title("Place Cell Memory")
    ax_cann = fig.add_subplot(133); ax_cann.set_title("Pose CANN Belief")
    
    _draw_room(ax_map, env.obstacles)
    
    # 🌟 NEW: Initialize the Biological SOG with the expanded canvas
    sog = SpikingOccupancyGrid(map_size_m=30.0, res=0.10, offset_m=10.0)
    sog_state = sog.init_state()
    
    # Render the membrane potentials as a glowing image!
    sog_img = ax_map.imshow(np.zeros((sog.grid_w, sog.grid_h)), 
                            cmap='magma', origin='lower', 
                            extent=[-sog.offset_m, 30.0 - sog.offset_m, -sog.offset_m, 30.0 - sog.offset_m], # 🌟 UPDATED
                            vmin=-0.2, vmax=1.0, alpha=0.8, zorder=2)
    
    fov_poly_gt = plt.Polygon(np.zeros((3, 2)), color='deepskyblue', alpha=0.15, zorder=1)
    ax_map.add_patch(fov_poly_gt)
    tof_rays_gt = [ax_map.plot([], [], color='blue', linestyle='--', lw=2.0, alpha=0.6, zorder=2)[0] for _ in range(3)]

    fov_poly = plt.Polygon(np.zeros((3, 2)), color='gold', alpha=0.3, zorder=3)
    ax_map.add_patch(fov_poly)
    tof_rays = [ax_map.plot([], [], 'r-', lw=2.5, alpha=0.8, zorder=4)[0] for _ in range(3)]
    
    gt_traj, = ax_map.plot([], [], 'b--', lw=1.5, alpha=0.3, label="Ground Truth", zorder=5)
    gt_head, = ax_map.plot([], [], 'bo', ms=6, alpha=0.5, zorder=6)
    
    live_traj, = ax_map.plot([], [], 'g-', lw=1.5, alpha=0.3, label="SNN Belief Trail", zorder=7)
    current_live_traj, = ax_map.plot([], [], color='#27AE60', lw=4.0, label="Current Belief", zorder=9)
    live_head, = ax_map.plot([], [], 'go', ms=10, zorder=10)
    ax_map.legend(loc='upper right', fontsize=8)

    brain_img = ax_brain.imshow(np.zeros((MAP_SIZE, MAP_SIZE)), cmap='magma', origin='lower', vmin=0, vmax=1.0)
    cann_img = ax_cann.imshow(np.zeros((CANN_SIZE, CANN_SIZE)), cmap='plasma', origin='lower', vmin=0, vmax=1.0)
    plt.show(block=False)

    print("\n 🟢 LIVE SLAM RUNNING! Press Ctrl+C in terminal to stop and generate PNGs.\n")
    
    step = 0
    tof_angles = np.array([-np.pi/4, 0.0, np.pi/4])
    t0 = time.time()

    # 🌟 V4: Graph Data Structures
    graph_poses = []
    graph_odom_edges = []
    loop_closures = []
    place_to_node = {}
    MAX_LOOPS = 200

    try:
        while True:
            ev_t, kin_t, tof_t, gt_pos, gt_th, intensity = env.step()
            ev_jax, kin_jax, tof_jax = jnp.array([ev_t]), jnp.array([kin_t]), jnp.array([tof_t])
            
            inject_drift = step >= live_drift_start

            if step > 0:
                omega_b = kin_t[2] + (DRIFT_OMEGA if inject_drift else 0.0)
                vx_w = kin_t[0] * np.cos(th_imu) - kin_t[1] * np.sin(th_imu)
                vy_w = kin_t[0] * np.sin(th_imu) + kin_t[1] * np.cos(th_imu)
                x_imu += vx_w * DT
                y_imu += vy_w * DT
                th_imu = (th_imu + omega_b * DT) % (2 * np.pi)

            pose_ol, _, _ = system_ol.forward_step_open_loop(ev_jax, kin_jax, tof_jax, inject_drift=inject_drift)
            pose_cl, r_place, r_ring, is_confident, peak_idx_place, debug_gates = system_cl.forward_step(ev_jax, kin_jax, tof_jax, inject_drift=inject_drift)
            
            cx, cy, cth = float(pose_cl[0, 0]), float(pose_cl[0, 1]), float(pose_cl[0, 2])

            # 🌟 V4.1: Is it a Keyframe?
            is_keyframe = False
            if len(graph_poses) == 0:
                is_keyframe = True
            else:
                prev_cx, prev_cy, prev_cth = last_kf_cann
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                ang = np.abs((cth - prev_cth + np.pi) % (2*np.pi) - np.pi)
                if dist > KEYFRAME_DIST or ang > KEYFRAME_ANG:
                    is_keyframe = True

            # 🌟 V4.1: Only build the graph on Keyframes!
            if is_keyframe:
                current_node_id = len(graph_poses)
                
                if current_node_id == 0:
                    graph_poses.append([cx, cy, cth])
                else:
                    dx, dy = cx - prev_cx, cy - prev_cy
                    dth = (cth - prev_cth + np.pi) % (2*np.pi) - np.pi
                    graph_odom_edges.append([dx, dy, dth])
                    
                    last_x, last_y, last_th = graph_poses[-1]
                    graph_poses.append([last_x + dx, last_y + dy, last_th + dth])
                
                last_kf_cann = (cx, cy, cth)
                
                # Save the RAW ToF hits attached to this specific node
                node_tof_hits.append(tof_t.copy())

                # 🌟 Trigger Topological Relaxation!
                if is_confident[0]:
                    place_id = int(peak_idx_place[0])
                    if place_id in place_to_node:
                        matched_node = place_to_node[place_id]
                        # Compare node IDs, not timesteps! Must be at least 20 nodes ago.
                        if current_node_id - matched_node > 20: 
                            loop_closures.append([matched_node, current_node_id])
                            
                            p_jax = jnp.array(graph_poses)
                            o_jax = jnp.array(graph_odom_edges)
                            lc_padded = np.zeros((MAX_LOOPS, 2), dtype=np.int32)
                            lc_mask = np.zeros(MAX_LOOPS, dtype=np.float32)
                            num_lc = min(len(loop_closures), MAX_LOOPS)
                            if num_lc > 0:
                                lc_padded[:num_lc] = np.array(loop_closures[-num_lc:])
                                lc_mask[:num_lc] = 1.0
                            
                            # RUN PHYSICS ENGINE
                            relaxed_p = relax_graph(p_jax, o_jax, jnp.array(lc_padded), jnp.array(lc_mask))
                            graph_poses = np.array(relaxed_p).tolist()
                            
                    place_to_node[place_id] = current_node_id

            # 🌟 V4.2 FIX: Map every frame to its Keyframe node so length matches Ground Truth
            active_node_id = len(graph_poses) - 1
            history['frame_to_node'].append(active_node_id)
            
            # Overwrite the plotting history with the relaxed graph, expanded to full length
            history['cl_pose'] = [np.array(graph_poses[nid]) for nid in history['frame_to_node']]
            history['raw_cann'].append(np.array(pose_cl[0]))
            
            # Use the relaxed tip for standard UI projection
            rel_cx, rel_cy, rel_cth = graph_poses[-1]

            history['gt_pos'].append(gt_pos); history['gt_th'].append(gt_th)
            history['imu_pos'].append([x_imu, y_imu]); history['imu_th'].append(th_imu)
            history['ol_pose'].append(np.array(pose_ol[0]))
            history['conf'].append(float(is_confident[0]))
            history['pc_act'].append(np.array(r_place[0]))
            history['cann_act'].append(np.array(system_cl.pose.get_state_flat()[0]))
            history['intensities'].append(intensity)

            # ---------------------------------------------------------
            # 🌟 V4: UPDATE THE BIOLOGICAL MEMORY (EVERY SINGLE STEP)
            # ---------------------------------------------------------
            # Cast rays using the newly aligned robot tip
            hit_idx, free_idx = get_ray_indices(
                rel_cx, rel_cy, rel_cth, 
                tof_t, tof_angles, res=sog.res, grid_size=sog.grid_w, offset_m=sog.offset_m
            )
            # Update the neurons!
            sog_state = sog.update(sog_state, jnp.array(hit_idx), jnp.array(free_idx))

            # ---------------------------------------------------------
            # 🌟 V4: RESTORED LIVE PLOTTING BLOCK
            # ---------------------------------------------------------
            if step % 15 == 0:
                d = debug_gates
                print(f"\r--- Step {step} | Snap Triggered: {d['Final_Conf'][0]} | Drift: {inject_drift} | Loops: {len(loop_closures)} ---        ", end="")

                gt_arr = np.array(history['gt_pos'])
                gt_traj.set_data(gt_arr[:, 0], gt_arr[:, 1])
                gt_head.set_data([gt_pos[0]], [gt_pos[1]])
                
                cl_arr = np.array(history['cl_pose'])
                
                # Live Umeyama Alignment (so the plot doesn't spin wildly)
                R_align, t_align = get_optimal_alignment_2d(cl_arr[:, :2], gt_arr)
                delta_theta = np.arctan2(R_align[1, 0], R_align[0, 0])
                
                aligned_cl_path = (R_align @ cl_arr[:, :2].T).T + t_align
                live_traj.set_data(aligned_cl_path[:, 0], aligned_cl_path[:, 1])
                
                recent_cl = aligned_cl_path[-100:] 
                current_live_traj.set_data(recent_cl[:, 0], recent_cl[:, 1])
                
                aligned_cx, aligned_cy = aligned_cl_path[-1, 0], aligned_cl_path[-1, 1]
                aligned_cth = rel_cth + delta_theta
                
                live_head.set_data([aligned_cx], [aligned_cy])
                
                # 🌟 V4.1: Dynamic Point Projection (The Skin)
                # Recalculate the entire map instantly from the relaxed nodes!
                current_map_pts = []
                
                # Update the UI image with the current membrane potentials
                sog_img.set_data(np.array(sog_state.v_mem).T) # Transpose for correct XY rendering
                
                # 🌟 THE FIX: Twist and shift the live SOG image to match the aligned green robot!
                trans_data = mtransforms.Affine2D().rotate(delta_theta).translate(t_align[0], t_align[1]) + ax_map.transData
                sog_img.set_transform(trans_data)
                
                R_fov = 3.0
                fov_rad = np.radians(FOV_DEG)
                gx, gy = gt_pos[0], gt_pos[1]
                
                for i in range(3):
                    gx_end = gx + tof_t[i] * np.cos(gt_th + tof_angles[i])
                    gy_end = gy + tof_t[i] * np.sin(gt_th + tof_angles[i])
                    tof_rays_gt[i].set_data([gx, gx_end], [gy, gy_end])
                    
                    rx_end = aligned_cx + tof_t[i] * np.cos(aligned_cth + tof_angles[i])
                    ry_end = aligned_cy + tof_t[i] * np.sin(aligned_cth + tof_angles[i])
                    tof_rays[i].set_data([aligned_cx, rx_end], [aligned_cy, ry_end])

                pt1_gt = [gx, gy]
                pt2_gt = [gx + R_fov * np.cos(gt_th + fov_rad/2), gy + R_fov * np.sin(gt_th + fov_rad/2)]
                pt3_gt = [gx + R_fov * np.cos(gt_th - fov_rad/2), gy + R_fov * np.sin(gt_th - fov_rad/2)]
                fov_poly_gt.set_xy([pt1_gt, pt2_gt, pt3_gt])

                pt1 = [aligned_cx, aligned_cy]
                pt2 = [aligned_cx + R_fov * np.cos(aligned_cth + fov_rad/2), aligned_cy + R_fov * np.sin(aligned_cth + fov_rad/2)]
                pt3 = [aligned_cx + R_fov * np.cos(aligned_cth - fov_rad/2), aligned_cy + R_fov * np.sin(aligned_cth - fov_rad/2)]
                fov_poly.set_xy([pt1, pt2, pt3])
                
                pc_act = np.array(r_place[0]).reshape(MAP_SIZE, MAP_SIZE)
                brain_img.set_data(pc_act / (pc_act.max() + 1e-8))
                
                cann_act = history['cann_act'][-1].reshape(CANN_SIZE, CANN_SIZE)
                cann_img.set_data(cann_act / (cann_act.max() + 1e-8))
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
            step += 1

    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n 🛑 Halted! Simulated {step} steps in {elapsed:.1f}s.")
        print(" 💾 Compiling logs and generating final PNG plots...")
        plt.ioff()
        plt.close(fig)

    # --- Compile & GLOBALLY ALIGN final results dictionary ---
    gt_arr = np.array(history['gt_pos'])
    imu_arr = np.array(history['imu_pos'])
    ol_arr = np.array(history['ol_pose'])
    cl_arr = np.array(history['cl_pose'])
    th_gt, th_imu = np.array(history['gt_th']), np.array(history['imu_th'])

    # Align Open-Loop SNN
    R_ol, t_ol = get_optimal_alignment_2d(ol_arr[:, :2], gt_arr)
    ol_arr_aligned = (R_ol @ ol_arr[:, :2].T).T + t_ol
    
    # Align Closed-Loop SNN
    R_cl, t_cl = get_optimal_alignment_2d(cl_arr[:, :2], gt_arr)
    cl_arr_aligned = (R_cl @ cl_arr[:, :2].T).T + t_cl
    delta_th_cl = np.arctan2(R_cl[1, 0], R_cl[0, 0])
    
    # 🌟 V4.1: Align Map Points Dynamically
    final_map_pts = []
    for node_idx, (nx, ny, nth) in enumerate(graph_poses):
        hits = node_tof_hits[node_idx]
        for i in range(3):
            if hits[i] < 7.5:
                hx = nx + hits[i] * np.cos(nth + tof_angles[i])
                hy = ny + hits[i] * np.sin(nth + tof_angles[i])
                final_map_pts.append([hx, hy])
                
    map_pts = np.array(final_map_pts) if len(final_map_pts) > 0 else np.zeros((0, 2))
    if len(map_pts) > 0:
        map_pts = (R_cl @ map_pts.T).T + t_cl

    # Calculate True Absolute Trajectory Error (ATE) using aligned paths!
    pos_err_imu = np.sqrt((imu_arr[:, 0] - gt_arr[:, 0])**2 + (imu_arr[:, 1] - gt_arr[:, 1])**2)
    pos_err_ol = np.sqrt((ol_arr_aligned[:, 0] - gt_arr[:, 0])**2 + (ol_arr_aligned[:, 1] - gt_arr[:, 1])**2)
    pos_err_cl = np.sqrt((cl_arr_aligned[:, 0] - gt_arr[:, 0])**2 + (cl_arr_aligned[:, 1] - gt_arr[:, 1])**2)

    def angle_err(a, b):
        diff = np.abs(a - b)
        return np.minimum(diff, 2*np.pi - diff)

    return {
        'B': 1, 'time_steps': step, 'drift_start': live_drift_start, 'obstacles': env.obstacles,
        'x_gt': gt_arr[None, :, 0], 'y_gt': gt_arr[None, :, 1], 'th_gt': th_gt[None, :],
        'x_imu': imu_arr[None, :, 0], 'y_imu': imu_arr[None, :, 1], 'th_imu': th_imu[None, :],
        'x_cl_raw': cl_arr[None, :, 0], 'y_cl_raw': cl_arr[None, :, 1],
        'x_ol': ol_arr_aligned[None, :, 0], 'y_ol': ol_arr_aligned[None, :, 1], 'th_ol': ol_arr[None, :, 2],
        'x_cl': cl_arr_aligned[None, :, 0], 'y_cl': cl_arr_aligned[None, :, 1], 'th_cl': (cl_arr[:, 2] + delta_th_cl)[None, :],
        'pos_err_imu': pos_err_imu[None, :], 'pos_err_ol': pos_err_ol[None, :], 'pos_err_cl': pos_err_cl[None, :],
        'theta_err_imu': angle_err(th_imu, th_gt)[None, :], 'theta_err_ol': angle_err(ol_arr[:, 2], th_gt)[None, :], 'theta_err_cl': angle_err(cl_arr[:, 2], th_gt)[None, :],
        'loop_conf': np.array(history['conf'])[None, :],
        'pc_top_conf': np.zeros((1, step)), 'pc_x_decoded': np.zeros((1, step)), 'pc_y_decoded': np.zeros((1, step)),
        'sog_grid': np.array(sog_state.v_mem)
    }

# ============================================================================
#  🎨  4-PANEL VISUALIZATION
# ============================================================================

def visualize_4panel(results, save_path=None, ev_save_path=None):
    B = results['B']
    T = results['time_steps']
    n_show = min(N_TRAJ_SHOW, B)
    ds = results['drift_start']
    ev_shape = results.get('ev_shape', (B, T, N_PIXELS))
    N_PIX = ev_shape[2]

    gt_colors = plt.cm.Blues(np.linspace(0.5, 0.9, n_show))
    imu_color  = '#E74C3C'  
    ol_color   = '#E67E22'  
    cl_color   = '#27AE60'  
    pc_star_c  = '#9B59B6'  

    t_arr = np.arange(T) * DT
    fig = plt.figure(figsize=(24, 7))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, figure=fig, wspace=0.35, left=0.04, right=0.98, top=0.90, bottom=0.14)

    # ── Panel 1: IMU-only ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_room(ax1, results['obstacles'])
    for i in range(n_show):
        ax1.plot(results['x_gt'][i, ::4], results['y_gt'][i, ::4], 'o-', color=gt_colors[i], ms=3, lw=1.5, alpha=0.7, label=f'GT {i}' if i == 0 else None)
        ax1.plot(results['x_imu'][i, ::4], results['y_imu'][i, ::4], 's--', color=imu_color, ms=3, lw=2.0, alpha=0.85, label=f'IMU-only {i}' if i == 0 else None)
        ax1.plot(results['x_gt'][i, 0], results['y_gt'][i, 0], 'D', color='lime', ms=10, zorder=10)
        ax1.plot(results['x_imu'][i, 0], results['y_imu'][i, 0], 'X', color=imu_color, ms=10, zorder=10)

    ax1.set_title('Panel 1: IMU-Only\n(Pure velocity integration — no SNN)', fontsize=11, fontweight='bold', color=imu_color)
    ax1.set_xlabel('x (m)', fontsize=9); ax1.set_ylabel('y (m)', fontsize=9)
    ax1.legend(fontsize=7, loc='upper right'); ax1.set_xlim(-0.5, ROOM_W + 0.5); ax1.set_ylim(-0.5, ROOM_H + 0.5)

    # ── Panel 2: Open-loop SNN ───────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_room(ax2, results['obstacles'])
    for i in range(n_show):
        ax2.plot(results['x_gt'][i, ::4], results['y_gt'][i, ::4], 'o-', color=gt_colors[i], ms=3, lw=1.5, alpha=0.7)
        ax2.plot(results['x_ol'][i, ::4], results['y_ol'][i, ::4], '^--', color=ol_color, ms=3, lw=2.0, alpha=0.85, label=f'OL SNN {i}' if i == 0 else None)
        ax2.plot(results['x_gt'][i, 0], results['y_gt'][i, 0], 'D', color='lime', ms=10, zorder=10)
        ax2.plot(results['x_ol'][i, 0], results['y_ol'][i, 0], 'X', color=ol_color, ms=10, zorder=10)

    ax2.set_title('Panel 2: Open-Loop SNN\n(Pose-CANN, Globally Aligned)', fontsize=11, fontweight='bold', color=ol_color)
    ax2.set_xlabel('x (m)', fontsize=9); ax2.set_ylabel('y (m)', fontsize=9)
    ax2.legend(fontsize=7, loc='upper right'); ax2.set_xlim(-0.5, ROOM_W + 0.5); ax2.set_ylim(-0.5, ROOM_H + 0.5)

    # ── Panel 3: Closed-loop SNN ────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _draw_room(ax3, results['obstacles'])
    for i in range(n_show):
        ax3.plot(results['x_gt'][i, ::4], results['y_gt'][i, ::4], 'o-', color=gt_colors[i], ms=3, lw=1.5, alpha=0.7)
        ax3.plot(results['x_cl'][i, ::4], results['y_cl'][i, ::4], '^-', color=cl_color, ms=3, lw=2.0, alpha=0.85, label=f'CL SNN {i}' if i == 0 else None)

    conf = results['loop_conf']
    for i in range(n_show):
        for t in range(T):
            if conf[i, t] > 0.1:
                ax3.plot(results['x_cl'][i, t], results['y_cl'][i, t], '.', color='#F39C12', ms=6, alpha=0.8, zorder=8)

    ax3.set_title('Panel 3: Closed-Loop SNN SLAM v3\n(Globally Aligned, • = loop closure)', fontsize=11, fontweight='bold', color=cl_color)
    ax3.set_xlabel('x (m)', fontsize=9); ax3.set_ylabel('y (m)', fontsize=9)
    ax3.legend(fontsize=7, loc='upper right'); ax3.set_xlim(-0.5, ROOM_W + 0.5); ax3.set_ylim(-0.5, ROOM_H + 0.5)

    # ── Panel 4: Event camera intensity images ───────────────────────────
    ev_examples = results.get('ev_examples', [])
    if ev_examples:
        n_r, n_c = 2, 4
        displayed = min(len(ev_examples), n_r * n_c)
        ev_gs = gs[0, 3].subgridspec(n_r, n_c, wspace=0.15, hspace=0.35)
        for idx in range(displayed):
            r, c = idx // n_c, idx % n_c
            ax_ev = fig.add_subplot(ev_gs[r, c])
            ax_ev.imshow(ev_examples[idx][None, :], aspect='auto', cmap='gray_r', vmin=0, vmax=1.5, interpolation='nearest')
            ax_ev.set_yticks([]); ax_ev.set_xticks([0, N_PIX//2, N_PIX])
            ax_ev.set_xticklabels(['0', str(N_PIX//2), str(N_PIX)], fontsize=5)
            ax_ev.tick_params(pad=1); ax_ev.set_title(f't={idx}', fontsize=6)
        fig.text(0.895, 0.96, 'Panel 4: Event Camera\n(continuous intensity)', ha='center', va='top', fontsize=11, fontweight='bold', color='#333')
    else:
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')

    # ── Error comparison subplot (ATE) ─────────────────────────────────────────────
    fig_err = plt.figure(figsize=(20, 5))
    ax_err = fig_err.add_subplot(1, 1, 1)

    mean_imu = results['pos_err_imu'].mean(axis=0)
    mean_ol  = results['pos_err_ol'].mean(axis=0)
    mean_cl  = results['pos_err_cl'].mean(axis=0)

    ax_err.plot(t_arr, mean_imu, color=imu_color, lw=2.5, label=f'Raw IMU (mean={mean_imu.mean():.3f}m)', ls='--', alpha=0.8)
    ax_err.plot(t_arr, mean_ol,  color=ol_color,  lw=2.5, label=f'Open-Loop ATE (mean={mean_ol.mean():.3f}m)', ls='-.', alpha=0.8)
    ax_err.plot(t_arr, mean_cl,  color=cl_color,  lw=3.0, label=f'Closed-Loop ATE (mean={mean_cl.mean():.3f}m)', ls='-')

    if ds < T:
        ax_err.axvline(ds * DT, color='gray', ls=':', lw=1.5, alpha=0.7)
        ax_err.text(ds * DT + 0.05, 0.95, 'Drift\nstarts', fontsize=8, color='gray', transform=ax_err.get_xaxis_transform(), verticalalignment='top')

    ax_err.fill_between(t_arr, mean_imu, mean_cl, where=(mean_imu > mean_cl), color=cl_color, alpha=0.08, label='IMU→CL Improvement')

    ax_err.set_title('Absolute Trajectory Error (ATE) over Time (meters, lower is better)', fontsize=12, fontweight='bold')
    ax_err.set_xlabel('Time (s)', fontsize=10); ax_err.set_ylabel('Position Error (m)', fontsize=10)
    ax_err.legend(fontsize=9, loc='upper left'); ax_err.grid(alpha=0.25, linestyle='--')
    ax_err.set_xlim(0, T * DT); ax_err.set_ylim(bottom=0)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n  💾 Saved trajectory figure: {save_path}")

    err_path = save_path.replace('.png', '_error.png') if save_path else None
    if err_path:
        fig_err.savefig(err_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  💾 Saved error figure: {err_path}")

    return fig, fig_err


def _draw_room(ax, obstacles):
    ax.set_aspect('equal'); ax.set_xlim(-0.3, ROOM_W + 0.3); ax.set_ylim(-0.3, ROOM_H + 0.3)
    ax.grid(alpha=0.2, linestyle='--')
    ax.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H, lw=2.5, edgecolor='#333', facecolor='#f8f8f5', alpha=0.5))
    if obstacles is not None:
        for o in obstacles:
            ax.add_patch(Rectangle((float(o[0]), float(o[1])), float(o[2]-o[0]), float(o[3]-o[1]), facecolor='#888', edgecolor='#222', lw=1.0, alpha=0.85))
    ax.set_xlabel('x (m)', fontsize=9); ax.set_ylabel('y (m)', fontsize=9)

# ============================================================================
#  🗺️  WORLD MAP VISUALIZATION
# ============================================================================

def visualize_world_map(results, save_path="snn_slam_world_map.png"):
    print(f"\n 🗺️ Rendering Global Spiking Occupancy Map to {save_path}...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Plot the native SOG Image (drawn using relaxed poses)
    sog_grid = results['sog_grid']
    offset_m = 10.0
    map_size_m = 30.0
    extent = [-offset_m, map_size_m - offset_m, -offset_m, map_size_m - offset_m]
    
    ax.imshow(sog_grid.T, cmap='magma', origin='lower', 
              extent=extent, vmin=-0.2, vmax=1.0)
    
    # 2. Extract Data
    gt_arr_x = results['x_gt'][0]
    gt_arr_y = results['y_gt'][0]
    
    # We use the CLOSED-LOOP path because the SOG walls were drawn using the relaxed graph
    snn_x = results['x_cl'][0] 
    snn_y = results['y_cl'][0] 
    
    # 3. Plot Ground Truth Room NORMALLY (Clean and straight!)
    if results['obstacles'] is not None:
        for o in results['obstacles']:
            w, h = float(o[2]-o[0]), float(o[3]-o[1])
            ax.add_patch(Rectangle((float(o[0]), float(o[1])), w, h, 
                                   facecolor='none', edgecolor='cyan', lw=1.5, ls='--', alpha=0.5))
                                   
    ax.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H, facecolor='none', edgecolor='cyan', lw=1.5, ls='--'))
    
    # Plot the Trajectories
    ax.plot(gt_arr_x, gt_arr_y, color='#3498DB', lw=1.5, alpha=0.6, label='Ground Truth')
    ax.plot(snn_x, snn_y, color='lime', lw=2.0, alpha=0.9, label='Closed-Loop SNN')

    ax.set_aspect('equal')
    ax.set_xlim(-2, ROOM_W + 2); ax.set_ylim(-2, ROOM_H + 2)
    ax.set_title('Phase 3: Spiking Occupancy Grid', fontsize=14, fontweight='bold')
    ax.set_xlabel('x (meters)'); ax.set_ylabel('y (meters)')
    ax.legend(loc='upper right')
    
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#222', edgecolor='none')
    print(f"  ✅ Map saved successfully!")
    plt.close(fig)

# ============================================================================
#  🚀  MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("  🦊  LIVE SNN SLAM System v3 — Continuous Exploration")
    print("=" * 65)

    seed_val = int(time.time() * 1000) % (2**31)
    key = random.PRNGKey(seed_val)
    print(f"  🎲 Generated New Random Room Seed: {seed_val}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    FIG_PATH = os.path.join(current_dir, "snn_slam_4panel.png")
    MAP_PATH = os.path.join(current_dir, "snn_slam_world_map.png")

    results = run_live_slam(key)

    print(f"\n🎨 Generating final High-Res offline visualizations...")
    visualize_4panel(results, save_path=FIG_PATH)
    # 🌟 THE FIX: Look for 'sog_grid' instead of 'world_map_points'
    if 'sog_grid' in results:
        visualize_world_map(results, save_path=MAP_PATH)

    print(f"\n{'='*65}")
    print(f"  ✅ SYSTEM SHUTDOWN COMPLETE")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()