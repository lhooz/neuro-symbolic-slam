#!/usr/bin/env python3
"""
snn_slam_system.py — Neuromorphic SLAM Orchestrator (v7)

Integrates three biological modules into a closed-loop navigation system:

  1. VisionCSNN   (256 feature neurons) — event-based edge features
  2. PoseCANN     (1024 spatial + 64 heading) — dead-reckoning via IMU
  3. PlaceCellNetwork + Parallel Ring Memory — depth-aware spatial memory

================================================================
  PARALLEL RING MEMORY + DUAL-KEY GATING — INSECT BRAIN v7
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
import io          # 🌟 NEW: For safe memory buffers
import imageio     # 🌟 NEW: For GIF generation

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

N_VISION        = 256     # VisionCSNN feature neurons

N_DEPTH_PER_RAY = 64
N_DEPTH         = N_DEPTH_PER_RAY * 3  # 192 Total Apical Dendrites

TOF_MIN         = 0.1    # meters
TOF_MAX         = 9.9    # meters
TOF_SIGMA       = 0.1    # 10cm precision

DRIFT_START     = 5000     # (Offline Default) step at which drift kicks in
DRIFT_OMEGA     = 0.001  # rad/s artificial yaw drift per timestep

N_TRAJ_SHOW     = 1
SAVE_FIG        = True

BASE_LC_FACTOR = 2.0
LC_MATURITY    = 0.65
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
    def __init__(self, map_size_m=30.0, res=0.10, offset_m=10.0, v_max=None): # 🌟 UPDATED
        self.res = res
        self.offset_m = offset_m # 🌟 NEW
        self.grid_w = int(map_size_m / res)
        self.grid_h = int(map_size_m / res)
        
        # LIF Dynamics
        self.v_th = 1.0         # Spiking threshold
        self.v_reset = 0.0      # Post-spike reset
        self.v_rest = 0.0       # Resting potential
        self.beta = 0.999        # Leak rate (0.98 = forgets unconfirmed hits slowly)
        self.w_exc = 0.35       # Excitatory weight (ToF hit)
        self.w_inh = -0.15      # Inhibitory weight (Free space)
        self.v_max = v_max if v_max is not None else float('inf')  # 🌟 V_MAX cap — biological membrane limit

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
        # v_next = jnp.where(spikes > 0.5, self.v_reset, v_next)
        
        # 🌟 V_MAX CEILING: Biological membrane potential cap
        # Prevents rotational smearing — voltage cannot exceed physical cell limits
        v_next = jnp.minimum(v_next, self.v_max)

        return SpikingMapState(v_mem=v_next, spikes=spikes)

def wrap_angle(theta):
    """Keeps angles bound between -pi and pi to prevent winding spring tension."""
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

@partial(jax.jit, static_argnames=['iterations'])
def relax_graph(poses, odom_edges, odom_mask, loop_closures, loop_offsets, loop_weights, loop_mask, is_frozen, iterations=3000):
    """
    3DOF Force-directed graph relaxation (X, Y, Theta).
    Now upgraded with SIMULATED ANNEALING (Dynamic Damping)!
    """
    k_odom_pos = 0.20
    k_odom_th = 0.15 
    
    # 🌟 DELETED: damping = 0.85 

    velocities = jnp.zeros((poses.shape[0], 3))

    def step_fn(i, state):
        p, v = state
        
        # =================================================================
        # 🌟 THE UPGRADE: Simulated Annealing (JAX-Safe Dynamic Damping)
        # =================================================================
        phase_1_end = iterations // 3        # Step 1000
        phase_2_end = (2 * iterations) // 3  # Step 2000

        dynamic_damping = jnp.where(
            i < phase_1_end,
            0.98,  # Phase 1 (Water): Near-frictionless. The energy wave hits Node 800 instantly!
            jnp.where(
                i < phase_2_end,
                0.85,  # Phase 2 (Oil): Catch the macro-swings and stabilize the map.
                0.60   # Phase 3 (Molasses): Aggressive freeze to lock in sub-millimeter precision.
            )
        )

        # --- 1. Odometry Springs (SE2 Kinematics) ---
        p_A = p[:-1]
        p_B = p[1:]
        
        th_A = p_A[:, 2]
        expected_dx = odom_edges[:, 0] * jnp.cos(th_A) - odom_edges[:, 1] * jnp.sin(th_A)
        expected_dy = odom_edges[:, 0] * jnp.sin(th_A) + odom_edges[:, 1] * jnp.cos(th_A)
        
        err_x = (p_B[:, 0] - p_A[:, 0]) - expected_dx
        err_y = (p_B[:, 1] - p_A[:, 1]) - expected_dy
        err_th = wrap_angle((p_B[:, 2] - p_A[:, 2]) - odom_edges[:, 2])

        f_x = err_x * k_odom_pos * odom_mask
        f_y = err_y * k_odom_pos * odom_mask
        f_th = err_th * k_odom_th * odom_mask

        # 🌟 THE SE(2) UPGRADE: The Lever Arm Effect!
        # Torque = r x F = (expected_dx * f_y) - (expected_dy * f_x)
        # We multiply by 0.25 to heavily dampen the torque and prevent Euler explosion
        torque_A = (expected_dx * f_y - expected_dy * f_x) * 0.25 * odom_mask

        # Apply translational forces exactly as before
        dp_odom_x = jnp.pad(f_x, (0, 1)) + jnp.pad(-f_x, (1, 0))
        dp_odom_y = jnp.pad(f_y, (0, 1)) + jnp.pad(-f_y, (1, 0))

        # 🌟 APPLY THE TORQUE
        # Node B receives standard IMU rotational correction (-f_th)
        # Node A receives standard IMU correction (+f_th) PLUS the physical lever arm torque!
        dp_odom_th_A = f_th + torque_A
        dp_odom_th_B = -f_th
        dp_odom_th = jnp.pad(dp_odom_th_A, (0, 1)) + jnp.pad(dp_odom_th_B, (1, 0))

        # --- 2. Loop Closure Springs ---
        lc_A = p[loop_closures[:, 0]]
        lc_B = p[loop_closures[:, 1]]
        
        # 🌟 SOLUTION 1: Apply Relative Transforms (Offsets)
        th_A_lc = lc_A[:, 2]
        expected_lc_dx = loop_offsets[:, 0] * jnp.cos(th_A_lc) - loop_offsets[:, 1] * jnp.sin(th_A_lc)
        expected_lc_dy = loop_offsets[:, 0] * jnp.sin(th_A_lc) + loop_offsets[:, 1] * jnp.cos(th_A_lc)

        lc_err_x = (lc_B[:, 0] - lc_A[:, 0] - expected_lc_dx) * loop_mask
        lc_err_y = (lc_B[:, 1] - lc_A[:, 1] - expected_lc_dy) * loop_mask
        lc_err_th = wrap_angle((lc_B[:, 2] - lc_A[:, 2]) - loop_offsets[:, 2]) * loop_mask

        # 🌟 SOLUTION 2: Apply Biological Confidence Weights
        lc_f_x = lc_err_x * loop_weights[:, 0]
        lc_f_y = lc_err_y * loop_weights[:, 0]
        lc_f_th = lc_err_th * loop_weights[:, 1]

        # Accumulate forces
        dp_loop_x = jax.ops.segment_sum(lc_f_x, loop_closures[:, 0], num_segments=p.shape[0]) - \
                    jax.ops.segment_sum(lc_f_x, loop_closures[:, 1], num_segments=p.shape[0])
        dp_loop_y = jax.ops.segment_sum(lc_f_y, loop_closures[:, 0], num_segments=p.shape[0]) - \
                    jax.ops.segment_sum(lc_f_y, loop_closures[:, 1], num_segments=p.shape[0])
        dp_loop_th = jax.ops.segment_sum(lc_f_th, loop_closures[:, 0], num_segments=p.shape[0]) - \
                     jax.ops.segment_sum(lc_f_th, loop_closures[:, 1], num_segments=p.shape[0])

        # 🌟 THE EXPLOSION FIX: Clip the max velocity step to guarantee Euler stability!
        dp_loop_x = jnp.clip(dp_loop_x, -0.10, 0.10)
        dp_loop_y = jnp.clip(dp_loop_y, -0.10, 0.10)
        dp_loop_th = jnp.clip(dp_loop_th, -0.05, 0.05)

        # --- 3. Integrate Kinematics ---
        # 🌟 Multiply by our new dynamic_damping!
        v_new_x = (v[:, 0] + dp_odom_x + dp_loop_x) * dynamic_damping
        v_new_y = (v[:, 1] + dp_odom_y + dp_loop_y) * dynamic_damping
        v_new_th = (v[:, 2] + dp_odom_th + dp_loop_th) * dynamic_damping

        p_new_x = p[:, 0] + v_new_x
        p_new_y = p[:, 1] + v_new_y
        p_new_th = wrap_angle(p[:, 2] + v_new_th)

        # 🌟 THE SLIDING WINDOW FIX: Use the is_frozen mask!
        p_new_x = jnp.where(is_frozen, poses[:, 0], p_new_x)
        p_new_y = jnp.where(is_frozen, poses[:, 1], p_new_y)
        p_new_th = jnp.where(is_frozen, poses[:, 2], p_new_th)

        v_new_x = jnp.where(is_frozen, 0.0, v_new_x)
        v_new_y = jnp.where(is_frozen, 0.0, v_new_y)
        v_new_th = jnp.where(is_frozen, 0.0, v_new_th)

        p_new = jnp.stack([p_new_x, p_new_y, p_new_th], axis=1)
        v_new = jnp.stack([v_new_x, v_new_y, v_new_th], axis=1)

        return (p_new, v_new)

    final_p, final_v = jax.lax.fori_loop(0, iterations, step_fn, (poses, velocities))
    return final_p

# ============================================================================
#  🌊 SOTA EVENT-CAMERA MENTAL ROTATION (PHASE CORRELATION)
# ============================================================================
@jax.jit
def get_phase_correlation(live_csnn, mem_csnn):
    """
    Computes scale/contrast-invariant correlation using 1D Phase Correlation.
    Strips away event amplitude variance and returns pure structural overlap.
    """
    # 1. Transform to Frequency Domain
    F_live = jnp.fft.fft(live_csnn)
    F_mem = jnp.fft.fft(mem_csnn)
    
    # 2. Calculate the Cross-Power Spectrum (Mem * conj(Live))
    cross_power = F_mem * jnp.conj(F_live)
    
    # 3. THE SOTA UPGRADE: Normalize by magnitude to extract pure Phase
    cross_power_norm = cross_power / (jnp.abs(cross_power) + 1e-8)
    
    # 4. Inverse FFT back to Spatial Domain
    r = jnp.fft.ifft(cross_power_norm)
    return jnp.abs(r) # Return the real magnitude of the Dirac peaks

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

        # 🌟 FIX 3: Pass the true hardware FOV into the network!
        self.place = PlaceCellNetwork(n_csnn=256, n_stdp=256, n_depth=self.n_depth, fov_deg=FOV_DEG)

        self.vision_state = None
        self.place_state = None
        self._initialized = False
        self._step = 0
        
        # 🌟 NEW: The Cerebellum's internal calibration model
        self.learned_omega_bias = 0.0

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

    def calibrate_cerebellum(self, accumulated_error_rads, time_elapsed_sec):
        """Phase 3 Plasticity: Learn the systematic hardware drift!"""
        if time_elapsed_sec <= 0: return
        
        # Calculate the drift rate in rad/s
        drift_rate = accumulated_error_rads / time_elapsed_sec
        
        # Gentle Hebbian update (EMA) to prevent overreacting to one noisy loop closure
        learning_rate = 0.05
        self.learned_omega_bias = (1.0 - learning_rate) * self.learned_omega_bias + (learning_rate * drift_rate)

    def phase_perception(self, events_t, tof_t):
        self.vision_state, dual_vis_features = self.vision(self.vision_state, events_t, tof_t[:, 1], learn=True)
        tof_pop = self.tof_coder(tof_t)
        return dual_vis_features, tof_pop

    def phase_inference(self, dual_vis_features, tof_features, pose_xy, current_heading_rads, ring_bump):
        vis_csnn, vis_stdp = dual_vis_features
        self.place_state, is_confident, peak_idx_place, debug_gates = self.place.compute_confidence_with_gates(
            self.place_state, vis_csnn, vis_stdp, tof_features, pose_xy, current_heading_rads, ring_bump
        )
        return is_confident, peak_idx_place, debug_gates

    def phase_odometry(self, kin_t, inject_drift=False):
        # 🌟 CEREBELLUM INTERVENTION: Subtract the learned bias from the raw hardware!
        corrected_omega = kin_t[:, 2] - self.learned_omega_bias
        kin_corrected = jnp.stack([kin_t[:, 0], kin_t[:, 1], corrected_omega], axis=1)

        if inject_drift:
            # We inject the factory drift onto the CORRECTED signal
            omega_drift = kin_corrected[:, 2] + DRIFT_OMEGA
            kin_injected = jnp.stack([kin_corrected[:, 0], kin_corrected[:, 1], omega_drift], axis=1)
        else:
            kin_injected = kin_corrected

        pose_est = self.pose(kin_injected)
        self.pose.update_cerebellum(kin_injected, pose_est[:, :2], pose_est[:, 2])

        pose_bump = self.pose.get_state_flat()
        ring_bump = self.pose.get_ring_activity()
        return pose_est, pose_bump, ring_bump

    # 🌟 UPGRADE: Add 'heading' to the signature
    def phase_mapping(self, dual_vis_features, tof_features, pose_bump, ring_bump, heading, angular_vel):
        vis_csnn, vis_stdp = dual_vis_features
        
        self.place_state, (r_place, r_ring) = self.place.forward_mapping(
            self.place_state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump=ring_bump, heading=heading, angular_vel=angular_vel, learn=True
        )
        
        return r_place, r_ring

    def forward_step(self, events_t, kin_t, tof_t, inject_drift=False):
        dual_vis_features, tof_features = self.phase_perception(events_t, tof_t)

        pose_xy = self.pose.estimate_position()
        current_heading_rads = self.pose.estimate_heading()
        
        # 🌟 Grab the CANN belief BEFORE inference
        ring_bump_prior = self.pose.get_ring_activity()

        is_confident, peak_idx_place, debug_gates = self.phase_inference(
            dual_vis_features, tof_features, pose_xy, current_heading_rads, ring_bump_prior
        )

        pose_est, pose_bump, ring_bump = self.phase_odometry(kin_t, inject_drift)
        
        # 🌟 UPGRADE: Pass the continuous heading (pose_est[:, 2]) into the mapping phase!
        r_place, r_ring = self.phase_mapping(dual_vis_features, tof_features, pose_bump, ring_bump, pose_est[:, 2], kin_t[:, 2])

        self._step += 1
        return pose_est, r_place, r_ring, is_confident, peak_idx_place, debug_gates

    def forward_step_open_loop(self, events_t, kin_t, tof_t, inject_drift=False):
        dual_vis_features, tof_features = self.phase_perception(events_t, tof_t)

        pose_est, pose_bump, ring_bump = self.phase_odometry(kin_t, inject_drift=inject_drift)

        self._step += 1
        return pose_est, None, None

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
    node_tof_hits = []       
    loop_closures = []
    loop_offsets_list = []  # 🌟 NEW: Track Relative Transforms
    loop_weights_list = []  # 🌟 NEW: Track Confidence Springs
    place_to_node = collections.defaultdict(list)
    
    MAX_LOOPS = 200
    KEYFRAME_DIST = 0.15     # Add node every 15cm
    KEYFRAME_ANG = 0.20      # Or every 0.20 radians (~11 degrees)
    last_kf_cann = None      # Tracks the CANN state at the last keyframe
    lc_refractory_timer = 0

    # ---------------------------------------------------------
    # 🎨 SETUP LIVE PLOTTING (Upgraded to Overlapped UI)
    # ---------------------------------------------------------
    plt.ion()
    fig = plt.figure(figsize=(18, 10)) # 🌟 Slightly taller to fit the error panel
    
    # 🌟 NEW: 3-row layout. Map on top left, Brains on top right, Error spans the bottom.
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1.5, 1.0, 1.0])
    
    ax_map = fig.add_subplot(gs[:2, 0]); ax_map.set_title("Phase 3: Real-Time Map (Live Umeyama Alignment)")
    
    # 🌟 Overlapped Subplots
    ax_2d = fig.add_subplot(gs[0, 1]); ax_2d.set_title("2D Space: Memory (Magma) vs CANN Belief (Cyan)")
    ax_1d = fig.add_subplot(gs[1, 1]); ax_1d.set_title("1D Heading: Memory (Purple) vs CANN Belief (Orange)")

    # 🌟 NEW PANEL 4: Live Error Subplot
    ax_err = fig.add_subplot(gs[2, :]); ax_err.set_title("Live Absolute Trajectory Error (ATE)")
    ax_err.set_xlabel("Time (s)"); ax_err.set_ylabel("Position Error (meters)")
    ax_err.grid(alpha=0.3, ls='--')
    
    line_err_imu, = ax_err.plot([], [], color='#E74C3C', lw=1.5, ls='--', label="Raw IMU Drift")
    line_err_ol, = ax_err.plot([], [], color='#E67E22', lw=1.5, ls='-.', label="Open-Loop ATE")
    line_err_cl, = ax_err.plot([], [], color='#27AE60', lw=2.5, label="Closed-Loop ATE")
    ax_err.legend(loc='upper left')
    
    _draw_room(ax_map, env.obstacles)
    
    # 🌟 NEW: Initialize the Biological SOG with the expanded canvas
    sog = SpikingOccupancyGrid(map_size_m=30.0, res=0.10, offset_m=10.0)
    sog_state = sog.init_state()
    
    # Render the membrane potentials as a glowing image!
    sog_img = ax_map.imshow(np.zeros((sog.grid_w, sog.grid_h)), 
                            cmap='magma', origin='lower', 
                            extent=[-sog.offset_m, 30.0 - sog.offset_m, -sog.offset_m, 30.0 - sog.offset_m], 
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

    # ==========================================================
    # 🌟 NEW: PLASTICITY STATUS INDICATOR
    # ==========================================================
    learning_indicator = ax_map.text(0.02, 0.98, '👁️ Plasticity: ON', 
                                     transform=ax_map.transAxes, fontsize=12, fontweight='bold', color='lime',
                                     verticalalignment='top', 
                                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'), zorder=20)

    # 🌟 2D Overlapped Matrices
    brain_img = ax_2d.imshow(np.zeros((MAP_SIZE, MAP_SIZE)), cmap='magma', origin='lower', vmin=0, vmax=1.0)
    
    # 🌟 THE GUI FIX: Initialize the CANN image as a pure 4-channel RGBA matrix!
    cann_img = ax_2d.imshow(np.zeros((CANN_SIZE, CANN_SIZE, 4)), origin='lower', zorder=5)
    
    # 🌟 1D Overlapped Line Plots
    x_ring = np.arange(RING_N)
    line_ring_mem, = ax_1d.plot(x_ring, np.zeros(RING_N), color='#9B59B6', lw=3, label='Memory')
    line_ring_cann, = ax_1d.plot(x_ring, np.zeros(RING_N), color='#E67E22', lw=2, ls='--', label='CANN Belief')
    ax_1d.set_ylim(-0.1, 1.1); ax_1d.set_xticks([])
    ax_1d.legend(loc='upper right')

    plt.show(block=False)

    print("\n 🟢 LIVE SLAM RUNNING! Press Ctrl+C in terminal to stop and generate PNGs.\n")
    
    step = 0
    steps_since_last_lc = 0  # 🌟 NEW: The Drift Timer
    tof_angles = np.array([-np.pi/4, 0.0, np.pi/4])
    t0 = time.time()

    ui_smooth_th = 0.0
    ui_smooth_t = np.zeros(2)
    history['live_offsets'] = [] 
    
    # ==========================================================
    # 🌟 NEW: INITIALIZE GIF WRITER
    # ==========================================================
    gif_filename = "snn_live_run.gif"
    print(f" 🎥 Recording live UI to {gif_filename} (Capturing every 15 steps)...")
    gif_writer = imageio.get_writer(gif_filename, fps=10) # Adjust FPS to speed up/slow down the GIF
    
    try:
        while True:
            ev_t, kin_t, tof_t, gt_pos, gt_th, intensity = env.step()
            ev_jax, kin_jax, tof_jax = jnp.array([ev_t]), jnp.array([kin_t]), jnp.array([tof_t])
            
            steps_since_last_lc += 1  # 🌟 Tick the timer!
            
            # 🌟 NEW: Cool down the Loop Closure Engine
            if lc_refractory_timer > 0:
                lc_refractory_timer -= 1

            inject_drift = step >= live_drift_start

            if step > 0:
                # 1. Constant Bias (The factory offset)
                bias = DRIFT_OMEGA if inject_drift else 0.0
    
                # 2. Random Walk (The electrical noise)
                noise_std_dev = 0.005 if inject_drift else 0.0
                random_noise = np.random.normal(0.0, noise_std_dev)
    
                omega_b = kin_t[2] + bias + random_noise
                vx_w = kin_t[0] * np.cos(th_imu) - kin_t[1] * np.sin(th_imu)
                vy_w = kin_t[0] * np.sin(th_imu) + kin_t[1] * np.cos(th_imu)
                x_imu += vx_w * DT
                y_imu += vy_w * DT
                th_imu = wrap_angle(th_imu + omega_b * DT)

            pose_ol, _, _ = system_ol.forward_step_open_loop(ev_jax, kin_jax, tof_jax, inject_drift=inject_drift)
            pose_cl, r_place, r_ring, is_confident, peak_idx_place, debug_gates = system_cl.forward_step(ev_jax, kin_jax, tof_jax, inject_drift=inject_drift)
            
            cx, cy, cth = float(pose_cl[0, 0]), float(pose_cl[0, 1]), float(pose_cl[0, 2])

            # 🌟 V4.6 FIX: Protect Frame 0 from the NoneType crash!
            if last_kf_cann is None:
                last_kf_cann = (cx, cy, cth)

            # 🌟 V4.5 FIX: Calculate the high-frequency UI offset FIRST, using the un-teleported math!
            kf_x, kf_y, kf_th = last_kf_cann
            dx = cx - kf_x
            dy = cy - kf_y
            local_dx = dx * np.cos(-kf_th) - dy * np.sin(-kf_th)
            local_dy = dx * np.sin(-kf_th) + dy * np.cos(-kf_th)
            local_dth = (cth - kf_th + np.pi) % (2*np.pi) - np.pi

            # The logic that decides when to drop a node
            is_keyframe = False
            if len(graph_poses) == 0:
                is_keyframe = True
            else:
                dist = np.sqrt(dx**2 + dy**2)
                ang = np.abs(local_dth)
                
                # 🌟 THE UPGRADE: Event-Driven Keyframing!
                # If the robot moves 15cm, OR turns 11 deg, OR the SNN suddenly 
                # achieves a confident memory lock (and isn't asleep), drop a node immediately!
                if dist > KEYFRAME_DIST or ang > KEYFRAME_ANG or (is_confident[0] and lc_refractory_timer == 0):
                    is_keyframe = True

            if is_keyframe:
                current_node_id = len(graph_poses)
                
                if current_node_id == 0:
                    graph_poses.append([cx, cy, cth])
                else:
                    graph_odom_edges.append([local_dx, local_dy, local_dth])
                    graph_poses.append([cx, cy, cth])
                
                last_kf_cann = (cx, cy, cth)
                node_tof_hits.append(tof_t.copy())

                # 1. What room does the VISION think we are in?
                recalled_place_id = int(peak_idx_place[0])

                # 🌟 THE BUG FIX: Only allow LC if the SNN is confident AND the system is rested!
                if is_confident[0] and lc_refractory_timer == 0:
                    matched_node = None
                    min_tof_diff = 999.0
                    curr_tof = np.array(tof_t) # 🌟 Moved this up so we can use it!
                    
                    # 2. Search the folder of the RECALLED room
                    if recalled_place_id in place_to_node:
                        for nid in place_to_node[recalled_place_id]:
                            if current_node_id - nid > 50:  
                                _, _, nth = graph_poses[nid]
                                dth = abs(wrap_angle(cth - nth))
                                
                                if dth < 0.30:
                                    # 🌟 THE SYNC FIX: Make Python pick the exact Node the SNN matched!
                                    mem_tof = np.array(node_tof_hits[nid])
                                    tof_diff = np.sum(np.abs(curr_tof - mem_tof))
                                    
                                    if tof_diff < min_tof_diff:
                                        min_tof_diff = tof_diff
                                        matched_node = nid

                    # 🌟 FIX 2: The Absolute Geometry Gate!
                    # If the sum of absolute errors across all 3 lasers is > 0.80 meters, 
                    # it is a corridor hallucination. Abort!
                    if min_tof_diff > 0.80:
                        matched_node = None 
                    
                    if matched_node is not None:
                        maturity = float(debug_gates['Maturity_Lvl'][0])
                        if maturity >= LC_MATURITY: 
                            matched_x, matched_y, matched_th = graph_poses[matched_node]
                            
                            # =================================================================
                            # 🌟 THE SCOPE LEAK FIX: Re-fetch the correct arrays!
                            # =================================================================
                            mem_tof = np.array(node_tof_hits[matched_node])
                            tof_diff = np.sum(np.abs(curr_tof - mem_tof))
                            
                            # =================================================================
                            # 🌟 THE ALIASING FIX: The Aperture Gate
                            # =================================================================
                            raw_jump = np.sqrt((cx - matched_x)**2 + (cy - matched_y)**2)
                            
                            # 1. Establish a hardware drift noise floor (e.g., 15cm)
                            # We don't want micro-drifts triggering the math.
                            drift_noise_floor = 0.15
                            
                            # 2. The Physics Law of Translation:
                            # If a robot moves D meters, 3 divergent lasers should change 
                            # by 1.0*D to 3.8*D. If they change by less than 50% of the jump, 
                            # we are mathematically sliding parallel to an unobservable wall!
                            expected_min_tof_diff = raw_jump * 0.50
                            
                            if raw_jump > drift_noise_floor and tof_diff < expected_min_tof_diff:
                                print(f"\n ⚠️ Aborted: Aperture Problem! Claimed {raw_jump:.2f}m jump, but ToF only shifted {tof_diff:.2f}m (Expected > {expected_min_tof_diff:.2f}m).")
                                matched_node = None
                                continue # Skip the rest of the loop closure logic

                            # =================================================================
                            # 🌟 STEP 1: SOLVE ROTATION (1D Fourier Phase Correlation)
                            # =================================================================
                            input_csnn = jnp.array(debug_gates["Debug_Input_CSNN"][0])
                            mem_csnn_ring = jnp.array(debug_gates["Debug_Mem_CSNN_Ring"][0])
                            
                            # Run the JIT-compiled Phase Correlation
                            r_real = np.array(get_phase_correlation(input_csnn, mem_csnn_ring))
                            N_VIS = len(input_csnn)
                            
                            # 🌟 THE FIX: The FFT Search Window
                            # Phase correlation wraps around. Shift 0 is at index 0. 
                            # Shift +1 is index 1. Shift -1 is index 255.
                            # We force the algorithm to IGNORE shifts > 24 pixels (~17 deg)
                            search_radius = 24
                            
                            # Create a masked array that zeroes out physically impossible shifts
                            r_masked = np.zeros_like(r_real)
                            r_masked[:search_radius + 1] = r_real[:search_radius + 1]       # Positive shifts [0, 24]
                            r_masked[-search_radius:] = r_real[-search_radius:]             # Negative shifts [-24, -1]
                            
                            # Find the peak INSIDE the safe window
                            peak_idx = np.argmax(r_masked)
                            
                            # True sub-pixel interpolation (Parabolic fit on the Dirac spike)
                            # Use modulo arithmetic to safely wrap the indices around the edges!
                            y1 = r_real[(peak_idx - 1) % N_VIS]
                            y2 = r_real[peak_idx] 
                            y3 = r_real[(peak_idx + 1) % N_VIS]
                            
                            sub_pixel_offset = (y1 - y3) / (2.0 * (y1 - 2.0 * y2 + y3) + 1e-8)
                            
                            # Convert FFT peak index back to integer shift [-24, 24]
                            shift_int = peak_idx if peak_idx <= search_radius else peak_idx - N_VIS
                            
                            pixel_shift = shift_int + sub_pixel_offset 
                            pixel_ang_res = np.radians(FOV_DEG) / N_VISION
                            
                            # =================================================================
                            # 🌟 THE UPGRADE: True Sub-Pixel Cartesian Physics
                            # =================================================================
                            
                            # 1. Calculate the SNN's raw sub-pixel visual rotation (The FFT Offset)
                            sub_pixel_th = float(np.clip(-pixel_shift * pixel_ang_res, -0.30, 0.30))
                            
                            # 2. Extract the exact continuous angle burned into the SNN memory!
                            # 🌟 No more 5.6-degree Ring Cell quantization bins!
                            true_burned_angle = float(debug_gates['Peak_Theta_Burn'][0])
                            
                            # 3. Calculate the True Absolute Compass Heading according to Vision
                            vision_th = float(wrap_angle(true_burned_angle + sub_pixel_th))
                            
                            # 4. Calculate the True Crossing Angle for the Graph Optimizer
                            lc_offset_th = float(wrap_angle(vision_th - matched_th))
                            
                            # 5. Extract r_idx just for the Phase 2 Plasticity update later
                            r_idx = int(debug_gates['Peak_Ring'][0])

                            # =================================================================
                            # 🌟 THE ROTATIONAL CONTRADICTION GATE (Dynamic Physics Formula)
                            # =================================================================
                            # 1. Calculate the expected ToF shift based on trig (2 * L * dTh)
                            # 🌟 THE BUG FIX: Do not include 8.0m artifacts in the mean!
                            valid_rays_for_mean = curr_tof[curr_tof < 7.4]
                            mean_tof_dist = np.mean(valid_rays_for_mean) if len(valid_rays_for_mean) > 0 else 1.0
                            
                            expected_tof_diff = 2.0 * mean_tof_dist * abs(lc_offset_th)
                            
                            # 2. Establish a strict hardware noise floor (e.g. 10cm total noise)
                            noise_floor = 0.10 
                            
                            # 3. The Gate: If physics expected a massive shift, but the actual 
                            # lasers saw less than 30% of that expectation, Vision is lying!
                            if expected_tof_diff > noise_floor and tof_diff < (0.30 * expected_tof_diff):
                                print(f"\n ⚠️ Aborted: Rotational Contradiction! Expected {expected_tof_diff:.2f}m shift, but saw {tof_diff:.2f}m.")
                                matched_node = None
                                continue
                            
                            # =================================================================
                            # 🌟 THE ROTATIONAL SIGNATURE GATE (Directional Physics Check)
                            # =================================================================
                            # The magnitude gate above can be fooled if the robot translates forward.
                            # We must explicitly check the physical direction of the sweeping beams!
                            
                            delta_L = curr_tof[0] - mem_tof[0]
                            delta_R = curr_tof[2] - mem_tof[2]
                            
                            # A positive signature means the robot physically rotated LEFT.
                            # A negative signature means the robot physically rotated RIGHT.
                            rotational_signature = delta_L - delta_R
                            
                            # We only enforce this strict direction check if the claimed rotation 
                            # is large enough (> 5 degrees) to avoid flat-wall noise.
                            if abs(lc_offset_th) > 0.08:
                                # If Vision claims LEFT (+), but physical signature is RIGHT (-)
                                if lc_offset_th > 0 and rotational_signature < -0.10:
                                    print(f"\n ⚠️ Aborted: Directional Contradiction! Vision claims LEFT (+{lc_offset_th:.2f}), but lasers swept RIGHT ({rotational_signature:.2f}).")
                                    matched_node = None
                                    continue
                                    
                                # If Vision claims RIGHT (-), but physical signature is LEFT (+)
                                elif lc_offset_th < 0 and rotational_signature > 0.10:
                                    print(f"\n ⚠️ Aborted: Directional Contradiction! Vision claims RIGHT ({lc_offset_th:.2f}), but lasers swept LEFT (+{rotational_signature:.2f}).")
                                    matched_node = None
                                    continue

                            # =================================================================
                            # 🌟 STEP 2: SOLVE TRANSLATION (Scan-to-Map / Point-to-Plane)
                            # =================================================================
                            curr_tof = np.array(tof_t)
                            mem_tof = np.array(node_tof_hits[matched_node])
                                
                            # 1. Extract the SOG map as a fast NumPy array
                            v_mem_np = np.array(sog_state.v_mem)
                            # 🌟 THE BINARIZATION FIX: Destroy the temporal slope!
                            # Treat everything above 0.5 as a pure 1.0 solid block.
                            v_mem_np = (v_mem_np > 0.5).astype(np.float32)
                            map_w, map_h = v_mem_np.shape
                                
                            normals_x, normals_y = [], []
                                
                            for i in range(3):
                                # 2. Where did the MEMORIZED laser hit the global map?
                                hit_x = matched_x + mem_tof[i] * np.cos(matched_th + tof_angles[i])
                                hit_y = matched_y + mem_tof[i] * np.sin(matched_th + tof_angles[i])
                                    
                                # 3. Convert to Map Grid Indices
                                ix = int((hit_x + sog.offset_m) / sog.res)
                                iy = int((hit_y + sog.offset_m) / sog.res)
                                    
                                # 4. Calculate the Wall Normal (3x3 Sobel Spatial Gradient)
                                if 1 <= ix < map_w - 1 and 1 <= iy < map_h - 1:
                                        
                                    # 🌟 UPGRADE: 3x3 Sobel Filter for the X-Gradient
                                    map_dx = (v_mem_np[ix + 1, iy - 1] + 2 * v_mem_np[ix + 1, iy] + v_mem_np[ix + 1, iy + 1]) - \
                                             (v_mem_np[ix - 1, iy - 1] + 2 * v_mem_np[ix - 1, iy] + v_mem_np[ix - 1, iy + 1])
                                                 
                                    # 🌟 UPGRADE: 3x3 Sobel Filter for the Y-Gradient
                                    map_dy = (v_mem_np[ix - 1, iy + 1] + 2 * v_mem_np[ix, iy + 1] + v_mem_np[ix + 1, iy + 1]) - \
                                             (v_mem_np[ix - 1, iy - 1] + 2 * v_mem_np[ix, iy - 1] + v_mem_np[ix + 1, iy - 1])
                                        
                                    norm = np.sqrt(map_dx**2 + map_dy**2)
                                        
                                    # Slightly higher threshold because Sobel sums 6 pixels instead of 2
                                    if norm > 0.15: 
                                        global_nx = map_dx / norm
                                        global_ny = map_dy / norm
                                        
                                        # 🌟 THE BUG FIX: Rotate Global Map Normals into the Node's Local Frame!
                                        nx = global_nx * np.cos(-matched_th) - global_ny * np.sin(-matched_th)
                                        ny = global_nx * np.sin(-matched_th) + global_ny * np.cos(-matched_th)
                                    else:
                                        # Fallback to ray angle (This was already safely in the Local Frame!)
                                        ray_ang = tof_angles[i] + lc_offset_th
                                        nx, ny = np.cos(ray_ang), np.sin(ray_ang)
                                else:
                                    # Out of bounds fallback
                                    ray_ang = tof_angles[i] + lc_offset_th
                                    nx, ny = np.cos(ray_ang), np.sin(ray_ang)
                                        
                                normals_x.append(nx)
                                normals_y.append(ny)

                            # 5. The Superior A-Matrix (Point-to-Plane)
                            A = np.stack([normals_x, normals_y], axis=1)
                            
                            # Convert scalar lengths to local Cartesian points (X, Y)
                            P_mx = mem_tof * np.cos(tof_angles)
                            P_my = mem_tof * np.sin(tof_angles)
                            
                            # 🌟 THE CURE FOR THE SWEEPING BEAM 🌟
                            # Pre-rotate the current rays by the known heading error!
                            P_cx = curr_tof * np.cos(tof_angles + lc_offset_th)
                            P_cy = curr_tof * np.sin(tof_angles + lc_offset_th)
                            
                            # The true perpendicular Cartesian error (Dot Product of distance and wall normal)
                            b = (P_mx - P_cx) * np.array(normals_x) + (P_my - P_cy) * np.array(normals_y)
                                
                            # =================================================================
                            # 🌟 THE DEPTH DISCONTINUITY GATE (Outlier Rejection)
                            # =================================================================
                            # 1. We can now safely restore the 7.4m long-range vision!
                            # 2. Obstacle Limit: If the perpendicular error is > 0.60m, it hit a 
                            #    new obstacle (like a chair). Drop this specific ray from the math!
                            valid_mask = (curr_tof < 7.4) & (mem_tof < 7.4) & (np.abs(b) < 0.20)
                                
                            # 🌟 THE FALLBACK GATE
                            if np.sum(valid_mask) >= 2:
                                A_valid = A[valid_mask]
                                b_valid = b[valid_mask]
                                
                                AtA = A_valid.T @ A_valid
                                
                                # =================================================================
                                # 🌟 THE SINGULARITY FIX: The True Aperture Gate
                                # =================================================================
                                # Calculate the determinant of the un-damped matrix.
                                # If it's near zero, all valid rays hit parallel walls.
                                # We cannot solve for both X and Y! Abort!
                                det_AtA = np.linalg.det(AtA)
                                if det_AtA < 0.25:
                                    print(f"\n ⚠️ Aborted: Matrix Singularity! Det(AtA) = {det_AtA:.3f}. Not enough geometric cross-constraints.")
                                    matched_node = None
                                    continue
                                
                                # If the geometry is structurally sound, proceed with solving!
                                Atb = A_valid.T @ b_valid
                                
                                # Damped Least Squares (Tikhonov Regularization)
                                lambda_damp = 0.05  
                                AtA_damped = AtA + lambda_damp * np.eye(2)
                                offset_xy = np.linalg.solve(AtA_damped, Atb)
                                
                                lc_offset_x = float(np.clip(offset_xy[0], -0.75, 0.75))
                                lc_offset_y = float(np.clip(offset_xy[1], -0.75, 0.75))
                                
                                conc_p   = float(debug_gates['Conc_Place'][0])
                                conc_r   = float(debug_gates['Conc_Ring'][0])
                                w_pos = (maturity * conc_p) * 0.2 * BASE_LC_FACTOR
                                w_th  = (maturity * conc_r) * 0.15 * BASE_LC_FACTOR
                                
                                print(f"\n\n 💥 LOOP CLOSURE SNAP (Node {current_node_id} -> Node {matched_node})!")
                                print(f"  ↳ CANN Belief : X={cx:.2f}m, Y={cy:.2f}m, Th={cth:.2f} rad")
                                print(f"  ↳ Memory Node : X={matched_x:.2f}m, Y={matched_y:.2f}m, Th={matched_th:.2f} rad")
                                print(f"  ↳ Raw ToF Now : L={curr_tof[0]:.3f}m, C={curr_tof[1]:.3f}m, R={curr_tof[2]:.3f}m")
                                print(f"  ↳ Raw ToF Mem : L={mem_tof[0]:.3f}m, C={mem_tof[1]:.3f}m, R={mem_tof[2]:.3f}m")
                                print(f"  ↳ Calc'd Shift: dX={lc_offset_x:.3f}m, dY={lc_offset_y:.3f}m, dTh={lc_offset_th:.3f} rad")
                                print(f"  ↳ Delta (Err) : dX={(cx-matched_x):.2f}m, dY={(cy-matched_y):.2f}m, dTh={wrap_angle(cth-matched_th):.2f} rad")
                                print(f"  ↳ Tension Wgt : W_Pos={w_pos:.2f}, W_Th={w_th:.2f}\n")

                                # =================================================================
                                # 🌟 MENTAL ROTATION & PHASE 2 PLASTICITY
                                # =================================================================
                                # Extract specific Place & Ring IDs
                                p_idx = int(peak_idx_place[0])
                                r_idx = int(debug_gates['Peak_Ring'][0])

                                # Grab the current live camera frames
                                current_csnn = np.array(debug_gates["Debug_Input_CSNN"][0])
                                current_stdp = np.array(debug_gates["Debug_Input_STDP"][0])

                                # Calculate the integer pixel shift (invert it to align with memory)
                                pixel_shift_int = int(-np.round(pixel_shift))
                                
                                aligned_csnn = np.zeros_like(current_csnn)
                                aligned_stdp = np.zeros_like(current_stdp)
                                fov_mask = np.zeros_like(current_csnn)
                                
                                # Perform the Zero-Padded Linear Shift
                                if pixel_shift_int > 0:
                                    aligned_csnn[pixel_shift_int:] = current_csnn[:-pixel_shift_int]
                                    aligned_stdp[pixel_shift_int:] = current_stdp[:-pixel_shift_int]
                                    fov_mask[pixel_shift_int:] = 1.0
                                elif pixel_shift_int < 0:
                                    aligned_csnn[:pixel_shift_int] = current_csnn[-pixel_shift_int:]
                                    aligned_stdp[:pixel_shift_int] = current_stdp[-pixel_shift_int:]
                                    fov_mask[:pixel_shift_int] = 1.0
                                else:
                                    aligned_csnn = current_csnn.copy()
                                    aligned_stdp = current_stdp.copy()
                                    fov_mask[:] = 1.0
                                    
                                # Trigger the Masked Hebbian Update inside the SNN!
                                # (We pass them as JAX arrays with a batch dimension)
                                system_cl.place_state = system_cl.place.apply_post_relaxation_update(
                                    system_cl.place_state,
                                    jnp.array([p_idx]), 
                                    jnp.array([r_idx]),
                                    jnp.array([aligned_csnn]),
                                    jnp.array([aligned_stdp]),
                                    jnp.array([fov_mask]),
                                    ring_lr=0.05
                                )
                                print(f"  ↳ 🧠 Phase 2 Plasticity: Applied 5% Hebbian update to Node {matched_node} memory with a {pixel_shift_int}px mental rotation!")

                                # =================================================================
                                # 🌟 CEREBELLAR CALIBRATION & PHASE 3 PLASTICITY
                                # =================================================================
                                # 1. How much heading error accumulated since the last correction?
                                accumulated_heading_error = wrap_angle((cth - matched_th) - lc_offset_th)
                                
                                # 2. Convert steps to seconds
                                time_elapsed_sec = steps_since_last_lc * DT
                                
                                # 🌟 THE BUG FIX: Prevent the Divide-by-Zero Explosion!
                                # Only learn if we have been exploring blindly for at least 5 seconds.
                                # This ignores consecutive frame snaps that cause hallucinatory drift rates.
                                if time_elapsed_sec > 2.0:
                                    # 3. Teach the Cerebellum!
                                    system_cl.calibrate_cerebellum(accumulated_heading_error, time_elapsed_sec)
                                    print(f"  ↳ 🧠 Phase 3 Plasticity: Cerebellum learned! Current Hardware Bias = {system_cl.learned_omega_bias:.6f} rad/s")
                                else:
                                    print(f"  ↳ 🧠 Phase 3 Plasticity: Skipped. (Time elapsed {time_elapsed_sec:.1f}s too short, preventing bias explosion).")

                                # 4. Reset the drift timer to 0!
                                steps_since_last_lc = 0

                                # ==================================================
                                # 🌟 AUTOMATIC DEBUG PLOTTING (UPGRADED 4x2 GRID)
                                # (Leave your entire debug plotting block exactly as it is here)
                                # ==================================================
                                if "Debug_Input_CSNN" in debug_gates:
                                    pass # (Your plotting code remains unchanged here)

                                # 🌟 THE APPEND FIX: Only add the spring if we reach this point!
                                loop_closures.append([matched_node, current_node_id])
                                loop_offsets_list.append([lc_offset_x, lc_offset_y, lc_offset_th])
                                loop_weights_list.append([w_pos, w_th])

                                # 🌟 THE REFRACTORY FIX: Go to sleep for 20 frames (1 second)
                                # This prevents the "Black Hole" graph explosion!
                                lc_refractory_timer = 20

                            # 🌟 THE ABORT: If we don't have 2 valid lasers, throw the match in the trash!
                            else:
                                print(f"\n ⚠️ Aborted Loop Closure ({current_node_id} -> {matched_node}): ToF lasers out of range (>7.4m).")

                            # ==================================================
                            # 🌟 AUTOMATIC DEBUG PLOTTING (UPGRADED 4x2 GRID)
                            # ==================================================
                            if "Debug_Input_CSNN" in debug_gates:
                                # 1. Extract Camera & ToF Inputs
                                input_csnn = np.array(debug_gates["Debug_Input_CSNN"][0])
                                input_stdp = np.array(debug_gates["Debug_Input_STDP"][0])
                                input_tof  = np.array(debug_gates["Debug_Input_ToF"][0]) # 🌟 NEW
                                
                                # 2. Extract Place Cell Memories
                                mem_csnn_place = np.array(debug_gates["Debug_Mem_CSNN"][0])
                                mem_stdp_place = np.array(debug_gates["Debug_Mem_STDP"][0])
                                mem_tof_place  = np.array(debug_gates["Debug_Mem_ToF"][0]) # 🌟 NEW
                                i_place = np.array(debug_gates["Debug_I_Place"][0])
                                
                                # 3. Extract Ring Cell Memories
                                mem_csnn_ring = np.array(debug_gates["Debug_Mem_CSNN_Ring"][0])
                                mem_stdp_ring = np.array(debug_gates["Debug_Mem_STDP_Ring"][0])
                                mem_tof_ring  = np.array(debug_gates["Debug_Mem_ToF_Ring"][0]) # 🌟 NEW
                                i_ring  = np.array(debug_gates["Debug_I_Ring"][0]) 
                                
                                match_score = float(debug_gates["Raw_Match"][0])
                                
                                # 🌟 Build a taller 4x2 side-by-side layout
                                fig_debug, axs = plt.subplots(4, 2, figsize=(18, 13), gridspec_kw={'hspace': 0.5, 'wspace': 0.2})
                                
                                # ── LEFT COLUMN: PLACE CELL (WHERE) ──
                                # ROW 1: CSNN
                                axs[0, 0].plot(input_csnn, label="Camera Reality", color="#3498DB", lw=2)
                                axs[0, 0].plot(mem_csnn_place, label="Place Memory", color="#E67E22", linestyle="--", lw=2)
                                axs[0, 0].set_title(f"CSNN Place Anchor | Match: {match_score:.2f}", fontweight='bold')
                                axs[0, 0].legend(loc="upper right", fontsize=8); axs[0, 0].grid(alpha=0.3, linestyle="--")
                                
                                # ROW 2: ToF (Depth)
                                axs[1, 0].plot(input_tof, label="ToF Reality", color="#1ABC9C", lw=2)
                                axs[1, 0].plot(mem_tof_place, label="ToF Memory", color="#E74C3C", linestyle="--", lw=2)
                                axs[1, 0].set_title("ToF Depth Population Code (192 dims)", fontweight='bold')
                                axs[1, 0].legend(loc="upper right", fontsize=8); axs[1, 0].grid(alpha=0.3, linestyle="--")

                                # ROW 3: STDP
                                axs[2, 0].plot(input_stdp, label="Camera Reality", color="#9B59B6", lw=2)
                                axs[2, 0].plot(mem_stdp_place, label="Place Memory", color="#2ECC71", linestyle="--", lw=2)
                                axs[2, 0].set_title("STDP Place Plasticity", fontweight='bold')
                                axs[2, 0].legend(loc="upper right", fontsize=8); axs[2, 0].grid(alpha=0.3, linestyle="--")
                                
                                # ROW 4: Soma
                                axs[3, 0].plot(i_place, color="#E74C3C", lw=2)
                                peak_idx = np.argmax(i_place) 
                                axs[3, 0].axvline(peak_idx, color='gold', linestyle='--', lw=2, label=f"Winning Place ({peak_idx})")
                                axs[3, 0].set_title("Spatial Soma Activation", fontweight='bold')
                                axs[3, 0].legend(loc="upper right", fontsize=8); axs[3, 0].grid(alpha=0.3, linestyle="--")

                                # ── RIGHT COLUMN: RING CELL (WHICH WAY) ──
                                # ROW 1: CSNN
                                axs[0, 1].plot(input_csnn, label="Camera Reality", color="#3498DB", lw=2)
                                axs[0, 1].plot(mem_csnn_ring, label="Ring Memory", color="#E67E22", linestyle="--", lw=2)
                                axs[0, 1].set_title(f"CSNN Ring Anchor (Conjunctive)", fontweight='bold')
                                axs[0, 1].legend(loc="upper right", fontsize=8); axs[0, 1].grid(alpha=0.3, linestyle="--")
                                
                                # ROW 2: ToF (Depth)
                                axs[1, 1].plot(input_tof, label="ToF Reality", color="#1ABC9C", lw=2)
                                axs[1, 1].plot(mem_tof_ring, label="Ring Memory", color="#E74C3C", linestyle="--", lw=2)
                                axs[1, 1].set_title("ToF Ring Geometry (Conjunctive)", fontweight='bold')
                                axs[1, 1].legend(loc="upper right", fontsize=8); axs[1, 1].grid(alpha=0.3, linestyle="--")

                                # ROW 3: STDP
                                axs[2, 1].plot(input_stdp, label="Camera Reality", color="#9B59B6", lw=2)
                                axs[2, 1].plot(mem_stdp_ring, label="Ring Memory", color="#2ECC71", linestyle="--", lw=2)
                                axs[2, 1].set_title("STDP Ring Plasticity (Conjunctive)", fontweight='bold')
                                axs[2, 1].legend(loc="upper right", fontsize=8); axs[2, 1].grid(alpha=0.3, linestyle="--")
                                
                                # ROW 4: Soma
                                axs[3, 1].plot(i_ring, color="#F39C12", lw=2)
                                peak_ring = np.argmax(i_ring)
                                axs[3, 1].axvline(peak_ring, color='blue', linestyle='--', lw=2, label=f"Winning Ring ({peak_ring})")
                                axs[3, 1].set_title("Heading Soma Activation", fontweight='bold')
                                axs[3, 1].legend(loc="upper right", fontsize=8); axs[3, 1].grid(alpha=0.3, linestyle="--")
                                
                                # Save and destroy
                                debug_filename = f"debug_csnn_step.png"
                                plt.savefig(debug_filename, bbox_inches='tight', dpi=100)
                                plt.close(fig_debug)
                                print(f"  ↳ 💾 Auto-saved multi-stream brain scan to {debug_filename}\n")
                            else:
                                print("\n")
                            
                            p_jax = jnp.array(graph_poses)
                            o_jax = jnp.array(graph_odom_edges)
                            
                            MAX_NODES = 1000
                            WINDOW_SIZE = 800
                            
                            N_total = len(graph_poses)
                            start_idx = max(0, N_total - WINDOW_SIZE)
                            
                            active_nodes = graph_poses[start_idx:]
                            num_active = len(active_nodes)
                            active_odom = graph_odom_edges[start_idx:] if start_idx > 0 else graph_odom_edges
                                
                            mapped_closures, mapped_offsets, mapped_weights = [], [], []
                            frozen_anchors = []
                            
                            recent_lcs = loop_closures[-MAX_LOOPS:]
                            recent_offs = loop_offsets_list[-MAX_LOOPS:]
                            recent_wgts = loop_weights_list[-MAX_LOOPS:]
                            
                            for i in range(len(recent_lcs)):
                                target_id, current_id = recent_lcs[i]
                                if current_id < start_idx: 
                                    continue 
                                    
                                mapped_current = current_id - start_idx
                                if target_id >= start_idx:
                                    mapped_target = target_id - start_idx
                                else:
                                    mapped_target = num_active + len(frozen_anchors)
                                    frozen_anchors.append(graph_poses[target_id])
                                    
                                mapped_closures.append([mapped_target, mapped_current])
                                mapped_offsets.append(recent_offs[i])
                                mapped_weights.append(recent_wgts[i])
                                
                            padded_poses = np.zeros((MAX_NODES, 3), dtype=np.float32)
                            padded_poses[:num_active] = np.array(active_nodes)
                            if frozen_anchors:
                                padded_poses[num_active:num_active+len(frozen_anchors)] = np.array(frozen_anchors)
                                
                            padded_odom = np.zeros((MAX_NODES - 1, 3), dtype=np.float32)
                            if num_active > 1:
                                padded_odom[:num_active-1] = np.array(active_odom)
                                
                            odom_mask = np.zeros(MAX_NODES - 1, dtype=np.float32)
                            if num_active > 1:
                                odom_mask[:num_active-1] = 1.0
                                
                            is_frozen = np.zeros(MAX_NODES, dtype=bool)
                            is_frozen[0] = True 
                            if frozen_anchors:
                                is_frozen[num_active:num_active+len(frozen_anchors)] = True
                                
                            lc_padded = np.zeros((MAX_LOOPS, 2), dtype=np.int32)
                            lc_offsets_padded = np.zeros((MAX_LOOPS, 3), dtype=np.float32)
                            lc_weights_padded = np.zeros((MAX_LOOPS, 2), dtype=np.float32)
                            lc_mask = np.zeros(MAX_LOOPS, dtype=np.float32)
                            
                            num_lc = min(len(mapped_closures), MAX_LOOPS)
                            if num_lc > 0:
                                lc_padded[:num_lc] = np.array(mapped_closures[-num_lc:])
                                lc_offsets_padded[:num_lc] = np.array(mapped_offsets[-num_lc:])
                                lc_weights_padded[:num_lc] = np.array(mapped_weights[-num_lc:])
                                lc_mask[:num_lc] = 1.0
                            
                            relaxed_p = relax_graph(
                                jnp.array(padded_poses), jnp.array(padded_odom), jnp.array(odom_mask),
                                jnp.array(lc_padded), jnp.array(lc_offsets_padded), 
                                jnp.array(lc_weights_padded), jnp.array(lc_mask),
                                jnp.array(is_frozen)
                            )
                            
                            relaxed_active = np.array(relaxed_p[:num_active]).tolist()
                            graph_poses[start_idx:] = relaxed_active
                            
                            corr_x, corr_y, corr_th = graph_poses[-1]
                            system_cl.initialize_from_gt(jnp.array([[corr_x, corr_y]]), jnp.array([corr_th]))
                            last_kf_cann = (corr_x, corr_y, corr_th)
                            
                            local_dx, local_dy, local_dth = 0.0, 0.0, 0.0

                # 🌟 FIX 1: Database Poisoning!
                # Calculate exactly which Place Cell the physical CANN is standing on right now.
                grid_x = np.clip(int(cx / (ROOM_W / MAP_SIZE)), 0, MAP_SIZE - 1)
                grid_y = np.clip(int(cy / (ROOM_H / MAP_SIZE)), 0, MAP_SIZE - 1)
                true_physical_place_id = grid_y * MAP_SIZE + grid_x

                # Save the new memory to the True Physical Folder, NEVER the hallucinated folder!
                place_to_node[true_physical_place_id].append(current_node_id)

                # 🌟 V4.7 FIX: This must be outside the loop closure block!
                # If we drop ANY keyframe, we are perfectly on the anchor. Zero the offset!
                local_dx, local_dy, local_dth = 0.0, 0.0, 0.0

            # --- Outside the keyframe block ---
            active_node_id = len(graph_poses) - 1
            history['frame_to_node'].append(active_node_id)
            history['live_offsets'].append((local_dx, local_dy, local_dth))
            
            # 🌟 V4.3 FIX B: Reconstruct the ultra-smooth live trajectory (The FLAWLESS Rubber Band!)
            smooth_cl_poses = []
            
            # 1. Find the start index of every node in the history array
            node_start_indices = [0] * len(graph_poses)
            for i, nid in enumerate(history['frame_to_node']):
                if i == 0 or history['frame_to_node'][i-1] != nid:
                    node_start_indices[nid] = i

            # 2. Reconstruct the path with curved stretching
            for i, nid in enumerate(history['frame_to_node']):
                rx, ry, rth = graph_poses[nid]
                
                # Get the raw, beautifully curved IMU step for this specific frame
                ldx, ldy, ldth = history['live_offsets'][i]
                gdx_raw = ldx * np.cos(rth) - ldy * np.sin(rth)
                gdy_raw = ldx * np.sin(rth) + ldy * np.cos(rth)
                
                # If this node has a "next" node, smoothly stretch the curve to meet it
                if nid + 1 < len(graph_poses):
                    nx, ny, nth = graph_poses[nid + 1]
                    
                    start_i = node_start_indices[nid]
                    end_i = node_start_indices[nid + 1]
                    steps = max(1, end_i - start_i)
                    alpha = (i - start_i) / steps
                    
                    # 🌟 THE ZIG-ZAG FIX: Use the exact Odometry Edge to calculate the gap!
                    # This completely bypasses the 1-frame boundary missing-delta bug.
                    edge_dx, edge_dy, edge_dth = graph_odom_edges[nid]
                    
                    tip_gdx = edge_dx * np.cos(rth) - edge_dy * np.sin(rth)
                    tip_gdy = edge_dx * np.sin(rth) + edge_dy * np.cos(rth)
                    
                    # Calculate the exact "Solver Tension" gap
                    gap_x = nx - (rx + tip_gdx)
                    gap_y = ny - (ry + tip_gdy)
                    gap_th = wrap_angle(nth - (rth + edge_dth))
                    
                    # Add the raw curved IMU step + a proportional percentage of the tension!
                    final_x = rx + gdx_raw + (alpha * gap_x)
                    final_y = ry + gdy_raw + (alpha * gap_y)
                    final_th = wrap_angle(rth + ldth + (alpha * gap_th))
                    
                    smooth_cl_poses.append(np.array([final_x, final_y, final_th]))
                    
                # If this is the active tail, just use the raw curve!
                else:
                    smooth_cl_poses.append(np.array([rx + gdx_raw, ry + gdy_raw, wrap_angle(rth + ldth)]))
                
            history['cl_pose'] = smooth_cl_poses
            history['raw_cann'].append(np.array(pose_cl[0]))
            
            # 🌟 V4.3 FIX C: Use the SMOOTH pose for the map drawing/raycasting!
            rel_cx, rel_cy, rel_cth = smooth_cl_poses[-1]

            history['gt_pos'].append(gt_pos); history['gt_th'].append(gt_th)
            history['imu_pos'].append([x_imu, y_imu]); history['imu_th'].append(th_imu)
            history['ol_pose'].append(np.array(pose_ol[0]))
            history['conf'].append(float(is_confident[0]))
            
            # 🌟 NEW: Track 1D Ring Memory & Ring CANN Arrays
            history['pc_act'].append(np.array(r_place[0]))
            history['ring_mem_act'].append(np.array(r_ring[0])) 
            history['cann_act'].append(np.array(system_cl.pose.get_state_flat()[0]))
            history['ring_cann_act'].append(np.array(system_cl.pose.get_ring_activity()[0])) 
            history['intensities'].append(intensity)

            hit_idx, free_idx = get_ray_indices(
                rel_cx, rel_cy, rel_cth, 
                tof_t, tof_angles, res=sog.res, grid_size=sog.grid_w, offset_m=sog.offset_m
            )
            sog_state = sog.update(sog_state, jnp.array(hit_idx), jnp.array(free_idx))

            if step % 15 == 0:
                d = debug_gates
                
                # 🌟 CHECK IF THE ROBOT IS BLINKING
                # kin_t[2] is the current rotational velocity (omega)
                is_learning = abs(kin_t[2]) < system_cl.place.dynamic_saccade_thresh
                
                if is_learning:
                    learning_indicator.set_text('[ PLASTICITY: ON ]')
                    learning_indicator.set_color('lime')
                else:
                    learning_indicator.set_text('[ PLASTICITY: OFF ]')
                    learning_indicator.set_color('#E74C3C') # Bright Red

                # Extract the gating values
                snap = bool(d['Final_Conf'][0])
                maturity = float(d['Maturity_Lvl'][0])
                
                # 🌟 Extract the raw biological metrics
                vis_act = float(d['Raw_Vis_Act'][0])
                match_val = float(d['Raw_Match'][0])
                c_place = float(d['Conc_Place'][0])
                c_ring = float(d['Conc_Ring'][0])
                tof_score = float(d['ToF_Score'][0]) 
                
                # 🌟 Upgraded Terminal Status Line
                print(f"\r--- Step {step:04d} | Learn: {bool(is_learning)} | Snap: {snap} | VisAct: {vis_act:.2f} | Match: {match_val:.2f} | ToFMatch: {tof_score:.2f} | Conc(P/R): {c_place:.2f}/{c_ring:.2f} | Mat: {maturity:.2f} ---        ", end="", flush=True)

                gt_arr = np.array(history['gt_pos'])
                gt_traj.set_data(gt_arr[:, 0], gt_arr[:, 1])
                gt_head.set_data([gt_pos[0]], [gt_pos[1]])
                
                cl_arr = np.array(history['cl_pose'])
                ol_arr = np.array(history['ol_pose'])
                imu_arr = np.array(history['imu_pos'])
                
                # 🌟 UPGRADE: Live Umeyama Alignment!
                # Continuously align the entire history of Closed-Loop and Open-Loop to Ground Truth
                R_cl, t_cl = get_optimal_alignment_2d(cl_arr[:, :2], gt_arr)
                R_ol, t_ol = get_optimal_alignment_2d(ol_arr[:, :2], gt_arr)
                
                aligned_cl_path = (R_cl @ cl_arr[:, :2].T).T + t_cl
                aligned_ol_path = (R_ol @ ol_arr[:, :2].T).T + t_ol
                
                # Extract the dynamic rotation angle for the FOV and SOG Map transform
                delta_theta = np.arctan2(R_cl[1, 0], R_cl[0, 0])
                
                live_traj.set_data(aligned_cl_path[:, 0], aligned_cl_path[:, 1])
                
                recent_cl = aligned_cl_path[-100:] 
                current_live_traj.set_data(recent_cl[:, 0], recent_cl[:, 1])
                
                aligned_cx, aligned_cy = aligned_cl_path[-1, 0], aligned_cl_path[-1, 1]
                aligned_cth = rel_cth + delta_theta
                live_head.set_data([aligned_cx], [aligned_cy])
                
                sog_img.set_data(np.array(sog_state.v_mem).T) 
                trans_data = mtransforms.Affine2D().rotate(delta_theta).translate(t_cl[0], t_cl[1]) + ax_map.transData
                sog_img.set_transform(trans_data)

                # 🌟 NEW PANEL 4 MATH: Use raw global coordinates! 
                # This stops Umeyama from retroactively changing past errors.
                err_imu = np.sqrt(np.sum((imu_arr[:, :2] - gt_arr)**2, axis=1))
                err_ol = np.sqrt(np.sum((ol_arr[:, :2] - gt_arr)**2, axis=1))
                err_cl = np.sqrt(np.sum((cl_arr[:, :2] - gt_arr)**2, axis=1))
                
                t_axis = np.arange(len(err_cl)) * DT
                line_err_imu.set_data(t_axis, err_imu)
                line_err_ol.set_data(t_axis, err_ol)
                line_err_cl.set_data(t_axis, err_cl)
                
                # Dynamically scale the Error plot's X and Y axes
                if len(t_axis) > 0:
                    ax_err.set_xlim(0, max(10.0, t_axis[-1]))
                    ax_err.set_ylim(0, max(0.5, np.max(err_imu) * 1.1))
                
                # --- The R_fov and fov_poly_gt drawing code remains the same below this ---
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
                
                # Update 2D Overlapped Brain Maps
                pc_act = np.array(r_place[0]).reshape(MAP_SIZE, MAP_SIZE)
                brain_img.set_data(pc_act / (pc_act.max() + 1e-8))
                # brain_img.set_data(pc_act)
                
                cann_act = history['cann_act'][-1].reshape(CANN_SIZE, CANN_SIZE)
                cann_norm = cann_act / (cann_act.max() + 1e-8)
                
                # 🌟 THE GUI FIX: Build the RGBA transparency explicitly. No more np.nan!
                cann_rgba = plt.cm.cool(cann_norm)                     # Apply the Cyan/Magenta color map
                cann_rgba[..., 3] = np.where(cann_norm > 0.15, 0.8, 0.0) # Set Alpha channel (0.8 = visible, 0.0 = clear)
                cann_img.set_data(cann_rgba)
                
                # Update 1D Overlapped Ring Plots
                ring_mem = history['ring_mem_act'][-1].flatten()
                line_ring_mem.set_ydata(ring_mem / (ring_mem.max() + 1e-8))

                ring_cann = history['ring_cann_act'][-1].flatten()
                line_ring_cann.set_ydata(ring_cann / (ring_cann.max() + 1e-8))
                
                # Align both overlaps to the same rotation
                tr_brain = mtransforms.Affine2D().translate(-MAP_SIZE/2, -MAP_SIZE/2).rotate(delta_theta).translate(MAP_SIZE/2, MAP_SIZE/2) + ax_2d.transData
                brain_img.set_transform(tr_brain)
                
                tr_cann = mtransforms.Affine2D().translate(-CANN_SIZE/2, -CANN_SIZE/2).rotate(delta_theta).translate(CANN_SIZE/2, CANN_SIZE/2) + ax_2d.transData
                cann_img.set_transform(tr_cann)
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                # ==========================================================
                # 🌟 NEW: CAPTURE FRAME TO GIF
                # ==========================================================
                buf = io.BytesIO()
                # 🌟 FIX 1: Save as 'png' so it builds a proper image header
                fig.savefig(buf, format='png', dpi=fig.dpi) 
                buf.seek(0)
                # 🌟 FIX 2: Explicitly tell imageio to expect a PNG format
                frame = imageio.v3.imread(buf, extension=".png") 
                gif_writer.append_data(frame)
                buf.close()
                
            step += 1

    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n 🛑 Halted! Simulated {step} steps in {elapsed:.1f}s.")
        
        # ==========================================================
        # 🌟 NEW: CLOSE AND SAVE THE GIF
        # ==========================================================
        gif_writer.close()
        print(f" 💾 Saved live animation to {gif_filename}")
        
        print(" 💾 Compiling logs and generating final PNG plots...")
        plt.ioff()
        plt.close(fig)

    # --- Compile & GLOBALLY ALIGN final results dictionary ---
    # 🌟 THE CTRL+C FIX: Find the shortest array and truncate everything else!
    min_len = min(len(history['gt_pos']), len(history['imu_pos']), 
                  len(history['ol_pose']), len(history['cl_pose']))
    
    gt_arr = np.array(history['gt_pos'][:min_len])
    imu_arr = np.array(history['imu_pos'][:min_len])
    ol_arr = np.array(history['ol_pose'][:min_len])
    cl_arr = np.array(history['cl_pose'][:min_len])
    th_gt = np.array(history['gt_th'][:min_len])
    th_imu = np.array(history['imu_th'][:min_len])
    
    # Sync the confidence array and step counter to match
    history['conf'] = history['conf'][:min_len]
    step = min_len

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
        'x_cl_raw': cl_arr[None, :, 0], 'y_cl_raw': cl_arr[None, :, 1], 'th_cl_raw': cl_arr[None, :, 2],
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
    # 🌟 THE FIX: Measure the actual array length in case of Ctrl+C mismatch!
    T = results['x_gt'].shape[1]
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
    print(f"\n 🗺️ Rendering Global Spiking Occupancy Map (Egocentric 'Latest Point' Pin) to {save_path}...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Extract Latest Poses (The Anchors)
    gt_x_end = results['x_gt'][0, -1]
    gt_y_end = results['y_gt'][0, -1]
    gt_th_end = results['th_gt'][0, -1]
    
    snn_x_end = results['x_cl_raw'][0, -1]
    snn_y_end = results['y_cl_raw'][0, -1]
    snn_th_end = results['th_cl_raw'][0, -1]
    
    # 2. Calculate the "Pin at Latest" Transform
    # Rotate the SNN map so its final heading perfectly matches the GT final heading
    delta_th = gt_th_end - snn_th_end
    R_align = np.array([
        [np.cos(delta_th), -np.sin(delta_th)],
        [np.sin(delta_th),  np.cos(delta_th)]
    ])
    
    # Translate the SNN map so its final X,Y perfectly matches the GT final X,Y
    t_align = np.array([gt_x_end, gt_y_end]) - R_align @ np.array([snn_x_end, snn_y_end])
    
    # 3. Plot the Native SOG Image, Twisted by the Anchor Transform
    sog_grid = results['sog_grid']
    offset_m = 10.0
    map_size_m = 30.0
    extent = [-offset_m, map_size_m - offset_m, -offset_m, map_size_m - offset_m]
    
    # Force Matplotlib to rotate the SOG matrix using our calculated transform
    trans_data = mtransforms.Affine2D().rotate(delta_th).translate(t_align[0], t_align[1]) + ax.transData
    
    ax.imshow(sog_grid.T, cmap='magma', origin='lower', 
              extent=extent, vmin=-0.2, vmax=1.0, transform=trans_data)
    
    # 4. Transform the entire SNN history path
    raw_snn_pts = np.stack([results['x_cl_raw'][0], results['y_cl_raw'][0]], axis=1)
    pinned_snn_pts = (R_align @ raw_snn_pts.T).T + t_align
    
    # 5. Plot Ground Truth Room (Clean and straight!)
    if results['obstacles'] is not None:
        for o in results['obstacles']:
            w, h = float(o[2]-o[0]), float(o[3]-o[1])
            ax.add_patch(Rectangle((float(o[0]), float(o[1])), w, h, 
                                   facecolor='none', edgecolor='cyan', lw=1.5, ls='--', alpha=0.5))
                                   
    ax.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H, facecolor='none', edgecolor='cyan', lw=1.5, ls='--'))
    
    # Plot the Trajectories
    ax.plot(results['x_gt'][0], results['y_gt'][0], color='#3498DB', lw=1.5, alpha=0.6, label='Ground Truth')
    ax.plot(pinned_snn_pts[:, 0], pinned_snn_pts[:, 1], color='lime', lw=2.0, alpha=0.9, label='SNN (Pinned to Current)')
    
    # Plot a giant gold star to represent the "Now" Anchor
    ax.plot(gt_x_end, gt_y_end, marker='*', color='gold', ms=18, markeredgecolor='red', label='Current Position (Anchor)')

    ax.set_aspect('equal')
    ax.set_xlim(-2, ROOM_W + 2); ax.set_ylim(-2, ROOM_H + 2)
    ax.set_title('Phase 3: SOG (Egocentric "Latest Point" Pin)', fontsize=14, fontweight='bold')
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