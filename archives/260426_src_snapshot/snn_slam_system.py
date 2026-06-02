#!/usr/bin/env python3
"""
snn_slam_system.py — Neuromorphic SLAM Orchestrator (v3)

Integrates three biological modules into a closed-loop navigation system:

  1. VisionSTDP   (256 feature neurons) — event-based edge features
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

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import time, sys, os


# ============================================================================
# WORKSPACE
# ============================================================================
sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')

from src.sparse_forest import (
    generate_fixed_room_dataset,
    N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H, FOV_DEG,
    VX_RANGE, VY_RANGE, OMEGA_RANGE,
)
from src.snn_vision_stdp import VisionSTDP
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

BATCH_SIZE      = 4       # trajectories in parallel
N_VISION        = 256     # VisionSTDP feature neurons
N_DEPTH         = 8       # ToF Gaussian population coding depth channels
N_FUSED         = N_VISION * N_DEPTH  # 256 × 8 = 2048 fused features

TOF_MIN         = 0.1    # meters
TOF_MAX         = 9.9    # meters
TOF_SIGMA       = 0.5    # Gaussian σ in meters

DRIFT_START     = 80     # step at which drift kicks in
DRIFT_OMEGA     = 0.005  # rad/s artificial yaw drift per timestep

N_TRAJ_SHOW     = 3
SAVE_FIG        = True


# ============================================================================
#  🧠  TOF POPULATION CODER (Gaussian RBF)
# ============================================================================

class ToFPopulationCoder:
    """Convert scalar ToF depth → Gaussian population code."""

    def __init__(self, n_depth=N_DEPTH, tof_min=TOF_MIN, tof_max=TOF_MAX, sigma=TOF_SIGMA):
        self.n_depth  = n_depth
        self.tof_min  = tof_min
        self.tof_max  = tof_max
        self.sigma    = sigma
        self.centers = np.linspace(tof_min, tof_max, n_depth)

    def __call__(self, tof_scalar):
        tof = jnp.asarray(tof_scalar)
        if tof.ndim == 0:
            tof = tof[None]
        B = tof.shape[0]
        diff = tof[:, None] - jnp.array(self.centers)[None, :]
        activations = jnp.exp(-(diff ** 2) / (2 * self.sigma ** 2))
        activations = activations / (activations.max(axis=1, keepdims=True) + 1e-8)
        return activations


# ============================================================================
#  🧠  MASTER SYSTEM CLASS
# ============================================================================

class SNNSLAMSystem:
    """Full neuromorphic SLAM orchestrator with Dual-Key sensor fusion. v3."""

    def __init__(self, key, n_depth=N_DEPTH):
        self.n_depth = n_depth
        self.n_fused = N_VISION * n_depth

        W_cann        = build_2d_cann_weights()
        W_ring        = build_1d_ring_weights()
        W_ring_asym   = build_asymmetric_ring_weights()
        W_cann_asym_x = build_asymmetric_cann_weights_x()
        W_cann_asym_y = build_asymmetric_cann_weights_y()

        k_vision, key = random.split(key)
        self.vision = VisionSTDP(k_vision, n_input=N_PIXELS,
                                  n_hidden=N_VISION, k_wta=20)

        self.tof_coder = ToFPopulationCoder(n_depth=n_depth)

        k_pose, key = random.split(key)
        self.pose = PoseCANN(k_pose, W_cann, W_ring,
                              W_cann_asym_x, W_cann_asym_y, W_ring_asym)

        k_place, key = random.split(key)
        self.place = PlaceCellNetwork(k_place, n_fused=self.n_fused)

        self._initialized = False
        self._step = 0

    def reset(self, B):
        self.vision.reset(B)
        self.pose.reset(B)
        self.place.reset(B)
        self._initialized = False
        self._step = 0

    def initialize_from_gt(self, gt_pos, gt_heading):
        self.pose.initialize_from_gt(gt_pos, gt_heading)
        pose_bump = self.pose.get_state_flat()
        ring_bump = self.pose.get_ring_activity()
        self.place.initialize_from_pose(pose_bump, ring_bump=ring_bump)
        self._initialized = True

    def phase_perception(self, events_t, tof_t):
        vision_spikes, _ = self.vision(events_t, tof_t, learn=True)
        tof_pop = self.tof_coder(tof_t)
        fused = jnp.einsum('bf,bd->bfd', vision_spikes, tof_pop)
        fused = fused.reshape(vision_spikes.shape[0], -1)
        return fused

    def phase_inference(self, fused_features, pose_xy, current_heading_rads):
        I_spatial_place = self.place.compute_spatial_correction(fused_features, pose_xy, current_heading_rads)
        I_spatial_ring  = self.place.compute_ring_correction(fused_features, current_heading_rads)

        blend = self.place.compute_confidence_with_gates(
            fused_features, pose_xy, current_heading_rads
        )

        # ✅ FIX: Match the 2D (B, 1024) shape with a 2D (B, 1) multiplier
        I_gated_place = I_spatial_place * blend[:, None]
        I_gated_ring  = I_spatial_ring  * blend[:, None]

        return I_gated_place, I_gated_ring, blend

    def phase_odometry(self, kin_t, I_gated_place, I_gated_ring, inject_drift=False):
        if inject_drift:
            omega_drift = kin_t[:, 2] + DRIFT_OMEGA
            kin_injected = jnp.stack([kin_t[:, 0], kin_t[:, 1], omega_drift], axis=1)
        else:
            kin_injected = kin_t

        pose_est = self.pose(
            kin_injected,
            map_correction_place=I_gated_place,
            map_correction_ring=I_gated_ring * 0.04,
        )

        # =================================================================
        # 🚀 THE FINAL STEP: FIRE THE CEREBELLUM LEARNING LOOP
        # =================================================================
        # We pass the COMMANDED velocity (kin_injected) and the ACTUAL 
        # resulting pose (x, y, theta) so the Purkinje cells can calculate 
        # the climbing fiber error and update their speed synapses!
        self.pose.update_cerebellum(kin_injected, pose_est[:, :2], pose_est[:, 2])
        # =================================================================

        pose_bump = self.pose.get_state_flat()
        ring_bump = self.pose.get_ring_activity()

        return pose_est, pose_bump, ring_bump

    def phase_mapping(self, fused_features, pose_bump, ring_bump, loop_conf=None):
        # 🟢 FIX: Pass loop_conf down to the place cell network as 'confidence'
        self.place.forward_mapping(fused_features, pose_bump, ring_bump=ring_bump, learn=True, confidence=loop_conf)
        r_place = self.place.get_place_activity_flat()
        r_ring  = self.place.get_ring_activity_flat()
        return r_place, r_ring

    def forward_step(self, events_t, kin_t, tof_t, inject_drift=False):
        fused_features = self.phase_perception(events_t, tof_t)

        pose_xy = self.pose.estimate_position()
        current_heading_rads = self.pose.estimate_heading()

        I_gated_place, I_gated_ring, loop_conf = self.phase_inference(
            fused_features, pose_xy, current_heading_rads
        )

        pose_est, pose_bump, ring_bump = self.phase_odometry(
            kin_t, I_gated_place, I_gated_ring, inject_drift
        )

        # 🟢 FIX: Pass loop_conf into the mapping phase
        r_place, r_ring = self.phase_mapping(fused_features, pose_bump, ring_bump, loop_conf)

        self._step += 1
        return pose_est, r_place, r_ring, loop_conf

    def forward_step_open_loop(self, events_t, kin_t, tof_t, inject_drift=False):
        """Open-loop SNN: pose-CANN odometry WITHOUT place cell corrections."""
        fused_features = self.phase_perception(events_t, tof_t)

        B = fused_features.shape[0]
        I_zero_place = jnp.zeros((B, MAP_SIZE, MAP_SIZE))
        I_zero_ring  = jnp.zeros((B, RING_N))

        pose_est, pose_bump, ring_bump = self.phase_odometry(
            kin_t, I_zero_place, I_zero_ring, inject_drift
        )

        # 🟢 FIX: Open loop has zero confidence (100% novelty), so it learns continuously
        zero_conf = jnp.zeros(B)
        r_place, r_ring = self.phase_mapping(fused_features, pose_bump, ring_bump, loop_conf=zero_conf)

        self._step += 1
        return pose_est, r_place, r_ring

    def get_pc_preferred_locs(self):
        """Return (N_PLACE, 2) place cell preferred locations in world meters."""
        return np.array(self.place.pc_preferred_locs)


# ============================================================================
#  🧮  DATASET
# ============================================================================

def generate_looping_trajectories(key, n_samples=4, time_steps=200,
                                   drift_start=DRIFT_START, drift_omega=DRIFT_OMEGA):
    print(f"\n  🌊 Generating {n_samples} looping trajectories "
          f"({time_steps} steps, drift @ t={drift_start})...")

    events, labels, tof_dists, positions, headings, obstacles, segments, intensities = \
        generate_fixed_room_dataset(key, n_samples, time_steps=time_steps)

    B, T, N = events.shape
    T = min(T, time_steps)
    events = events[:, :T, :]
    tof_dists = tof_dists[:, :T]
    positions = positions[:, :T, :]
    intensities = intensities[:, :T, :]

    # ------------------------------------------------------------------
    # Derive DYNAMIC kinematics from GT positions (Bug 2 fix)
    # GT positions come from the B-spline -- use them to get proper
    # time-varying omega, heading, and body-frame velocities.
    # ------------------------------------------------------------------
    # 1. Extract GT positions and true holonomic headings
    pos_gt = positions[:, :, :2]  
    th_gt = headings              

    # 2. Extract and un-normalize the true kinematics from the labels array
    vx_body = labels[:, :, 0] * abs(VX_RANGE[1])
    vy_body = labels[:, :, 1] * abs(VY_RANGE[1])
    omega_raw = labels[:, :, 2] * abs(OMEGA_RANGE[1])

    # 3. Stack into the kinematic tensor for the SNN
    kin = jnp.stack([vx_body, vy_body, omega_raw], axis=2)

    x_ol = jnp.zeros((B, time_steps)).at[:, 0].set(pos_gt[:, 0, 0])
    y_ol = jnp.zeros((B, time_steps)).at[:, 0].set(pos_gt[:, 0, 1])
    th_ol = jnp.zeros((B, time_steps))
    th_ol = th_ol.at[:, 0].set(th_gt[:, 0])

    for t in range(1, time_steps):
        vx_b = kin[:, t, 0]
        vy_b = kin[:, t, 1]
        omega_b = kin[:, t, 2]
        if t >= drift_start:
            omega_b = omega_b + drift_omega
        cos_h = jnp.cos(th_ol[:, t-1])
        sin_h = jnp.sin(th_ol[:, t-1])
        vx_w = vx_b * cos_h - vy_b * sin_h
        vy_w = vx_b * sin_h + vy_b * cos_h
        x_ol = x_ol.at[:, t].set(x_ol[:, t-1] + vx_w * DT)
        y_ol = y_ol.at[:, t].set(y_ol[:, t-1] + vy_w * DT)
        th_ol = th_ol.at[:, t].set((th_ol[:, t-1] + omega_b * DT) % (2 * jnp.pi))

    print(f"     GT bounds: x=[{float(pos_gt[:,:,0].min()):.2f}, "
          f"{float(pos_gt[:,:,0].max()):.2f}]  "
          f"y=[{float(pos_gt[:,:,1].min()):.2f}, "
          f"{float(pos_gt[:,:,1].max()):.2f}]")
    print(f"     Drift: omega_drift={drift_omega:.4f} rad/step starting @ t={drift_start}")

    return {
        'events': np.array(events),
        'intensities': np.array(intensities),  # raw event-camera pixel intensities
        'tof': np.array(tof_dists),
        'kin': np.array(kin),
        'pos_gt': np.array(pos_gt),
        'th_gt': np.array(th_gt),
        'obstacles': np.array(obstacles),
        'labels': np.array(labels),
    }


# ============================================================================
#  📊  EVALUATION
# ============================================================================

def evaluate_system(key, n_samples=BATCH_SIZE, time_steps=TIME_STEPS,
                    drift_start=DRIFT_START):

    data = generate_looping_trajectories(
        key, n_samples=n_samples,
        time_steps=time_steps,
        drift_start=drift_start
    )

    ev     = data['events']
    intensities = data['intensities']  # (B, T, N_PIXELS) raw intensities
    tof    = data['tof']
    kin    = data['kin']
    pos_gt = data['pos_gt']
    th_gt  = data['th_gt']
    obs    = data['obstacles']
    labels = data['labels']

    B, T = n_samples, time_steps

    print(f"\n 🧠 Initializing Twin SNN SLAM Systems v3 "
          f"(N_FUSED={N_FUSED}, N_DEPTH={N_DEPTH})...")
          
    # 1. Instantiate TWO independent systems
    system_ol = SNNSLAMSystem(random.PRNGKey(42), n_depth=N_DEPTH)
    system_cl = SNNSLAMSystem(random.PRNGKey(43), n_depth=N_DEPTH)
    
    system_ol.reset(B)
    system_cl.reset(B)

    # Use GT heading from spline directly (time-varying omega integrated correctly)
    headings_init = np.array(th_gt[:, 0])   # (B,) -- from true time-varying omega

    # 2. Initialize BOTH systems at GT pose
    system_ol.initialize_from_gt(jnp.array(pos_gt[:, 0]), jnp.array(headings_init))
    system_cl.initialize_from_gt(jnp.array(pos_gt[:, 0]), jnp.array(headings_init))

    # ---- Storage for three trajectory types ----
    x_imu = np.zeros((B, T))   # IMU-only: pure velocity integration
    y_imu = np.zeros((B, T))
    th_imu = np.zeros((B, T))

    x_ol = np.zeros((B, T))    # Open-loop SNN: pose-CANN, NO corrections
    y_ol = np.zeros((B, T))
    th_ol = np.zeros((B, T))

    x_cl = np.zeros((B, T))    # Closed-loop SNN: pose-CANN + corrections
    y_cl = np.zeros((B, T))
    th_cl = np.zeros((B, T))

    loop_conf_log = np.zeros((B, T))

    # ---- Place cell decoded positions (population vector) ----
    pc_x_decoded = np.zeros((B, T))   # decoded x from place cells
    pc_y_decoded = np.zeros((B, T))   # decoded y from place cells
    pc_top_conf  = np.zeros((B, T))   # confidence of top decoded cell
    
    # 🌟 NEW: Array to log raw place cell activity for the GIF
    pc_activity_log = np.zeros((B, T, MAP_SIZE * MAP_SIZE))

    # 🌟 NEW: Array to log raw place cell activity for the GIF
    pc_activity_log = np.zeros((B, T, MAP_SIZE * MAP_SIZE))
    
    # 👇 ADD THIS LINE 👇
    cann_activity_log = np.zeros((B, T, CANN_SIZE * CANN_SIZE))

    # ---- Event camera examples (store raw event frames for batch 0) ----
    ev_examples = []

    print(f"\n  ⚡ Running SNN SLAM simulation ({B}×{T} steps)...")
    t0 = time.time()

    for t in range(T):
        ev_t  = ev[:, t, :]
        kin_t = kin[:, t, :]
        tof_t = tof[:, t]

        inject_drift = (t >= drift_start)

        # ---- IMU-only: pure velocity integration (kin only, no SNN) ----
        if t == 0:
            x_imu[:, 0] = pos_gt[:, 0, 0]
            y_imu[:, 0] = pos_gt[:, 0, 1]
            th_imu[:, 0] = headings_init   # CORRECT: start at GT heading
        else:
            vx_b = kin[:, t, 0]
            vy_b = kin[:, t, 1]
            omega_b = kin[:, t, 2]
            if inject_drift:
                omega_b = omega_b + DRIFT_OMEGA
            cos_h = np.cos(th_imu[:, t-1])
            sin_h = np.sin(th_imu[:, t-1])
            vx_w = vx_b * cos_h - vy_b * sin_h
            vy_w = vx_b * sin_h + vy_b * cos_h
            x_imu[:, t] = x_imu[:, t-1] + vx_w * DT
            y_imu[:, t] = y_imu[:, t-1] + vy_w * DT
            th_imu[:, t] = (th_imu[:, t-1] + omega_b * DT) % (2 * np.pi)

        # ---- Open-loop SNN: no place cell corrections ----
        pose_est_ol, r_place_ol, r_ring_ol = system_ol.forward_step_open_loop(
            ev_t, kin_t, tof_t, inject_drift=inject_drift
        )
        x_ol[:, t] = np.array(pose_est_ol[:, 0])
        y_ol[:, t] = np.array(pose_est_ol[:, 1])
        th_ol[:, t] = np.array(pose_est_ol[:, 2])

        # ---- Closed-loop SNN: full system with corrections ----
        pose_est, r_place, r_ring, loop_conf = system_cl.forward_step(
            ev_t, kin_t, tof_t, inject_drift=inject_drift
        )
        x_cl[:, t] = np.array(pose_est[:, 0])
        y_cl[:, t] = np.array(pose_est[:, 1])
        th_cl[:, t] = np.array(pose_est[:, 2])
        loop_conf_log[:, t] = np.array(loop_conf)

        # ---- Decode place cell positions from smoothed recall current ----
        # Extract from the Closed-Loop system ONLY
        r_place_smooth = np.array(system_cl.place.get_place_activity_flat())
        pc_decoded = system_cl.place.decode_position(r_place_smooth)
        pc_x_decoded[:, t] = pc_decoded[:, 0]
        pc_y_decoded[:, t] = pc_decoded[:, 1]
        # Normalizes the peak so it measures memory "clarity" [0, 1]
        pc_top_conf[:, t] = r_place_smooth.max(axis=1) / (r_place_smooth.sum(axis=1) + 1e-8)
        
        # 🌟 NEW: Log the continuous place cell activity for heatmap visualization
        pc_activity_log[:, t, :] = r_place_smooth

        # 🌟 NEW: Log the continuous place cell activity for heatmap visualization
        pc_activity_log[:, t, :] = r_place_smooth

        # 👇 ADD THIS LINE 👇 (Grabs the flat CANN state directly from the pose module)
        cann_activity_log[:, t, :] = np.array(system_cl.pose.get_state_flat())

        # ---- Store raw intensity frames from batch 0 at key timesteps ----
        if t in [0, 10, 50, 100, 200, 300, 400, T-1]:
            ev_examples.append(np.array(intensities[0, t]))   # batch 0, raw intensity (N_PIXELS,)

        if t < 3 or t == drift_start or t % 100 == 0:
            lc = "🔄" if loop_conf[0] > 0.1 else "   "
            print(f'    t={t:3d}: pose=({pose_est[0,0]:.2f}, {pose_est[0,1]:.2f}, '
                  f'{np.degrees(pose_est[0,2]):.1f}°) conf={loop_conf[0]:.3f} {lc}')

    elapsed = time.time() - t0
    print(f"\n  ✅ Done in {elapsed:.1f}s "
          f"({elapsed/(B*T)*1000:.2f} ms/timestep)")

    # ---- Error metrics ----
    pos_err_imu = np.sqrt(
        (x_imu - pos_gt[:, :, 0])**2 + (y_imu - pos_gt[:, :, 1])**2
    )
    pos_err_ol = np.sqrt(
        (x_ol - pos_gt[:, :, 0])**2 + (y_ol - pos_gt[:, :, 1])**2
    )
    pos_err_cl = np.sqrt(
        (x_cl - pos_gt[:, :, 0])**2 + (y_cl - pos_gt[:, :, 1])**2
    )

    th_err_imu = np.abs(th_imu - th_gt)
    th_err_imu = np.minimum(th_err_imu, 2*np.pi - th_err_imu)
    th_err_ol = np.abs(th_ol - th_gt)
    th_err_ol = np.minimum(th_err_ol, 2*np.pi - th_err_ol)
    th_err_cl = np.abs(th_cl - th_gt)
    th_err_cl = np.minimum(th_err_cl, 2*np.pi - th_err_cl)

    print(f"\n📊 RESULTS ({B} trajectories, {T} steps, drift @ t={drift_start}):")
    print(f"  IMU-Only (raw integration):  mean_err={pos_err_imu.mean():.3f}m  "
          f"final_err={pos_err_imu[:,-1].mean():.3f}m")
    print(f"  Open-Loop SNN (CANN only):   mean_err={pos_err_ol.mean():.3f}m  "
          f"final_err={pos_err_ol[:,-1].mean():.3f}m")
    print(f"  Closed-Loop SNN SLAM v3:     mean_err={pos_err_cl.mean():.3f}m  "
          f"final_err={pos_err_cl[:,-1].mean():.3f}m")
    print(f"  Angular — IMU: {np.degrees(th_err_imu.mean()):.1f}°  "
          f"OL: {np.degrees(th_err_ol.mean()):.1f}°  "
          f"CL: {np.degrees(th_err_cl.mean()):.1f}°")

    return {
        # IMU-only (raw velocity integration)
        'x_imu': x_imu, 'y_imu': y_imu, 'th_imu': th_imu,
        'pos_err_imu': pos_err_imu, 'theta_err_imu': th_err_imu,
        # Open-loop SNN (pose-CANN without corrections)
        'x_ol': x_ol, 'y_ol': y_ol, 'th_ol': th_ol,
        'pos_err_ol': pos_err_ol, 'theta_err_ol': th_err_ol,
        # Closed-loop SNN (pose-CANN + place cell corrections)
        'x_cl': x_cl, 'y_cl': y_cl, 'th_cl': th_cl,
        'pos_err_cl': pos_err_cl, 'theta_err_cl': th_err_cl,
        # Ground truth
        'x_gt': pos_gt[:, :, 0], 'y_gt': pos_gt[:, :, 1],
        'th_gt': th_gt,
        # Environment
        'obstacles': obs,
        'labels': labels,
        'loop_conf': loop_conf_log,
        # Place cell decoded positions
        'pc_x_decoded': pc_x_decoded, 'pc_y_decoded': pc_y_decoded,
        'pc_top_conf': pc_top_conf,
        # Event camera examples
        'ev_examples': ev_examples,
        # Metadata
        'time_steps': T,
        'drift_start': drift_start,
        'B': B,
        'ev_shape': ev.shape,   # (B, T, N_PIXELS)
        'intensities': intensities,  # (B, T, N_PIXELS) raw intensity images
        # 🌟 NEW: Add ToF and PC Activity to the returns
        'tof': tof,
        'pc_activity': pc_activity_log,
        # 🌟 NEW: Add ToF and PC Activity to the returns
        'tof': tof,
        'pc_activity': pc_activity_log,
        # 👇 ADD THIS LINE 👇
        'cann_activity': cann_activity_log
    }


# ============================================================================
#  🎨  4-PANEL VISUALIZATION
# ============================================================================

def visualize_4panel(results, save_path=None, ev_save_path=None):
    """4-panel visualization addressing all 4 user concerns:

    Panel 1: IMU-only vs GT — raw velocity integration (the TRUE baseline)
    Panel 2: Open-loop SNN — pose-CANN WITHOUT place cell corrections
    Panel 3: Closed-loop SNN — full system WITH place cell corrections
             + ★ purple stars = decoded place cell firing locations
    Panel 4: Event camera — what the event sensor actually sees
    """

    B = results['B']
    T = results['time_steps']
    n_show = min(N_TRAJ_SHOW, B)
    ds = results['drift_start']
    ev_shape = results.get('ev_shape', (B, T, N_PIXELS))
    N_PIX = ev_shape[2]

    gt_colors = plt.cm.Blues(np.linspace(0.5, 0.9, n_show))
    imu_color  = '#E74C3C'  # red — IMU-only
    ol_color   = '#E67E22'  # orange — open-loop SNN
    cl_color   = '#27AE60'  # green — closed-loop SNN
    pc_star_c  = '#9B59B6'  # purple — place cell stars

    t_arr = np.arange(T) * DT

    fig = plt.figure(figsize=(24, 7))

    # Use GridSpec for precise control — Panels 1-3 span the grid
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, figure=fig, wspace=0.35,
                  left=0.04, right=0.98, top=0.90, bottom=0.14)

    # ── Panel 1: IMU-only (pure velocity integration) ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_room(ax1, results['obstacles'])

    for i in range(n_show):
        ax1.plot(results['x_gt'][i, ::4], results['y_gt'][i, ::4],
                 'o-', color=gt_colors[i], ms=3, lw=1.5, alpha=0.7,
                 label=f'GT {i}' if i == 0 else None)
        ax1.plot(results['x_imu'][i, ::4], results['y_imu'][i, ::4],
                 's--', color=imu_color, ms=3, lw=2.0, alpha=0.85,
                 label=f'IMU-only {i}' if i == 0 else None)

    # Mark starting positions explicitly
    for i in range(n_show):
        ax1.plot(results['x_gt'][i, 0], results['y_gt'][i, 0],
                 'D', color='lime', ms=10, zorder=10, label=f'Start GT' if i == 0 else None)
        ax1.plot(results['x_imu'][i, 0], results['y_imu'][i, 0],
                 'X', color=imu_color, ms=10, zorder=10, label=f'Start IMU' if i == 0 else None)

    ax1.set_title('Panel 1: IMU-Only\n(Pure velocity integration — no SNN)',
                   fontsize=11, fontweight='bold', color=imu_color)
    ax1.set_xlabel('x (m)', fontsize=9)
    ax1.set_ylabel('y (m)', fontsize=9)
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_xlim(-0.5, ROOM_W + 0.5)
    ax1.set_ylim(-0.5, ROOM_H + 0.5)

    # ── Panel 2: Open-loop SNN (pose-CANN, no corrections) ───────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_room(ax2, results['obstacles'])

    for i in range(n_show):
        ax2.plot(results['x_gt'][i, ::4], results['y_gt'][i, ::4],
                 'o-', color=gt_colors[i], ms=3, lw=1.5, alpha=0.7)
        ax2.plot(results['x_ol'][i, ::4], results['y_ol'][i, ::4],
                 '^--', color=ol_color, ms=3, lw=2.0, alpha=0.85,
                 label=f'OL SNN {i}' if i == 0 else None)

    # Starting positions
    for i in range(n_show):
        ax2.plot(results['x_gt'][i, 0], results['y_gt'][i, 0],
                 'D', color='lime', ms=10, zorder=10)
        ax2.plot(results['x_ol'][i, 0], results['y_ol'][i, 0],
                 'X', color=ol_color, ms=10, zorder=10)

    ax2.set_title('Panel 2: Open-Loop SNN\n(Pose-CANN odometry, no corrections)',
                   fontsize=11, fontweight='bold', color=ol_color)
    ax2.set_xlabel('x (m)', fontsize=9)
    ax2.set_ylabel('y (m)', fontsize=9)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.set_xlim(-0.5, ROOM_W + 0.5)
    ax2.set_ylim(-0.5, ROOM_H + 0.5)

    # ── Panel 3: Closed-loop SNN + place cell stars ────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _draw_room(ax3, results['obstacles'])

    for i in range(n_show):
        ax3.plot(results['x_gt'][i, ::4], results['y_gt'][i, ::4],
                 'o-', color=gt_colors[i], ms=3, lw=1.5, alpha=0.7)
        ax3.plot(results['x_cl'][i, ::4], results['y_cl'][i, ::4],
                 '^-', color=cl_color, ms=3, lw=2.0, alpha=0.85,
                 label=f'CL SNN {i}' if i == 0 else None)

    # ★ Place cell firing stars — decoded from population activity ★
    # Show top place cell decoded positions (purple stars at decoded locations)
    pc_conf = results['pc_top_conf']
    pc_x = results['pc_x_decoded']
    pc_y = results['pc_y_decoded']
    for i in range(n_show):
        for t in range(T):
            # Only show stars where place cell confidence is meaningful
            if pc_conf[i, t] > pc_conf.max() * 0.3:
                ax3.plot(pc_x[i, t], pc_y[i, t],
                         '*', color=pc_star_c, ms=8, alpha=0.5, zorder=7)

    # Loop confidence dots (where SNN is correcting)
    conf = results['loop_conf']
    for i in range(n_show):
        for t in range(T):
            if conf[i, t] > 0.1:
                ax3.plot(results['x_cl'][i, t], results['y_cl'][i, t],
                         '.', color='#F39C12', ms=6, alpha=0.8, zorder=8)

    ax3.set_title('Panel 3: Closed-Loop SNN SLAM v3\n(★ = decoded place cell firing, • = loop closure)',
                   fontsize=11, fontweight='bold', color=cl_color)
    ax3.set_xlabel('x (m)', fontsize=9)
    ax3.set_ylabel('y (m)', fontsize=9)
    ax3.legend(fontsize=7, loc='upper right')
    ax3.set_xlim(-0.5, ROOM_W + 0.5)
    ax3.set_ylim(-0.5, ROOM_H + 0.5)

    # ── Panel 4: Event camera intensity images ───────────────────────────
    # Show continuous pixel intensities — what the event camera continuously sees
    ev_examples = results.get('ev_examples', [])
    if ev_examples:
        n_r, n_c = 2, 4
        displayed = min(len(ev_examples), n_r * n_c)
        ev_gs = gs[0, 3].subgridspec(n_r, n_c, wspace=0.15, hspace=0.35)
        for idx in range(displayed):
            r, c = idx // n_c, idx % n_c
            ax_ev = fig.add_subplot(ev_gs[r, c])
            intensity_img = ev_examples[idx]  # (N_PIXELS,) — raw 1D intensity signal
            # Render as a horizontal strip (like a 1-pixel tall image)
            ax_ev.imshow(intensity_img[None, :], aspect='auto',
                         cmap='gray_r', vmin=0, vmax=1.5,
                         interpolation='nearest')
            ax_ev.set_yticks([])
            ax_ev.set_xticks([0, N_PIX//2, N_PIX])
            ax_ev.set_xticklabels(['0', str(N_PIX//2), str(N_PIX)], fontsize=5)
            ax_ev.tick_params(pad=1)
            ax_ev.set_title(f't={idx}', fontsize=6)
        fig.text(0.895, 0.96, 'Panel 4: Event Camera\n(continuous intensity)',
                  ha='center', va='top', fontsize=11, fontweight='bold', color='#333')
    else:
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.text(0.5, 0.5, 'No intensity\nexamples', ha='center', va='center',
                  fontsize=12, transform=ax4.transAxes, color='gray')
        ax4.axis('off')

    # ── Error comparison subplot ─────────────────────────────────────────────
    fig_err = plt.figure(figsize=(20, 5))
    ax_err = fig_err.add_subplot(1, 1, 1)

    mean_imu = results['pos_err_imu'].mean(axis=0)
    mean_ol  = results['pos_err_ol'].mean(axis=0)
    mean_cl  = results['pos_err_cl'].mean(axis=0)

    ax_err.plot(t_arr, mean_imu, color=imu_color, lw=2.5,
                label=f'IMU-Only (mean={mean_imu.mean():.3f}m)', ls='--', alpha=0.8)
    ax_err.plot(t_arr, mean_ol,  color=ol_color,  lw=2.5,
                label=f'Open-Loop SNN (mean={mean_ol.mean():.3f}m)', ls='-.', alpha=0.8)
    ax_err.plot(t_arr, mean_cl,  color=cl_color,  lw=3.0,
                label=f'Closed-Loop SNN (mean={mean_cl.mean():.3f}m)', ls='-')

    ax_err.axvline(ds * DT, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax_err.text(ds * DT + 0.05, ax_err.get_ylim()[1] * 0.95 * (ax_err.get_ylim()[1] or 1),
                f'Drift\nstarts', fontsize=8, color='gray')

    ax_err.fill_between(t_arr, mean_imu, mean_cl,
                        where=(mean_imu > mean_cl),
                        color=cl_color, alpha=0.08, label='IMU→CL Improvement')

    ax_err.set_title('Position Error over Time — All Three Baselines (meters, lower is better)',
                      fontsize=12, fontweight='bold')
    ax_err.set_xlabel('Time (s)', fontsize=10)
    ax_err.set_ylabel('Position Error (m)', fontsize=10)
    ax_err.legend(fontsize=9, loc='upper left')
    ax_err.grid(alpha=0.25, linestyle='--')
    ax_err.set_xlim(0, T * DT)
    ax_err.set_ylim(bottom=0)

    fig_err.text(0.5, 0.02,
                  f'IMU-Only: {mean_imu[-1]:.3f}m  |  '
                  f'Open-Loop SNN: {mean_ol[-1]:.3f}m  |  '
                  f'Closed-Loop SNN: {mean_cl[-1]:.3f}m',
                  ha='center', fontsize=11, style='italic', color='#333')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n  💾 Saved trajectory figure: {save_path}")

    err_path = save_path.replace('.png', '_error.png') if save_path else None
    if err_path:
        fig_err.savefig(err_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  💾 Saved error figure: {err_path}")

    return fig, fig_err


def _draw_room(ax, obstacles):
    ax.set_aspect('equal')
    ax.set_xlim(-0.3, ROOM_W + 0.3)
    ax.set_ylim(-0.3, ROOM_H + 0.3)
    ax.grid(alpha=0.2, linestyle='--')
    ax.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H,
                            lw=2.5, edgecolor='#333',
                            facecolor='#f8f8f5', alpha=0.5))
    for o in obstacles:
        w = float(o[2]-o[0])
        h = float(o[3]-o[1])
        ax.add_patch(Rectangle((float(o[0]), float(o[1])), w, h,
                              facecolor='#888', edgecolor='#222',
                              lw=1.0, alpha=0.85))
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('y (m)', fontsize=9)


# ============================================================================
#  🎬  REAL-TIME GIF VISUALIZATION
# ============================================================================

import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

def create_slam_gif(results, batch_idx=0, step_skip=10, save_path="slam_realtime.gif"):
    print(f"\n 🎬 Rendering GIF to {save_path} (skipping every {step_skip} frames)...")
    
    T = results['time_steps']
    N_PIXELS = results['intensities'].shape[2]
    MAP_SIZE = int(np.sqrt(results['pc_activity'].shape[2]))
    CANN_SIZE = int(np.sqrt(results['cann_activity'].shape[2])) # 🌟 NEW
    
    # Wide figure (18x12) for a 2x3 Grid
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.5, 1, 1])
    
    # 1. Trajectory gets massive left column (spanning both rows)
    ax_traj = fig.add_subplot(gs[:, 0])
    
    # 2. Sensors in middle column
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_tof = fig.add_subplot(gs[1, 1])
    
    # 3. Brains in right column
    ax_brain = fig.add_subplot(gs[0, 2])
    ax_cann  = fig.add_subplot(gs[1, 2]) # 🌟 NEW CANN AXIS

    # --- 1. Trajectory Setup ---
    _draw_room(ax_traj, results['obstacles'])
    ax_traj.set_title("Live Trajectory (Closed-Loop vs GT)", fontweight='bold')
    traj_gt_line, = ax_traj.plot([], [], 'b--', lw=1.5, alpha=0.5, label="Ground Truth")
    traj_gt_head, = ax_traj.plot([], [], 'bo', ms=6, alpha=0.5)
    traj_line, = ax_traj.plot([], [], 'g-', lw=2, label="SNN Belief")
    traj_head, = ax_traj.plot([], [], 'go', ms=8)
    ax_traj.legend(loc='upper right', fontsize=8)
    
    # --- 2. Camera Setup ---
    ax_cam.set_title("Event Camera Intensity", fontweight='bold')
    ax_cam.set_ylim(0, 1.5) 
    ax_cam.set_xlim(0, N_PIXELS)
    cam_line, = ax_cam.plot([], [], 'k-', lw=2)
    ax_cam.set_ylabel("Intensity")
    ax_cam.set_xlabel("Pixel Index")
    
    # --- 3. ToF Setup ---
    ax_tof.set_title("ToF Depth", fontweight='bold')
    ax_tof.set_xlim(0, T)
    ax_tof.set_ylim(0, 10)
    tof_line, = ax_tof.plot([], [], 'b-', lw=2)
    tof_current, = ax_tof.plot([], [], 'ro', ms=6)
    ax_tof.set_ylabel("Distance (m)")
    ax_tof.set_xlabel("Time Step")
    
    # --- 4. Place Cell Memory Setup ---
    ax_brain.set_title("Place Cell Map (Where am I in Memory?)", fontweight='bold')
    brain_img = ax_brain.imshow(np.zeros((MAP_SIZE, MAP_SIZE)), 
                                cmap='magma', origin='lower', vmin=0, vmax=1.0)
    ax_brain.axis('off')
    conf_text = ax_brain.text(0.05, 0.95, '', transform=ax_brain.transAxes, 
                              color='white', fontsize=12, fontweight='bold', va='top', ha='left')

    # --- 5. CANN Bump Setup (🌟 NEW) ---
    ax_cann.set_title("CANN Pose (Where am I in Reality?)", fontweight='bold')
    # Using 'plasma' colormap so it visually looks different from the Memory Map
    cann_img = ax_cann.imshow(np.zeros((CANN_SIZE, CANN_SIZE)), 
                              cmap='plasma', origin='lower', vmin=0, vmax=1.0)
    ax_cann.axis('off')
    
    fig.tight_layout()

    def update(frame):
        # 1. Trajectory
        x_gt_curr, y_gt_curr = results['x_gt'][batch_idx, :frame], results['y_gt'][batch_idx, :frame]
        traj_gt_line.set_data(x_gt_curr, y_gt_curr)
        x_curr, y_curr = results['x_cl'][batch_idx, :frame], results['y_cl'][batch_idx, :frame]
        traj_line.set_data(x_curr, y_curr)
        if frame > 0:
            traj_gt_head.set_data([x_gt_curr[-1]], [y_gt_curr[-1]])
            traj_head.set_data([x_curr[-1]], [y_curr[-1]])
            
        # 2. Sensors
        cam_line.set_data(range(N_PIXELS), results['intensities'][batch_idx, frame])
        tof_history = results['tof'][batch_idx, :frame]
        tof_line.set_data(range(frame), tof_history)
        if frame > 0:
            tof_current.set_data([frame], [tof_history[-1]])
            
        # 3. Brain Maps
        pc_act = results['pc_activity'][batch_idx, frame].reshape(MAP_SIZE, MAP_SIZE)
        brain_img.set_data(pc_act / (pc_act.max() + 1e-8))
        
        conf = results['loop_conf'][batch_idx, frame]
        conf_text.set_text(f"Confidence: {conf:.3f}")
        conf_text.set_color('lime' if conf > 0.1 else 'white')

        # 4. CANN Bump (🌟 NEW)
        cann_act = results['cann_activity'][batch_idx, frame].reshape(CANN_SIZE, CANN_SIZE)
        cann_img.set_data(cann_act / (cann_act.max() + 1e-8))
        
        return traj_gt_line, traj_gt_head, traj_line, traj_head, cam_line, tof_line, tof_current, brain_img, conf_text, cann_img

    frames = list(range(0, T, step_skip))
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(save_path, writer='pillow', fps=15)
    print(f"  ✅ GIF successfully saved to {save_path}")

# ============================================================================
#  🚀  MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("  🦊  SNN SLAM System v3 — Parallel Ring Memory + Dual-Key Gating")
    print("  VisionSTDP + ToF RBF Population + PlaceCellNetwork")
    print("=" * 65)

    key = random.PRNGKey(0xFACEC0DE)

    N_SAMPLES   = 3  # reduced from 4
    TIME_STEPS  = 3000   # bumped from 500; B-spline supports long looping trajectories
    DRIFT_START = 80
    FIG_PATH    = "/Users/lhooz/.openclaw/workspace/snn_slam_4panel.png"
    GIF_PATH    = "/Users/lhooz/.openclaw/workspace/snn_slam_realtime.gif" # 🌟 NEW: GIF path

    print(f"\n⚙️  Config:")
    print(f"   Batch: {N_SAMPLES}  |  Steps: {TIME_STEPS}  |  Drift @ t={DRIFT_START}")
    print(f"   VisionSTDP: {N_VISION} features")
    print(f"   ToF Population: {N_DEPTH} Gaussian RBF channels (σ={TOF_SIGMA}m)")
    print(f"   Fused features: {N_FUSED} = {N_VISION}×{N_DEPTH}")
    print(f"   PlaceCellNetwork: {N_FUSED}×{N_PLACE} Hebbian + {N_FUSED}×{RING_N} Ring")
    print(f"   Dual-Key Gates: SPARSITY + MATCH(dual) + ANTI-ALIASING(dual) + PLAUSIBILITY + SELF-MATCH(1.5m)")
    print(f"   Global Divisive Norm: k_cann=0.05, k_ring=0.1")

    results = evaluate_system(
        key, n_samples=N_SAMPLES,
        time_steps=TIME_STEPS,
        drift_start=DRIFT_START
    )

    print(f"\n🎨 Generating 4-panel visualization (with event camera + place cell stars)...")
    fig, fig_err = visualize_4panel(results, save_path=FIG_PATH)

    # 🌟 NEW: Call the GIF generator (skipping frames to save memory/render time)
    create_slam_gif(results, batch_idx=0, step_skip=15, save_path=GIF_PATH)

    print(f"\n{'='*65}")
    print(f"  ✅ SYSTEM TEST COMPLETE")
    print(f"{'='*65}")
    return results, fig, fig_err


if __name__ == '__main__':
    results, fig, fig_err = main()
