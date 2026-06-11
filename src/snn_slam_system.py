#!/usr/bin/env python3
"""
snn_slam_system.py — Neuromorphic SLAM Orchestrator (v10)

Integrates three biological modules into a closed-loop navigation system:

  1. VisionCSNN   (256 feature neurons) — event-based edge features
  2. PoseCANN     (579 grid cell spatial + 64 heading) — dead-reckoning via IMU
  3. PlaceCellNetwork + Parallel Ring Memory — depth-aware spatial memory

================================================================
  PARALLEL RING MEMORY + DUAL-KEY GATING — INSECT BRAIN v10
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
    CANN_SIZES, WRAP_SCALES, TOTAL_GRID_DIM, RING_N,
    ring_readout,
)
from src.snn_place_cells import (
    PlaceCellNetwork,
    N_PLACE,
)

# ============================================================================
#  🌟 NEW: HDC SPATIAL HASH MAP CONFIGURATION
# ============================================================================
HDC_CONFIG = {
    "num_bits": 256,         # 🌟 UPGRADE: Stops the mathematical False Positive collisions
    "active_spikes_k": 8,   # 🌟 UPGRADE: ~3% Sparsity
    "match_threshold": 5,    # 🌟 FIX: Raised from 5→6 to reduce birthday-paradox false collisions on large maps
}

# ============================================================================
#  🎛️  HYPERPARAMETERS
# ============================================================================

N_VISION        = 256     # VisionCSNN feature neurons

N_DEPTH_PER_RAY = 64
N_DEPTH         = N_DEPTH_PER_RAY * 3  # 192 Total Apical Dendrites

TOF_MIN         = 0.1    # meters
TOF_MAX         = 2.83   # meters — max diagonal of 2m×2m room (√(2²+2²))
TOF_SIGMA       = 0.25   # tof precision

DRIFT_START     = 5000     # (Offline Default) step at which drift kicks in
DRIFT_OMEGA     = 0.001  # rad/s artificial yaw drift per timestep

N_TRAJ_SHOW     = 1
SAVE_FIG        = True

BASE_LC_FACTOR = 1.20
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

@jax.jit
def decode_grid_to_xy(grid_key_flat, prior_xy):
    """
    Local Phase Unwrapping: Slices the 579-dim key back into 3 modules.
    Finds the closest valid modulo coordinate to the previous known position,
    completely eliminating harmonic alias "teleportation".
    """
    B = grid_key_flat.shape[0]
    
    # 1. Slice the 579-dim vector back into 121, 169, 289 arrays
    s1, s2 = CANN_SIZES[0]**2, CANN_SIZES[1]**2
    r1 = grid_key_flat[:, :s1].reshape(B, CANN_SIZES[0], CANN_SIZES[0])
    r2 = grid_key_flat[:, s1:s1+s2].reshape(B, CANN_SIZES[1], CANN_SIZES[1])
    r3 = grid_key_flat[:, s1+s2:].reshape(B, CANN_SIZES[2], CANN_SIZES[2])
    
    modules = [r1, r2, r3]
    phases_x, phases_y = [], []
    
    # 2. Extract local phase (in meters) for each module
    for i, (size, scale) in enumerate(zip(CANN_SIZES, WRAP_SCALES)):
        angles = jnp.arange(size, dtype=jnp.float32) * (2 * jnp.pi / size)
        sin_a, cos_a = jnp.sin(angles), jnp.cos(angles)
        
        p = modules[i] / (modules[i].sum(axis=(1, 2), keepdims=True) + 1e-8)
        cx_angle = jnp.arctan2((p.sum(axis=1) * sin_a).sum(axis=1), (p.sum(axis=1) * cos_a).sum(axis=1)) % (2 * jnp.pi)
        cy_angle = jnp.arctan2((p.sum(axis=2) * sin_a).sum(axis=1), (p.sum(axis=2) * cos_a).sum(axis=1)) % (2 * jnp.pi)
        
        phases_x.append((cx_angle / (2 * jnp.pi)) * scale)
        phases_y.append((cy_angle / (2 * jnp.pi)) * scale)
        
    phases_x = jnp.stack(phases_x, axis=1) # [B, 3]
    phases_y = jnp.stack(phases_y, axis=1)
    
    scale_arr = jnp.array(WRAP_SCALES)[None, :] # [1, 3]
    
    # 3. Local Phase Unwrapping (The Temporal Prior)
    def unwrap_closest(phases, prior):
        prior_exp = prior[:, None] # [B, 1]
        # Find the shortest distance from the current phase to the expected prior phase
        delta = phases - (prior_exp % scale_arr)
        # Wrap delta cleanly between -scale/2 and +scale/2
        delta_wrapped = (delta + scale_arr / 2.0) % scale_arr - scale_arr / 2.0
        
        # Apply the shortest path delta to the prior, and average the 3 independent pendulums
        unwrapped_candidates = prior_exp + delta_wrapped 
        return jnp.mean(unwrapped_candidates, axis=1) 

    global_x = unwrap_closest(phases_x, prior_xy[:, 0])
    global_y = unwrap_closest(phases_y, prior_xy[:, 1])
    
    return jnp.stack([global_x, global_y], axis=1)

class SpikingOccupancyGrid:
    """A 2D sheet of Leaky Integrate-and-Fire (LIF) neurons for spatial mapping."""
    def __init__(self, map_size_m=30.0, res=0.10, offset_m=10.0, v_max=None): 
        self.res = res
        self.offset_m = offset_m 
        self.grid_w = int(map_size_m / res)
        self.grid_h = int(map_size_m / res)
        
        self.v_th = 1.0         
        self.v_reset = 0.0      
        self.v_rest = 0.0       
        self.beta = 0.999        
        self.w_exc = 0.35       
        self.w_inh = -0.15      
        self.v_max = v_max if v_max is not None else float('inf')  

    def init_state(self):
        return SpikingMapState(
            v_mem=jnp.full((self.grid_w, self.grid_h), self.v_rest, dtype=jnp.float32),
            spikes=jnp.zeros((self.grid_w, self.grid_h), dtype=jnp.float32)
        )

    @partial(jax.jit, static_argnames=['self'])
    def update(self, state: SpikingMapState, hit_idx, free_idx):
        v_next = state.v_mem * self.beta
        
        # 🌟 JAX FIX: Use mode='drop' to ignore our padded '-1' arrays without recompiling
        v_next = v_next.at[free_idx[:, 0], free_idx[:, 1]].add(self.w_inh, mode='drop')
        v_next = v_next.at[hit_idx[:, 0], hit_idx[:, 1]].add(self.w_exc, mode='drop')
        
        v_next = jnp.maximum(v_next, -0.5)
        spikes = jnp.where(v_next >= self.v_th, 1.0, 0.0)
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

        # 🌟 BUG FIX 3: Dynamic Covariance Scaling (DCS) — SOTA outlier rejection!
        # (Agarwal, Olson, Stachniss, 2013)
        # Automatically downweights false loop closures based on residual magnitude.
        # A correct LC has small residual → dcs_weight ≈ 1.0 (full force)
        # A false LC has large residual → dcs_weight → 0.0 (neutralized)
        dcs_phi = 0.5  # Kernel width — smaller = more aggressive outlier rejection
        residual_sq = lc_err_x**2 + lc_err_y**2 + lc_err_th**2
        dcs_weight = dcs_phi / (dcs_phi + residual_sq)

        # 🌟 SOLUTION 2: Apply Biological Confidence Weights × DCS Robust Kernel
        lc_f_x = lc_err_x * loop_weights[:, 0] * dcs_weight
        lc_f_y = lc_err_y * loop_weights[:, 0] * dcs_weight
        lc_f_th = lc_err_th * loop_weights[:, 1] * dcs_weight

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
        # 🌟 THE EULER EXPLOSION FIX: Hard Clamp the Integration Velocity
        v_new_x = jnp.clip((v[:, 0] + dp_odom_x + dp_loop_x) * dynamic_damping, -0.05, 0.05)
        v_new_y = jnp.clip((v[:, 1] + dp_odom_y + dp_loop_y) * dynamic_damping, -0.05, 0.05)
        v_new_th = jnp.clip((v[:, 2] + dp_odom_th + dp_loop_th) * dynamic_damping, -0.02, 0.02)

        p_new_x = p[:, 0] + v_new_x
        p_new_y = p[:, 1] + v_new_y
        p_new_th = wrap_angle(p[:, 2] + v_new_th)

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
    Upgraded to LINEAR Phase Correlation via Zero-Padding (2N) to prevent 
    circular wrap-around without destroying edge features.
    """
    N = live_csnn.shape[-1]
    
    # 🌟 THE LINEAR CORRELATION FIX: Zero-pad to 2N
    # Creates "empty space" so the shifted features don't wrap around,
    # completely eliminating the need for a feature-destroying Hanning window!
    pad_width = [(0, 0)] * (live_csnn.ndim - 1) + [(0, N)]
    
    live_padded = jnp.pad(live_csnn, pad_width)
    mem_padded = jnp.pad(mem_csnn, pad_width)
    
    # Transform to Frequency Domain
    F_live = jnp.fft.fft(live_padded)
    F_mem = jnp.fft.fft(mem_padded)
    
    # Calculate the Cross-Power Spectrum (Mem * conj(Live))
    cross_power = F_mem * jnp.conj(F_live)
    
    # Normalize by magnitude to extract pure Phase
    cross_power_norm = cross_power / (jnp.abs(cross_power) + 1e-8)
    
    # Inverse FFT back to Spatial Domain
    r = jnp.fft.ifft(cross_power_norm)
    return jnp.abs(r)


@jax.jit
def get_dvs_rotation_shift(curr_ts, prev_ts):
    """
    Computes visual rotation shift and Peak-to-Sidelobe Ratio (PSR) confidence
    between consecutive time surfaces using 1D phase correlation.
    """
    curr_on = curr_ts[:, :N_PIXELS]
    curr_off = curr_ts[:, N_PIXELS:]
    prev_on = prev_ts[:, :N_PIXELS]
    prev_off = prev_ts[:, N_PIXELS:]
    
    r_on = get_phase_correlation(curr_on, prev_on)
    r_off = get_phase_correlation(curr_off, prev_off)
    r_real = r_on + r_off
    
    N_PAD = N_PIXELS * 2
    search_radius = 15
    
    idx = jnp.arange(N_PAD)
    mask = (idx <= search_radius) | (idx >= N_PAD - search_radius)
    r_masked = jnp.where(mask[None, :], r_real, -1e9)
    
    peak_idx = jnp.argmax(r_masked, axis=1)
    
    y2 = jnp.take_along_axis(r_real, peak_idx[:, None], axis=1)[:, 0]
    y1 = jnp.take_along_axis(r_real, ((peak_idx - 1) % N_PAD)[:, None], axis=1)[:, 0]
    y3 = jnp.take_along_axis(r_real, ((peak_idx + 1) % N_PAD)[:, None], axis=1)[:, 0]
    
    denom = 2.0 * (y1 - 2.0 * y2 + y3)
    sub_pixel_offset = jnp.clip((y1 - y3) / (denom - 1e-8), -1.0, 1.0)
    
    shift_int = jnp.where(peak_idx <= search_radius, peak_idx, peak_idx - N_PAD)
    pixel_shift = shift_int + sub_pixel_offset
    
    pixel_ang_res = jnp.radians(FOV_DEG) / N_PIXELS
    sub_pixel_th = -pixel_shift * pixel_ang_res
    
    # Calculate PSR (Peak-to-Sidelobe Ratio) as confidence
    mean_val = jnp.mean(r_real, axis=1)
    psr = y2 / (mean_val + 1e-8)
    
    return sub_pixel_th, psr


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

        # 🌟 THE UPGRADE: Build LISTS of weight matrices for the 3 spatial modules!
        W_cann_list = [build_2d_cann_weights(size) for size in CANN_SIZES]
        W_cann_asym_x_list = [build_asymmetric_cann_weights_x(size) for size in CANN_SIZES]
        W_cann_asym_y_list = [build_asymmetric_cann_weights_y(size) for size in CANN_SIZES]
        
        # The Ring Attractor heading circuit stays single-scale!
        W_ring        = build_1d_ring_weights()
        W_ring_asym   = build_asymmetric_ring_weights()

        k_vision, key = random.split(key)
        self.vision = DualStreamVisionCortex(k_vision, n_pixels=N_PIXELS)

        self.tof_coder = ToFPopulationCoder(n_depth_per_ray=self.n_depth // 3)

        k_pose, key = random.split(key)
        # Pass the lists into PoseCANN instead of a single matrix
        self.pose = PoseCANN(k_pose, W_cann_list, W_ring,
                              W_cann_asym_x_list, W_cann_asym_y_list, W_ring_asym)

        # 🌟 FIX: Pass the HDC config into the brain!
        k_place, key = random.split(key)
        self.place = PlaceCellNetwork(
            key=k_place, 
            n_csnn=256, 
            n_stdp=256, 
            n_depth=self.n_depth, 
            fov_deg=FOV_DEG,
            n_place=HDC_CONFIG["num_bits"],        # 👈 Injects 1024
            k_spikes=HDC_CONFIG["active_spikes_k"] # 👈 Injects 8
        )

        self.vision_state = None
        self.place_state = None
        self._initialized = False
        self._step = 0
        
        # 🌟 NEW: The Cerebellum's internal calibration model
        self.learned_omega_bias = 0.0
        self.last_decoded_xy = None   # temporal anchor for phase unwrapping
        self.prev_decoded_xy = None   # one step earlier — used for linear extrapolation
        self.prev_time_surface = None # previous event time surface for visual odometry

    def reset(self, B):
        self.vision_state = self.vision.init_state(B)
        self.place_state = self.place.init_state(B)
        self.pose.reset(B)
        self._initialized = False
        self._step = 0
        self.last_decoded_xy = None
        self.prev_decoded_xy = None
        self.prev_time_surface = None
        self._theta_gravity = jnp.zeros((B,))
        self._smooth_acc = None

    def inject_stdp_memory(self, recovered_stdp_weights):
        """🌟 NEW: Overwrites the live working memory with the episodic Hash Map snapshot."""
        new_stdp_state = self.vision_state.stdp_state._replace(W=recovered_stdp_weights)
        self.vision_state = self.vision_state._replace(stdp_state=new_stdp_state)

    def initialize_pose(self, gt_pos, gt_heading):
        self.pose.initialize_pose(gt_pos, gt_heading)
        pose_bump = self.pose.get_state_flat()
        ring_bump = self.pose.get_ring_activity()
        self.place_state = self.place.initialize_from_pose(self.place_state, pose_bump, ring_bump=ring_bump)
        self.last_decoded_xy = gt_pos[:, :2]
        self.prev_decoded_xy = gt_pos[:, :2]  # bootstrap: prev == last so first-frame step = 0
        self._theta_gravity = gt_heading
        self._smooth_acc = None
        self._initialized = True

    def calibrate_cerebellum(self, accumulated_error_rads, time_elapsed_sec):
        """Phase 3 Plasticity: Learn the systematic hardware drift!"""
        if time_elapsed_sec <= 0: return
        
        # Calculate the drift rate in rad/s
        drift_rate = accumulated_error_rads / time_elapsed_sec
        
        # Gentle Hebbian update (EMA) to prevent overreacting to one noisy loop closure
        learning_rate = 0.05
        self.learned_omega_bias = (1.0 - learning_rate) * self.learned_omega_bias + (learning_rate * drift_rate)

    def phase_perception(self, events_t, tof_t, learn=True):
        self.vision_state, dual_vis_features = self.vision(self.vision_state, events_t, tof_t[:, 1], learn=learn)
        tof_pop = self.tof_coder(tof_t)
        return dual_vis_features, tof_pop

    def phase_inference(self, dual_vis_features, tof_features, pose_bump, current_heading_rads, ring_bump): 
        vis_csnn, vis_stdp = dual_vis_features
        self.place_state, is_confident, peak_idx_place, debug_gates = self.place.compute_confidence_with_gates(
            self.place_state, vis_csnn, vis_stdp, tof_features, pose_bump, current_heading_rads, ring_bump 
        )
        return is_confident, peak_idx_place, debug_gates

    def phase_odometry(self, kin_t, theta_gravity=None, inject_drift=False):
        # 🌟 CEREBELLUM INTERVENTION: Subtract the learned bias from the raw hardware!
        corrected_omega = kin_t[:, 2] - self.learned_omega_bias
        
        # DVS Visual Odometry: calculate relative rotation shift using SOTA phase correlation
        # on the spatial time surface between consecutive frames.
        curr_ts = self.vision_state.time_surface
        
        if self.prev_time_surface is not None:
            # Grab current visual activity level from vis_csnn
            csnn_clean = jnp.maximum(0.0, self.vision_state.csnn_trace)
            vis_csnn = csnn_clean / (jnp.linalg.norm(csnn_clean, axis=-1, keepdims=True) + 1e-8)
            top16_vis = jnp.sort(vis_csnn, axis=1)[:, -16:]
            vis_act = jnp.mean(top16_vis, axis=1)
            
            # Compute phase correlation, sub-pixel shift, and peak PSR confidence
            sub_pixel_th, psr = get_dvs_rotation_shift(curr_ts, self.prev_time_surface)
            omega_vis = jnp.clip(sub_pixel_th / DT, -6.0, 6.0)
            
            # Calculate dynamic blending weight based on PSR confidence
            # Below PSR = 4.0: trust is 0. Above PSR = 8.0: trust is 1.0 (fully trusted)
            vis_trust = jnp.clip((psr - 4.0) / 4.0, 0.0, 1.0)
            
            # Blend visual velocity with IMU yaw rate using the dynamic confidence weight.
            # Max blending weight is 0.40.
            base_gamma = 0.40
            gamma = jnp.where(vis_act >= 0.08, base_gamma * vis_trust, 0.0)
            
            # Apply dynamic complementary blending
            fused_omega = (1.0 - gamma) * corrected_omega + gamma * omega_vis
        else:
            fused_omega = corrected_omega
            
        self.prev_time_surface = curr_ts
        
        kin_corrected = jnp.stack([kin_t[:, 0], kin_t[:, 1], fused_omega], axis=1)

        if inject_drift:
            # We inject the factory drift onto the CORRECTED signal
            omega_drift = kin_corrected[:, 2] + DRIFT_OMEGA
            kin_injected = jnp.stack([kin_corrected[:, 0], kin_corrected[:, 1], omega_drift], axis=1)
        else:
            kin_injected = kin_corrected

        # Capture the pre-update heading so the predicted displacement is in the
        # correct global frame (matches what the CANN __call__ uses internally).
        theta_pre = self.pose.estimate_heading()  # ring readout BEFORE CANN update

        pose_est = self.pose(kin_injected, theta_gravity=theta_gravity)
        
        # 🌟 THE UPGRADE: Grab the raw 579-dim Grid Key and decode it!
        pose_bump = self.pose.get_state_flat()
        
        # 🌟 FIX v3: Zero-extrapolation prior — use last decoded position directly.
        #
        # WHY THIS WORKS: The bump moves at most ~3 cm/step (1.5 m/s × 0.02s),
        # while the smallest alias distance is WRAP_SCALES[0]/2 = 0.30 m.
        # That's a 10× safety margin — the prior is ALWAYS well within the
        # correct Voronoi cell of the true phase.
        #
        # WHY THE OLD APPROACH FAILED:
        # 1. Linear extrapolation assumed constant velocity direction in the
        #    global frame.  During turns the extrapolated prior diverged from
        #    the actual bump direction.
        # 2. The jump guard (which rejected decoded positions far from the
        #    extrapolation) created a fatal feedback loop: once it fired, the
        #    prior was locked to the (wrong) extrapolation, causing the guard
        #    to fire every subsequent step — permanently disconnecting the
        #    decoded position from the actual CANN bump.
        # 3. The "invisible boundary" the user saw was the first turn, where
        #    extrapolation direction ≠ bump direction triggered the loop.
        # Rotate local IMU velocity to global frame for a velocity-based prior prediction
        vx, vy = kin_injected[:, 0], kin_injected[:, 1]
        cos_th = jnp.cos(theta_pre)
        sin_th = jnp.sin(theta_pre)
        V_map_x = vx * cos_th - vy * sin_th
        V_map_y = vx * sin_th + vy * cos_th
        predicted_xy = self.last_decoded_xy  # 🌟 Zero-extrapolation prior (prevents turn/drift divergence)
        
        decoded_xy = decode_grid_to_xy(pose_bump, predicted_xy)
        
        # Advance the two-frame history (kept for cerebellum velocity computation)
        self.prev_decoded_xy = self.last_decoded_xy
        self.last_decoded_xy = decoded_xy
        
        # Update Cerebellum using the newly decoded position
        self.pose.update_cerebellum(kin_injected, decoded_xy, pose_est[:, 2])

        ring_bump = self.pose.get_ring_activity()
        
        # Replace the "dummy" CANN output with our beautifully unwrapped coordinates!
        final_pose_est = jnp.stack([decoded_xy[:, 0], decoded_xy[:, 1], pose_est[:, 2]], axis=1)
        
        return final_pose_est, pose_bump, ring_bump

    # 🌟 UPGRADE: Add 'heading' to the signature
    def phase_mapping(self, dual_vis_features, tof_features, pose_bump, ring_bump, heading, angular_vel):
        vis_csnn, vis_stdp = dual_vis_features
        
        self.place_state, (r_place, r_ring) = self.place.forward_mapping(
            self.place_state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump=ring_bump, heading=heading, angular_vel=angular_vel, learn=True
        )
        
        return r_place, r_ring

    def forward_step(self, events_t, kin_t, tof_t, acc_t=None, inject_drift=False, autopilot_on=True):
        # Pass the flag into phase_perception to pause STDP when surprised
        dual_vis_features, tof_features = self.phase_perception(events_t, tof_t, learn=autopilot_on)

        # 🌟 THE UPGRADE: Decode the Grid Key instead of calling estimate_position
        pose_bump_prior = self.pose.get_state_flat()
        pose_xy = decode_grid_to_xy(pose_bump_prior, self.last_decoded_xy)
        
        current_heading_rads = self.pose.estimate_heading()
        
        # 🌟 Grab the CANN belief BEFORE inference
        ring_bump_prior = self.pose.get_ring_activity()

        is_confident, peak_idx_place, debug_gates = self.phase_inference(
            dual_vis_features, tof_features, pose_bump_prior, current_heading_rads, ring_bump_prior # 🌟 CHANGED
        )

        # Complementary Filter state estimator for gravity direction (pitch correction)
        if acc_t is not None:
            # 1. Low-pass filter the accelerometer readings (EMA) to suppress high-frequency flapping vibration
            alpha_acc = 0.1
            if self._smooth_acc is None or self._smooth_acc.shape[0] != acc_t.shape[0]:
                self._smooth_acc = acc_t
            else:
                self._smooth_acc = (1.0 - alpha_acc) * self._smooth_acc + alpha_acc * acc_t
            
            # 2. Extract pitch angle from proper acceleration (acc_x, acc_z)
            ax = self._smooth_acc[:, 0]
            az = self._smooth_acc[:, 1]
            theta_accel = jnp.arctan2(ax, az)
            theta_accel = wrap_angle(theta_accel)
            
            # 3. Integrate gyroscope rate (corrected for learned bias)
            corrected_omega = kin_t[:, 2] - self.learned_omega_bias
            theta_gyro = self._theta_gravity + corrected_omega * 0.02
            theta_gyro = wrap_angle(theta_gyro)
            
            # 4. Fuse using Complementary Filter
            alpha_fuse = 0.05
            diff = wrap_angle(theta_accel - theta_gyro)
            self._theta_gravity = wrap_angle(theta_gyro + alpha_fuse * diff)
            
            theta_gravity_val = self._theta_gravity
        else:
            theta_gravity_val = None

        pose_est, pose_bump, ring_bump = self.phase_odometry(
            kin_t, theta_gravity=theta_gravity_val, inject_drift=inject_drift
        )
        
        # 🌟 FIX: Update last_decoded_xy so the next frame unwraps around the NEW position!
        self.last_decoded_xy = pose_est[:, :2]

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
