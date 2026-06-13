#!/usr/bin/env python3
"""
snn_pose_cann.py — The "Where Am I?" Module

2D Spatial CANN (x, y) + 1D Ring Attractor (θ) with ANALYTICAL DoG weights.
Receives IMU velocity injection and corrective current from the Map module.

Key changes for v3 (Repaired) — Parallel Ring Memory + Global Divisive Normalization:
  - Dual map correction inputs: map_correction_place AND map_correction_ring
  - Ring injection: map_correction_ring is NO LONGER zeroed out
  - Stabilized Global Divisive Normalization (prevents gain collapse & 2-bump extinction)
  - Exact Exponential Euler Integration (prevents blowups)
  - Auto-freezing Cerebellum during loop closures

Architecture:
  IMU [vx, vy, ω] → velocity_injection (asymmetric DoG) → bump shift
                          ↑
  I_map_corr_place  ← ghost bump for CANN (loop closure position)
  I_map_corr_ring   ← ghost bump for Ring (loop closure heading)
                          ↓
  2D CANN sheet (DoG weights) ←→ recurrent (x,y) bump
                          ↓
  1D Ring Attractor (DoG weights) ←→ recurrent θ bump
                          ↓
  Readout → [x̂, ŷ, θ̂]

Author: Ada 🦊
Mathematical framework: Mexican Hat / Difference of Gaussians (DoG)
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sparse_forest import (
    N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H,
    VX_RANGE, VY_RANGE, OMEGA_RANGE,
)

# ============================================================================
# Configuration (same as snn_slam_twin.py — the proven DoG parameters)
# ============================================================================

# Multi-Module Grid Cells (2D spatial sheets)
CANN_SIZES = [11, 13, 17]        # Neurons per edge for Module 1, 2, 3
WRAP_SCALES = [0.6, 0.8, 1.06]   # ×0.2 from 10m-room [3.0,4.0,5.3]; same WRAP/ROOM_W ratio
TOTAL_GRID_DIM = sum([s**2 for s in CANN_SIZES]) # 121 + 169 + 289 = 579 neurons

A_EXC = 0.5             
A_INH = 0.125           
SIGMA_EXC = 1.0         
SIGMA_INH = 2.0         
TAU_U = 0.05

# Ring Attractor (1D heading)
RING_N = 64             # 64 neurons for 360°
RING_A_EXC = 1.0        # excitatory amplitude
RING_A_INH = 0.50       # inhibitory amplitude (Mexican Hat)
RING_SIGMA_EXC = 2.0    # excitatory spread (neuron units)
RING_SIGMA_INH = 4.0    # inhibitory spread (neuron units)
RING_TAU_U = 0.03

# Spiking LIF
BETA_LIF = 0.85
V_TH = 1.0

# Velocity injection gains
# VEL_GAIN_XY must be scaled with WRAP_SCALES: density_factor = c_size/scale.
# Old 10m room: WRAP_SCALES[0]=3.0, slam_scale=10.0 → density_factor=3.67, V_map_x is 10x larger.
# New  2m room: WRAP_SCALES[0]=0.6, slam_scale=1.0  → density_factor=18.3, V_map_x is 10x smaller.
# The product (V_map_x * density_factor) is actually 2x smaller than before.
# Therefore, we need to INCREASE the gain (0.10 → 0.15) and allow it to learn higher.
VEL_GAIN_XY = 0.035     # velocity → bump shift (1:1 bump speed matching)
VEL_GAIN_TH = 0.157     # omega → ring shift (approx 1:1 bump speed matching)




# ============================================================================
# 1. Analytical DoG Weight Matrices
# ============================================================================

def build_2d_cann_weights(cann_size, a_exc=A_EXC, a_inh=A_INH,
                          sigma_exc=SIGMA_EXC, sigma_inh=SIGMA_INH):
    """Build 2D CANN recurrent weight matrix: W_ij = A_exc·G_exc(d_ij) - A_inh·G_inh(d_ij)"""
    x_1d = jnp.arange(cann_size, dtype=jnp.float32)

    src_x_4d = jnp.broadcast_to(x_1d[None, None, None, :], (cann_size, cann_size, cann_size, cann_size))
    src_y_4d = jnp.broadcast_to(x_1d[None, None, :, None], (cann_size, cann_size, cann_size, cann_size))
    dst_x_4d = jnp.broadcast_to(x_1d[None, :, None, None], (cann_size, cann_size, cann_size, cann_size))
    dst_y_4d = jnp.broadcast_to(x_1d[:, None, None, None], (cann_size, cann_size, cann_size, cann_size))

    dx = jnp.minimum(jnp.abs(src_x_4d - dst_x_4d), cann_size - jnp.abs(src_x_4d - dst_x_4d))
    dy = jnp.minimum(jnp.abs(src_y_4d - dst_y_4d), cann_size - jnp.abs(src_y_4d - dst_y_4d))
    d2 = dx**2 + dy**2

    W_4d_unnorm = (a_exc * jnp.exp(-d2 / (2 * sigma_exc**2))
                   - a_inh * jnp.exp(-d2 / (2 * sigma_inh**2)))

    self_conn_unnorm = a_exc - a_inh
    W_4d = (W_4d_unnorm / (self_conn_unnorm + 1e-8))

    W = W_4d.reshape(cann_size * cann_size, cann_size * cann_size)
    return W


def build_1d_ring_weights(ring_n=RING_N, a_exc=RING_A_EXC, a_inh=RING_A_INH,
                          sigma_exc=RING_SIGMA_EXC, sigma_inh=RING_SIGMA_INH):
    """Build 1D ring attractor weight matrix using INTEGER INDEX distance."""
    rows = []
    for i in range(ring_n):
        d = jnp.arange(ring_n, dtype=jnp.float32)
        d = jnp.abs(d - i)
        d = jnp.minimum(d, ring_n - d)
        w = (a_exc * jnp.exp(-d**2 / (2 * sigma_exc**2))
             - a_inh * jnp.exp(-d**2 / (2 * sigma_inh**2)))
        rows.append(w)
    W = jnp.stack(rows)
    W = W / (W[0, 0] + 1e-8)
    return W


def build_asymmetric_ring_weights(ring_n=RING_N, sigma=RING_SIGMA_EXC):
    """Asymmetric weight matrix for ω → bump shift."""
    rows = []
    for i in range(ring_n):
        n = jnp.arange(ring_n, dtype=jnp.float32)
        diff = n - i
        diff = jnp.where(diff > ring_n / 2, diff - ring_n, diff)
        diff = jnp.where(diff < -ring_n / 2, diff + ring_n, diff)
        w_asym = diff * jnp.exp(-diff**2 / (2 * sigma**2))
        rows.append(w_asym)
    W_asym = jnp.stack(rows)
    W_asym = W_asym / (jnp.abs(W_asym).max() + 1e-8)
    return W_asym


def build_asymmetric_cann_weights_x(cann_size, sigma=SIGMA_EXC):
    """Asymmetric weight matrix for vx → bump shift along x-axis."""
    x_c = jnp.arange(cann_size, dtype=jnp.float32)
    y_c = jnp.arange(cann_size, dtype=jnp.float32)

    dx = x_c[:, None] - x_c[None, :]
    dx = jnp.where(dx > cann_size/2, dx - cann_size, dx)
    dx = jnp.where(dx < -cann_size/2, dx + cann_size, dx)
    k_x = dx * jnp.exp(-dx**2 / (2 * sigma**2))

    dy = y_c[:, None] - y_c[None, :]
    dy = jnp.where(dy > cann_size/2, dy - cann_size, dy)
    dy = jnp.where(dy < -cann_size/2, dy + cann_size, dy)
    G_y = jnp.exp(-dy**2 / (2 * sigma**2))

    k_x_4d = k_x[None, :, None, :]
    G_y_4d = G_y[:, None, :, None]
    W_4d = k_x_4d * G_y_4d

    W = W_4d.reshape(cann_size * cann_size, cann_size * cann_size)
    W = W / (jnp.abs(W).max() + 1e-8)
    return W


def build_asymmetric_cann_weights_y(cann_size, sigma=SIGMA_EXC):
    """Asymmetric weight matrix for vy → bump shift along y-axis."""
    x_c = jnp.arange(cann_size, dtype=jnp.float32)
    y_c = jnp.arange(cann_size, dtype=jnp.float32)

    dx = x_c[:, None] - x_c[None, :]
    dx = jnp.where(dx > cann_size/2, dx - cann_size, dx)
    dx = jnp.where(dx < -cann_size/2, dx + cann_size, dx)
    G_x = jnp.exp(-dx**2 / (2 * sigma**2))

    dy = y_c[:, None] - y_c[None, :]
    dy = jnp.where(dy > cann_size/2, dy - cann_size, dy)
    dy = jnp.where(dy < -cann_size/2, dy + cann_size, dy)
    k_y = dy * jnp.exp(-dy**2 / (2 * sigma**2))

    G_x_4d = G_x[None, :, None, :]
    k_y_4d = k_y[:, None, :, None]
    W_4d = G_x_4d * k_y_4d

    W = W_4d.reshape(cann_size * cann_size, cann_size * cann_size)
    W = W / (jnp.abs(W).max() + 1e-8)
    return W


# ============================================================================
# 2. Neural Field Dynamics
# ============================================================================

def neural_field_update(u, r, W, I_ext, dt=DT, tau=0.05):
    """Discrete neural field: Exact Exponential Integration to prevent instability."""
    decay = jnp.exp(-dt / tau)
    drive = (jnp.einsum('ij,bj->bi', W, r) + I_ext) * (1.0 - decay)
    return decay * u + drive


def lif_step(v, beta=BETA_LIF, v_th=V_TH):
    """LIF spike generation with reset."""
    v_new = beta * v + (1 - beta) * v_th
    spike = jnp.clip(v_new - v_th, 0.0, 1.0)
    v_new = v_new - spike * v_th
    return v_new, spike


# ============================================================================
# 3. State Readout from Attractors
# ============================================================================

def ring_readout(state, ring_n=RING_N):
    """Read θ from ring using circular statistics."""
    angles = jnp.arange(ring_n, dtype=jnp.float32) * (2 * jnp.pi / ring_n)
    p = state / (state.sum(axis=1, keepdims=True) + 1e-8)
    sin_sum = (jnp.sin(angles) * p).sum(axis=1)
    cos_sum = (jnp.cos(angles) * p).sum(axis=1)
    theta = jnp.arctan2(sin_sum, cos_sum)  # naturally in (-π, π) — no modulo needed
    return theta

# ============================================================================
# CEREBELLUM: Granule Cell Population Coder
# ============================================================================
class VelocityPopulationCoder:
    """Converts scalar velocity magnitude into a 1D Gaussian population code."""
    def __init__(self, num_neurons=32, min_v=0.0, max_v=1.5, sigma=0.1):
        self.num_neurons = num_neurons
        self.centers = jnp.linspace(min_v, max_v, num_neurons)
        self.sigma = sigma

    def __call__(self, v_mag):
        v_mag_clamped = jnp.clip(v_mag, self.centers[0], self.centers[-1])
        diff = v_mag_clamped[:, None] - self.centers[None, :]
        activations = jnp.exp(-(diff ** 2) / (2 * self.sigma ** 2))
        return activations / (activations.sum(axis=1, keepdims=True) + 1e-8)


# ============================================================================
# 5. Pose CANN Class
# ============================================================================

class PoseCANN:
    # Notice the W_cann parameters are now expected to be lists!
    def __init__(self, key, W_cann_list, W_ring, W_cann_asym_x_list, W_cann_asym_y_list, W_ring_asym):
        k1, k2 = random.split(key)
        self.W_cann_list = W_cann_list
        self.W_ring = W_ring
        self.W_cann_asym_x_list = W_cann_asym_x_list
        self.W_cann_asym_y_list = W_cann_asym_y_list
        self.W_ring_asym = W_ring_asym

        # State arrays are now lists holding 3 separate JAX arrays
        self._u_canns = [None, None, None]
        self._r_canns = [None, None, None]
        
        self._u_ring = None
        self._r_ring = None
        self._smooth_omega = None
        
        self.n_speed_neurons = 32
        self.vel_coder_xy = VelocityPopulationCoder(self.n_speed_neurons, 0.0, 3.0, 0.2)
        self.vel_coder_th = VelocityPopulationCoder(self.n_speed_neurons, 0.0, 3.14, 0.2)
        
        self.prev_pose_xy = None
        self.prev_heading = None

    def reset(self, B):
        """Reset to centered Gaussian bumps (fallback initialization)."""
        self.prev_pose_xy = None
        self.prev_heading = None
        self.lagged_v_imu = None
        self.lagged_w_imu = None
        self._smooth_omega = None

        # 1. Loop through all 3 modules to initialize their flat states
        for i, c_size in enumerate(CANN_SIZES):
            xx, yy = jnp.meshgrid(
                jnp.arange(c_size, dtype=jnp.float32),
                jnp.arange(c_size, dtype=jnp.float32),
                indexing='ij'
            )
            # Center the bump in the middle of the local module
            bump = jnp.exp(-((xx - c_size//2)**2 + (yy - c_size//2)**2) / (2 * SIGMA_EXC**2))
            bump = bump / (bump.max() + 1e-8)
            
            self._u_canns[i] = jnp.tile(0.5 * bump[None, :, :], (B, 1, 1))
            self._r_canns[i] = jnp.clip(self._u_canns[i], 0, 1.0)

        idx = jnp.arange(RING_N, dtype=jnp.float32)
        d = jnp.abs(idx - 0.0)
        d = jnp.minimum(d, RING_N - d)
        bump_r = jnp.exp(-d**2 / (2 * RING_SIGMA_EXC**2))
        bump_r = bump_r / (bump_r.sum() + 1e-8)
        self._u_ring = jnp.tile(0.5 * bump_r[None, :], (B, 1))
        self._r_ring = jnp.clip(self._u_ring, 0, 1.0)
        
        if getattr(self, 'W_cereb_xy', None) is None or self.W_cereb_xy.shape[0] != B:
            self.W_cereb_xy = jnp.ones((B, self.n_speed_neurons)) * VEL_GAIN_XY
        if getattr(self, 'W_cereb_th', None) is None or self.W_cereb_th.shape[0] != B:
            self.W_cereb_th = jnp.ones((B, self.n_speed_neurons)) * VEL_GAIN_TH
        
        self.prev_pose_xy = None
        self.prev_heading = None
        self.lagged_v_imu = None   # 🌟 ADD THIS
        self.lagged_w_imu = None   # 🌟 ADD THIS

    def initialize_pose(self, gt_pos, gt_heading):
        B = gt_pos.shape[0]
        
        # 1. Initialize the 3 Grid Modules using Modulo Math
        for i, (c_size, scale) in enumerate(zip(CANN_SIZES, WRAP_SCALES)):
            local_x = gt_pos[:, 0] % scale
            local_y = gt_pos[:, 1] % scale
            
            cx_float = (local_x / scale) * c_size
            cy_float = (local_y / scale) * c_size

            xx, yy = jnp.meshgrid(
                jnp.arange(c_size, dtype=jnp.float32),
                jnp.arange(c_size, dtype=jnp.float32),
                indexing='xy'
            )

            cx_exp = cx_float[:, None, None]
            cy_exp = cy_float[:, None, None]
            
            dx = jnp.minimum(jnp.abs(xx[None, :, :] - cx_exp), c_size - jnp.abs(xx[None, :, :] - cx_exp))
            dy = jnp.minimum(jnp.abs(yy[None, :, :] - cy_exp), c_size - jnp.abs(yy[None, :, :] - cy_exp))
            d2 = dx**2 + dy**2
            
            bumps = jnp.exp(-d2 / (2 * SIGMA_EXC**2))
            bumps = bumps / (bumps.max(axis=(1, 2), keepdims=True) + 1e-8)
            
            self._u_canns[i] = jnp.clip(bumps, 0, 1.0)
            self._r_canns[i] = self._u_canns[i]

        th_idx_float = (gt_heading % (2 * jnp.pi)) * (RING_N / (2 * jnp.pi))
        idx = jnp.arange(RING_N, dtype=jnp.float32)[None, :]
        th_exp = th_idx_float[:, None]
        d = jnp.abs(idx - th_exp)
        d = jnp.minimum(d, RING_N - d)
        bumps_r = jnp.exp(-d**2 / (2 * RING_SIGMA_EXC**2))
        bumps_r = bumps_r / (bumps_r.max(axis=1, keepdims=True) + 1e-8)
        self._u_ring = jnp.clip(bumps_r, 0, 1.0)
        self._r_ring = self._u_ring

        # 🌟 THE TELEPORT FIX: Reset Cerebellum tracking so the robot doesn't instantly 
        # calculate a 50m/s pseudo-velocity and wipe out its learned IMU gains!
        self.prev_pose_xy = gt_pos
        self.prev_heading = gt_heading
        self.lagged_v_imu = None   # 🌟 ADD THIS
        self.lagged_w_imu = None   # 🌟 ADD THIS

    def __call__(self, kin_t, theta_gravity=None, dt=DT):
        B = kin_t.shape[0]
        vx, vy, omega = kin_t[:, 0], kin_t[:, 1], kin_t[:, 2]

        # Low-pass filter angular velocity (gyro) input using Exponential Moving Average (EMA)
        # to filter out high-frequency (115Hz) sinusoidal wingbeat vibrations.
        # At 50Hz CANN update rate, alpha=0.25 corresponds to a time constant of ~80ms,
        # which effectively dampens wingbeat wobble while preserving intentional turns.
        alpha = 0.25
        if self._smooth_omega is None or self._smooth_omega.shape[0] != B:
            self._smooth_omega = omega
        else:
            self._smooth_omega = (1.0 - alpha) * self._smooth_omega + alpha * omega
        omega_filtered = self._smooth_omega

        # 1. Read the CURRENT heading from the Ring Attractor
        theta_current = ring_readout(self._r_ring)
        
        # 🌟 THE UPGRADE: Apply a Predictive Phase Lead to cancel Leaky Integrator lag!
        predictive_lead = omega_filtered * RING_TAU_U 
        theta_compensated = theta_current + predictive_lead
        
        # 2nd-Order Midpoint Integration on the COMPENSATED heading
        theta_mid = theta_compensated + (omega_filtered * dt) / 2.0
        
        # 2. Calculate the rotation matrices
        cos_t_mid = jnp.cos(theta_mid)
        sin_t_mid = jnp.sin(theta_mid)

        # 3. Rotate the local velocities into the global map frame
        V_map_x = vx * cos_t_mid - vy * sin_t_mid
        V_map_y = vx * sin_t_mid + vy * cos_t_mid

        v_mag_xy = jnp.sqrt(V_map_x**2 + V_map_y**2)

        spikes_xy = self.vel_coder_xy(v_mag_xy)
        spikes_th = self.vel_coder_th(jnp.abs(omega_filtered))

        dynamic_gain_xy = jnp.sum(spikes_xy * self.W_cereb_xy, axis=1)
        dynamic_gain_th = jnp.sum(spikes_th * self.W_cereb_th, axis=1)

        # Iterate over the 3 spatial modules
        # Scale velocity injection by dt/DT so bump displacement matches real elapsed time,
        # while neural field dynamics (decay, tau) remain at intrinsic DT.
        vel_time_scale = dt / DT
        for i, (c_size, scale) in enumerate(zip(CANN_SIZES, WRAP_SCALES)):
            r_flat = self._r_canns[i].reshape(B, -1)
            
            # Density scaling: Smaller wrap scales mean the bump must traverse neurons faster!
            density_factor = c_size / scale 
            scaled_vel_x = V_map_x * density_factor * vel_time_scale
            scaled_vel_y = V_map_y * density_factor * vel_time_scale

            I_vel_x = dynamic_gain_xy[:, None] * jnp.einsum('ij,bj->bi', self.W_cann_asym_x_list[i], r_flat) * scaled_vel_x[:, None]
            I_vel_y = dynamic_gain_xy[:, None] * jnp.einsum('ij,bj->bi', self.W_cann_asym_y_list[i], r_flat) * scaled_vel_y[:, None]

            u_flat = self._u_canns[i].reshape(B, -1)
            u_new = neural_field_update(
                u_flat, r_flat + 1e-8, self.W_cann_list[i],
                I_vel_x + I_vel_y, 
                dt=DT, tau=TAU_U
            )

            self._u_canns[i] = u_new.reshape(B, c_size, c_size)

            # Global Divisive Normalization for this specific module
            k_global_cann = 0.05
            r_raw = jnp.maximum(0.0, self._u_canns[i])
            raw_sum = r_raw.sum(axis=(1, 2), keepdims=True)
            global_inhibition = 1.0 + k_global_cann * raw_sum 
            baseline_inhibition = 1.0 + k_global_cann * 10.0 
            self._r_canns[i] = (r_raw / global_inhibition) * baseline_inhibition

        # ---- Ring: angular velocity injection ----
        I_vel_raw = -dynamic_gain_th[:, None] * omega_filtered[:, None] * vel_time_scale * jnp.einsum('ij,bj->bi', self.W_ring_asym, self._r_ring)
        I_vel_smooth = (jnp.roll(I_vel_raw, 1, axis=1) + I_vel_raw + jnp.roll(I_vel_raw, -1, axis=1)) / 3.0

        I_ext = I_vel_smooth
        if theta_gravity is not None:
            # preferred angles of 64 Ring Attractor neurons representing [-pi, pi]
            angles = jnp.arange(RING_N, dtype=jnp.float32) * (2.0 * jnp.pi / RING_N)
            diff = angles[None, :] - theta_gravity[:, None]
            diff_wrapped = jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi
            
            K_GRAVITY = 0.50
            SIGMA_GRAVITY = 0.25
            I_gravity = K_GRAVITY * jnp.exp(- (diff_wrapped ** 2) / (2.0 * (SIGMA_GRAVITY ** 2)))
            I_ext = I_ext + I_gravity

        u_ring_new = neural_field_update(
            self._u_ring, self._r_ring + 1e-8, self.W_ring,
            I_ext, 
            dt=DT, tau=RING_TAU_U
        )

        self._u_ring = u_ring_new

        # === REPAIRED GLOBAL DIVISIVE NORMALIZATION FOR RING ===
        k_global_ring = 0.1
        r_ring_raw = jnp.maximum(0.0, self._u_ring)
        ring_sum = r_ring_raw.sum(axis=1, keepdims=True)
        global_inhibition_ring = 1.0 + k_global_ring * ring_sum 
        baseline_ring_inh = 1.0 + k_global_ring * 6.0
        self._r_ring = (r_ring_raw / global_inhibition_ring) * baseline_ring_inh

        # ---- Readout ----
        # The CANN no longer decodes X/Y natively. The Orchestrator will do Phase Unwrapping!
        dummy_pos = jnp.zeros((B, 2)) 
        th = ring_readout(self._r_ring)[:, None]

        return jnp.concatenate([dummy_pos, th], axis=1)

    def update_cerebellum(self, kin_t, pose_xy, current_heading, dt=DT):
        # 🌟 BULLETPROOF INIT: Check if the lagged variables exist yet!
        # This catches cases where initialize_pose() pre-filled prev_pose_xy.
        if getattr(self, 'lagged_v_imu', None) is None or self.prev_pose_xy is None:
            self.prev_pose_xy = pose_xy
            self.prev_heading = current_heading
            
            # Prime the low-pass IMU target filters safely on the first frame
            self.lagged_v_imu = kin_t[:, 0]
            self.lagged_w_imu = kin_t[:, 2]
            return

        v_bump_x = (pose_xy[:, 0] - self.prev_pose_xy[:, 0]) / dt
        v_bump_y = (pose_xy[:, 1] - self.prev_pose_xy[:, 1]) / dt
        cos_t = jnp.cos(self.prev_heading)
        sin_t = jnp.sin(self.prev_heading)
        v_bump_forward = v_bump_x * cos_t + v_bump_y * sin_t

        v_bump_omega = (current_heading - self.prev_heading + jnp.pi) % (2 * jnp.pi) - jnp.pi
        v_bump_omega = v_bump_omega / dt

        v_imu_forward = kin_t[:, 0]
        v_imu_omega = kin_t[:, 2]

        # 🌟 Apply Neural Inertia (Low-Pass) to the IMU Targets
        decay_xy = jnp.exp(-dt / TAU_U)
        decay_th = jnp.exp(-dt / RING_TAU_U)
        
        self.lagged_v_imu = decay_xy * self.lagged_v_imu + (1 - decay_xy) * v_imu_forward
        self.lagged_w_imu = decay_th * self.lagged_w_imu + (1 - decay_th) * v_imu_omega

        # Calculate Error (Comparing Apples to Apples!)
        error_forward = jnp.clip(self.lagged_v_imu - v_bump_forward, -3.0, 3.0)
        error_omega = jnp.clip(self.lagged_w_imu - v_bump_omega, -3.0, 3.0)

        v_imu_mag = jnp.sqrt(kin_t[:, 0]**2 + kin_t[:, 1]**2)
        speed_spikes_xy = self.vel_coder_xy(v_imu_mag)
        speed_spikes_th = self.vel_coder_th(jnp.abs(v_imu_omega))

        base_eta_xy = 0.1  # Set learning rate to 0.1
        base_eta_th = 0.15  # Set learning rate to 0.15


        
        # Soften the adrenaline so it amplifies without exploding
        adrenaline_xy = 1.0 + 1.0 * jnp.abs(error_forward[:, None])
        adrenaline_th = 1.0 + 1.0 * jnp.abs(error_omega[:, None])

        dynamic_eta_xy = base_eta_xy * adrenaline_xy
        dynamic_eta_th = base_eta_th * adrenaline_th
        
        delta_W_xy = dynamic_eta_xy * error_forward[:, None] * jnp.sign(v_imu_forward)[:, None] * speed_spikes_xy
        delta_W_th = dynamic_eta_th * error_omega[:, None] * jnp.sign(v_imu_omega)[:, None] * speed_spikes_th

        self.W_cereb_xy = jnp.clip(self.W_cereb_xy + delta_W_xy, 0.001, 1.0)   # Allows correct gain scaling in 2m room
        self.W_cereb_th = jnp.clip(self.W_cereb_th + delta_W_th, 0.01,  0.40)  # Raised max heading gain to 0.40


        self.prev_pose_xy = pose_xy
        self.prev_heading = current_heading

    def get_state_flat(self):
        """Returns the 579-dim Cryptographic Grid Key"""
        B = self._r_canns[0].shape[0]
        flats = [r.reshape(B, -1) for r in self._r_canns]
        return jnp.concatenate(flats, axis=1)

    def get_ring_activity(self):
        return self._r_ring

    def estimate_heading(self):
        return ring_readout(self._r_ring)