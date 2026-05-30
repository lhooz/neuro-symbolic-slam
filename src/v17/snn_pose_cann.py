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

import sys
sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')

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
WRAP_SCALES = [3.0, 4.0, 5.3]    # Physical meters before wrapping
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

# Velocity injection gains (calibrated from snn_slam_twin.py)
VEL_GAIN_XY = 0.098      # velocity → bump shift — restore original
VEL_GAIN_TH = 0.172      # omega → ring shift — restore original

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
    theta = jnp.arctan2(sin_sum, cos_sum) % (2 * jnp.pi)
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
        diff = v_mag[:, None] - self.centers[None, :]
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
        
        self.n_speed_neurons = 32
        self.vel_coder_xy = VelocityPopulationCoder(self.n_speed_neurons, 0.0, 1.5, 0.1)
        self.vel_coder_th = VelocityPopulationCoder(self.n_speed_neurons, 0.0, 3.14, 0.2)
        
        self.prev_pose_xy = None
        self.prev_heading = None

    def reset(self, B):
        """Reset to centered Gaussian bumps (fallback initialization)."""

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
        
        self.W_cereb_xy = jnp.ones((B, self.n_speed_neurons)) * VEL_GAIN_XY
        self.W_cereb_th = jnp.ones((B, self.n_speed_neurons)) * VEL_GAIN_TH
        
        self.prev_pose_xy = None
        self.prev_heading = None

    def initialize_from_gt(self, gt_pos, gt_heading):
        B = gt_pos.shape[0]
        
        # 1. Initialize the 3 Grid Modules using Modulo Math
        for i, (c_size, scale) in enumerate(zip(CANN_SIZES, WRAP_SCALES)):
            # Find the local phase (0 to scale)
            local_x = gt_pos[:, 0] % scale
            local_y = gt_pos[:, 1] % scale
            
            # Map physical phase to neuron indices
            cx_float = (local_x / scale) * c_size
            cy_float = (local_y / scale) * c_size

            xx, yy = jnp.meshgrid(
                jnp.arange(c_size, dtype=jnp.float32),
                jnp.arange(c_size, dtype=jnp.float32),
                indexing='xy'
            )

            cx_exp = cx_float[:, None, None]
            cy_exp = cy_float[:, None, None]
            
            # Draw the bump on the toroidal surface (accounting for edges)
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

    def __call__(self, kin_t):
        B = kin_t.shape[0]
        vx, vy, omega = kin_t[:, 0], kin_t[:, 1], kin_t[:, 2]

        theta = ring_readout(self._r_ring)
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)

        V_map_x = vx * cos_t - vy * sin_t
        V_map_y = vx * sin_t + vy * cos_t

        v_mag_xy = jnp.sqrt(V_map_x**2 + V_map_y**2)

        spikes_xy = self.vel_coder_xy(v_mag_xy)
        spikes_th = self.vel_coder_th(jnp.abs(omega))

        dynamic_gain_xy = jnp.sum(spikes_xy * self.W_cereb_xy, axis=1)
        dynamic_gain_th = jnp.sum(spikes_th * self.W_cereb_th, axis=1)

        # Iterate over the 3 spatial modules
        for i, (c_size, scale) in enumerate(zip(CANN_SIZES, WRAP_SCALES)):
            r_flat = self._r_canns[i].reshape(B, -1)
            
            # Density scaling: Smaller wrap scales mean the bump must traverse neurons faster!
            density_factor = c_size / scale 
            scaled_vel_x = V_map_x * density_factor
            scaled_vel_y = V_map_y * density_factor

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
        I_vel_raw = -dynamic_gain_th[:, None] * omega[:, None] * jnp.einsum('ij,bj->bi', self.W_ring_asym, self._r_ring)
        I_vel_smooth = (jnp.roll(I_vel_raw, 1, axis=1) + I_vel_raw + jnp.roll(I_vel_raw, -1, axis=1)) / 3.0

        u_ring_new = neural_field_update(
            self._u_ring, self._r_ring + 1e-8, self.W_ring,
            I_vel_smooth, 
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
        if self.prev_pose_xy is None:
            self.prev_pose_xy = pose_xy
            self.prev_heading = current_heading
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

        # 🌟 CEREBELLUM ENHANCEMENT 1: Expand the Error Bounds!
        # Let the cerebellum actually "feel" the massive error of a sudden jerk.
        error_forward = jnp.clip(v_imu_forward - v_bump_forward, -3.0, 3.0)
        error_omega = jnp.clip(v_imu_omega - v_bump_omega, -3.0, 3.0)

        v_imu_mag = jnp.sqrt(kin_t[:, 0]**2 + kin_t[:, 1]**2)
        speed_spikes_xy = self.vel_coder_xy(v_imu_mag)
        speed_spikes_th = self.vel_coder_th(jnp.abs(v_imu_omega))

        # 🌟 CEREBELLUM ENHANCEMENT 2: Neuromodulation (Dynamic ETA)
        base_eta_xy = 0.1  
        base_eta_th = 0.22
        
        # Calculate Adrenaline: Scale learning rate by the severity of the mistake!
        # If the error is near 0, learning rate stays normal. 
        adrenaline_xy = 1.0 + 1.0 * jnp.abs(error_forward[:, None])
        adrenaline_th = 1.0 + 1.0 * jnp.abs(error_omega[:, None])

        dynamic_eta_xy = base_eta_xy * adrenaline_xy
        dynamic_eta_th = base_eta_th * adrenaline_th
        
        delta_W_xy = dynamic_eta_xy * error_forward[:, None] * jnp.sign(v_imu_forward)[:, None] * speed_spikes_xy
        delta_W_th = dynamic_eta_th * error_omega[:, None] * jnp.sign(v_imu_omega)[:, None] * speed_spikes_th

        self.W_cereb_xy = jnp.clip(self.W_cereb_xy + delta_W_xy, 0.01, 0.60)
        self.W_cereb_th = jnp.clip(self.W_cereb_th + delta_W_th, 0.01, 0.60)

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