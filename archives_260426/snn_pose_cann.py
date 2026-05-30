#!/usr/bin/env python3
"""
snn_pose_cann.py — The "Where Am I?" Module

2D Spatial CANN (x, y) + 1D Ring Attractor (θ) with ANALYTICAL DoG weights.
Receives IMU velocity injection and corrective current from the Map module.

Key changes for v3 — Parallel Ring Memory + Global Divisive Normalization:
  - Dual map correction inputs: map_correction_place AND map_correction_ring
  - Ring injection: map_correction_ring is NO LONGER zeroed out
  - Max-norm clipping → GLOBAL DIVISIVE NORMALIZATION (k_cann=0.05, k_ring=0.1)

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

# CANN (2D spatial sheet)
CANN_SIZE = 32          # 32×32 = 1024 neurons
A_EXC = 0.5             # area_exc = 0.5×1² = 0.5
A_INH = 0.125           # area_inh = 0.125×2² = 0.5 (area-balanced!)
SIGMA_EXC = 1.0         # excitatory radius (cells)
SIGMA_INH = 2.0         # inhibitory radius (cells)
TAU_U = 0.05            # membrane time constant (seconds)

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

def build_2d_cann_weights(cann_size=CANN_SIZE, a_exc=A_EXC, a_inh=A_INH,
                           sigma_exc=SIGMA_EXC, sigma_inh=SIGMA_INH):
    """Build 2D CANN recurrent weight matrix: W_ij = A_exc·G_exc(d_ij) - A_inh·G_inh(d_ij)

    Fully vectorized: no Python loops. Uses broadcasting to compute all pairwise
    toroidal distances and DoG responses simultaneously.
    Returns: (cann_size², cann_size²) matrix
    """
    x_1d = jnp.arange(cann_size, dtype=jnp.float32)

    # Source coords: (1, 1, src_y, src_x) broadcast over all destinations
    src_x_4d = jnp.broadcast_to(x_1d[None, None, None, :], (cann_size, cann_size, cann_size, cann_size))
    src_y_4d = jnp.broadcast_to(x_1d[None, None, :, None], (cann_size, cann_size, cann_size, cann_size))

    # Destination coords: (dst_y, dst_x, 1, 1) broadcast over all sources
    dst_x_4d = jnp.broadcast_to(x_1d[None, :, None, None], (cann_size, cann_size, cann_size, cann_size))
    dst_y_4d = jnp.broadcast_to(x_1d[:, None, None, None], (cann_size, cann_size, cann_size, cann_size))

    # Toroidal wrapped coordinate differences: min(|Δ|, N-|Δ|)
    dx = jnp.minimum(jnp.abs(src_x_4d - dst_x_4d), cann_size - jnp.abs(src_x_4d - dst_x_4d))
    dy = jnp.minimum(jnp.abs(src_y_4d - dst_y_4d), cann_size - jnp.abs(src_y_4d - dst_y_4d))
    d2 = dx**2 + dy**2

    # DoG (unnormalized): a_exc·G_exc - a_inh·G_inh
    W_4d_unnorm = (a_exc * jnp.exp(-d2 / (2 * sigma_exc**2))
                   - a_inh * jnp.exp(-d2 / (2 * sigma_inh**2)))

    # Normalize: self-conn = (a_exc - a_inh) = 0.375, final self-conn = 0.5
    self_conn_unnorm = a_exc - a_inh  # = 0.375
    W_4d = (W_4d_unnorm / (self_conn_unnorm + 1e-8))

    # Reshape: (N_dest_y, N_dest_x, N_src_y, N_src_x) → (N, N)
    W = W_4d.reshape(cann_size * cann_size, cann_size * cann_size)
    return W


def build_1d_ring_weights(ring_n=RING_N, a_exc=RING_A_EXC, a_inh=RING_A_INH,
                          sigma_exc=RING_SIGMA_EXC, sigma_inh=RING_SIGMA_INH):
    """Build 1D ring attractor weight matrix using INTEGER INDEX distance.

    CRITICAL: σ values are in neuron units, NOT radian units.
    At d=0:  W = A_exc - A_inh = +0.50 (self-excitation)
    At d=4-5: W < 0 (inhibitory surround → Mexican Hat!)
    """
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
    """Asymmetric weight matrix for ω → bump shift.

    W_asym[i,j] = d/dn G(n_i - n_j) — odd function.
    Positive when j is "ahead" (positive omega direction).
    Returns: (ring_n, ring_n)
    """
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


def build_asymmetric_cann_weights_x(cann_size=CANN_SIZE, sigma=SIGMA_EXC):
    """Asymmetric weight matrix for vx → bump shift along x-axis.

    W_asym[(y,x), (yc,xc)] = d/dx G(x-xc) · G(y-yc)
    Fully vectorized: einsum computes all 4D outer products at once.
    Returns: (N, N) where N = cann_size²
    """
    x_c = jnp.arange(cann_size, dtype=jnp.float32)
    y_c = jnp.arange(cann_size, dtype=jnp.float32)

    dx = x_c[:, None] - x_c[None, :]
    dx = jnp.where(dx > cann_size/2, dx - cann_size, dx)
    dx = jnp.where(dx < -cann_size/2, dx + cann_size, dx)
    k_x = dx * jnp.exp(-dx**2 / (2 * sigma**2))  # DoG derivative along x

    dy = y_c[:, None] - y_c[None, :]
    dy = jnp.where(dy > cann_size/2, dy - cann_size, dy)
    dy = jnp.where(dy < -cann_size/2, dy + cann_size, dy)
    G_y = jnp.exp(-dy**2 / (2 * sigma**2))  # symmetric Gaussian along y

    k_x_4d = k_x[None, :, None, :]
    G_y_4d = G_y[:, None, :, None]
    W_4d = k_x_4d * G_y_4d  # (y, x, yc, xc)

    W = W_4d.reshape(cann_size * cann_size, cann_size * cann_size)
    W = W / (jnp.abs(W).max() + 1e-8)
    return W


def build_asymmetric_cann_weights_y(cann_size=CANN_SIZE, sigma=SIGMA_EXC):
    """Asymmetric weight matrix for vy → bump shift along y-axis.

    W_asym[(y,x), (yc,xc)] = G(x-xc) · d/dy G(y-yc)
    Fully vectorized: same broadcasting trick as _x variant.
    Returns: (N, N) where N = cann_size²
    """
    x_c = jnp.arange(cann_size, dtype=jnp.float32)
    y_c = jnp.arange(cann_size, dtype=jnp.float32)

    dx = x_c[:, None] - x_c[None, :]
    dx = jnp.where(dx > cann_size/2, dx - cann_size, dx)
    dx = jnp.where(dx < -cann_size/2, dx + cann_size, dx)
    G_x = jnp.exp(-dx**2 / (2 * sigma**2))  # symmetric Gaussian along x

    dy = y_c[:, None] - y_c[None, :]
    dy = jnp.where(dy > cann_size/2, dy - cann_size, dy)
    dy = jnp.where(dy < -cann_size/2, dy + cann_size, dy)
    k_y = dy * jnp.exp(-dy**2 / (2 * sigma**2))  # DoG derivative along y

    G_x_4d = G_x[None, :, None, :]  # (1, x, 1, xc)
    k_y_4d = k_y[:, None, :, None]  # (y, 1, yc, 1)
    W_4d = G_x_4d * k_y_4d  # broadcasting → (y, x, yc, xc)

    W = W_4d.reshape(cann_size * cann_size, cann_size * cann_size)
    W = W / (jnp.abs(W).max() + 1e-8)
    return W


# ============================================================================
# 2. Neural Field Dynamics
# ============================================================================

def neural_field_update(u, r, W, I_ext, dt=DT, tau=0.05):
    """Discrete neural field: τ·du/dt = -u + W@r + I_ext

    r: (B, N), W: (N, N), I_ext: (B, N)
    """
    decay = 1.0 - dt / tau
    drive = (jnp.einsum('ij,bj->bi', W, r) + I_ext) * (dt / tau)
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

def cann_readout(state, cann_size=CANN_SIZE):
    """Read (x, y) from CANN using circular statistics.

    state: (B, cann_size, cann_size)
    Returns: (B, 2) in world meters
    """
    angles = jnp.arange(cann_size, dtype=jnp.float32) * (2 * jnp.pi / cann_size)
    sin_a = jnp.sin(angles)
    cos_a = jnp.cos(angles)

    p = state / (state.sum(axis=(1, 2), keepdims=True) + 1e-8)
    p_x = p.sum(axis=1)
    p_y = p.sum(axis=2)

    cx_angle = jnp.arctan2((p_x * sin_a).sum(axis=1), (p_x * cos_a).sum(axis=1)) % (2 * jnp.pi)
    cx = cx_angle * (cann_size / (2 * jnp.pi))

    cy_angle = jnp.arctan2((p_y * sin_a).sum(axis=1), (p_y * cos_a).sum(axis=1)) % (2 * jnp.pi)
    cy = cy_angle * (cann_size / (2 * jnp.pi))

    mpc = ROOM_W / cann_size
    center = cann_size / 2
    x = (cx - center) * mpc + ROOM_W / 2
    y = (cy - center) * mpc + ROOM_H / 2
    return jnp.stack([x, y], axis=1)


def ring_readout(state, ring_n=RING_N):
    """Read θ from ring using circular statistics (arctan2 of sin/cos).

    Correctly handles 0°/360° boundary.
    state: (B, ring_n)
    Returns: (B,) in radians
    """
    angles = jnp.arange(ring_n, dtype=jnp.float32) * (2 * jnp.pi / ring_n)
    p = state / (state.sum(axis=1, keepdims=True) + 1e-8)
    sin_sum = (jnp.sin(angles) * p).sum(axis=1)
    cos_sum = (jnp.cos(angles) * p).sum(axis=1)
    theta = jnp.arctan2(sin_sum, cos_sum) % (2 * jnp.pi)
    return theta


# ============================================================================
# 4. Map Correction Injection (Loop Closure Signal)
# ============================================================================

def map_correction_to_cann(map_centroid, pose_est, cann_size=CANN_SIZE):
    """Convert Map CANN centroid to corrective current for Pose CANN.

    The Map's active region centroid (in world coords) is compared to
    the Pose CANN's current estimate, and a Gaussian correction bump
    is injected to pull the pose estimate toward the map's belief.

    map_centroid: (B, 2) — world coords [x, y] of map activity centroid
    pose_est:     (B, 3) — current pose [x, y, θ]

    Returns: (B, CANN_SIZE, CANN_SIZE) Gaussian correction bump
    """
    mx, my = map_centroid[:, 0], map_centroid[:, 1]

    # World → cell conversion (same as cann_readout inverse)
    mpc = float(ROOM_W) / cann_size
    center = cann_size / 2.0

    cx_cell = mx / mpc
    cy_cell = my / mpc

    xx, yy = jnp.meshgrid(
        jnp.arange(cann_size, dtype=jnp.float32),
        jnp.arange(cann_size, dtype=jnp.float32),
        indexing='xy'
    )

    def _bump(cx, cy):
        d2 = (xx - cx)**2 + (yy - cy)**2
        sigma = 2.0  # correction spread in cells
        bump = jnp.exp(-d2 / (2 * sigma**2))
        return bump / (bump.sum() + 1e-8)

    corr = jax.vmap(_bump)(cx_cell, cy_cell)
    return corr * MAP_CORR_GAIN_XY


def map_correction_to_ring(map_heading, pose_est, ring_n=RING_N):
    """Convert Map-derived heading belief to corrective ring current."""
    current_th = pose_est[:, 2]  # current ring heading
    delta_th = (map_heading - current_th + jnp.pi) % (2 * jnp.pi) - jnp.pi  # signed diff

    # Target is converted to neuron index space [0, 64)
    target_idx_float = (current_th + delta_th) * (ring_n / (2 * jnp.pi))

    # Base array is also in neuron index space [0, 64)
    indices = jnp.arange(ring_n, dtype=jnp.float32)

    def _bump(tgt_idx):
        # Everything is calculated in index units now
        d = jnp.abs(indices - tgt_idx)
        d = jnp.minimum(d, ring_n - d)
        sigma = 2.0  # sigma is simply 2.0 neurons
        bump = jnp.exp(-d**2 / (2 * sigma**2))
        return bump / (bump.sum() + 1e-8)

    # vmap only needs to map over the target index
    corr = jax.vmap(_bump)(target_idx_float)
    return corr * MAP_CORR_GAIN_TH

# ============================================================================
# CEREBELLUM: Granule Cell Population Coder
# ============================================================================
class VelocityPopulationCoder:
    """Converts scalar velocity magnitude into a 1D Gaussian population code."""
    def __init__(self, num_neurons=32, min_v=0.0, max_v=1.5, sigma=0.1):
        self.num_neurons = num_neurons
        # The preferred speeds of the 32 neurons
        self.centers = jnp.linspace(min_v, max_v, num_neurons)
        self.sigma = sigma

    def __call__(self, v_mag):
        # v_mag: (B,)
        diff = v_mag[:, None] - self.centers[None, :]
        activations = jnp.exp(-(diff ** 2) / (2 * self.sigma ** 2))
        # Normalize so the population always sums to 1.0
        return activations / (activations.sum(axis=1, keepdims=True) + 1e-8)

# ============================================================================
# 5. Pose CANN Class
# ============================================================================

class PoseCANN:
    """Neuromorphic Pose Estimator — "Where Am I?"

    2D CANN (x, y) + 1D Ring Attractor (θ) with analytical DoG weights.
    Receives IMU velocity injection for dead-reckoning and DUAL corrective
    currents from Map CANN for loop closure (position AND heading).

    v3 Changes:
      - Dual map correction inputs: map_correction_place + map_correction_ring
      - Ring injection: ring correction NO LONGER zeroed — actually injected
      - Global Divisive Normalization replaces max-norm clipping throughout
    """

    def __init__(self, key, W_cann, W_ring, W_cann_asym_x, W_cann_asym_y, W_ring_asym):
        k1, k2 = random.split(key)
        self.W_cann = W_cann
        self.W_ring = W_ring
        self.W_cann_asym_x = W_cann_asym_x
        self.W_cann_asym_y = W_cann_asym_y
        self.W_ring_asym = W_ring_asym

        self._v_lif = None
        self._u_cann = None
        self._r_cann = None
        self._u_ring = None
        self._r_ring = None
        # 🧠 SNN Cerebellum Setup
        self.n_speed_neurons = 32
        self.vel_coder_xy = VelocityPopulationCoder(self.n_speed_neurons, 0.0, 1.5, 0.1)
        self.vel_coder_th = VelocityPopulationCoder(self.n_speed_neurons, 0.0, 3.14, 0.2)
        
        # State tracking for the Efference Copy
        self.prev_pose_xy = None
        self.prev_heading = None

    def reset(self, B):
        """Reset to centered Gaussian bump (fallback initialization)."""
        self._v_lif = jnp.zeros((B, 1))  # placeholder, no LIF used in path integration

        xx, yy = jnp.meshgrid(
            jnp.arange(CANN_SIZE, dtype=jnp.float32),
            jnp.arange(CANN_SIZE, dtype=jnp.float32),
            indexing='ij'
        )
        bump = jnp.exp(-((xx - CANN_SIZE//2)**2 + (yy - CANN_SIZE//2)**2) / (2 * SIGMA_EXC**2))
        bump = bump / (bump.max() + 1e-8)
        self._u_cann = jnp.tile(0.5 * bump[None, :, :], (B, 1, 1))
        self._r_cann = jnp.clip(self._u_cann, 0, 1)

        idx = jnp.arange(RING_N, dtype=jnp.float32)
        d = jnp.abs(idx - 0.0)
        d = jnp.minimum(d, RING_N - d)
        bump_r = jnp.exp(-d**2 / (2 * RING_SIGMA_EXC**2))
        bump_r = bump_r / (bump_r.sum() + 1e-8)
        self._u_ring = jnp.tile(0.5 * bump_r[None, :], (B, 1))
        self._r_ring = jnp.clip(self._u_ring, 0, 1.0)
        # 🧠 Initialize the plastic Purkinje Synapses (Weights)
        # Shape: (Batch, N_Speed_Neurons). They start at the baseline gain.
        self.W_cereb_xy = jnp.ones((B, self.n_speed_neurons)) * VEL_GAIN_XY
        self.W_cereb_th = jnp.ones((B, self.n_speed_neurons)) * VEL_GAIN_TH
        
        self.prev_pose_xy = None
        self.prev_heading = None

    def initialize_from_gt(self, gt_pos, gt_heading):
        """Initialize CANN and Ring at GT pose for proper trajectory tracking.

        gt_pos:     (B, 2) in world meters [x, y]
        gt_heading:  (B,) in radians
        """
        B = gt_pos.shape[0]
        mpc = float(ROOM_W) / CANN_SIZE
        cx_float = (gt_pos[:, 0] - ROOM_W / 2) / mpc + CANN_SIZE / 2
        cy_float = (gt_pos[:, 1] - ROOM_H / 2) / mpc + CANN_SIZE / 2

        xx, yy = jnp.meshgrid(
            jnp.arange(CANN_SIZE, dtype=jnp.float32),
            jnp.arange(CANN_SIZE, dtype=jnp.float32),
            indexing='xy'
        )

        cx_exp = cx_float[:, None, None]
        cy_exp = cy_float[:, None, None]
        d2 = (xx[None, :, :] - cx_exp)**2 + (yy[None, :, :] - cy_exp)**2
        bumps = jnp.exp(-d2 / (2 * SIGMA_EXC**2))
        bumps = bumps / (bumps.max(axis=(1, 2), keepdims=True) + 1e-8)
        self._u_cann = jnp.clip(bumps, 0, 1.0)
        self._r_cann = self._u_cann

        th_idx_float = (gt_heading % (2 * jnp.pi)) * (RING_N / (2 * jnp.pi))
        idx = jnp.arange(RING_N, dtype=jnp.float32)[None, :]
        th_exp = th_idx_float[:, None]
        d = jnp.abs(idx - th_exp)
        d = jnp.minimum(d, RING_N - d)
        bumps_r = jnp.exp(-d**2 / (2 * RING_SIGMA_EXC**2))
        bumps_r = bumps_r / (bumps_r.max(axis=1, keepdims=True) + 1e-8)
        self._u_ring = jnp.clip(bumps_r, 0, 1.0)
        self._r_ring = self._u_ring

    def __call__(self, kin_t, map_correction_place=None, map_correction_ring=None):
        """One timestep update of Pose CANN with DUAL map corrections.

        kin_t:               (B, 3) — [vx, vy, omega] from IMU (m/s, rad/s)
        map_correction_place: (B, CANN_SIZE, CANN_SIZE) OR None —
                              corrective current from PlaceCellNetwork (position)
        map_correction_ring:  (B, RING_N) OR None —
                              corrective current from PlaceCellNetwork (heading)

        Both map corrections default to zero if None (no loop closure signal).

        Returns: (B, 3) — [x̂, ŷ, θ̂] pose estimate
        """
        B = kin_t.shape[0]
        vx, vy, omega = kin_t[:, 0], kin_t[:, 1], kin_t[:, 2]

        # ---- CANN: Body-frame → Global-frame velocity rotation ----
        theta = ring_readout(self._r_ring)  # (B,)
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)

        V_map_x = vx * cos_t - vy * sin_t
        V_map_y = vx * sin_t + vy * cos_t

        # 🧠 SNN CEREBELLUM INJECTION (Fixed Vector Math!)
        # 1. Calculate True Speed (Vector Magnitude) to prevent X/Y decoupling
        v_mag_xy = jnp.sqrt(V_map_x**2 + V_map_y**2)

        # 2. Activate Granule Cells based on True Speed
        spikes_xy = self.vel_coder_xy(v_mag_xy)
        spikes_th = self.vel_coder_th(jnp.abs(omega))

        # 3. Purkinje cell output: Dynamic Multiplier
        dynamic_gain_xy = jnp.sum(spikes_xy * self.W_cereb_xy, axis=1)
        dynamic_gain_th = jnp.sum(spikes_th * self.W_cereb_th, axis=1)

        # 4. Asymmetric weight injection 
        # (RESTORED V_map_x/y multiplier to preserve the analog heading vector!)
        r_flat = self._r_cann.reshape(B, -1)
        I_vel_x = dynamic_gain_xy[:, None] * jnp.einsum('ij,bj->bi', self.W_cann_asym_x, r_flat) * V_map_x[:, None]
        I_vel_y = dynamic_gain_xy[:, None] * jnp.einsum('ij,bj->bi', self.W_cann_asym_y, r_flat) * V_map_y[:, None]

        # Place map correction (loop closure position signal)
        if map_correction_place is not None:
            I_map_corr_place = map_correction_place.reshape(B, -1)
        else:
            I_map_corr_place = jnp.zeros((B, CANN_SIZE * CANN_SIZE))

        # Neural field update
        u_flat = self._u_cann.reshape(B, -1)
        u_new = neural_field_update(
            u_flat, r_flat + 1e-8, self.W_cann,
            I_vel_x + I_vel_y + I_map_corr_place,
            dt=DT, tau=TAU_U
        )

        self._u_cann = u_new.reshape(B, CANN_SIZE, CANN_SIZE)

        # === GLOBAL DIVISIVE NORMALIZATION (v3) ===
        # Replaces: r_cann = jnp.clip(u_cann, 0, max_val) / max_val
        # Prevents runaway excitation while preserving relative competition
        k_global_cann = 0.05  # strength of global inhibition
        r_raw = jnp.maximum(0.0, self._u_cann)
        global_inhibition = 1.0 + k_global_cann * r_raw.sum(axis=(1, 2), keepdims=True)
        self._r_cann = r_raw / global_inhibition

        # ---- Ring: angular velocity injection ----
        # (Restored the raw omega multiplier here too!)
        I_vel_raw = -dynamic_gain_th[:, None] * omega[:, None] * jnp.einsum('ij,bj->bi', self.W_ring_asym, self._r_ring)
        I_vel_smooth = (jnp.roll(I_vel_raw, 1, axis=1) + I_vel_raw + jnp.roll(I_vel_raw, -1, axis=1)) / 3.0

        # Ring map correction (loop closure heading signal) — NOW ACTUALLY INJECTED!
        if map_correction_ring is not None:
            I_map_corr_ring = map_correction_ring
        else:
            I_map_corr_ring = jnp.zeros((B, RING_N))

        u_ring_new = neural_field_update(
            self._u_ring, self._r_ring + 1e-8, self.W_ring,
            I_vel_smooth + I_map_corr_ring,
            dt=DT, tau=RING_TAU_U
        )

        self._u_ring = u_ring_new

        # === GLOBAL DIVISIVE NORMALIZATION (v3) ===
        # Replaces: r_ring = jnp.clip(u_ring, 0, max_val) / max_val
        k_global_ring = 0.1  # slightly stronger for ring (fewer neurons)
        r_ring_raw = jnp.maximum(0.0, self._u_ring)
        global_inhibition_ring = 1.0 + k_global_ring * r_ring_raw.sum(axis=1, keepdims=True)
        self._r_ring = r_ring_raw / global_inhibition_ring

        # ---- Readout ----
        pos = cann_readout(self._r_cann)
        th = ring_readout(self._r_ring)[:, None]

        return jnp.concatenate([pos, th], axis=1)

    def update_cerebellum(self, kin_t, pose_xy, current_heading, dt=DT):
        """Efference Copy Learning: Updates specific speed synapses based on bump drag."""
        if self.prev_pose_xy is None:
            self.prev_pose_xy = pose_xy
            self.prev_heading = current_heading
            return

        # 1. ACTUAL Bump Velocity (Proprioception Reality)
        v_bump_x = (pose_xy[:, 0] - self.prev_pose_xy[:, 0]) / dt
        v_bump_y = (pose_xy[:, 1] - self.prev_pose_xy[:, 1]) / dt
        cos_t = jnp.cos(self.prev_heading)
        sin_t = jnp.sin(self.prev_heading)
        v_bump_forward = v_bump_x * cos_t + v_bump_y * sin_t

        v_bump_omega = (current_heading - self.prev_heading + jnp.pi) % (2 * jnp.pi) - jnp.pi
        v_bump_omega = v_bump_omega / dt

        # 2. COMMANDED Velocity (Efference Copy)
        v_imu_forward = kin_t[:, 0]
        v_imu_omega = kin_t[:, 2]

        # 3. ERROR (Climbing Fiber Spikes)
        # 👇 You are likely missing these two lines! 👇
        error_forward = v_imu_forward - v_bump_forward
        error_omega = v_imu_omega - v_bump_omega

        # 🛑 THE FIX 1: Error Clipping (Gradient Clipping)
        # Prevent massive 1-frame velocity spikes from blowing up the weights
        error_forward = jnp.clip(error_forward, -1.0, 1.0)
        error_omega = jnp.clip(error_omega, -1.0, 1.0)

        # 4. RE-ACTIVATE GRANULE CELLS (Which speed neurons are responsible?)
        speed_spikes_xy = self.vel_coder_xy(jnp.abs(v_imu_forward))
        speed_spikes_th = self.vel_coder_th(jnp.abs(v_imu_omega))

        # 5. SNN HEBBIAN PLASTICITY
        # 🛑 THE FIX 2: Lower the learning rate! 
        # The Cerebellum needs to learn patiently over seconds, not instantly in 1 frame.
        eta_xy = 0.05  # lowered from 0.05
        eta_th = 0.05  # lowered from 0.05
        
        # Matrix multiply: Error * Direction * Presynaptic Spikes
        delta_W_xy = eta_xy * error_forward[:, None] * jnp.sign(v_imu_forward)[:, None] * speed_spikes_xy
        delta_W_th = eta_th * error_omega[:, None] * jnp.sign(v_imu_omega)[:, None] * speed_spikes_th

        # Update and clip to biological limits
        self.W_cereb_xy = jnp.clip(self.W_cereb_xy + delta_W_xy, 0.01, 0.60)
        self.W_cereb_th = jnp.clip(self.W_cereb_th + delta_W_th, 0.01, 0.60)

        # Update state
        self.prev_pose_xy = pose_xy
        self.prev_heading = current_heading

    def get_state_flat(self):
        """Return flattened CANN activity (B, CANN_SIZE²) for Hebbian weight updates."""
        return self._r_cann.reshape(-1, CANN_SIZE * CANN_SIZE)

    def get_ring_activity(self):
        """Return Ring activity (B, RING_N) for Hebbian weight updates."""
        return self._r_ring

    def estimate_position(self):
        """Read (x, y) position from CANN bump — returns (B, 2) in world meters."""
        return cann_readout(self._r_cann)

    def estimate_heading(self):
        """Read θ heading from Ring bump — returns (B,) in radians."""
        return ring_readout(self._r_ring)
