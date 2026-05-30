#!/usr/bin/env python3
"""
snn_map_cann.py — The Hippocampus: Place Cell Map with Hebbian Memory

Split-Brain Architecture: The Digital Twin's spatial memory module.

Biological analogy:
  • Place Cells  → 2D CANN sheet (32×32 = 1024 neurons)
  • Dentate Gyrus → Sparse vision features (STDP frontend)
  • Perforant Path → Hebbian W_vis_to_map memory matrix
  • CA3 Recurrent → DoG lateral inhibition on place cell sheet

Key design decisions:
  1. NO direct sensory input (Events/ToF go to STDP first)
  2. Vision arrives as sparse STDP spike features (24 winners, 256-dim)
  3. Pose arrives as flattened CANN bump (1024-dim, one-hot-ish)
  4. Hebbian outer product: ΔW = η · (pose_bump ⊗ vision_spikes) − decay·W
  5. Loop closure: I_correction = vision_spikes @ W_vis_to_map.T
     — creates "ghost bump" at remembered location
     — inject into Pose CANN to overpower IMU drift

Memory trace (EMA of co-activations):
  trace = α·trace + (1−α)·(pose_bump ⊗ vision_spikes)
  ΔW = η·(trace − γ·W)   [Oja's rule with forgetting]

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import sys
sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')

from src.sparse_forest import DT, ROOM_W, ROOM_H, FOV_DEG, N_PIXELS

# ============================================================================
# Configuration
# ============================================================================

MAP_SIZE = 32           # 32×32 = 1024 place cells
N_PLACE = MAP_SIZE * MAP_SIZE
N_STDP_FEATURES = 256  # matches VisionSTDP hidden neurons

# Place Cell recurrent DoG weights
A_EXC_MAP = 0.5
A_INH_MAP = 0.125
SIGMA_EXC_MAP = 1.0
SIGMA_INH_MAP = 2.0
TAU_U_MAP = 0.05

# Spiking
BETA_LIF_MAP = 0.85
V_TH_MAP = 1.0

# Hebbian Memory Matrix: W_vis_to_map[place_idx, feature_idx]
ETA_HEBB = 0.03
MEM_DECAY = 0.9995    # per-step weight decay (slow forgetting)
GAMMA_HEBB = 0.999    # Oja forgetting factor

# Memory trace (EMA of co-activations)
TRACE_ALPHA = 0.85

# Loop closure
CORRECTION_GAIN = 0.05       # moderate ghost bump strength (subtle corrections)
CORRECTION_SMOOTH = 0.80

USE_PLASTIC_MAP = True


# ============================================================================
# 1. Analytical DoG Weight Matrix (4D broadcasting, same as snn_pose_cann.py)
# ============================================================================

def build_map_dog_weights(map_size=MAP_SIZE,
                          a_exc=A_EXC_MAP, a_inh=A_INH_MAP,
                          sigma_exc=SIGMA_EXC_MAP, sigma_inh=SIGMA_INH_MAP):
    """Build 2D Place Cell recurrent weight matrix.

    4D broadcasting for correct toroidal DoG. All neurons share the same
    translation-invariant DoG profile. lambda_max ≈ 1.97 (stable attractor).
    """
    x_1d = jnp.arange(map_size, dtype=jnp.float32)

    src_x_4d = jnp.broadcast_to(x_1d[None, None, None, :], (map_size, map_size, map_size, map_size))
    src_y_4d = jnp.broadcast_to(x_1d[None, None, :, None], (map_size, map_size, map_size, map_size))
    dst_x_4d = jnp.broadcast_to(x_1d[None, :, None, None], (map_size, map_size, map_size, map_size))
    dst_y_4d = jnp.broadcast_to(x_1d[:, None, None, None], (map_size, map_size, map_size, map_size))

    dx = jnp.minimum(jnp.abs(src_x_4d - dst_x_4d), map_size - jnp.abs(src_x_4d - dst_x_4d))
    dy = jnp.minimum(jnp.abs(src_y_4d - dst_y_4d), map_size - jnp.abs(src_y_4d - dst_y_4d))
    d2 = dx**2 + dy**2

    W_4d = (a_exc * jnp.exp(-d2 / (2 * sigma_exc**2))
              - a_inh * jnp.exp(-d2 / (2 * sigma_inh**2)))

    self_conn_analytical = a_exc - a_inh  # = 0.375
    W_4d = (W_4d / (self_conn_analytical + 1e-8)) * 0.5

    W = W_4d.reshape(map_size * map_size, map_size * map_size)
    return W


# ============================================================================
# 2. Neural Field Dynamics
# ============================================================================

def neural_field_update(u, r, W, I_ext, dt=DT, tau=TAU_U_MAP):
    """Discrete neural field: τ·du/dt = −u + W @ r + I_ext"""
    decay = 1.0 - dt / tau
    drive = (jnp.einsum('ij,bj->bi', W, r) + I_ext) * (dt / tau)
    return decay * u + drive


def lif_step_map(v, beta=BETA_LIF_MAP, v_th=V_TH_MAP):
    """LIF spike for place cells."""
    v_new = beta * v + (1 - beta) * v_th
    spike = jnp.clip(v_new - v_th, 0.0, v_th)
    v_new = v_new - spike
    return v_new, spike


# ============================================================================
# 3. Place Cell Map — The Hippocampus
# ============================================================================

class PlaceCellMap:
    """Neuromorphic Spatial Memory — Hippocampal Place Cell Network.

    Forward flow (encoding):
      vision_spikes (B, 256) + pose_bump (B, 1024)
        → memory_trace (EMA of pose⊗vision)
        → ΔW = η·(trace − γ·W)  [Oja's rule with forgetting]
        → DoG recurrent dynamics on place cell sheet
        → r_map (B, 32, 32)

    Backward flow (retrieval):
      vision_spikes (B, 256) @ W_vis_to_map.T
        → I_correction (B, 1024)  [ghost bump]
        → inject into Pose CANN to fix IMU drift
    """

    def __init__(self, key, W_dog):
        self.W_dog = W_dog
        self.N_PLACE = N_PLACE
        self.map_size = MAP_SIZE
        self.n_features = N_STDP_FEATURES

        self._u_map = None
        self._r_map = None

        # Hebbian memory: (N_PLACE, N_STDP_FEATURES) = (1024, 256)
        k_w, _ = random.split(key)
        self.W_vis_to_map = jnp.zeros((N_PLACE, N_STDP_FEATURES), dtype=jnp.float32)
        self.W_vis_to_map = self.W_vis_to_map + random.uniform(
            k_w, (N_PLACE, N_STDP_FEATURES), dtype=jnp.float32) * 1e-4

        self._trace = None
        self._I_correction_smooth = None

    def reset(self, B):
        self._u_map = jnp.zeros((B, MAP_SIZE, MAP_SIZE))
        self._r_map = jnp.zeros((B, MAP_SIZE, MAP_SIZE))
        self._trace = jnp.zeros((B, N_PLACE), dtype=jnp.float32)
        self._I_correction_smooth = jnp.zeros((B, N_PLACE), dtype=jnp.float32)

    def initialize_from_pose(self, pose_bump):
        u_init = pose_bump.reshape(-1, MAP_SIZE, MAP_SIZE)
        self._u_map = u_init
        self._r_map = jnp.clip(u_init, 0, 1.0)

    def forward_mapping(self, vision_spikes, pose_bump, learn=USE_PLASTIC_MAP):
        """Learn vision→place associations. Update Hebbian memory + DoG dynamics."""
        B = vision_spikes.shape[0]

        # 1. Memory trace update (EMA of co-activation)
        vision_mean = vision_spikes.mean(axis=1, keepdims=True)
        co_activation = pose_bump * vision_mean
        self._trace = TRACE_ALPHA * self._trace + (1 - TRACE_ALPHA) * co_activation

        # 2. Hebbian weight update (Oja's rule with forgetting)
        if learn:
            dW_batch = ETA_HEBB * jnp.einsum('bi,bj->bij', self._trace, vision_spikes)
            dW = dW_batch.mean(axis=0)
            W_forget = GAMMA_HEBB * self.W_vis_to_map
            self.W_vis_to_map = jnp.clip(
                self.W_vis_to_map + dW - ETA_HEBB * W_forget,
                0.0, 2.0
            )

        # 3. Vision-driven input to place cells
        I_vis = jnp.einsum('ij,bj->bi', self.W_vis_to_map, vision_spikes)

        # 4. DoG recurrent dynamics
        u_flat = self._u_map.reshape(B, -1)
        r_flat = self._r_map.reshape(B, -1) + 1e-8
        u_new = neural_field_update(u_flat, r_flat, self.W_dog, I_vis, dt=DT, tau=TAU_U_MAP)
        self._u_map = u_new.reshape(B, MAP_SIZE, MAP_SIZE)
        self._r_map = jnp.clip(self._u_map, 0, 1.0)

        return self._r_map

    def compute_loop_closure(self, vision_spikes):
        """Compute loop closure correction current — the "ghost bump"."""
        I_corr_raw = jnp.einsum('bj,ij->bi', vision_spikes, self.W_vis_to_map)

        if self._I_correction_smooth is None:
            self._I_correction_smooth = I_corr_raw
        else:
            self._I_correction_smooth = (
                CORRECTION_SMOOTH * self._I_correction_smooth
                + (1 - CORRECTION_SMOOTH) * I_corr_raw
            )

        I_corr = CORRECTION_GAIN * self._I_correction_smooth
        I_corr = I_corr / (I_corr.max(axis=1, keepdims=True) + 1e-8)
        return I_corr

    def compute_confidence(self, vision_spikes):
        """Compute loop-closure confidence from RAW (unsmoothed) recalled energy.

        Returns (B,) confidence in [0, 1].
        Uses absolute thresholds on raw match energy, NOT the smoothed/normalized output.
        """
        # Raw recalled energy (before smoothing and gain scaling)
        I_raw = jnp.einsum('bj,ij->bi', vision_spikes, self.W_vis_to_map)  # (B, N_PLACE)

        # Sparsity: how many vision features are active?
        vision_activity = vision_spikes.sum(axis=1) / float(N_STDP_FEATURES)  # (B,)

        # Match quality: total recalled energy / N_PLACE (average place-cell activation)
        recalled_activity = I_raw.sum(axis=1) / float(N_PLACE)  # (B,)

        # Absolute thresholds
        SPARSITY_THRESH = 0.05   # ≥5% of vision features active → distinctive
        MATCH_THRESH    = 0.012  # ≥1.2% avg recalled → fires ~20-40% of time from t=200+

        is_distinctive = vision_activity > SPARSITY_THRESH
        is_match       = recalled_activity > MATCH_THRESH

        confidence = jnp.where(
            is_distinctive & is_match,
            recalled_activity / (MATCH_THRESH * 5.0),  # 0-1 scale
            0.0
        )
        return jnp.clip(confidence, 0.0, 1.0)

    def compute_spatial_correction(self, vision_spikes, gain=CORRECTION_GAIN):
        """Compute spatially-structured ghost bump for CANN injection.

        Creates a Gaussian bump centered on the recalled place cell location —
        the same spatial structure as IMU velocity injection. This way the
        CANN's Mexican Hat dynamics can properly compete the two bumps.

        Returns: (B, MAP_SIZE, MAP_SIZE) structured correction current
        """
        # Raw recalled place-cell activation
        I_corr_raw = jnp.einsum('bj,ij->bi', vision_spikes, self.W_vis_to_map)  # (B, 1024)

        # Smooth over time
        if self._I_correction_smooth is None:
            self._I_correction_smooth = I_corr_raw
        else:
            self._I_correction_smooth = (
                CORRECTION_SMOOTH * self._I_correction_smooth
                + (1 - CORRECTION_SMOOTH) * I_corr_raw
            )

        # Confidence
        confidence = self.compute_confidence(vision_spikes)  # (B,)

        # Find the recalled position: index of max activation → (cx, cy) grid coords
        peak_idx = jnp.argmax(self._I_correction_smooth, axis=1)  # (B,)
        cx_float = (peak_idx % self.map_size).astype(jnp.float32)  # (B,)
        cy_float = (peak_idx // self.map_size).astype(jnp.float32)  # (B,)

        # Create Gaussian ghost bump centered on recalled position
        y_grid, x_grid = jnp.meshgrid(
            jnp.arange(self.map_size, dtype=jnp.float32),
            jnp.arange(self.map_size, dtype=jnp.float32),
            indexing='ij'
        )  # each (32, 32)

        # σ = 1.5 cells — tight spatial precision for the ghost bump
        sigma_ghost = 1.5
        cx_exp = cx_float[:, None, None]  # (B, 1, 1)
        cy_exp = cy_float[:, None, None]

        d2 = (x_grid[None, :, :] - cx_exp)**2 + (y_grid[None, :, :] - cy_exp)**2
        ghost_bump = jnp.exp(-d2 / (2 * sigma_ghost**2))  # (B, 32, 32)
        ghost_bump = ghost_bump / (ghost_bump.max(axis=(1, 2), keepdims=True) + 1e-8)  # normalize

        # Scale by confidence and CORRECTION_GAIN
        I_spatial = gain * confidence[:, None, None] * ghost_bump  # (B, 32, 32)

        return I_spatial

    # =========================================================================
    #  SOTA AUTONOMOUS GATES
    # =========================================================================

    # Gate hyperparameters (can be overridden per-instance)
    MASK_RADIUS_CELLS = 3.0   # [deprecated — kept for API compat]
    SPARSITY_THRESH   = 0.05   # ≥5% of vision features active → distinctive
    MATCH_THRESH      = 0.012  # ≥1.2% avg recalled → genuine match

    def compute_confidence_with_gates(self, vision_spikes, pose_bump_meters):
        """Compute loop-closure confidence with SOTA autonomous gates.

        1. SPARSITY GATE — distinctive visual scene?
           (≥5% vision features active = not a blank/noisy frame)
        2. MATCH GATE — genuine memory activation?
           (recalled_activity > threshold = real association exists)
        3. ANTI-ALIASING GATE — concentration of recalled activity.
           If top-16 cells hold <40% of total activity → aliasing → reject.
           If ≥40% → clean unimodal match → accept.

        The concentration gate defeats perceptual aliasing without requiring
        the ghost bump to be spatially far from the current pose.

        pose_bump_meters: (B, 2) — current pose [x, y] in world meters (unused, kept for API)

        Returns: (B,) gated confidence in [0, 1]
        """
        # ---- Raw recalled place-cell activation ----
        I_raw = jnp.einsum('bj,ij->bi', vision_spikes, self.W_vis_to_map)  # (B, 1024)

        # Smooth over time
        if self._I_correction_smooth is None:
            self._I_correction_smooth = I_raw
        else:
            self._I_correction_smooth = (
                CORRECTION_SMOOTH * self._I_correction_smooth
                + (1 - CORRECTION_SMOOTH) * I_raw
            )

        # ---- Gate 1: SPARSITY — distinctive visual scene? ----
        vision_activity = vision_spikes.sum(axis=1) / float(N_STDP_FEATURES)  # (B,)
        is_distinctive = vision_activity > self.SPARSITY_THRESH  # (B,)

        # ---- Gate 2: MATCH — genuine memory activation? ----
        recalled_activity = I_raw.sum(axis=1) / float(N_PLACE)  # (B,)
        is_match = recalled_activity > self.MATCH_THRESH  # (B,)

        # ---- Gate 3: ANTI-ALIASING (Sharpness) — concentration gate ----
        # Sort descending and measure what fraction top-K cells hold
        I_sorted = jnp.sort(I_raw, axis=1)[:, ::-1]  # descending (B, 1024)
        total = I_sorted.sum(axis=1) + 1e-8  # (B,)
        K = 16
        top_k = I_sorted[:, :K].sum(axis=1)  # (B,)
        concentration = top_k / total  # (B,) — 1.0 = one dominant cell, ~0.16 = uniform

        # Anti-aliasing: if activity is too evenly spread → aliasing → reject
        is_anti_aliased = concentration > 0.35  # (B,) — ≥35% in top 16 cells

        # ---- Combine gates (AND logic) ----
        is_confident = is_distinctive & is_match & is_anti_aliased  # (B,)

        # Confidence: linear ramp from MATCH_THRESH to strong recall
        conf_raw = recalled_activity / (self.MATCH_THRESH * 5.0)  # (B,)
        confidence = jnp.where(is_confident, conf_raw, 0.0)

        return jnp.clip(confidence, 0.0, 1.0)

    def __call__(self, vision_spikes, pose_bump, learn=True):
        r_map = self.forward_mapping(vision_spikes, pose_bump, learn)
        I_correction = self.compute_loop_closure(vision_spikes)
        return r_map, I_correction

    def get_active_centroid(self):
        """Read centroid of place cell activity → (B, 2) world coords."""
        r = self._r_map
        p = r / (r.sum(axis=(1, 2), keepdims=True) + 1e-8)

        xx, yy = jnp.meshgrid(
            jnp.arange(MAP_SIZE, dtype=jnp.float32),
            jnp.arange(MAP_SIZE, dtype=jnp.float32),
            indexing='xy'
        )

        cx_cell = (p * xx[None, :, :]).sum(axis=(1, 2))
        cy_cell = (p * yy[None, :, :]).sum(axis=(1, 2))

        mpc = float(ROOM_W) / MAP_SIZE
        center = MAP_SIZE / 2.0
        x = (cx_cell - center) * mpc + ROOM_W / 2.0
        y = (cy_cell - center) * mpc + ROOM_H / 2.0
        return jnp.stack([x, y], axis=1)

    def get_place_activity_flat(self):
        return self._r_map.reshape(-1, N_PLACE)


# ============================================================================
# 4. Injection Snippet for snn_pose_cann.py
# ============================================================================
"""
HOW TO INJECT I_correction INTO snn_pose_cann.py:

In snn_pose_cann.py __call__, AFTER IMU velocity injection and BEFORE
the neural_field_update, add:

    # ---- Loop Closure: inject map correction from PlaceCellMap ----
    if map_correction is not None:
        # map_correction: (B, 1024) = I_correction from PlaceCellMap
        # Reshape to CANN sheet (B, 32, 32) and add as excitatory current
        I_loop_closure = map_correction.reshape(B, CANN_SIZE, CANN_SIZE)

        # Confidence weighting: only apply when signal is strong
        loop_energy = map_correction.sum(axis=1, keepdims=True) / (CANN_SIZE * CANN_SIZE)
        blend = jnp.clip(loop_energy * 10.0, 0.0, 1.0)
        I_cann_correction = I_loop_closure * blend[:, None, None]
    else:
        I_cann_correction = jnp.zeros((B, CANN_SIZE, CANN_SIZE))

    # In neural_field_update:
    u_new = neural_field_update(
        u_flat, r_flat + 1e-8, self.W_cann,
        I_vel_x + I_vel_y + I_cann_correction.reshape(B, -1),
        dt=DT, tau=TAU_U
    )

MATH: du/dt = −u + W_cann@r + I_vel + I_correction
If IMU drifts: bump at WRONG pose. I_correction fires at CORRECT pose.
Attractor dynamics → bump snaps to correct location. Loop closed! 🎯
"""


# ============================================================================
# 5. Diagnostic Visualization
# ============================================================================

def visualize_place_cell_map(pcm, title="Place Cell Activity + Loop Closure"):
    r = np.array(pcm._r_map[0])
    W_mem = np.array(pcm.W_vis_to_map)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    im = ax.imshow(r, cmap='hot', vmin=0, vmax=1.0,
                   extent=[0, ROOM_W, 0, ROOM_H], origin='lower',
                   interpolation='bilinear')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('Place Cell Activity', fontweight='bold')
    ax.set_xlim(0, ROOM_W); ax.set_ylim(0, ROOM_H)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    W_max = W_mem.max(axis=1).reshape(MAP_SIZE, MAP_SIZE)
    im2 = ax.imshow(W_max, cmap='plasma', vmin=0,
                    extent=[0, ROOM_W, 0, ROOM_H], origin='lower')
    ax.set_title('Memory Strength per Place Cell', fontweight='bold')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_xlim(0, ROOM_W); ax.set_ylim(0, ROOM_H)
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax, fraction=0.046)

    ax = axes[2]
    if pcm._I_correction_smooth is not None:
        I_corr = np.array(pcm._I_correction_smooth[0]).reshape(MAP_SIZE, MAP_SIZE)
        vmax = max(I_corr.max() * 0.9, 1e-8)
        im3 = ax.imshow(I_corr, cmap='RdYlGn', vmin=0, vmax=vmax,
                        extent=[0, ROOM_W, 0, ROOM_H], origin='lower')
        ax.set_title('Loop Closure Signal (Ghost Bump)', fontweight='bold')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_xlim(0, ROOM_W); ax.set_ylim(0, ROOM_H)
        ax.set_aspect('equal')
        plt.colorbar(im3, ax=ax, fraction=0.046)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_memory_matrix_diagnostic(pcm, title="Hebbian Memory Matrix W_vis_to_map"):
    W = np.array(pcm.W_vis_to_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    vmax = W[W > 0].max() * 0.9 if W.max() > 0 else 0.01
    im = ax.imshow(W, cmap='hot', aspect='auto',
                   vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('Vision Feature Index')
    ax.set_ylabel('Place Cell Index')
    ax.set_title('W_vis_to_map: Full Memory Matrix\n(place ← visual feature)', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Synaptic Weight')

    ax = axes[1]
    w_flat = W.ravel()
    active = (w_flat > 1e-4).sum()
    ax.hist(w_flat[w_flat > 1e-5], bins=80, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Weight Distribution\nActive: {active:,}/{W.size:,} ({100*active/W.size:.1f}%)')
    ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# 6. Evaluation
# ============================================================================

def evaluate_place_cell_map(key, n_samples=8, n_steps=100):
    from src.sparse_forest import generate_fixed_room_dataset, TIME_STEPS
    from src.snn_vision_stdp import VisionSTDP, N_INPUT, N_HIDDEN

    events, labels, tof_dists, positions, obstacles, *_, intensities = \
        generate_fixed_room_dataset(key, n_samples=n_samples)

    ev = np.array(events)
    tof = np.array(tof_dists)
    pos_gt = np.array(positions[:, :, :2])
    T = min(n_steps, TIME_STEPS)
    B = n_samples

    W_dog = build_map_dog_weights()
    pcm = PlaceCellMap(random.PRNGKey(42), W_dog)
    pcm.reset(B)

    vision_net = VisionSTDP(random.PRNGKey(0), n_input=N_INPUT, n_hidden=N_HIDDEN)
    vision_net.reset(B)

    def pose_bump_from_pos(positions_2d, map_size=MAP_SIZE, sigma=1.5):
        B2, T2, _ = positions_2d.shape
        mpc = float(ROOM_W) / map_size
        center = map_size / 2.0
        cx_float = (positions_2d[:, :, 0] - ROOM_W/2) / mpc + center
        cy_float = (positions_2d[:, :, 1] - ROOM_H/2) / mpc + center
        xx, yy = np.meshgrid(np.arange(map_size), np.arange(map_size), indexing='xy')
        bumps = []
        for b in range(B2):
            row = []
            for t in range(T2):
                cx, cy = float(cx_float[b, t]), float(cy_float[b, t])
                d2 = (xx - cx)**2 + (yy - cy)**2
                bump = np.exp(-d2 / (2 * sigma**2))
                row.append(bump.ravel())
            bumps.append(np.stack(row, axis=0))
        return np.stack(bumps, axis=0)

    pose_bumps_raw = pose_bump_from_pos(pos_gt[:, :T, :])

    print(f"\n  🦊 Place Cell Map Evaluation (B={B}, T={T})")
    print(f"     Vision: STDP features ({N_HIDDEN}d) → Place cells ({N_PLACE})")
    print(f"     Hebbian: η={ETA_HEBB}, decay={MEM_DECAY}, γ={GAMMA_HEBB}")
    print(f"     Memory trace: α={TRACE_ALPHA}")
    print(f"     Loop closure gain: {CORRECTION_GAIN}")

    I_corr_history = np.zeros((B, T, N_PLACE))
    r_map_history = np.zeros((B, T, MAP_SIZE, MAP_SIZE))

    for t in range(T):
        ev_t = ev[:, t, :]
        tof_t = tof[:, t]
        pose_b = jnp.array(pose_bumps_raw[:, t, :])

        vision_spikes, _ = vision_net(ev_t, tof_t, learn=True)
        r_map, I_corr = pcm(vision_spikes, pose_b, learn=True)

        I_corr_history[:, t, :] = np.array(I_corr)
        r_map_history[:, t, :] = np.array(r_map)

        if t % 25 == 0:
            centroid = pcm.get_active_centroid()
            mem_str = pcm.get_memory_strength()
            print(f"    t={t:3d}: centroid=({float(centroid[0,0]):.2f}, "
                  f"{float(centroid[0,1]):.2f})  "
                  f"|W|_max={float(mem_str.max()):.4f}  "
                  f"r_map_max={float(pcm._r_map[0].max()):.4f}")

    W_final = np.array(pcm.W_vis_to_map)
    r_map_mean = r_map_history.mean(axis=(0, 1))
    active_places = (r_map_mean > 0.05).sum()

    print(f"\n  📊 Place Cell Map Summary:")
    print(f"     Active place cells (>5% mean): {active_places}/{N_PLACE}")
    print(f"     Memory W: [{W_final.min():.6f}, {W_final.max():.6f}], mean={W_final.mean():.6f}")
    active_syn = (W_final > 1e-4).sum()
    print(f"     Active synapses (W>1e-4): {active_syn:,}/{W_final.size:,} "
          f"({100*active_syn/W_final.size:.1f}%)")

    return {
        'pcm': pcm, 'vision_net': vision_net,
        'I_corr_history': I_corr_history,
        'r_map_history': r_map_history,
        'pose_bumps': pose_bumps_raw,
    }


# ============================================================================
# 7. Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  🦊 Place Cell Map — Hebbian Hippocampus")
    print("    STDP Vision (256-feat) → W_vis_to_map → Place Cells (32×32)")
    print("=" * 65)

    t0 = time.time()

    W_dog = build_map_dog_weights()
    print(f"\n  Place Cell CANN: {W_dog.shape} recurrent DoG weights")
    print(f"  Memory Matrix: ({N_PLACE}, {N_STDP_FEATURES}) = {N_PLACE*N_STDP_FEATURES:,} synapses")

    results = evaluate_place_cell_map(random.PRNGKey(99), n_samples=8, n_steps=100)

    print(f"\n  Generating visualizations...")
    pcm = results['pcm']

    fig_map = visualize_place_cell_map(pcm)
    out_map = '/Users/lhooz/.openclaw/workspace/results/place_cell_map.png'
    fig_map.savefig(out_map, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"  📸 Saved: {out_map}")
    plt.close(fig_map)

    fig_mem = plot_memory_matrix_diagnostic(pcm)
    out_mem = '/Users/lhooz/.openclaw/workspace/results/place_cell_memory.png'
    fig_mem.savefig(out_mem, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"  📸 Saved: {out_mem}")
    plt.close(fig_mem)

    total = time.time() - t0
    print(f"\n  ✅ Place Cell Map done in {total:.1f}s")
    print(f"  📸 results/place_cell_map.png")
    print(f"  📸 results/place_cell_memory.png")
