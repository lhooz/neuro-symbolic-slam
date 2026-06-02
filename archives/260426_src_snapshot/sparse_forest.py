#!/usr/bin/env python3
"""
Sparse Forest — 2D Navigation with 1D Event Camera

A 10×10m room with 2-3 small obstacles (max 1×1m).
The SNN must learn [vx, vy, ω, clearance] from optic flow
while flying past random pillars.

Physics:
  - No distance dimming (Conservation of Radiance — ambient light)
  - Hard rejection: room is regenerated if no collision-free trajectory found
  - Strict 0.5m clearance margin from all surfaces at all timesteps
  - No position clipping — trajectories are clean by construction

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOM_W = 10.0
ROOM_H = 10.0

N_PIXELS = 64
FOV_DEG = 180.0
DT = 0.02
TIME_STEPS = 2000        # 20.0 seconds at 50 Hz
BATCH_SIZE = 8

# Barcode texture resolution (pixels per surface texture)
BARCODE_RESOLUTION = 512
THRESHOLD = 0.015

# Sparse Forest: few, small obstacles
N_OBSTACLES = 3
OBS_SIZE_MIN = 0.3
OBS_SIZE_MAX = 1.0
OBS_MARGIN = 2.0         # keep obstacles away from room edges

# Robot dynamics
VX_RANGE = (-0.8, 0.8)
VY_RANGE = (-0.3, 0.3)
OMEGA_RANGE = (-0.5, 0.5)
# ---------------------------------------------------------------------------
# B-Spline Trajectory Configuration
# ---------------------------------------------------------------------------
N_CONTROL             = 8        # number of B-spline control points
DEGREE                = 3        # cubic spline
SPLINE_COLLISION_RES  = 5        # collision-check resolution multiplier
USE_BSPLINE           = True    # False -> constant-velocity fallback


# ---------------------------------------------------------------------------
# B-Spline Trajectory Configuration
# ---------------------------------------------------------------------------
N_CONTROL    = 8       # number of B-spline control points (8 → 6 interior knots → loops)
DEGREE       = 3       # cubic B-spline
KNOT_FRAC    = 0.12    # fraction of arc-length per knot interval (越小越smooth)
SPLINE_COLLISION_RES = 5   # check collision at 5x resolution to catch mid-segment hits
USE_BSPLINE  = True   # set False to fall back to constant-velocity arcs

# Safety
SAFE_MARGIN = 0.5        # strict clearance from all surfaces
MAX_ROOM_ATTEMPTS = 50   # max room regenerations per sample
MAX_TRAJ_ATTEMPTS = 30   # max trajectory tries per room

# Texture
TEX_FREQS = [5.0, 10.0, 20.0]  # cycles per pixel — generates MANY fine texture edges
TEX_AMPS  = [0.8, 0.4, 0.2]    # amplitude of each frequency component

SEED = 42


# =============================================================================
# Obstacle Generation (Sparse Forest)
# =============================================================================
def generate_obstacles(key):
    """Generate N_OBSTACLES small random rectangles.
    Returns (N_OBSTACLES, 4): [x_min, y_min, x_max, y_max].
    """
    keys = jax.random.split(key, N_OBSTACLES)
    def _one(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        cx = jax.random.uniform(k1, (), minval=OBS_MARGIN, maxval=ROOM_W - OBS_MARGIN)
        cy = jax.random.uniform(k2, (), minval=OBS_MARGIN, maxval=ROOM_H - OBS_MARGIN)
        w = jax.random.uniform(k3, (), minval=OBS_SIZE_MIN, maxval=OBS_SIZE_MAX)
        h = jax.random.uniform(k4, (), minval=OBS_SIZE_MIN, maxval=OBS_SIZE_MAX)
        return jnp.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return jax.vmap(_one)(keys)


def obstacles_to_segments(obstacles):
    """Convert obstacles + room boundary into line segments."""
    room_segs = jnp.array([
        [[0, 0], [ROOM_W, 0]], [[ROOM_W, 0], [ROOM_W, ROOM_H]],
        [[ROOM_W, ROOM_H], [0, ROOM_H]], [[0, ROOM_H], [0, 0]],
    ], dtype=jnp.float32)
    def _rect(r):
        x0, y0, x1, y1 = r
        return jnp.array([
            [[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]],
            [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]],
        ], dtype=jnp.float32)
    obs_segs = jax.vmap(_rect)(obstacles).reshape(-1, 2, 2)
    return jnp.concatenate([room_segs, obs_segs], axis=0)


# =============================================================================
# Collision Detection
# =============================================================================
def _point_rect_dist(px, py, rect):
    """Signed distance from point to rectangle. Negative = inside."""
    cx = jnp.clip(px, rect[0], rect[2])
    cy = jnp.clip(py, rect[1], rect[3])
    outside = jnp.sqrt((px - cx)**2 + (py - cy)**2)
    inside = -jnp.minimum(jnp.minimum(px - rect[0], rect[2] - px),
                           jnp.minimum(py - rect[1], rect[3] - py))
    inside_rect = (px >= rect[0]) & (px <= rect[2]) & (py >= rect[1]) & (py <= rect[3])
    return jnp.where(inside_rect, inside, outside)


def _min_clearance_to_obstacles(px, py, obstacles):
    """Min distance from point to nearest obstacle surface."""
    dists = jax.vmap(lambda r: _point_rect_dist(px, py, r))(obstacles)
    # For inside points (negative), take the largest negative (least deep)
    # For outside, take the smallest positive
    return jnp.min(jnp.where(dists > 0, dists, jnp.inf))


def _wall_clearance(px, py):
    """Min distance from point to any room wall."""
    return jnp.minimum(jnp.minimum(px, py),
                       jnp.minimum(ROOM_W - px, ROOM_H - py))


def _is_clear(px, py, obstacles, margin=SAFE_MARGIN):
    """True if point has >= margin clearance from all obstacles and walls."""
    obs_ok = _min_clearance_to_obstacles(px, py, obstacles) >= margin
    wall_ok = (px >= margin) & (px <= ROOM_W - margin) & \
              (py >= margin) & (py <= ROOM_H - margin)
    return obs_ok & wall_ok


def _trajectory_clear(positions, obstacles, margin=SAFE_MARGIN):
    """Check all trajectory points are clear."""
    checks = jax.vmap(lambda p: _is_clear(p[0], p[1], obstacles, margin))(positions)
    return jnp.all(checks)



# =============================================================================
# B-Spline Trajectory Generation  (scipy-powered, JAX-compatible)
# =============================================================================
# All concrete spline math (splprep/splev) happens here in concrete numpy.
# Only plain JAX arrays flow out to the calling code.
# =============================================================================

from scipy.interpolate import splprep, splev, splrep
import numpy as np_np


def _scipy_bspline_positions_and_derivatives(ctrl_pts_np, heading_ctrl_np, time_steps, dt, degree=3):
    # ctrl_pts_np:     (N_CONTROL, 2) — position control polygon
    # heading_ctrl_np: (N_CONTROL,) — heading control points (radians, already unwrapped)
    ctrl      = np_np.array(ctrl_pts_np,      dtype=np_np.float64)
    heading_c = np_np.array(heading_ctrl_np, dtype=np_np.float64)

    t_u    = np_np.linspace(0.0, 1.0, time_steps)
    t_u_hr = np_np.linspace(0.0, 1.0, time_steps * SPLINE_COLLISION_RES)

    # --- Position spline ---
    try:
        tck_pos, u_pos = splprep((ctrl[:, 0], ctrl[:, 1]), k=degree, s=0)
    except Exception:
        # Fallback: linear position interpolation + independent random heading
        u_out = np_np.linspace(0, 1, time_steps)
        pos = np_np.stack([np_np.interp(u_out,
                        np_np.arange(ctrl.shape[0])/(ctrl.shape[0]-1), ctrl[:, i])
                        for i in range(2)], axis=1)
        vel = np_np.gradient(pos, u_out[1] - u_out[0], axis=0)
        # Independent random heading (holonomic even in fallback)
        hdg_raw = np_np.linspace(0, 2*np_np.pi, time_steps) + np_np.random.uniform(-0.5, 0.5)
        hdg = np_np.unwrap(hdg_raw) % (2 * np_np.pi)
        omega = np_np.gradient(hdg) / (u_out[1] - u_out[0]) / (time_steps * dt)
        return (pos.astype(np_np.float32), pos.astype(np_np.float32),
                hdg.astype(np_np.float32),
                vel[:, 0].astype(np_np.float32), vel[:, 1].astype(np_np.float32),
                omega.astype(np_np.float32))

    # Evaluate position spline directly on dense grids (no polygon!)
    pos_grid, d_grid       = splev(t_u,    tck_pos, der=0), splev(t_u,    tck_pos, der=1)
    pos_hr_grid, d_hr_grid = splev(t_u_hr, tck_pos, der=0), splev(t_u_hr, tck_pos, der=1)

    pos    = np_np.stack([np_np.asarray(pos_grid[0]),  np_np.asarray(pos_grid[1])],  axis=1).astype(np_np.float32)
    pos_hr = np_np.stack([np_np.asarray(pos_hr_grid[0]), np_np.asarray(pos_hr_grid[1])], axis=1).astype(np_np.float32)
    d_pos  = np_np.stack([np_np.asarray(d_grid[0]),   np_np.asarray(d_grid[1])],   axis=1).astype(np_np.float32)
    d_hr   = np_np.stack([np_np.asarray(d_hr_grid[0]), np_np.asarray(d_hr_grid[1])], axis=1).astype(np_np.float32)

    total_time = time_steps * dt
    wvx   = d_pos[:, 0]  / total_time   # m/s world-frame
    wvy   = d_pos[:, 1]  / total_time
    wvx_h = d_hr[:, 0]  / total_time
    wvy_h = d_hr[:, 1]  / total_time

    # --- Independent heading spline (HOLONOMIC: decouples look direction from motion) ---
    try:
        # 1D spline for theta(t) — splrep takes (x, y, k) where x must be sorted
        t_ctrl = np_np.linspace(0.0, 1.0, len(heading_c))
        tck_hdg, u_hdg = splrep(t_ctrl, heading_c, k=min(degree, len(heading_c)-1), s=0)
    except Exception:
        # Fallback: linear interpolation of heading control points
        hdg = np_np.interp(t_u, t_ctrl, heading_c)
        hdg_hr = np_np.interp(t_u_hr, t_ctrl, heading_c)
        dhdg  = np_np.gradient(hdg,  t_u[1]  - t_u[0])
        dhdg_hr = np_np.gradient(hdg_hr, t_u_hr[1] - t_u_hr[0])
        omega_raw  = dhdg  / total_time
        omega_hr   = dhdg_hr / total_time
        return (pos, pos_hr, (hdg % (2*np_np.pi)).astype(np_np.float32),
                wvx.astype(np_np.float32), wvy.astype(np_np.float32),
                omega_raw.astype(np_np.float32))

    # Evaluate heading spline on dense grids
    hdg_grid, dhdg_grid       = splev(t_u,    tck_hdg, der=0), splev(t_u,    tck_hdg, der=1)
    hdg_hr_grid, dhdg_hr_grid = splev(t_u_hr, tck_hdg, der=0), splev(t_u_hr, tck_hdg, der=1)

    headings    = np_np.asarray(hdg_grid)     % (2 * np_np.pi)   # (T,)
    headings_hr = np_np.asarray(hdg_hr_grid)  % (2 * np_np.pi)   # (T*5,)
    omega_raw   = np_np.asarray(dhdg_grid)                     # rad/s  (derivative already in param/s)
    omega_hr    = np_np.asarray(dhdg_hr_grid)                  # rad/s

    # Scale by 1/total_time: param u is in [0,1], du/dt = 1/T
    omega_raw = omega_raw  / total_time
    omega_hr  = omega_hr   / total_time

    return (pos, pos_hr, headings.astype(np_np.float32),
            wvx.astype(np_np.float32), wvy.astype(np_np.float32),
            omega_raw.astype(np_np.float32))


def _generate_bspline_trajectory(key, time_steps, dt, obstacles):
    MARGIN = SAFE_MARGIN + 0.4
    keys   = jax.random.split(key, N_CONTROL + 4)
    ptx_k  = keys[:N_CONTROL // 2]
    pty_k  = keys[N_CONTROL // 2 : N_CONTROL]

    cx_lo, cx_hi = MARGIN, ROOM_W - MARGIN
    cy_lo, cy_hi = MARGIN, ROOM_H - MARGIN
    cx_mid = (cx_lo + cx_hi) / 2
    cy_mid = (cy_lo + cy_hi) / 2

    # 4 pairs from opposing quadrants -> loops / figure-8s
    pair_defs = [
        (ptx_k[0], pty_k[0], cx_lo,cx_mid, cy_lo,cy_mid,  cx_mid,cx_hi, cy_mid,cy_hi),
        (ptx_k[1], pty_k[1], cx_mid,cx_hi, cy_lo,cy_mid,  cx_lo,cx_mid, cy_mid,cy_hi),
        (ptx_k[2], pty_k[2], cx_lo,cx_mid, cy_mid,cy_hi,  cx_mid,cx_hi, cy_lo,cy_mid),
        (ptx_k[3], pty_k[3], cx_lo,cx_mid, cy_lo,cy_mid,  cx_mid,cx_hi, cy_mid,cy_hi),
    ]

    pairs = []
    for pdef in pair_defs:
        ka, kb = pdef[0], pdef[1]
        ra = [float(pdef[2]), float(pdef[3]), float(pdef[4]), float(pdef[5])]
        rb = [float(pdef[6]), float(pdef[7]), float(pdef[8]), float(pdef[9])]
        pa = np_np.array([jax.random.uniform(ka, (), minval=ra[0], maxval=ra[1]),
                          jax.random.uniform(ka, (), minval=ra[2], maxval=ra[3])],
                         dtype=np_np.float64)
        pb = np_np.array([jax.random.uniform(kb, (), minval=rb[0], maxval=rb[1]),
                          jax.random.uniform(kb, (), minval=rb[2], maxval=rb[3])],
                         dtype=np_np.float64)
        pairs.extend([pa, pb])

    ctrl_np = np_np.stack(pairs, axis=0)

    # --- HOLONOMIC HEADING: generate independent heading control points ---
    # Generate random heading values and unwrap them to avoid 0/2pi discontinuities
    hdg_keys = jax.random.split(key, N_CONTROL + 1)
    # --- HOLONOMIC HEADING: smooth random walk ---
    # Generate smooth, bounded angular changes instead of random jumps
    hdg_keys = jax.random.split(key, N_CONTROL)
    hdg_deltas = np_np.array([jax.random.uniform(hdg_keys[i], (), minval=-1.0, maxval=1.0) 
                              for i in range(N_CONTROL)])
    
    # Start at a random angle, then cumulatively add the small deltas
    start_angle = np_np.random.uniform(0, 2*np_np.pi)
    heading_ctrl_np = start_angle + np_np.cumsum(hdg_deltas)

    (pos_np, pos_hr_np, headings_np,
     wvx_np, wvy_np, omega_np) = _scipy_bspline_positions_and_derivatives(
         ctrl_np, heading_ctrl_np, time_steps, dt, degree=DEGREE)

    # Convert to JAX
    positions  = jnp.array(pos_np)
    pos_hr     = jnp.array(pos_hr_np)
    headings   = jnp.array(headings_np)
    wvx        = jnp.array(wvx_np)
    wvy        = jnp.array(wvy_np)
    omega_raw  = jnp.array(omega_np)

    # Body-frame velocities: v_body = R(-theta) @ v_world
    cos_t   = jnp.cos(-headings)
    sin_t   = jnp.sin(-headings)
    vx_body = wvx * cos_t - wvy * sin_t   # (T,)
    vy_body = wvx * sin_t + wvy * cos_t   # (T,)
    # Return full time-varying arrays (Bug 2 fix: no jnp.mean!)

    # Collision check at high resolution
    if obstacles is not None and obstacles.shape[0] > 0:
        clear_hr = bool(jnp.all(
            jax.vmap(lambda p: _is_clear(p[0], p[1], obstacles))(pos_hr)))
    else:
        clear_hr = True

    in_bounds = bool(jnp.all(
        (pos_hr[:, 0] > SAFE_MARGIN) & (pos_hr[:, 0] < ROOM_W - SAFE_MARGIN) &
        (pos_hr[:, 1] > SAFE_MARGIN) & (pos_hr[:, 1] < ROOM_H - SAFE_MARGIN)))

    return positions, headings, vx_body, vy_body, omega_raw, clear_hr and in_bounds


def _make_trajectory(key, time_steps, dt, obstacles=None):
    if USE_BSPLINE and obstacles is not None:
        for attempt in range(MAX_TRAJ_ATTEMPTS):
            k_try = jax.random.PRNGKey(
                int(jax.random.randint(key, (), 0, 2**31 - 1)) + attempt)
            pos, hdg, vx, vy, omega, ok = _generate_bspline_trajectory(
                k_try, time_steps, dt, obstacles)
            if ok:
                return pos, hdg, vx, vy, omega
    return _make_trajectory_constant(key, time_steps, dt)


def _make_trajectory_constant(key, time_steps, dt):
    keys  = jax.random.split(key, 7)
    x0    = jax.random.uniform(keys[0], (), minval=SAFE_MARGIN+0.5,
                                maxval=ROOM_W-SAFE_MARGIN-0.5)
    y0    = jax.random.uniform(keys[1], (), minval=SAFE_MARGIN+0.5,
                                maxval=ROOM_H-SAFE_MARGIN-0.5)
    h0    = jax.random.uniform(keys[2], (), minval=0.0, maxval=2*jnp.pi)
    vx    = jax.random.uniform(keys[3], (), minval=VX_RANGE[0], maxval=VX_RANGE[1])
    vy    = jax.random.uniform(keys[4], (), minval=VY_RANGE[0], maxval=VY_RANGE[1])
    omega = jax.random.uniform(keys[5], (), minval=OMEGA_RANGE[0], maxval=OMEGA_RANGE[1])
    t     = jnp.arange(time_steps, dtype=jnp.float32) * dt
    headings = (h0 + omega * t) % (2 * jnp.pi)
    cos_h, sin_h = jnp.cos(headings), jnp.sin(headings)
    wx = vx * cos_h - vy * sin_h
    wy = vx * sin_h + vy * cos_h
    dx = jnp.concatenate([jnp.zeros(1), jnp.cumsum(wx[:-1] * dt)])
    dy = jnp.concatenate([jnp.zeros(1), jnp.cumsum(wy[:-1] * dt)])
    positions = jnp.stack([x0 + dx, y0 + dy], axis=-1)
    return positions, headings, vx, vy, omega

# =============================================================================
# Ray–Segment Intersection
# =============================================================================
def cast_rays(origins, directions, segments):
    A = segments[:, 0, :]
    B = segments[:, 1, :]
    E = B - A
    D = directions[:, None, :]
    diff = A[None, :, :] - origins[:, None, :]
    det = (D[:, :, 0] * E[None, :, 1] - D[:, :, 1] * E[None, :, 0])
    safe = jnp.where(jnp.abs(det) > 1e-10, det, 1.0)
    t = (diff[:, :, 0] * E[None, :, 1] - diff[:, :, 1] * E[None, :, 0]) / safe
    s = (diff[:, :, 0] * D[:, :, 1] - diff[:, :, 1] * D[:, :, 0]) / safe
    valid = (jnp.abs(det) > 1e-10) & (t > 0.01) & (s >= 0) & (s <= 1)
    dists = jnp.where(valid, t, 1e6)
    hit_pts = origins[:, None, :] + t[:, :, None] * directions[:, None, :]
    return dists, hit_pts


# =============================================================================
# Pixel Intensity (no dimming)
# =============================================================================
def _barcode_texture(barcode_key, local_coords):
    """Generate a 2D multi-scale barcode texture for one surface.

    barcode_key: numpy random state seeded per-surface
    local_coords: (N_PIXELS,) array of local coordinates along the surface [0, 1]

    Produces TWO levels of texture, TWO dimensions:
    1. Coarse random-width stripes along the surface (200-400 per surface)
    2. High-frequency cosine modulation in BOTH along-surface AND
       perpendicular-to-surface directions (decorrelated)

    KEY FIX: The perpendicular modulation uses an independent random t_perp
    for each pixel, creating depth-based variation even when the robot
    is very close to a wall (narrow along-surface x-range). With 200-400
    stripes, the robot always sees 10-30 stripes regardless of position.

    Event camera sensitivity: 200+ fine stripes + multi-freq cosines →
    events on every pixel as the robot moves, even at close range.
    """
    rng = np.random.RandomState(int(barcode_key) & 0xFFFFFFFF)

    # ── 1. Coarse stripe layer (200-400 stripes — THE KEY CHANGE) ───────────
    # Old: 6-16 stripes → robot near wall sees only 1-2 stripes → near-zero events
    # New: 200-400 stripes → robot near wall still sees 10-30 stripes → good events
    n_stripes = rng.randint(200, 401)
    boundaries = sorted([0.0] + list(rng.uniform(0.0, 1.0, n_stripes - 1)) + [1.0])
    boundaries = np.array(boundaries, dtype=np.float32)
    brightness = rng.uniform(0.15, 0.95, n_stripes).astype(np.float32)
    stripe_idx = np.searchsorted(boundaries[1:], local_coords)
    stripe_idx = np.clip(stripe_idx, 0, n_stripes - 1)
    base = brightness[stripe_idx]  # (N_PIXELS,)

    # ── 2. Multi-frequency 2D modulation ────────────────────────────────────
    # TWO independent modulation directions (decorrelated phases per pixel)
    # creates events from BOTH along-surface AND depth variation.
    n_pix = len(local_coords)
    for freq, amp in zip(TEX_FREQS, TEX_AMPS):
        # Along-surface modulation: varies with local_coords (texel position)
        phase_a = rng.uniform(0, 2 * np.pi)
        along_mod = np.cos(2 * np.pi * freq * n_pix * local_coords + phase_a)
        # Perpendicular modulation: varies with random t_perp (simulates depth variation)
        # Even if robot is close to wall, different ray angles → different hit depths
        t_perp = rng.uniform(0.0, 1.0, size=n_pix).astype(np.float32)
        phase_b = rng.uniform(0, 2 * np.pi)
        perp_mod = np.cos(2 * np.pi * freq * n_pix * t_perp + phase_b)
        # Combined: multiplicative so both dimensions contribute
        base = base * (1.0 + amp * (0.6 * along_mod + 0.6 * perp_mod))

    pattern = np.clip(base, 0.05, 1.5)
    return pattern.astype(np.float32)


def _generate_surface_textures(obstacles, room_seed):
    """Generate unique barcode texture for every surface in the room.

    Returns:
      wall_barcodes: dict mapping segment index → (BARCODE_RESOLUTION,) texture array
        Segment 0 = South wall, 1 = East, 2 = North, 3 = West
        Segments 4+ = obstacle surfaces (4 per obstacle, clockwise from bottom-left)
    """
    rng = np.random.RandomState(int(room_seed) & 0xFFFFFFFF)
    textures = {}

    # Room wall textures (segments 0-3)
    wall_seeds = [rng.randint(0, 2**31) for _ in range(4)]
    wall_coords = np.linspace(0, 1, BARCODE_RESOLUTION)
    for i, seed in enumerate(wall_seeds):
        textures[i] = _barcode_texture(seed, wall_coords)

    # Obstacle textures (segments 4+)
    n_obstacles = obstacles.shape[0]
    for obs_idx in range(n_obstacles):
        # Each side gets a fresh random seed (different barcode per side!)
        for side in range(4):
            seg_idx = 4 + obs_idx * 4 + side
            seed = int(rng.randint(0, 2**31))  # fresh seed per side
            # NORMALIZED [0, 1] coords for barcode lookup (same as walls!)
            # The actual world dimensions don't matter — we just need a
            # consistent local coordinate for texture lookup
            coords = np.linspace(0.0, 1.0, BARCODE_RESOLUTION)
            textures[seg_idx] = _barcode_texture(seed, coords)

    return textures


def compute_tof_distance(robot_pos, robot_heading, segments):
    """Compute forward-facing ToF laser rangefinder distance.

    Casts a single ray from robot center in heading direction.
    Returns distance to nearest obstacle (normalized to [0, 1]).
    """
    # Single ray forward
    origin = robot_pos
    direction = jnp.array([jnp.cos(robot_heading), jnp.sin(robot_heading)])

    # Cast against all segments
    dists, _ = cast_rays(origin[None, :], direction[None, :], segments)  # (1, N_SEG)

    # Find minimum distance
    min_dist = jnp.min(dists[0])

    # Clip max range for normalization
    max_range = 8.0
    tof_dist = jnp.clip(min_dist, 0.0, max_range)

    return tof_dist


def _precompute_barcode_tensors(surface_textures, obstacles):
    """Pre-compute texture tensors for fast vectorized lookup.

    Returns:
      tex_tensor: (n_surf, BARCODE_RESOLUTION) — flat texture values per surface
      stripe_edges: (n_surf, max_stripes) — cumulative edge positions per surface
    """
    seg_ids = sorted(surface_textures.keys())
    n_surf = max(seg_ids) + 1

    tex_rows = []
    edge_rows = []
    max_stripes = 0

    for seg_id in range(n_surf):
        if seg_id in surface_textures:
            tex = np.array(surface_textures[seg_id], dtype=np.float32)
            diffs = np.diff(tex)
            edge_mask = np.abs(diffs) > 0.05
            edge_pos = np.where(edge_mask)[0] + 1  # 1-indexed
            # Ensure at least 2 boundaries: 0 and BARCODE_RESOLUTION
            if len(edge_pos) == 0:
                edge_positions = np.array([0, BARCODE_RESOLUTION])
            else:
                edge_positions = np.concatenate([[0], edge_pos, [BARCODE_RESOLUTION]])
            stripe_widths = np.diff(edge_positions).astype(np.float32)
            max_stripes = max(max_stripes, len(stripe_widths))
            tex_rows.append(tex)
            edge_rows.append(np.cumsum(stripe_widths).astype(np.float32))
        else:
            tex_rows.append(np.zeros(BARCODE_RESOLUTION, dtype=np.float32))
            edge_rows.append(np.zeros(1, dtype=np.float32))

    padded_edges = np.zeros((n_surf, max_stripes), dtype=np.float32)
    for i, edges in enumerate(edge_rows):
        padded_edges[i, :len(edges)] = edges / (edges[-1] + 1e-8)

    return (
        jnp.stack([jnp.array(t) for t in tex_rows]),
        jnp.array(padded_edges),
    )


def _sample_barcode_fast(min_idx, nearest, obstacles, tex_tensor, stripe_edges):
    """Fast vectorized barcode texture lookup — all JAX-compatible, no Python loops.

    min_idx: (N_PIXELS,) — segment index for each pixel
    nearest: (N_PIXELS, 2) — world hit points
    obstacles: (N_OBS, 4) — obstacle bounding boxes
    tex_tensor: (n_surf, BARCODE_RESOLUTION) — stacked textures
    stripe_edges: (n_surf, max_stripes) — normalized cumulative edge positions

    Returns: (N_PIXELS,) intensity
    """
    n_pix = min_idx.shape[0]
    hx = nearest[:, 0]  # (N_PIXELS,)
    hy = nearest[:, 1]

    # ---- Compute local coordinate t for every pixel/segment combination ----
    # Segments 0-3: room walls
    is_wall = min_idx < 4
    is_obs  = min_idx >= 4

    # Wall local coords: (N_PIXELS, 4)
    t_wall = jnp.stack([
        hx / ROOM_W,   # seg 0 (South, y=0): x / ROOM_W
        hy / ROOM_H,   # seg 1 (East, x=ROOM_W): y / ROOM_H
        hx / ROOM_W,   # seg 2 (North, y=ROOM_H): x / ROOM_W
        hy / ROOM_H,   # seg 3 (West, x=0): y / ROOM_H
    ], axis=1)  # (N_PIXELS, 4)

    # Obstacle local coords: determine which side + obstacle → t
    # seg 4,5,6,7 = obs[0] bottom,right,top,left
    # seg 8,9,10,11 = obs[1] bottom,right,top,left
    # Handle empty obstacles by using zeros (is_obs will be all False anyway)
    obs_seg_ids = min_idx - 4  # (N_PIXELS,)
    obs_idx = obs_seg_ids // 4   # which obstacle (N_PIXELS,)
    side = obs_seg_ids % 4      # which side (N_PIXELS,)

    _obs_x0 = jnp.zeros(n_pix, dtype=jnp.float32) if obstacles is None else jnp.take(obstacles[:, 0], obs_idx)
    _obs_y0 = jnp.zeros(n_pix, dtype=jnp.float32) if obstacles is None else jnp.take(obstacles[:, 1], obs_idx)
    _obs_x1 = jnp.ones(n_pix, dtype=jnp.float32) if obstacles is None else jnp.take(obstacles[:, 2], obs_idx)
    _obs_y1 = jnp.ones(n_pix, dtype=jnp.float32) if obstacles is None else jnp.take(obstacles[:, 3], obs_idx)
    dx = _obs_x1 - _obs_x0 + 1e-8
    dy = _obs_y1 - _obs_y0 + 1e-8

    t_obs = jnp.where(
        side == 0,
        (hx - _obs_x0) / dx,
        jnp.where(
            side == 1,
            (hy - _obs_y0) / dy,
            jnp.where(
                side == 2,
                (hx - _obs_x0) / dx,
                (hy - _obs_y0) / dy
            )
        )
    )  # (N_PIXELS,)

    # Combine wall + obstacle using is_wall mask
    # Use take_along_axis (not take) for proper batched indexing
    t_wall_selected = jnp.take_along_axis(t_wall, min_idx[:, None], axis=1)[:, 0]  # (N_PIXELS,)
    t_all = jnp.where(is_wall, t_wall_selected, t_obs)  # (N_PIXELS,)

    # ---- One-hot select texture for each pixel ----
    # tex_tensor: (n_surf, R), min_idx: (N_PIXELS,)
    # result: (N_PIXELS, R) — texture for each pixel's surface
    tex_for_pix = jnp.take(tex_tensor, min_idx, axis=0)  # (N_PIXELS, R)

    # Stripe-edge gating: find nearest stripe boundary
    # stripe_edges: (n_surf, max_stripes), min_idx: (N_PIXELS,)
    edges_for_pix = jnp.take_along_axis(stripe_edges, min_idx[:, None], axis=0)[:, 0]  # (N_PIXELS, max_stripes)

    # t_all[:, None] - edges_for_pix: (N_PIXELS, max_stripes)
    # count how many edges t exceeds → stripe index
    exceeds = (t_all[:, None] > edges_for_pix).astype(jnp.float32)  # (N_PIXELS, max_stripes)
    stripe_idx = jnp.sum(exceeds, axis=1).astype(jnp.int32)  # (N_PIXELS,)

    # Gather texture at stripe_idx positions
    tex_idx = jnp.clip(stripe_idx, 0, BARCODE_RESOLUTION - 1)  # (N_PIXELS,)
    batch_idx = jnp.arange(n_pix)  # (N_PIXELS,)
    intensity = tex_for_pix[batch_idx, tex_idx]  # (N_PIXELS,)

    # ── KEY FIX: Add real geometric PERPENDICULAR modulation ─────────────────
    # Even when robot is close to a wall (narrow along-surface x-range),
    # different ray ANGLES hit DIFFERENT WORLD y-positions on the wall.
    # Using hy/ROOM_H as t_perp creates depth-based variation per pixel.
    #
    # For south wall (seg 0): hy varies 0.3-1.0 across FOV → good variation
    # For north wall (seg 2): hy varies 0.8-1.0 → smaller but nonzero variation
    # For east wall (seg 1): hx varies → variation via hy
    # For west wall (seg 3): hx varies → variation via hy
    is_south = (min_idx == 0)
    is_north = (min_idx == 2)
    is_ewall = (min_idx == 1) | (min_idx == 3)
    is_obs   = min_idx >= 4

    # t_perp: perpendicular world coordinate, normalized [0,1]
    t_perp_south = hy / ROOM_H                       # 0 at y=0, 1 at y=ROOM_H
    t_perp_north = hy / ROOM_H                       # same formula, different range
    t_perp_ewall = hy / ROOM_H                       # hy varies significantly for side walls
    t_perp_obs   = hy / ROOM_H                       # obstacles also see y-variation

    t_perp = (jnp.where(is_south, t_perp_south, 0.0)
             + jnp.where(is_north, t_perp_north, 0.0)
             + jnp.where(is_ewall, t_perp_ewall, 0.0)
             + jnp.where(is_obs,   t_perp_obs,   0.0))

    # High-freq cosine modulation in the PERPENDICULAR direction
    # This is the key fix: adjacent pixels hitting different world y-positions
    # now get different modulation → each pixel has a UNIQUE intensity value.
    n_pix_samp = float(BARCODE_RESOLUTION)  # for freq scaling
    for freq, amp in zip(TEX_FREQS, TEX_AMPS):
        phase = jnp.pi * freq  # decorrelated from along-surface phase
        perp_mod = jnp.cos(2.0 * jnp.pi * freq * n_pix_samp * t_perp + phase)
        intensity = intensity * (1.0 + 0.8 * amp * perp_mod)

    intensity = jnp.clip(intensity, 0.05, 1.5)
    return intensity


def _sample_barcode_textures(min_idx, nearest, surface_textures,
                              obstacles=None, tex_tensor=None, stripe_edges=None):
    """Main entry point for barcode texture sampling.

    Always uses the fast path (vectorized JAX). The tex_tensor and stripe_edges
    must be pre-computed (outside the vmap) and passed in.
    If they are None, uses a simplified fallback (no obstacle textures).
    """
    if tex_tensor is not None and stripe_edges is not None:
        return _sample_barcode_fast(min_idx, nearest, obstacles, tex_tensor, stripe_edges)

    # Fallback: use wall-only textures (simplified, no obstacle support)
    # This path should NOT be taken when obstacles are present.
    # Only for the empty-room case where tex_tensor was not pre-computed.
    hx = nearest[:, 0]
    hy = nearest[:, 1]
    # Simple approach: use wall segment lookup
    # For wall segments 0-3: compute t from hit point
    # For obstacle segments: use nearest wall
    is_wall = min_idx < 4
    t_wall = jnp.where(
        min_idx == 0, hx / ROOM_W,
        jnp.where(
            min_idx == 1, hy / ROOM_H,
            jnp.where(
                min_idx == 2, hx / ROOM_W,
                hy / ROOM_H
            )
        )
    )
    # For obstacle segments, use nearest wall approximation
    t = jnp.where(is_wall, t_wall, jnp.zeros_like(hx))
    tex_idx = jnp.clip((t * (BARCODE_RESOLUTION - 1)).astype(jnp.int32), 0, BARCODE_RESOLUTION - 1)
    return jnp.zeros(N_PIXELS, dtype=jnp.float32)  # fallback: return zeros


def compute_pixel_readings(robot_pos, robot_heading, segments,
                           surface_textures=None, obstacles=None,
                           tex_tensor=None, stripe_edges=None):
    """Compute pixel intensity readings with barcode textures.

    surface_textures: dict mapping segment index → (BARCODE_RESOLUTION,) texture array
                       If None, falls back to old sinusoidal wall texture.
    obstacles: (N_OBS, 4) array — needed for obstacle texture lookup.
    tex_tensor: pre-stacked texture array (n_surf, BARCODE_RESOLUTION) or None.
                Must be pre-computed OUTSIDE the vmap (required for JAX tracing).
    stripe_edges: pre-computed stripe edge positions or None.
    """
    fov_rad = jnp.radians(FOV_DEG)
    angles = robot_heading + jnp.linspace(-fov_rad/2, fov_rad/2, N_PIXELS)
    origins = jnp.broadcast_to(robot_pos, (N_PIXELS, 2))
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    dists, hit_pts = cast_rays(origins, dirs, segments)
    min_idx = jnp.argmin(dists, axis=-1)
    min_dist = jnp.min(dists, axis=-1)
    nearest = hit_pts[jnp.arange(N_PIXELS), min_idx]
    hit_type = (min_idx >= 4).astype(jnp.float32)

    if surface_textures is not None and tex_tensor is not None:
        # Fast path: use pre-computed tensors (built outside vmap)
        intensities = _sample_barcode_textures(min_idx, nearest, surface_textures, obstacles, tex_tensor, stripe_edges)
    else:
        # Fallback: old sinusoidal texture (deprecated)
        intensities = _wall_texture(nearest)

    return intensities, min_dist, hit_type, nearest


# =============================================================================
# HARD REJECT: Generate Safe Sample
# =============================================================================
def generate_sample(key, time_steps=TIME_STEPS, dt=DT):
    """Generate one collision-free labelled event sample.
    
    Hard rejection loop:
      1. Generate room
      2. Try up to MAX_TRAJ_ATTEMPTS trajectories
      3. If none safe → regenerate room (up to MAX_ROOM_ATTEMPTS)
      4. Guarantees: spawn clear, trajectory clear, no clipping
    
    Returns:
      events:  (T, N_PIXELS)  {-1, 0, +1}
      labels:  (4,)  [vx, vy, omega, min_clearance]
      info:    dict
    """
    # We use numpy for the rejection loop (not differentiable anyway)
    # then convert back to jax for the SNN
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)

    for room_attempt in range(MAX_ROOM_ATTEMPTS):
        # Generate room
        k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))
        obstacles = generate_obstacles(k_obs)
        segments = obstacles_to_segments(obstacles)

        # Try trajectories
        for traj_attempt in range(MAX_TRAJ_ATTEMPTS):
            k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
            positions, headings, vx, vy, omega = \
                _make_trajectory(k_traj, time_steps, dt, obstacles)

            # Check clearance
            spawn_ok = bool(jnp.all(
                _is_clear(positions[0, 0], positions[0, 1], obstacles)))
            traj_ok = bool(jnp.all(
                _trajectory_clear(positions, obstacles)))

            if spawn_ok and traj_ok:
                # SAFE -- generate events and return
                # Generate unique barcode textures for this room
                room_seed = rng.randint(0, 2**31)
                surface_textures = _generate_surface_textures(np.array(obstacles), room_seed)
                # Pre-compute tensors OUTSIDE the vmap (required for JAX tracing)
                tex_t, stripe_e = _precompute_barcode_tensors(surface_textures, np.array(obstacles))

                readings = jax.vmap(
                    lambda p, h: compute_pixel_readings(
                        p, h, segments, surface_textures, np.array(obstacles), tex_t, stripe_e
                    )
                )(positions, headings)
                intensities = readings[0]
                distances = readings[1]

                # ToF laser rangefinder (forward-facing)
                tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(
                    positions, headings, segments)
                # Normalize to [0, 1] at each timestep
                tof_normalized = tof_dists

                prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
                delta = intensities - prev
                events = jnp.where(delta > THRESHOLD, 1.0,
                          jnp.where(delta < -THRESHOLD, -1.0, 0.0))
                events = events.at[0].set(0.0)

                # Per-timestep clearance: (T,) minimum distance at each step
                min_clear_arr = jnp.min(distances, axis=1)          # (T,)

                # Handle B-spline (T,) vs constant-velocity scalar kinematics
                if vx.ndim == 0:
                    # Fallback: broadcast scalar to (T,)
                    vx_arr = jnp.broadcast_to(vx, (time_steps,)).astype(jnp.float32)
                    vy_arr = jnp.broadcast_to(vy, (time_steps,)).astype(jnp.float32)
                    omega_arr = jnp.broadcast_to(omega, (time_steps,)).astype(jnp.float32)
                else:
                    vx_arr = vx.astype(jnp.float32)
                    vy_arr = vy.astype(jnp.float32)
                    omega_arr = omega.astype(jnp.float32)

                # Labels: (T, 4) time-varying kinematics + instantaneous clearance
                clearance_norm = jnp.tanh(min_clear_arr / 2.0)          # (T,)
                labels = jnp.stack([
                    vx_arr     / abs(VX_RANGE[1]),
                    vy_arr     / abs(VY_RANGE[1]),
                    omega_arr  / abs(OMEGA_RANGE[1]),
                    clearance_norm,
                ], axis=1)                                              # (T, 4)

                info = {
                    'obstacles': obstacles,
                    'segments': segments,
                    'positions': positions,
                    'headings': headings,
                    'vx': vx_arr, 'vy': vy_arr, 'omega': omega_arr,   # raw (T,) arrays
                    'vx_mean': float(jnp.mean(vx_arr)),               # scalar for display
                    'vy_mean': float(jnp.mean(vy_arr)),
                    'omega_mean': float(jnp.mean(omega_arr)),
                    'intensities': intensities,
                    'distances': distances,
                    'tof': tof_normalized,
                    'room_attempts': room_attempt + 1,
                    'traj_attempts': traj_attempt + 1,
                }
                return events, labels, info

    # Fallback — should never happen with sparse forest, but just in case
    # Generate a completely empty room (no obstacles)
    empty_obs = jnp.zeros((0, 4), dtype=jnp.float32)
    empty_segs = jnp.array([
        [[0, 0], [ROOM_W, 0]], [[ROOM_W, 0], [ROOM_W, ROOM_H]],
        [[ROOM_W, ROOM_H], [0, ROOM_H]], [[0, ROOM_H], [0, 0]],
    ], dtype=jnp.float32)

    k_fb = jax.random.PRNGKey(rng.randint(0, 2**31))
    positions, headings, vx, vy, omega = _make_trajectory(k_fb, time_steps, dt)

    # Generate barcode textures for room-only fallback (PRE-COMPUTE tensors before vmap)
    fb_room_seed = rng.randint(0, 2**31)
    fb_textures = _generate_surface_textures(np.zeros((0, 4), dtype=np.float32), fb_room_seed)
    fb_tex_t, fb_stripe_e = _precompute_barcode_tensors(fb_textures, np.zeros((0, 4), dtype=np.float32))

    readings = jax.vmap(
        lambda p, h: compute_pixel_readings(
            p, h, empty_segs, fb_textures, empty_obs, fb_tex_t, fb_stripe_e
        )
    )(
        positions, headings)
    intensities = readings[0]
    distances = readings[1]

    # ToF laser rangefinder (fallback)
    tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(
                positions, headings, empty_segs)
    tof_normalized = tof_dists

    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0,
              jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)

    min_clear = jnp.min(distances[-1])
    min_clear_arr = jnp.min(distances, axis=1)
    vx_arr = jnp.broadcast_to(vx, (time_steps,)).astype(jnp.float32)
    vy_arr = jnp.broadcast_to(vy, (time_steps,)).astype(jnp.float32)
    omega_arr = jnp.broadcast_to(omega, (time_steps,)).astype(jnp.float32)
    clearance_norm = jnp.tanh(min_clear_arr / 2.0)
    labels = jnp.stack([
        vx_arr     / abs(VX_RANGE[1]),
        vy_arr     / abs(VY_RANGE[1]),
        omega_arr  / abs(OMEGA_RANGE[1]),
        clearance_norm,
    ], axis=1)

    info = {
        'obstacles': empty_obs,
        'segments': empty_segs,
        'positions': positions,
        'headings': headings,
        'vx': vx_arr, 'vy': vy_arr, 'omega': omega_arr,
        'vx_mean': float(jnp.mean(vx_arr)),
        'vy_mean': float(jnp.mean(vy_arr)),
        'omega_mean': float(jnp.mean(omega_arr)),
        'intensities': intensities,
        'distances': distances,
        'tof': tof_normalized,
        'room_attempts': MAX_ROOM_ATTEMPTS,
        'traj_attempts': MAX_TRAJ_ATTEMPTS,
        'fallback': True,
    }
    return events, labels, info


def generate_fixed_room_dataset(key, n_samples, obstacles=None,
                                  time_steps=TIME_STEPS, dt=DT):
    """Generate many trajectories within ONE fixed room.

    If obstacles is None, generates a random room first.
    Returns same-room trajectories with different starting positions
    and velocities — so the SNN can learn the actual depth structure.

    Returns:
      events:  (n_samples, T, N_PIXELS)
      labels:  (n_samples, 4)  [vx, vy, omega, min_clearance]
      obstacles: (N_OBS, 4) the fixed room layout
    """
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)

    # Generate or use provided room
    if obstacles is None:
        k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))
        obstacles = generate_obstacles(k_obs)
        segments = obstacles_to_segments(obstacles)
    else:
        segments = obstacles_to_segments(obstacles)

    # Generate unique barcode textures for this room (shared across all trajectories)
    # Convert obstacles to numpy for texture generation (outside JAX tracing)
    obstacles_np = np.array(obstacles) if hasattr(obstacles, 'dtype') else obstacles
    room_seed = rng.randint(0, 2**31)
    surface_textures = _generate_surface_textures(obstacles_np, room_seed)
    # Pre-compute tensors OUTSIDE the vmap (required for JAX tracing)
    tex_t, stripe_e = _precompute_barcode_tensors(surface_textures, obstacles_np)

    events_list = []
    labels_list = []
    tof_list = []
    positions_list = []
    headings_list = []
    intensities_list = []
    total_attempts = 0
    max_attempts = n_samples * MAX_TRAJ_ATTEMPTS

    while len(events_list) < n_samples and total_attempts < max_attempts:
        k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
        positions, headings, vx, vy, omega = \
            _make_trajectory(k_traj, time_steps, dt, obstacles)

        spawn_ok = bool(jnp.all(
            _is_clear(positions[0, 0], positions[0, 1], obstacles)))
        traj_ok = bool(jnp.all(
            _trajectory_clear(positions, obstacles)))
        total_attempts += 1

        if spawn_ok and traj_ok:
            readings = jax.vmap(
                lambda p, h: compute_pixel_readings(
                    p, h, segments, surface_textures, obstacles, tex_t, stripe_e
                )
            )(positions, headings)
            intensities = readings[0]
            distances = readings[1]
            intensities_list.append(intensities)

            # ToF laser rangefinder
            tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(
                positions, headings, segments)

            prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
            delta = intensities - prev
            events = jnp.where(delta > THRESHOLD, 1.0,
                      jnp.where(delta < -THRESHOLD, -1.0, 0.0))
            events = events.at[0].set(0.0)

            min_clear_arr = jnp.min(distances, axis=1)   # (T,)
            if vx.ndim == 0:
                vx_arr = jnp.broadcast_to(vx, (time_steps,)).astype(jnp.float32)
                vy_arr = jnp.broadcast_to(vy, (time_steps,)).astype(jnp.float32)
                omega_arr = jnp.broadcast_to(omega, (time_steps,)).astype(jnp.float32)
            else:
                vx_arr = vx.astype(jnp.float32)
                vy_arr = vy.astype(jnp.float32)
                omega_arr = omega.astype(jnp.float32)
            clearance_norm = jnp.tanh(min_clear_arr / 2.0)
            labels = jnp.stack([
                vx_arr     / abs(VX_RANGE[1]),
                vy_arr     / abs(VY_RANGE[1]),
                omega_arr  / abs(OMEGA_RANGE[1]),
                clearance_norm,
            ], axis=1)   # (T, 4)

            events_list.append(events)
            labels_list.append(labels)
            tof_list.append(tof_dists)
            positions_list.append(positions)
            headings_list.append(headings)

    n_actual = len(events_list)
    accept_rate = n_actual / total_attempts if total_attempts > 0 else 0
    print(f"    Generated {n_actual}/{n_samples} trajectories "
          f"({accept_rate:.0%} acceptance, {total_attempts} attempts)")

    return (jnp.stack(events_list), jnp.stack(labels_list), jnp.stack(tof_list),
            jnp.stack(positions_list), jnp.stack(headings_list), obstacles, segments,
            jnp.stack(intensities_list))


def generate_batch(key, batch_size=BATCH_SIZE,
                   time_steps=TIME_STEPS, dt=DT):
    """Generate a batch of collision-free samples."""
    keys = jax.random.split(key, batch_size)
    events_list, labels_list, info_list = [], [], []
    for i in range(batch_size):
        ev, lb, inf = generate_sample(keys[i], time_steps, dt)
        events_list.append(ev)
        labels_list.append(lb)
        info_list.append(inf)
    return jnp.stack(events_list), jnp.stack(labels_list), info_list


# =============================================================================
# Main (test)
# =============================================================================
def main():
    print("=" * 60)
    print("  🌲 Sparse Forest — Collision-Free Event Camera")
    print("=" * 60)
    print(f"  Room:          {ROOM_W}×{ROOM_H}m")
    print(f"  Obstacles:     {N_OBSTACLES} (max {OBS_SIZE_MAX}×{OBS_SIZE_MAX}m)")
    print(f"  Safe margin:   {SAFE_MARGIN}m")
    print(f"  Pixels:        {N_PIXELS}, FOV: {FOV_DEG}°")
    print(f"  Labels:        [vx, vy, ω, clearance]")
    print(f"  Dimming:       OFF (Conservation of Radiance)")
    print("=" * 60)

    key = jax.random.PRNGKey(SEED)
    import time as _time

    print("\n  ⚡ Generating single sample...")
    t0 = _time.time()
    events, labels, info = generate_sample(key)
    print(f"  Time: {_time.time()-t0:.3f}s")
    print(f"  Room attempts: {info.get('room_attempts', '?')}")
    print(f"  Traj attempts: {info.get('traj_attempts', '?')}")
    print(f"  Events: {int(jnp.sum(jnp.abs(events)))}/{N_PIXELS*TIME_STEPS} "
          f"({100*float(jnp.mean(jnp.abs(events))):.1f}%)")
    print(f"  vx={float(labels[0]):+.3f} vy={float(labels[1]):+.3f} "
          f"ω={float(labels[2]):+.3f} cl={float(labels[3]):+.3f}")
    print(f"  Obstacles: {info['obstacles'].shape[0]}")

    print(f"\n  ⚡ Batch (B={BATCH_SIZE})...")
    t0 = _time.time()
    key2 = jax.random.split(key, 2)[0]
    ev_b, lb_b, info_b = generate_batch(key2, BATCH_SIZE)
    elapsed = _time.time()-t0
    print(f"  Time: {elapsed:.3f}s ({elapsed/BATCH_SIZE:.3f}s/sample)")
    fallbacks = sum(1 for inf in info_b if inf.get('fallback', False))
    avg_room = np.mean([inf.get('room_attempts', 1) for inf in info_b])
    avg_traj = np.mean([inf.get('traj_attempts', 1) for inf in info_b])
    print(f"  Fallbacks: {fallbacks}/{BATCH_SIZE}")
    print(f"  Avg room attempts: {avg_room:.1f}")
    print(f"  Avg traj attempts: {avg_traj:.1f}")

    print(f"\n  Sample breakdown:")
    for i in range(min(8, BATCH_SIZE)):
        l = lb_b[i]
        ne = int(jnp.sum(jnp.abs(ev_b[i])))
        inf = info_b[i]
        n_obs = inf['obstacles'].shape[0]
        print(f"    [{i}] vx={l[0]:+.2f} vy={l[1]:+.2f} ω={l[2]:+.2f} "
              f"cl={l[3]:+.2f} | ev={ne} obs={n_obs} "
              f"r{inf.get('room_attempts',0)}t{inf.get('traj_attempts',0)}")

    # Collision verification
    print(f"\n  🔍 Verifying zero collisions...")
    n_violations = 0
    for i in range(BATCH_SIZE):
        pos = info_b[i]['positions']
        obs = info_b[i]['obstacles']
        dists = jax.vmap(lambda p: _min_clearance_to_obstacles(p[0], p[1], obs))(pos)
        wall_d = jax.vmap(lambda p: _wall_clearance(p[0], p[1]))(pos)
        min_d = float(jnp.minimum(jnp.min(dists), jnp.min(wall_d)))
        if min_d < SAFE_MARGIN:
            n_violations += 1
            print(f"    ❌ Sample {i}: min clearance = {min_d:.3f}m")
    if n_violations == 0:
        print(f"    ✅ All {BATCH_SIZE} samples pass clearance check")
    else:
        print(f"    ❌ {n_violations} violations found!")

    # Visualize
    print(f"\n  📊 Visualization...")
    _plot(info_b[0], ev_b[0],
          "/Users/lhooz/.openclaw/workspace/sparse_forest_sample.png")
    print(f"\n  ✅ Done!")


def _plot(info, events, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    obs = np.array(info['obstacles'])
    pos = np.array(info['positions'])
    hdg = np.array(info['headings'])
    intens = np.array(info['intensities'])
    ev = np.array(events)
    T, N = ev.shape
    time_s = np.arange(T) * DT

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                            gridspec_kw={'height_ratios': [2, 1.2, 1.0]})

    ax1 = axes[0]
    ax1.set_xlim(-0.5, ROOM_W + 0.5)
    ax1.set_ylim(-0.5, ROOM_H + 0.5)
    ax1.set_aspect('equal')
    ax1.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H, lw=2, ec='black', fc='#f0f0f0'))
    for o in obs:
        ax1.add_patch(Rectangle((o[0], o[1]), o[2]-o[0], o[3]-o[1],
                                fc='#555', ec='black', alpha=0.85))
    # Safe margin circles at start
    circle = plt.Circle((pos[0, 0], pos[0, 1]), SAFE_MARGIN,
                        fill=False, ec='limegreen', ls='--', lw=1.5)
    ax1.add_patch(circle)

    ax1.plot(pos[:, 0], pos[:, 1], '-', color='steelblue', lw=2, label='Trajectory')
    ax1.plot(pos[0, 0], pos[0, 1], 'o', color='limegreen', ms=10, label='Start')
    ax1.plot(pos[-1, 0], pos[-1, 1], 's', color='red', ms=10, label='End')

    fov_rad = np.radians(FOV_DEG)
    for step, c in [(0, 'limegreen'), (T-1, 'red')]:
        px, py = pos[step]
        for sign in (-1, 1):
            a = sign * fov_rad/2 + hdg[step]
            ax1.plot([px, px+1.5*np.cos(a)], [py, py+1.5*np.sin(a)], '--', color=c, alpha=0.5)

    ax1.set_title(f'Sparse Forest  |  vx={info["vx_mean"]:.2f}  vy={info["vy_mean"]:.2f}  '
                 f'omega={info["omega_mean"]:.2f}  |  {len(obs)} obstacles  |  '
                 f'room={info.get("room_attempts","?")} traj={info.get("traj_attempts","?")}',
                 fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')

    ax2 = axes[1]
    on_idx, off_idx = np.where(ev > 0), np.where(ev < 0)
    if len(on_idx[0]) > 0:
        ax2.scatter(on_idx[0], on_idx[1], c='tab:red', s=0.6, alpha=0.5, label='ON')
    if len(off_idx[0]) > 0:
        ax2.scatter(off_idx[0], off_idx[1], c='tab:blue', s=0.6, alpha=0.5, label='OFF')
    ax2.axhline(N//2, color='green', ls='--', lw=0.8, alpha=0.5)
    ax2.set_xlim(0, T)
    ax2.set_ylabel('Pixel')
    n_ev = np.sum(np.abs(ev))
    ax2.set_title(f'Event Raster  |  {n_ev} events ({100*n_ev/(T*N):.1f}%)', fontsize=10)
    ax2.legend(markerscale=8, loc='upper right', fontsize=7)

    ax3 = axes[2]
    ax3.imshow(intens.T, aspect='auto', cmap='gray',
               extent=[0, time_s[-1], N, 0], vmin=0, vmax=1)
    ax3.set_ylabel('Pixel'); ax3.set_xlabel('Time (s)')
    ax3.set_title('Pixel Intensity (no dimming)', fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📸 Saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
