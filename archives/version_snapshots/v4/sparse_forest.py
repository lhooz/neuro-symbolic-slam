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

N_PIXELS = 256
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
N_CONTROL            = 8        # number of B-spline control points (8 → 6 interior knots → loops)
DEGREE               = 3        # cubic B-spline
KNOT_FRAC            = 0.12     # fraction of arc-length per knot interval (越小越smooth)
SPLINE_COLLISION_RES = 5        # check collision at 5x resolution to catch mid-segment hits
USE_BSPLINE          = True     # False -> constant-velocity fallback

# Safety
SAFE_MARGIN = 0.5        # strict clearance from all surfaces
MAX_ROOM_ATTEMPTS = 50   # max room regenerations per sample
MAX_TRAJ_ATTEMPTS = 30   # max trajectory tries per room

# Texture
TEX_FREQS = [1.5, 3.0, 6.0]  # cycles per pixel — generates MANY fine texture edges
TEX_AMPS  = [0.8, 0.4, 0.2]  # amplitude of each frequency component

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
# KINEMATIC "ROOMBA" EXPLORER (Infinite Collision-Free Walk)
# =============================================================================
def _make_trajectory(key, time_steps, dt, obstacles=None):
    """
    Generates an infinite random walk. The robot drives forward and 
    pivots away when it senses a wall or obstacle.
    """
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)
    
    pos = np.zeros((time_steps, 2), dtype=np.float32)
    hdg = np.zeros(time_steps, dtype=np.float32)
    vx = np.zeros(time_steps, dtype=np.float32)
    vy = np.zeros(time_steps, dtype=np.float32)
    omega = np.zeros(time_steps, dtype=np.float32)
    
    margin = SAFE_MARGIN + 0.3
    obs_np = np.array(obstacles) if obstacles is not None else np.zeros((0, 4))
    
    # 🌟 CRITICAL FIX: Dynamically search for a safe spawn point!
    while True:
        sx = rng.uniform(margin, ROOM_W - margin)
        sy = rng.uniform(margin, ROOM_H - margin)
        hit = False
        for o in obs_np:
            if sx > o[0]-margin and sx < o[2]+margin and sy > o[1]-margin and sy < o[3]+margin:
                hit = True
                break
        if not hit:
            pos[0] = [sx, sy]
            break

    hdg[0] = rng.uniform(0, 2 * np.pi)
    
    v_forward = 0.6  # Cruising speed (m/s)
    current_omega = rng.uniform(-0.5, 0.5)
    
    for t in range(1, time_steps):
        # 1. Wander organically
        if rng.uniform() < 0.02: # 2% chance per frame to change steering
            current_omega = rng.uniform(-0.8, 0.8)
            
        # 2. Predict next position
        h = hdg[t-1] + current_omega * dt
        px = pos[t-1, 0] + v_forward * np.cos(h) * dt
        py = pos[t-1, 1] + v_forward * np.sin(h) * dt
        
        # 3. Collision Feeler (Check walls and obstacles)
        hit = False
        if px < margin or px > ROOM_W - margin or py < margin or py > ROOM_H - margin:
            hit = True
        else:
            for o in obs_np:
                if (px > o[0] - margin and px < o[2] + margin and 
                    py > o[1] - margin and py < o[3] + margin):
                    hit = True
                    break
                
        # 4. Evasive Maneuvers!
        if hit:
            # If we hit something, stop moving forward and pivot hard
            if abs(current_omega) < 0.1:
                current_omega = 2.0 if rng.uniform() > 0.5 else -2.0
            else:
                current_omega = 2.0 * np.sign(current_omega)
            
            px, py = pos[t-1, 0], pos[t-1, 1] # Freeze position
            v_actual = 0.0
        else:
            v_actual = v_forward
            
        # 5. Commit state
        pos[t] = [px, py]
        hdg[t] = h % (2 * np.pi)
        vx[t] = v_actual
        vy[t] = 0.0 # Holonomic slip is zero
        omega[t] = current_omega
        
    return jnp.array(pos), jnp.array(hdg), jnp.array(vx), jnp.array(vy), jnp.array(omega)

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
    """Generate a clean, 1D multi-scale barcode texture for one surface."""
    rng = np.random.RandomState(int(barcode_key) & 0xFFFFFFFF)

    n_stripes = rng.randint(15, 46)
    boundaries = sorted([0.0] + list(rng.uniform(0.0, 1.0, n_stripes - 1)) + [1.0])
    boundaries = np.array(boundaries, dtype=np.float32)
    brightness = rng.uniform(0.15, 0.95, n_stripes).astype(np.float32)
    stripe_idx = np.searchsorted(boundaries[1:], local_coords)
    stripe_idx = np.clip(stripe_idx, 0, n_stripes - 1)
    base = brightness[stripe_idx] 

    for freq, amp in zip(TEX_FREQS, TEX_AMPS):
        phase_a = rng.uniform(0, 2 * np.pi)
        along_mod = np.cos(2 * np.pi * freq * local_coords + phase_a)
        base = base * (1.0 + amp * 0.6 * along_mod)

    pattern = np.clip(base, 0.05, 1.5)
    return pattern.astype(np.float32)


def _generate_surface_textures(obstacles, room_seed):
    """Generate unique barcode texture for every surface in the room."""
    rng = np.random.RandomState(int(room_seed) & 0xFFFFFFFF)
    textures = {}

    wall_seeds = [rng.randint(0, 2**31) for _ in range(4)]
    wall_coords = np.linspace(0, 1, BARCODE_RESOLUTION)
    for i, seed in enumerate(wall_seeds):
        textures[i] = _barcode_texture(seed, wall_coords)

    n_obstacles = obstacles.shape[0]
    for obs_idx in range(n_obstacles):
        for side in range(4):
            seg_idx = 4 + obs_idx * 4 + side
            seed = int(rng.randint(0, 2**31)) 
            coords = np.linspace(0.0, 1.0, BARCODE_RESOLUTION)
            textures[seg_idx] = _barcode_texture(seed, coords)

    return textures


def compute_tof_distance(robot_pos, robot_heading, segments):
    """Compute 3-Ray ToF laser rangefinder distances (Whiskers)."""
    angles = robot_heading + jnp.array([-jnp.pi/4, 0.0, jnp.pi/4])
    origins = jnp.broadcast_to(robot_pos, (3, 2))
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    dists, _ = cast_rays(origins, directions, segments) 
    min_dists = jnp.min(dists, axis=-1) 

    max_range = 8.0
    tof_dists = jnp.clip(min_dists, 0.0, max_range)

    return tof_dists


def _precompute_barcode_tensors(surface_textures, obstacles):
    """Pre-compute texture tensors for fast vectorized lookup."""
    seg_ids = sorted(surface_textures.keys())
    n_surf = max(seg_ids) + 1 if seg_ids else 0

    tex_rows = []
    edge_rows = []
    max_stripes = 0

    for seg_id in range(n_surf):
        if seg_id in surface_textures:
            tex = np.array(surface_textures[seg_id], dtype=np.float32)
            diffs = np.diff(tex)
            edge_mask = np.abs(diffs) > 0.05
            edge_pos = np.where(edge_mask)[0] + 1 
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

    if max_stripes == 0:
        max_stripes = 1 # Avoid dimension issues on totally empty rooms
        
    padded_edges = np.zeros((n_surf, max_stripes), dtype=np.float32)
    for i, edges in enumerate(edge_rows):
        padded_edges[i, :len(edges)] = edges / (edges[-1] + 1e-8)

    return (
        jnp.stack([jnp.array(t) for t in tex_rows]) if tex_rows else jnp.zeros((0, BARCODE_RESOLUTION)),
        jnp.array(padded_edges) if padded_edges.shape[0] > 0 else jnp.zeros((0, max_stripes))
    )


def _sample_barcode_fast(min_idx, nearest, min_dist, obstacles, tex_tensor, stripe_edges):
    """Fast vectorized barcode texture lookup — all JAX-compatible."""
    n_pix = min_idx.shape[0]
    hx = nearest[:, 0]
    hy = nearest[:, 1]

    is_wall = min_idx < 4
    is_obs  = min_idx >= 4

    t_wall = jnp.stack([
        hx / ROOM_W,   
        hy / ROOM_H,   
        hx / ROOM_W,   
        hy / ROOM_H,   
    ], axis=1) 

    obs_seg_ids = min_idx - 4 
    obs_idx = obs_seg_ids // 4   
    side = obs_seg_ids % 4      

    _obs_x0 = jnp.zeros(n_pix, dtype=jnp.float32) if obstacles.shape[0] == 0 else jnp.take(obstacles[:, 0], obs_idx)
    _obs_y0 = jnp.zeros(n_pix, dtype=jnp.float32) if obstacles.shape[0] == 0 else jnp.take(obstacles[:, 1], obs_idx)
    _obs_x1 = jnp.ones(n_pix, dtype=jnp.float32)  if obstacles.shape[0] == 0 else jnp.take(obstacles[:, 2], obs_idx)
    _obs_y1 = jnp.ones(n_pix, dtype=jnp.float32)  if obstacles.shape[0] == 0 else jnp.take(obstacles[:, 3], obs_idx)
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
    ) 

    t_wall_selected = jnp.take_along_axis(t_wall, min_idx[:, None], axis=1)[:, 0]
    t_all = jnp.where(is_wall, t_wall_selected, t_obs) 

    tex_for_pix = jnp.take(tex_tensor, min_idx, axis=0) 
    edges_for_pix = jnp.take_along_axis(stripe_edges, min_idx[:, None], axis=0)[:, 0]

    exceeds = (t_all[:, None] > edges_for_pix).astype(jnp.float32) 
    stripe_idx = jnp.sum(exceeds, axis=1).astype(jnp.int32) 

    tex_idx = jnp.clip(stripe_idx, 0, BARCODE_RESOLUTION - 1) 
    batch_idx = jnp.arange(n_pix) 
    intensity = tex_for_pix[batch_idx, tex_idx]

    t_depth = min_dist / 14.14  
    
    for freq, amp in zip(TEX_FREQS, TEX_AMPS):
        phase = jnp.pi * freq 
        depth_mod = jnp.cos(2.0 * jnp.pi * freq * 6.0 * t_depth + phase)
        intensity = intensity * (1.0 + 0.6 * amp * depth_mod)

    intensity = jnp.clip(intensity, 0.05, 1.5)
    return intensity


def _sample_barcode_textures(min_idx, nearest, min_dist, surface_textures,
                             obstacles=None, tex_tensor=None, stripe_edges=None):
    if tex_tensor is not None and stripe_edges is not None:
        return _sample_barcode_fast(min_idx, nearest, min_dist, obstacles, tex_tensor, stripe_edges)

    return jnp.zeros(N_PIXELS, dtype=jnp.float32) 


def compute_pixel_readings(robot_pos, robot_heading, segments,
                           surface_textures=None, obstacles=None,
                           tex_tensor=None, stripe_edges=None):
    fov_rad = jnp.radians(FOV_DEG)
    angles = robot_heading + jnp.linspace(-fov_rad/2, fov_rad/2, N_PIXELS)
    origins = jnp.broadcast_to(robot_pos, (N_PIXELS, 2))
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    dists, hit_pts = cast_rays(origins, dirs, segments)
    min_idx = jnp.argmin(dists, axis=-1)
    min_dist = jnp.min(dists, axis=-1)
    nearest = hit_pts[jnp.arange(N_PIXELS), min_idx]
    hit_type = (min_idx >= 4).astype(jnp.float32)

    intensities = _sample_barcode_textures(min_idx, nearest, min_dist, surface_textures, obstacles, tex_tensor, stripe_edges)

    return intensities, min_dist, hit_type, nearest


# =============================================================================
# HARD REJECT: Generate Safe Sample
# =============================================================================
def generate_sample(key, time_steps=TIME_STEPS, dt=DT):
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)

    for room_attempt in range(MAX_ROOM_ATTEMPTS):
        k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))
        obstacles = generate_obstacles(k_obs)
        segments = obstacles_to_segments(obstacles)

        for traj_attempt in range(MAX_TRAJ_ATTEMPTS):
            k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
            positions, headings, vx, vy, omega = \
                _make_trajectory(k_traj, time_steps, dt, obstacles)

            spawn_ok = bool(jnp.all(
                _is_clear(positions[0, 0], positions[0, 1], obstacles)))
            traj_ok = bool(jnp.all(
                _trajectory_clear(positions, obstacles)))

            if spawn_ok and traj_ok:
                room_seed = rng.randint(0, 2**31)
                surface_textures = _generate_surface_textures(np.array(obstacles), room_seed)
                tex_t, stripe_e = _precompute_barcode_tensors(surface_textures, np.array(obstacles))

                jax_obstacles = jnp.asarray(obstacles)
                
                readings = jax.vmap(
                    lambda p, h: compute_pixel_readings(
                        p, h, segments, surface_textures, jax_obstacles, tex_t, stripe_e
                    )
                )(positions, headings)
                intensities = readings[0]
                distances = readings[1]

                tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(
                    positions, headings, segments)
                tof_normalized = tof_dists

                prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
                delta = intensities - prev
                events = jnp.where(delta > THRESHOLD, 1.0,
                          jnp.where(delta < -THRESHOLD, -1.0, 0.0))
                events = events.at[0].set(0.0)

                min_clear_arr = jnp.min(distances, axis=1)          

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
                ], axis=1)                                              

                info = {
                    'obstacles': obstacles,
                    'segments': segments,
                    'positions': positions,
                    'headings': headings,
                    'vx': vx_arr, 'vy': vy_arr, 'omega': omega_arr,   
                    'vx_mean': float(jnp.mean(vx_arr)),               
                    'vy_mean': float(jnp.mean(vy_arr)),
                    'omega_mean': float(jnp.mean(omega_arr)),
                    'intensities': intensities,
                    'distances': distances,
                    'tof': tof_normalized,
                    'room_attempts': room_attempt + 1,
                    'traj_attempts': traj_attempt + 1,
                }
                return events, labels, info

    empty_obs = jnp.zeros((0, 4), dtype=jnp.float32)
    empty_segs = jnp.array([
        [[0, 0], [ROOM_W, 0]], [[ROOM_W, 0], [ROOM_W, ROOM_H]],
        [[ROOM_W, ROOM_H], [0, ROOM_H]], [[0, ROOM_H], [0, 0]],
    ], dtype=jnp.float32)

    k_fb = jax.random.PRNGKey(rng.randint(0, 2**31))
    positions, headings, vx, vy, omega = _make_trajectory(k_fb, time_steps, dt)

    fb_room_seed = rng.randint(0, 2**31)
    fb_textures = _generate_surface_textures(np.zeros((0, 4), dtype=np.float32), fb_room_seed)
    fb_tex_t, fb_stripe_e = _precompute_barcode_tensors(fb_textures, np.zeros((0, 4), dtype=np.float32))

    readings = jax.vmap(
        lambda p, h: compute_pixel_readings(
            p, h, empty_segs, fb_textures, empty_obs, fb_tex_t, fb_stripe_e
        )
    )(positions, headings)
    
    intensities = readings[0]
    distances = readings[1]

    tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(
                positions, headings, empty_segs)
    tof_normalized = tof_dists

    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0,
              jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)

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
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)

    if obstacles is None:
        k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))
        obstacles = generate_obstacles(k_obs)
        segments = obstacles_to_segments(obstacles)
    else:
        segments = obstacles_to_segments(obstacles)

    obstacles_np = np.array(obstacles) if hasattr(obstacles, 'dtype') else obstacles
    room_seed = rng.randint(0, 2**31)
    surface_textures = _generate_surface_textures(obstacles_np, room_seed)
    tex_t, stripe_e = _precompute_barcode_tensors(surface_textures, obstacles_np)

    events_list = []
    labels_list = []
    tof_list = []
    positions_list = []
    headings_list = []
    intensities_list = []
    total_attempts = 0
    max_attempts = n_samples * MAX_TRAJ_ATTEMPTS

    jax_obstacles = jnp.asarray(obstacles)

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
                    p, h, segments, surface_textures, jax_obstacles, tex_t, stripe_e
                )
            )(positions, headings)
            
            intensities = readings[0]
            distances = readings[1]
            intensities_list.append(intensities)

            tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(
                positions, headings, segments)

            prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
            delta = intensities - prev
            events = jnp.where(delta > THRESHOLD, 1.0,
                      jnp.where(delta < -THRESHOLD, -1.0, 0.0))
            events = events.at[0].set(0.0)

            min_clear_arr = jnp.min(distances, axis=1)   
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
            ], axis=1)   

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
    print(f"  vx={float(labels[0, 0]):+.3f} vy={float(labels[0, 1]):+.3f} "
          f"ω={float(labels[0, 2]):+.3f} cl={float(labels[0, 3]):+.3f}")
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
        l = lb_b[i, 0]
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
    main();