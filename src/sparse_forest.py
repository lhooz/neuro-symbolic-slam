#!/usr/bin/env python3
"""
Sparse Forest — 2D Navigation with 1D Event Camera
(Upgraded: Texture Indexing Fix & Holonomic Kinematics)
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOM_W = 2.0
ROOM_H = 2.0

N_PIXELS = 256
FOV_DEG = 90.0
DT = 0.02
TIME_STEPS = 2000        
BATCH_SIZE = 8

BARCODE_RESOLUTION = 512
THRESHOLD = 0.015

N_OBSTACLES = 15
OBS_SIZE_MIN = 0.02        # 2cm twig (×0.2 from 10m-room scale)
OBS_SIZE_MAX = 0.14        # 14cm stem (×0.2 from 10m-room scale)
OBS_MARGIN = 0.4           # 40cm wall buffer (×0.2)

VX_RANGE = (-0.5, 0.5)     # physical hornet forward speed cap
VY_RANGE = (-0.15, 0.15)   # physical lateral speed cap
OMEGA_RANGE = (-1.0, 1.0)

SAFE_MARGIN = 0.1          # 10cm spawn clearance (×0.2)
MAX_ROOM_ATTEMPTS = 100  
MAX_TRAJ_ATTEMPTS = 50   

TEX_FREQS = [0.5, 1.0, 2.0]  
TEX_AMPS  = [0.8, 0.4, 0.2]

SEED = 42

# =============================================================================
# Obstacle Generation
# =============================================================================
def generate_obstacles(key):
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
    cx = jnp.clip(px, rect[0], rect[2])
    cy = jnp.clip(py, rect[1], rect[3])
    outside = jnp.sqrt((px - cx)**2 + (py - cy)**2)
    inside = -jnp.minimum(jnp.minimum(px - rect[0], rect[2] - px),
                           jnp.minimum(py - rect[1], rect[3] - py))
    inside_rect = (px >= rect[0]) & (px <= rect[2]) & (py >= rect[1]) & (py <= rect[3])
    return jnp.where(inside_rect, inside, outside)

def _min_clearance_to_obstacles(px, py, obstacles):
    dists = jax.vmap(lambda r: _point_rect_dist(px, py, r))(obstacles)
    # 🌟 JAX FIX: Use initial=inf to prevent zero-size array crashes
    return jnp.min(jnp.where(dists > 0, dists, jnp.inf), initial=jnp.inf)

def _wall_clearance(px, py):
    return jnp.minimum(jnp.minimum(px, py),
                       jnp.minimum(ROOM_W - px, ROOM_H - py))

def _is_clear(px, py, obstacles, margin=SAFE_MARGIN):
    obs_ok = _min_clearance_to_obstacles(px, py, obstacles) >= margin
    wall_ok = (px >= margin) & (px <= ROOM_W - margin) & \
              (py >= margin) & (py <= ROOM_H - margin)
    return obs_ok & wall_ok

def _trajectory_clear(positions, obstacles, margin=SAFE_MARGIN):
    checks = jax.vmap(lambda p: _is_clear(p[0], p[1], obstacles, margin))(positions)
    return jnp.all(checks)

# =============================================================================
# KINEMATIC "ROOMBA" EXPLORER
# =============================================================================
def _make_trajectory(key, time_steps, dt, obstacles=None):
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)
    
    pos = np.zeros((time_steps, 2), dtype=np.float32)
    hdg = np.zeros(time_steps, dtype=np.float32)
    vx = np.zeros(time_steps, dtype=np.float32)
    vy = np.zeros(time_steps, dtype=np.float32)
    omega = np.zeros(time_steps, dtype=np.float32)
    
    margin = SAFE_MARGIN + 0.05
    obs_np = np.array(obstacles) if obstacles is not None else np.zeros((0, 4))
    
    spawn_attempts = 0
    while spawn_attempts < 1000:
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
        spawn_attempts += 1
        
    if spawn_attempts >= 1000:
        return None 

    hdg[0] = rng.uniform(0, 2 * np.pi)
    v_forward = 0.3   # reduced for 2m room (was 0.6 in 10m room)
    current_omega = rng.uniform(-0.5, 0.5)
    
    for t in range(1, time_steps):
        # 1. Update intended commands
        if rng.uniform() < 0.02: 
            current_omega = rng.uniform(OMEGA_RANGE[0], OMEGA_RANGE[1])
            
        v_slip = rng.uniform(VY_RANGE[0], VY_RANGE[1]) if rng.uniform() > 0.6 else 0.0
        v_forward = 0.6
            
        # 2. Calculate intended next pose using RK2 (Midpoint Integration)
        h_mid = hdg[t-1] + (current_omega * dt) / 2.0
        h_next = hdg[t-1] + current_omega * dt
        
        px_next = pos[t-1, 0] + (v_forward * np.cos(h_mid) - v_slip * np.sin(h_mid)) * dt
        py_next = pos[t-1, 1] + (v_forward * np.sin(h_mid) + v_slip * np.cos(h_mid)) * dt
        
        # 3. Collision check on the intended pose
        hit = False
        if px_next < margin or px_next > ROOM_W - margin or py_next < margin or py_next > ROOM_H - margin:
            hit = True
        else:
            for o in obs_np:
                if (px_next > o[0] - margin and px_next < o[2] + margin and 
                    py_next > o[1] - margin and py_next < o[3] + margin):
                    hit = True
                    break
                
        # 4. Resolve Kinematics & Sync Logging
        if hit:
            # Re-roll a rotation to bounce away from the wall
            if abs(current_omega) < 0.1:
                current_omega = OMEGA_RANGE[1] if rng.uniform() > 0.5 else OMEGA_RANGE[0]
            else:
                current_omega = OMEGA_RANGE[1] * np.sign(current_omega)
            
            # Recalculate heading using the NEW bounce rotation
            h_next = hdg[t-1] + current_omega * dt
            
            # Freeze translation for this frame
            px_next, py_next = pos[t-1, 0], pos[t-1, 1] 
            v_actual_fwd = 0.0
            v_actual_slip = 0.0
        else:
            v_actual_fwd = v_forward
            v_actual_slip = v_slip
            
        # 5. Commit strictly synchronized states
        pos[t] = [px_next, py_next]
        hdg[t] = h_next % (2 * np.pi)
        
        vx[t] = v_actual_fwd
        vy[t] = v_actual_slip
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
    rng = np.random.RandomState(int(barcode_key) & 0xFFFFFFFF)
    n_stripes = rng.randint(3, 8) 
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
    angles = robot_heading + jnp.array([-jnp.pi/4, 0.0, jnp.pi/4])
    origins = jnp.broadcast_to(robot_pos, (3, 2))
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    dists, _ = cast_rays(origins, directions, segments) 
    min_dists = jnp.min(dists, axis=-1) 

    max_range = 2.83  # max diagonal of 2m×2m room = √(2²+2²)
    tof_dists = jnp.clip(min_dists, 0.0, max_range)

    return tof_dists

# 🌟 BUG FIX: Removed the convoluted 'stripe_edges' system entirely.
def _precompute_barcode_tensors(surface_textures, obstacles):
    """Pre-compute texture tensors for fast vectorized lookup."""
    seg_ids = sorted(surface_textures.keys())
    n_surf = max(seg_ids) + 1 if seg_ids else 0

    tex_rows = []
    for seg_id in range(n_surf):
        if seg_id in surface_textures:
            tex = np.array(surface_textures[seg_id], dtype=np.float32)
            tex_rows.append(tex)
        else:
            tex_rows.append(np.zeros(BARCODE_RESOLUTION, dtype=np.float32))

    return jnp.stack([jnp.array(t) for t in tex_rows]) if tex_rows else jnp.zeros((0, BARCODE_RESOLUTION))


def _sample_barcode_fast(min_idx, nearest, min_dist, obstacles, tex_tensor):
    n_pix = min_idx.shape[0]
    hx = nearest[:, 0]
    hy = nearest[:, 1]

    is_wall = min_idx < 4
    t_wall = jnp.stack([
        hx / ROOM_W,   
        hy / ROOM_H,   
        hx / ROOM_W,   
        hy / ROOM_H,   
    ], axis=1) 

    obs_seg_ids = min_idx - 4 
    obs_idx = obs_seg_ids // 4   
    side = obs_seg_ids % 4      

    # 🌟 SAFE INDEXING FIX: Use mode='clip' to prevent out-of-bounds JAX crashes
    max_obs_idx = max(0, obstacles.shape[0] - 1)
    safe_obs_idx = jnp.clip(obs_idx, 0, max_obs_idx)
    
    _obs_x0 = jnp.take(obstacles[:, 0], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.zeros(n_pix)
    _obs_y0 = jnp.take(obstacles[:, 1], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.zeros(n_pix)
    _obs_x1 = jnp.take(obstacles[:, 2], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.ones(n_pix)
    _obs_y1 = jnp.take(obstacles[:, 3], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.ones(n_pix)
    
    dx = _obs_x1 - _obs_x0 + 1e-8
    dy = _obs_y1 - _obs_y0 + 1e-8

    t_obs = jnp.where(
        side == 0, (hx - _obs_x0) / dx,
        jnp.where(
            side == 1, (hy - _obs_y0) / dy,
            jnp.where(
                side == 2, (hx - _obs_x0) / dx,
                (hy - _obs_y0) / dy
            )
        )
    ) 

    t_wall_selected = jnp.take_along_axis(t_wall, min_idx[:, None], axis=1)[:, 0]
    t_all = jnp.where(is_wall, t_wall_selected, t_obs) 

    tex_for_pix = jnp.take(tex_tensor, min_idx, axis=0, mode='clip') 

    # 🌟 THE TEXTURE SHREDDER FIX: Map coordinate natively (0.0 - 1.0) directly to the 512-dim pixel array
    tex_idx = jnp.clip(jnp.floor(t_all * BARCODE_RESOLUTION).astype(jnp.int32), 0, BARCODE_RESOLUTION - 1) 
    
    batch_idx = jnp.arange(n_pix) 
    intensity = tex_for_pix[batch_idx, tex_idx]

    t_depth = min_dist / 2.83   # normalise by room diagonal (was 14.14=√(10²+10²); now 2.83=√(2²+2²))
    
    for freq, amp in zip(TEX_FREQS, TEX_AMPS):
        phase = jnp.pi * freq 
        depth_mod = jnp.cos(2.0 * jnp.pi * freq * 6.0 * t_depth + phase)
        intensity = intensity * (1.0 + 0.6 * amp * depth_mod)

    return jnp.clip(intensity, 0.05, 1.5)


def _sample_barcode_textures(min_idx, nearest, min_dist, surface_textures, obstacles=None, tex_tensor=None):
    if tex_tensor is not None:
        return _sample_barcode_fast(min_idx, nearest, min_dist, obstacles, tex_tensor)
    return jnp.zeros(N_PIXELS, dtype=jnp.float32) 


def compute_pixel_readings(robot_pos, robot_heading, segments, surface_textures=None, obstacles=None, tex_tensor=None):
    fov_rad = jnp.radians(FOV_DEG)
    angles = robot_heading + jnp.linspace(-fov_rad/2, fov_rad/2, N_PIXELS)
    origins = jnp.broadcast_to(robot_pos, (N_PIXELS, 2))
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    
    dists, hit_pts = cast_rays(origins, dirs, segments)
    min_idx = jnp.argmin(dists, axis=-1)
    min_dist = jnp.min(dists, axis=-1)
    nearest = hit_pts[jnp.arange(N_PIXELS), min_idx]
    hit_type = (min_idx >= 4).astype(jnp.float32)

    intensities = _sample_barcode_textures(min_idx, nearest, min_dist, surface_textures, obstacles, tex_tensor)

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
            
            traj_result = _make_trajectory(k_traj, time_steps, dt, obstacles)
            if traj_result is None: continue 
                
            positions, headings, vx, vy, omega = traj_result

            spawn_ok = bool(jnp.all(_is_clear(positions[0, 0], positions[0, 1], obstacles)))
            traj_ok = bool(jnp.all(_trajectory_clear(positions, obstacles)))

            if spawn_ok and traj_ok:
                room_seed = rng.randint(0, 2**31)
                surface_textures = _generate_surface_textures(np.array(obstacles), room_seed)
                tex_t = _precompute_barcode_tensors(surface_textures, np.array(obstacles))

                jax_obstacles = jnp.asarray(obstacles)
                
                readings = jax.vmap(
                    lambda p, h: compute_pixel_readings(
                        p, h, segments, surface_textures, jax_obstacles, tex_t
                    )
                )(positions, headings)
                intensities = readings[0]
                distances = readings[1]

                tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, segments)

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

                clearance_norm = jnp.tanh(min_clear_arr / 0.4)          # 2m room: half of 0.8m mid-room clearance (was /2.0 in 10m room)
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
                    'tof': tof_dists,
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
    fb_tex_t = _precompute_barcode_tensors(fb_textures, np.zeros((0, 4), dtype=np.float32))

    readings = jax.vmap(
        lambda p, h: compute_pixel_readings(
            p, h, empty_segs, fb_textures, empty_obs, fb_tex_t
        )
    )(positions, headings)
    
    intensities = readings[0]
    distances = readings[1]

    tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, empty_segs)

    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0, jnp.where(delta < -THRESHOLD, -1.0, 0.0))
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
        'tof': tof_dists,
        'room_attempts': MAX_ROOM_ATTEMPTS,
        'traj_attempts': MAX_TRAJ_ATTEMPTS,
        'fallback': True,
    }
    return events, labels, info


def generate_fixed_room_dataset(key, n_samples, obstacles=None, time_steps=TIME_STEPS, dt=DT):
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)

    if obstacles is None:
        k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))
        obstacles = generate_obstacles(k_obs)
        segments = obstacles_to_segments(obstacles)
        room_seed = rng.randint(0, 2**31)
    else:
        segments = obstacles_to_segments(obstacles)
        # 🌟 THE TEXTURE HALLUCINATION FIX:
        # Hash the obstacles geometry to create a completely deterministic room texture seed!
        import hashlib
        obs_bytes = np.array(obstacles).tobytes()
        room_seed = int(hashlib.md5(obs_bytes).hexdigest(), 16) % (2**31 - 1)

    obstacles_np = np.array(obstacles) if hasattr(obstacles, 'dtype') else obstacles
    surface_textures = _generate_surface_textures(obstacles_np, room_seed)
    tex_t = _precompute_barcode_tensors(surface_textures, obstacles_np)

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
        traj_result = _make_trajectory(k_traj, time_steps, dt, obstacles)
        
        if traj_result is None:
            total_attempts += 1
            continue
            
        positions, headings, vx, vy, omega = traj_result

        spawn_ok = bool(jnp.all(_is_clear(positions[0, 0], positions[0, 1], obstacles)))
        traj_ok = bool(jnp.all(_trajectory_clear(positions, obstacles)))
        total_attempts += 1

        if spawn_ok and traj_ok:
            readings = jax.vmap(
                lambda p, h: compute_pixel_readings(
                    p, h, segments, surface_textures, jax_obstacles, tex_t
                )
            )(positions, headings)
            
            intensities = readings[0]
            distances = readings[1]
            intensities_list.append(intensities)

            tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, segments)

            prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
            delta = intensities - prev
            events = jnp.where(delta > THRESHOLD, 1.0, jnp.where(delta < -THRESHOLD, -1.0, 0.0))
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

    if len(events_list) == 0:
        # Fallback: ignore traj_ok (allow bounces) if we can't find a perfectly clear trajectory
        for attempt in range(max_attempts):
            k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
            traj_result = _make_trajectory(k_traj, time_steps, dt, obstacles)
            if traj_result is None: continue
            positions, headings, vx, vy, omega = traj_result
            spawn_ok = bool(jnp.all(_is_clear(positions[0, 0], positions[0, 1], obstacles)))
            if spawn_ok or attempt == max_attempts - 1:
                readings = jax.vmap(
                    lambda p, h: compute_pixel_readings(
                        p, h, segments, surface_textures, jax_obstacles, tex_t
                    )
                )(positions, headings)
                intensities = readings[0]
                distances = readings[1]
                intensities_list.append(intensities)
                tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, segments)
                prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
                delta = intensities - prev
                events = jnp.where(delta > THRESHOLD, 1.0, jnp.where(delta < -THRESHOLD, -1.0, 0.0))
                events = events.at[0].set(0.0)
                min_clear_arr = jnp.min(distances, axis=1)
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
                break

    return (jnp.stack(events_list), jnp.stack(labels_list), jnp.stack(tof_list),
            jnp.stack(positions_list), jnp.stack(headings_list), obstacles, segments,
            jnp.stack(intensities_list))


def generate_batch(key, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, dt=DT):
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
    
    key = jax.random.PRNGKey(SEED)
    import time as _time

    print("\n  ⚡ Generating single sample...")
    t0 = _time.time()
    events, labels, info = generate_sample(key)
    print(f"  Time: {_time.time()-t0:.3f}s")
    print(f"  Events: {int(jnp.sum(jnp.abs(events)))}/{N_PIXELS*TIME_STEPS}")

    print(f"\n  ⚡ Batch (B={BATCH_SIZE})...")
    t0 = _time.time()
    key2 = jax.random.split(key, 2)[0]
    ev_b, lb_b, info_b = generate_batch(key2, BATCH_SIZE)
    elapsed = _time.time()-t0
    print(f"  Time: {elapsed:.3f}s ({elapsed/BATCH_SIZE:.3f}s/sample)")

if __name__ == "__main__":
    main()