#!/usr/bin/env python3
"""
2D Navigation Environment with 1D Event Camera Simulator (Stripe-based)

A fly robot navigates a 2D room with rectangular obstacles.
A 1D horizontal event sensor (64 pixels, 180° FOV) observes the scene.
Events are generated from temporal contrast as the robot moves.

Instead of ray-casting, each pixel maps to a wall position via its angle.
The wall is painted with sharp stripes (like a barcode), so:
  - Translation creates optic flow: stripes scroll across pixels
  - Rotation sweeps the entire pattern across the sensor
  - The stripe pattern is deterministic per wall coordinate, no depth needed

For obstacle detection, each pixel checks if its ray hits an obstacle first.
If so, the obstacle surface texture is used instead of the wall.

Physics per pixel:
  1. Compute ray direction from robot heading + pixel angle
  2. Find wall intersection point (analytical: room boundary is axis-aligned)
  3. If obstacle blocks the ray first, use obstacle surface instead
  4. Read stripe texture at the hit point
  5. Events fire when intensity change > threshold C

Output:
  - events: (B, T, N) dense, values in {-1, 0, +1}
  - labels: (B, 4) normalized [vx, vy, omega, min_clearance]

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOM_W = 10.0
ROOM_H = 10.0

N_PIXELS = 64
FOV_DEG = 180.0          # fly-like wide FOV
DT = 0.02
TIME_STEPS = 100         # 2.0 seconds at 50 Hz
BATCH_SIZE = 8
THRESHOLD = 0.10         # contrast threshold C

N_OBSTACLES = 6
OBS_SIZE_MIN = 0.4
OBS_SIZE_MAX = 2.0
OBS_MARGIN = 1.8         # keep obstacles away from room edges

# Robot dynamics
VX_RANGE = (-0.8, 0.8)   # body-frame forward velocity (m/s)
VY_RANGE = (-0.3, 0.3)   # body-frame lateral velocity (m/s)
OMEGA_RANGE = (-0.5, 0.5) # yaw rate (rad/s)

# Wall stripe texture
STRIPE_PERIOD = 0.4      # meters — stripe period on walls

# Robot safety margin from room walls
ROBOT_MARGIN = 0.5

SEED = 42


# ---------------------------------------------------------------------------
# Wall stripe texture
# ---------------------------------------------------------------------------
def wall_texture(coord):
    """Sharp square-wave stripes painted on room walls.

    coord: wall coordinate (meters), any shape (...)
    Returns: intensity in [0.1, 0.9]

    Each wall face has a 1D stripe pattern along its length.
    Like a barcode sticker — deterministic, independent of robot position.
    """
    stripe = jnp.tanh(20.0 * jnp.sin(2 * jnp.pi * coord / STRIPE_PERIOD))
    return jnp.clip(0.4 * stripe + 0.5, 0.1, 0.9)


# ---------------------------------------------------------------------------
# Obstacle generation
# ---------------------------------------------------------------------------
def generate_obstacles(key):
    """Generate N_OBSTACLES random rectangular obstacles.
    Returns (N_OBSTACLES, 4): each row is [x_min, y_min, x_max, y_max].
    """
    keys = jax.random.split(key, N_OBSTACLES)

    def _one(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        cx = jax.random.uniform(k1, (), minval=OBS_MARGIN, maxval=ROOM_W - OBS_MARGIN)
        cy = jax.random.uniform(k2, (), minval=OBS_MARGIN, maxval=ROOM_H - OBS_MARGIN)
        w = jax.random.uniform(k3, (), minval=OBS_SIZE_MIN, maxval=OBS_SIZE_MAX)
        h = jax.random.uniform(k4, (), minval=OBS_SIZE_MIN, maxval=OBS_SIZE_MAX)
        return jnp.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    return jax.vmap(_one)(keys)


# ---------------------------------------------------------------------------
# Pixel → Wall mapping (analytical, no ray casting)
# ---------------------------------------------------------------------------
def _ray_room_hit(px, py, angle):
    """Find where a ray from (px, py) at angle hits the room boundary.

    Returns:
      hit_x, hit_y: intersection point
      wall_coord: 1D coordinate along that wall face (for texture lookup)
      wall_id: 0=bottom(x), 1=right(y), 2=top(x), 3=left(y)
    """
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    # Ray: (px + t*cos_a, py + t*sin_a), find positive t for each wall
    # Bottom wall (y=0): t = -py/sin_a (need sin_a < 0)
    t_bot = jnp.where(sin_a < 0, -py / sin_a, 1e9)
    # Top wall (y=ROOM_H): t = (ROOM_H - py)/sin_a (need sin_a > 0)
    t_top = jnp.where(sin_a > 0, (ROOM_H - py) / sin_a, 1e9)
    # Left wall (x=0): t = -px/cos_a (need cos_a < 0)
    t_left = jnp.where(cos_a < 0, -px / cos_a, 1e9)
    # Right wall (x=ROOM_W): t = (ROOM_W - px)/cos_a (need cos_a > 0)
    t_right = jnp.where(cos_a > 0, (ROOM_W - px) / cos_a, 1e9)

    # Find minimum positive t
    t_min = jnp.minimum(jnp.minimum(t_bot, t_top), jnp.minimum(t_left, t_right))
    t_min = jnp.maximum(t_min, 1e-6)  # avoid zero

    hit_x = px + t_min * cos_a
    hit_y = py + t_min * sin_a

    # Wall coordinate (distance along wall face)
    wall_id = jnp.argmin(jnp.array([t_bot, t_top, t_left, t_right]))
    wall_coord = jnp.array([
        hit_x,                                          # bottom: coord along x
        hit_x,                                          # top: coord along x
        hit_y,                                          # left: coord along y
        hit_y,                                          # right: coord along y
    ])
    # Select based on which wall we actually hit
    coord = jnp.switch(wall_id, [wall_coord[0], wall_coord[1],
                                  wall_coord[2], wall_coord[3]])
    return hit_x, hit_y, coord, wall_id, t_min


def _point_in_rect(px, py, rect):
    """Check if point is inside rectangle [x0, y0, x1, y1]."""
    return (px >= rect[0]) & (px <= rect[2]) & (py >= rect[1]) & (py <= rect[3])


def _ray_rect_intersect(px, py, cos_a, sin_a, rect):
    """Ray-rectangle intersection, returns t (distance) or inf if no hit."""
    x0, y0, x1, y1 = rect

    # Parametric: px + t*cos_a, py + t*sin_a
    # Clip to rect boundaries
    t_min = -1e9
    t_max = 1e9

    if cos_a != 0:
        t_x0 = (x0 - px) / cos_a
        t_x1 = (x1 - px) / cos_a
        t_enter_x = jnp.minimum(t_x0, t_x1)
        t_exit_x = jnp.maximum(t_x0, t_x1)
        t_min = jnp.maximum(t_min, t_enter_x)
        t_max = jnp.minimum(t_max, t_exit_x)

    if sin_a != 0:
        t_y0 = (y0 - py) / sin_a
        t_y1 = (y1 - py) / sin_a
        t_enter_y = jnp.minimum(t_y0, t_y1)
        t_exit_y = jnp.maximum(t_y0, t_y1)
        t_min = jnp.maximum(t_min, t_enter_y)
        t_max = jnp.minimum(t_max, t_exit_y)

    valid = (t_min < t_max) & (t_max > 1e-4)
    return jnp.where(valid, jnp.maximum(t_min, 1e-4), 1e9)


def compute_pixel_intensities(positions, headings, obstacles):
    """Compute per-pixel intensities for all timesteps.

    positions: (T, 2)
    headings:  (T,)
    obstacles: (N_OBS, 4)

    Returns:
      intensities: (T, N_PIXELS)
      min_distances: (T,) — min distance to any obstacle per timestep
    """
    T = positions.shape[0]
    fov_rad = jnp.radians(FOV_DEG)

    # Pixel angles relative to heading
    pix_angles = jnp.linspace(-fov_rad / 2, fov_rad / 2, N_PIXELS)  # (N,)

    # Full ray angles: (T, N)
    ray_angles = headings[:, None] + pix_angles[None, :]

    cos_a = jnp.cos(ray_angles)  # (T, N)
    sin_a = jnp.sin(ray_angles)  # (T, N)
    px = positions[:, 0]         # (T,)
    py = positions[:, 1]         # (T,)

    # --- Wall hits (vectorized) ---
    t_bot = jnp.where(sin_a < 0, -py[:, None] / (sin_a + 1e-30), 1e9)
    t_top = jnp.where(sin_a > 0, (ROOM_H - py[:, None]) / (sin_a - 1e-30), 1e9)
    t_left = jnp.where(cos_a < 0, -px[:, None] / (cos_a + 1e-30), 1e9)
    t_right = jnp.where(cos_a > 0, (ROOM_W - px[:, None]) / (cos_a - 1e-30), 1e9)

    t_wall = jnp.minimum(jnp.minimum(t_bot, t_top),
                         jnp.minimum(t_left, t_right))
    t_wall = jnp.maximum(t_wall, 1e-6)

    hit_x = px[:, None] + t_wall * cos_a  # (T, N)
    hit_y = py[:, None] + t_wall * sin_a

    # Wall coordinate for texture
    # Bottom/top walls → use x coordinate; left/right → use y coordinate
    is_horiz = (t_wall == t_bot) | (t_wall == t_top)
    wall_coord = jnp.where(is_horiz, hit_x, hit_y)

    # --- Obstacle hits ---
    # For each obstacle, check ray intersection
    def _obs_t(rect):
        """Compute ray-obstacle intersection distances for all (T, N) rays."""
        x0, y0, x1, y2 = rect

        # Clip parametric line to rect
        t_enter = jnp.full((T, N), -1e9)
        t_exit = jnp.full((T, N), 1e9)

        # X boundaries
        t_x0 = (x0 - px[:, None]) / (cos_a + 1e-30)
        t_x1 = (x1 - px[:, None]) / (cos_a + 1e-30)
        t_enter_x = jnp.minimum(t_x0, t_x1)
        t_exit_x = jnp.maximum(t_x0, t_x1)
        t_enter = jnp.maximum(t_enter, t_enter_x)
        t_exit = jnp.minimum(t_exit, t_exit_x)

        # Y boundaries
        t_y0 = (y0 - py[:, None]) / (sin_a + 1e-30)
        t_y1 = (y2 - py[:, None]) / (sin_a + 1e-30)
        t_enter_y = jnp.minimum(t_y0, t_y1)
        t_exit_y = jnp.maximum(t_y0, t_y1)
        t_enter = jnp.maximum(t_enter, t_enter_y)
        t_exit = jnp.minimum(t_exit, t_exit_y)

        valid = (t_enter < t_exit) & (t_exit > 1e-4)
        return jnp.where(valid, jnp.maximum(t_enter, 1e-4), 1e9)

    obs_ts = jax.vmap(_obs_t)(obstacles)  # (N_OBS, T, N)
    t_nearest_obs = jnp.min(obs_ts, axis=0)  # (T, N)

    # If obstacle is closer than wall, use obstacle
    obs_blocks = t_nearest_obs < t_wall
    obs_hit_x = px[:, None] + t_nearest_obs * cos_a
    obs_hit_y = py[:, None] + t_nearest_obs * sin_a

    # For obstacle surfaces, use distance from robot for texture
    obs_coord = t_nearest_obs  # just use distance as coordinate

    # Final hit coordinates
    final_coord = jnp.where(obs_blocks, obs_coord, wall_coord)
    final_t = jnp.where(obs_blocks, t_nearest_obs, t_wall)

    # Obstacles have a distinct flat texture (different from walls)
    # Use a different stripe frequency for obstacles
    obs_intensity = jnp.clip(
        0.4 * jnp.tanh(15.0 * jnp.sin(2 * jnp.pi * obs_coord / (STRIPE_PERIOD * 0.7))) + 0.5,
        0.1, 0.9)

    # Distance dimming
    dimming = 1.0 / (1.0 + (final_t / 4.0) ** 2)
    obs_dim = obs_intensity * dimming
    wall_dim = wall_texture(final_coord) * dimming
    intensity = jnp.where(obs_blocks, obs_dim, wall_dim)

    # Min distance to any obstacle per timestep
    min_distances = jnp.min(jnp.min(obs_ts, axis=-1), axis=-1)  # (T,)
    # Cap at room diagonal if no obstacle hit
    min_distances = jnp.minimum(min_distances, 15.0)

    return intensity, min_distances


# ---------------------------------------------------------------------------
# Trajectory generation (collision-safe)
# ---------------------------------------------------------------------------
def _point_rect_dist(px, py, rect):
    """Signed distance: positive outside, 0 at edge, negative inside."""
    cx = jnp.clip(px, rect[0], rect[2])
    cy = jnp.clip(py, rect[1], rect[3])
    dx, dy = px - cx, py - cy
    outside_dist = jnp.sqrt(dx * dx + dy * dy)
    inside = (rect[0] <= px) & (px <= rect[2]) & (rect[1] <= py) & (py <= rect[3])
    edge_d = jnp.minimum(jnp.minimum(px - rect[0], rect[2] - px),
                         jnp.minimum(py - rect[1], rect[3] - py))
    return jnp.where(inside, -edge_d, outside_dist)


def _min_obstacle_dist(px, py, obstacles):
    """Min distance from point to any obstacle. Positive = outside all."""
    dists = jax.vmap(lambda r: _point_rect_dist(px, py, r))(obstacles)
    return jnp.min(jnp.where(dists > 0, dists, jnp.inf))


def _is_clear(px, py, obstacles, margin=ROBOT_MARGIN):
    """True if point has >= margin clearance from all obstacles and walls."""
    obs_ok = _min_obstacle_dist(px, py, obstacles) >= margin
    wall_ok = (px >= margin) & (px <= ROOM_W - margin) & \
              (py >= margin) & (py <= ROOM_H - margin)
    return obs_ok & wall_ok


def _trajectory_clear(positions, obstacles, margin=ROBOT_MARGIN):
    """Check all trajectory points are clear."""
    checks = jax.vmap(lambda p: _is_clear(p[0], p[1], obstacles, margin))(positions)
    return jnp.all(checks)


def _generate_trajectory_inner(key, obstacles, time_steps, dt):
    """Generate one candidate trajectory."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    x0 = jax.random.uniform(k1, (), minval=ROBOT_MARGIN + 0.5,
                            maxval=ROOM_W - ROBOT_MARGIN - 0.5)
    y0 = jax.random.uniform(k2, (), minval=ROBOT_MARGIN + 0.5,
                            maxval=ROOM_H - ROBOT_MARGIN - 0.5)

    spawn_ok = _is_clear(x0, y0, obstacles)

    h0 = jax.random.uniform(k3, (), minval=0.0, maxval=2 * jnp.pi)
    vx = jax.random.uniform(k4, (), minval=VX_RANGE[0], maxval=VX_RANGE[1])
    vy = jax.random.uniform(k5, (), minval=VY_RANGE[0], maxval=VY_RANGE[1])
    omega = jax.random.uniform(jax.random.split(key, 6)[0], (),
                               minval=OMEGA_RANGE[0], maxval=OMEGA_RANGE[1])

    t = jnp.arange(time_steps, dtype=jnp.float32) * dt
    headings = (h0 + omega * t) % (2 * jnp.pi)

    cos_h, sin_h = jnp.cos(headings), jnp.sin(headings)
    wx = vx * cos_h - vy * sin_h
    wy = vx * sin_h + vy * cos_h

    dx = jnp.concatenate([jnp.zeros(1), jnp.cumsum(wx[:-1] * dt)])
    dy = jnp.concatenate([jnp.zeros(1), jnp.cumsum(wy[:-1] * dt)])
    positions = jnp.stack([x0 + dx, y0 + dy], axis=-1)

    traj_ok = _trajectory_clear(positions, obstacles)
    ok = spawn_ok & traj_ok

    dummy = jnp.zeros_like(positions)
    return jnp.where(ok[None, None], positions, dummy), headings, vx, vy, omega, ok


MAX_RESAMPLE = 20


def generate_trajectory_safe(key, obstacles, time_steps=TIME_STEPS, dt=DT):
    """Collision-safe trajectory with rejection sampling."""
    def body_fn(carry, _):
        k, _ = carry
        k_new, k_inner = jax.random.split(k, 2)
        positions, headings, vx, vy, omega, ok = \
            _generate_trajectory_inner(k_inner, obstacles, time_steps, dt)
        return (k_new, ok), (positions, headings, vx, vy, omega, ok)

    init = (key, jnp.array(False))
    (_, final_ok), (positions, headings, vx, vy, omega, _) = \
        jax.lax.scan(body_fn, init, None, length=MAX_RESAMPLE)

    return positions, headings, vx, vy, omega, final_ok


# ---------------------------------------------------------------------------
# Sample / Batch Generation
# ---------------------------------------------------------------------------
def generate_sample(key, time_steps=TIME_STEPS, dt=DT):
    """Generate one labelled event sample (collision-safe).

    Returns:
      events:  (T, N_PIXELS)  values in {-1, 0, +1}
      labels:  (4,)  normalized [vx, vy, omega, min_clearance]
      info:    dict with scene geometry + trajectory
    """
    k_obs, k_traj = jax.random.split(key, 2)

    obstacles = generate_obstacles(k_obs)
    positions, headings, vx, vy, omega, accepted = \
        generate_trajectory_safe(k_traj, obstacles, time_steps, dt)

    intensities, min_distances = compute_pixel_intensities(
        positions, headings, obstacles)

    # Temporal contrast → events
    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0,
              jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)

    # Labels
    min_clear = min_distances[-1]
    labels = jnp.array([
        vx / abs(VX_RANGE[1]),
        vy / abs(VY_RANGE[1]),
        omega / abs(OMEGA_RANGE[1]),
        jnp.tanh(min_clear / 2.0),
    ])

    info = {
        'obstacles': obstacles,
        'positions': positions,
        'headings': headings,
        'vx': vx, 'vy': vy, 'omega': omega,
        'intensities': intensities,
        'min_distances': min_distances,
    }
    return events, labels, info


def generate_batch(key, batch_size=BATCH_SIZE,
                   time_steps=TIME_STEPS, dt=DT):
    """Generate a batch of labelled event samples."""
    keys = jax.random.split(key, batch_size)
    events, labels, infos = jax.vmap(
        generate_sample, in_axes=(0, None, None)
    )(keys, time_steps, dt)

    info_list = [{k: v[i] for k, v in infos.items()} for i in range(batch_size)]
    return events, labels, info_list


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_scene(info, events, save_path=None, snap_steps=(0, 25, 50, 75, 99)):
    """Top-down scene view + event raster + intensity image."""
    obs = np.array(info['obstacles'])
    pos = np.array(info['positions'])
    hdg = np.array(info['headings'])
    intens = np.array(info['intensities'])
    dists = np.array(info['min_distances'])
    ev = np.array(events)
    T, N = ev.shape
    time_s = np.arange(T) * DT

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1.2, 1.0, 0.8],
                          hspace=0.35)

    # ---- 1. Top-down scene ----
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(-0.5, ROOM_W + 0.5)
    ax1.set_ylim(-0.5, ROOM_H + 0.5)
    ax1.set_aspect('equal')
    ax1.set_title(
        f'2D Navigation Scene (Stripe-based)  |  vx={info["vx"]:.2f} vy={info["vy"]:.2f} '
        f'ω={info["omega"]:.2f} rad/s',
        fontsize=11, fontweight='bold')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')

    room_rect = Rectangle((0, 0), ROOM_W, ROOM_H, linewidth=2,
                          edgecolor='black', facecolor='#f0f0f0')
    ax1.add_patch(room_rect)

    for o in obs:
        w, h = o[2] - o[0], o[3] - o[1]
        ax1.add_patch(Rectangle((o[0], o[1]), w, h,
                                facecolor='#555555', edgecolor='black',
                                linewidth=1.2, alpha=0.85))

    ax1.plot(pos[:, 0], pos[:, 1], '-', color='steelblue', lw=1.5,
             alpha=0.6, zorder=3, label='Trajectory')
    ax1.plot(pos[0, 0], pos[0, 1], 'o', color='limegreen', ms=8,
             zorder=5, label='Start')
    ax1.plot(pos[-1, 0], pos[-1, 1], 's', color='red', ms=8,
             zorder=5, label='End')

    fov_rad = np.radians(FOV_DEG)
    colors_snap = plt.cm.cool(np.linspace(0, 1, len(snap_steps)))
    for idx, step in enumerate(snap_steps):
        if step >= T:
            continue
        px, py = pos[step]
        h = hdg[step]
        half = fov_rad / 2
        for sign in (-1, 1):
            angle = h + sign * half
            dx = 1.5 * np.cos(angle)
            dy = 1.5 * np.sin(angle)
            ax1.annotate('', xy=(px + dx, py + dy), xytext=(px, py),
                         arrowprops=dict(arrowstyle='-', color=colors_snap[idx],
                                         lw=0.8, alpha=0.5))
        ax1.annotate('', xy=(px + 0.4 * np.cos(h), py + 0.4 * np.sin(h)),
                     xytext=(px, py),
                     arrowprops=dict(arrowstyle='->', color=colors_snap[idx],
                                     lw=1.5))
        ax1.text(px, py + 0.45, f't={step * DT:.1f}s',
                 fontsize=7, ha='center', color=colors_snap[idx], fontweight='bold')

    ax1.legend(loc='upper right', fontsize=8)

    # ---- 2. Event raster ----
    ax2 = fig.add_subplot(gs[1])
    on_idx = np.where(ev > 0)
    off_idx = np.where(ev < 0)
    if len(on_idx[0]) > 0:
        ax2.scatter(on_idx[0], on_idx[1], c='tab:red', s=0.6, alpha=0.5,
                    label='ON (+1)')
    if len(off_idx[0]) > 0:
        ax2.scatter(off_idx[0], off_idx[1], c='tab:blue', s=0.6, alpha=0.5,
                    label='OFF (−1)')
    n_ev = np.sum(np.abs(ev))
    ax2.set_xlim(0, T)
    ax2.set_ylabel('Pixel')
    ax2.set_title(f'Event Raster  |  {n_ev} events '
                  f'({100 * n_ev / (T * N):.1f}% active)', fontsize=10)
    ax2.legend(markerscale=8, loc='upper right', fontsize=7)

    # ---- 3. Intensity image ----
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(intens.T, aspect='auto', cmap='gray',
               extent=[0, time_s[-1], N, 0], vmin=0, vmax=1)
    ax3.set_ylabel('Pixel')
    ax3.set_title('Pixel Intensity (wall stripes + obstacle surfaces)', fontsize=10)

    # ---- 4. Min obstacle distance ----
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(time_s, dists, 'g-', lw=1.2)
    ax4.axhline(0.5, color='red', ls='--', lw=0.8, alpha=0.6, label='Danger zone')
    ax4.fill_between(time_s, 0, np.minimum(dists, 0.5), color='red', alpha=0.15)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Min distance (m)')
    ax4.set_ylim(0, None)
    ax4.legend(fontsize=7)
    ax4.set_title('Nearest Obstacle Distance', fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📸 Saved to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  🦊 2D Navigation — Stripe-based Event Camera")
    print("=" * 60)
    print(f"  Room:         {ROOM_W}m × {ROOM_H}m")
    print(f"  Obstacles:    {N_OBSTACLES} random rectangles")
    print(f"  Pixels:       {N_PIXELS}  (1D horizontal)")
    print(f"  FOV:          ±{FOV_DEG / 2:.0f}°")
    print(f"  Time steps:   {TIME_STEPS} ({TIME_STEPS * DT:.1f}s at {1 / DT:.0f} Hz)")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Threshold C:  {THRESHOLD}")
    print(f"  Stripe period: {STRIPE_PERIOD}m")
    print("-" * 60)
    print(f"  State:        [vx_body, vy_body, omega]")
    print(f"  vx range:     {VX_RANGE} m/s")
    print(f"  vy range:     {VY_RANGE} m/s")
    print(f"  ω range:      {OMEGA_RANGE} rad/s")
    print(f"  Labels:       [vx, vy, omega, min_clearance]")
    print("=" * 60)

    key = jax.random.PRNGKey(SEED)

    # Single sample
    print("\n  ⚡ Generating single sample...")
    t0 = time.time()
    events, labels, info = generate_sample(key)
    t1 = time.time()

    n_ev = jnp.sum(jnp.abs(events))
    print(f"  Events shape: {events.shape}")
    print(f"  Total events: {n_ev.item()} / {N_PIXELS * TIME_STEPS} "
          f"({n_ev / (N_PIXELS * TIME_STEPS):.2%})")
    print(f"  Labels: vx={labels[0]:+.3f}  vy={labels[1]:+.3f}  "
          f"ω={labels[2]:+.3f}  clear={labels[3]:+.3f}")
    print(f"  Generate time: {t1 - t0:.3f}s (includes JIT compile)")

    t0 = time.time()
    events2, labels2, info2 = generate_sample(jax.random.split(key, 2)[1])
    t1 = time.time()
    print(f"  Generate time: {t1 - t0:.3f}s (compiled)")

    print(f"\n  📊 Scene visualization...")
    plot_scene(info, events,
               save_path="/Users/lhooz/.openclaw/workspace/2d_nav_scene.png")

    # Batch
    print(f"\n  ⚡ Generating batch (B={BATCH_SIZE})...")
    t0 = time.time()
    key_b = jax.random.split(key, 3)[1]
    ev_batch, lb_batch, info_batch = generate_batch(key_b, BATCH_SIZE)
    t1 = time.time()

    n_ev_b = jnp.sum(jnp.abs(ev_batch))
    print(f"  Batch shape: {ev_batch.shape}")
    print(f"  Total events: {n_ev_b.item()} / {BATCH_SIZE * N_PIXELS * TIME_STEPS}")
    print(f"  Batch time: {t1 - t0:.3f}s (compiled)")

    print(f"\n  Sample breakdown:")
    for i in range(min(4, BATCH_SIZE)):
        l = lb_batch[i]
        ne = jnp.sum(jnp.abs(ev_batch[i])).item()
        sp = 1 - ne / (N_PIXELS * TIME_STEPS)
        print(f"    [{i}] vx={l[0]:+.2f} vy={l[1]:+.2f} "
              f"ω={l[2]:+.2f} clear={l[3]:+.2f} | "
              f"{ne} events  sparsity={sp:.1%}")

    print(f"\n  ⚡ Benchmark: batch B=128...")
    key_lg = jax.random.split(key, 4)[1]
    t0 = time.time()
    _ = generate_batch(key_lg, 128)
    t1 = time.time()
    print(f"  Batch gen (B=128): {t1 - t0:.3f}s")

    print("\n  ✅ Done!")


if __name__ == "__main__":
    main()
