#!/usr/bin/env python3
"""
The Box Environment — Curriculum Learning Stage 1

A simple 5×5m room with no internal obstacles. 4 textured walls.
Robot starts near the back wall, heading forward, randomizing only vx.

Side walls (left/right) provide constant Z for vx estimation.
Front wall provides looming expansion for clearance/time-to-contact prediction.

Labels: [vx, clearance] (2 outputs)

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOM_W = 5.0
ROOM_H = 5.0

N_PIXELS = 64
FOV_DEG = 180.0
DT = 0.02
TIME_STEPS = 100       # 2.0s at 50 Hz
BATCH_SIZE = 8
THRESHOLD = 0.015

# Robot dynamics
VX_RANGE = (-0.8, 0.8)   # forward velocity (m/s)
VY_RANGE = (-0.4, 0.4)   # lateral velocity (m/s)
OMEGA_RANGE = (-1.0, 1.0) # yaw rate (rad/s), ~±57°/s

# Texture (multi-frequency continuous)
TEX_FREQS = [2.0, 8.0, 20.0]
TEX_AMPS = [1.0, 0.5, 0.25]

# No distance dimming — Conservation of Radiance
# Ambiently lit rooms have constant surface irradiance.
# Dimming would let the SNN cheat via brightness → depth instead of true optic flow.

# Trajectory: start centered, 2m from front wall
START_Y = ROOM_H / 2.0   # centered laterally
START_X = 2.0            # 3m from back wall, 3m from front wall
HEADING = 0.0            # initial heading +x (toward front wall)

SEED = 42


# =============================================================================
# Wall Texture
# =============================================================================
def wall_texture(x_positions, y_positions=None):
    """Multi-frequency continuous noise on wall surfaces.
    
    For each wall, the texture varies along the wall's surface.
    We use x or y coordinate depending on wall orientation:
    - Back wall (x=0) and front wall (x=ROOM_W): texture varies along y
    - Left wall (y=0) and right wall (y=ROOM_H): texture varies along x
    
    x_positions: (..., ) wall surface coordinate 1
    y_positions: (..., ) wall surface coordinate 2 (optional, for cross-pattern)
    """
    pattern = jnp.zeros_like(x_positions, dtype=jnp.float32)
    phases = [0.0, 1.0, 2.5]
    for freq, amp, phase in zip(TEX_FREQS, TEX_AMPS, phases):
        pattern = pattern + amp * jnp.sin(freq * x_positions + phase)
        if y_positions is not None:
            pattern = pattern + amp * jnp.cos(freq * y_positions * 1.3 + phase * 0.7)
    max_amp = 2.0 * sum(TEX_AMPS)
    pattern = pattern / max_amp
    return jnp.clip(pattern * 0.4 + 0.5, 0.1, 0.9)


# =============================================================================
# Ray–Segment Intersection
# =============================================================================
def cast_box_rays(origin, heading, n_pixels=N_PIXELS):
    """Cast rays in a box room. 4 walls, no obstacles.
    
    Returns:
      intensities: (N_PIXELS,)
      distances:   (N_PIXELS,) — distance to nearest wall hit
      hit_pts:     (N_PIXELS, 2) — world coordinates of hit points
    """
    fov_rad = jnp.radians(FOV_DEG)
    angles = heading + jnp.linspace(-fov_rad / 2, fov_rad / 2, n_pixels)
    
    dirs_x = jnp.cos(angles)
    dirs_y = jnp.sin(angles)
    
    ox, oy = origin[0], origin[1]
    
    # Distance to each wall (only valid if ray points toward it)
    # Back wall  (x=0):      t = -ox / dx  (dx < 0)
    # Front wall (x=ROOM_W): t = (ROOM_W - ox) / dx  (dx > 0)
    # Left wall  (y=0):      t = -oy / dy  (dy < 0)
    # Right wall (y=ROOM_H): t = (ROOM_H - oy) / dy  (dy > 0)
    
    eps = 1e-6
    
    t_back  = jnp.where(dirs_x < -eps, -ox / dirs_x, 1e6)
    t_front = jnp.where(dirs_x >  eps, (ROOM_W - ox) / dirs_x, 1e6)
    t_left  = jnp.where(dirs_y < -eps, -oy / dirs_y, 1e6)
    t_right = jnp.where(dirs_y >  eps, (ROOM_H - oy) / dirs_y, 1e6)
    
    # Nearest wall hit
    t = jnp.minimum(jnp.minimum(t_back, t_front),
                    jnp.minimum(t_left, t_right))
    t = jnp.where(t < 1e-5, 1e6, t)
    
    # Hit points
    hit_x = ox + t * dirs_x
    hit_y = oy + t * dirs_y
    
    # Determine which wall was hit (for texture orientation)
    # Texture coordinate: the coordinate along the wall surface
    # Back/front walls: texture varies with y
    # Left/right walls: texture varies with x
    
    # Simple approach: use (hit_x, hit_y) as texture inputs.
    # The walls are axis-aligned so one coord is constant and the other varies.
    # wall_texture will create a 2D pattern that works on any wall.
    # Intensity from wall texture (no dimming — pure optic flow)
    intensities = wall_texture(hit_x, hit_y)
    
    return intensities, t, jnp.stack([hit_x, hit_y], axis=-1)


def cast_box_rays_batch(positions, headings, n_pixels=N_PIXELS):
    return jax.vmap(lambda p, h: cast_box_rays(p, h, n_pixels))(positions, headings)


# =============================================================================
# Trajectory
# =============================================================================
def generate_trajectory(key, time_steps=TIME_STEPS, dt=DT):
    """Trajectory with random vx, vy, and omega.
    
    Robot starts at (START_X, START_Y), heading HEADING.
    vx, vy: body-frame velocities (m/s), omega: yaw rate (rad/s)
    Euler integration in world frame:
      dx/dt = vx*cos(θ) - vy*sin(θ)
      dy/dt = vx*sin(θ) + vy*cos(θ)
      dθ/dt = omega
    """
    k1, k2, k3 = jax.random.split(key, 3)
    vx = jax.random.uniform(k1, (), minval=VX_RANGE[0], maxval=VX_RANGE[1])
    vy = jax.random.uniform(k2, (), minval=VY_RANGE[0], maxval=VY_RANGE[1])
    omega = jax.random.uniform(k3, (), minval=OMEGA_RANGE[0], maxval=OMEGA_RANGE[1])
    
    def step(carry, _):
        x, y, theta = carry
        # Body-to-world frame transform
        wx = vx * jnp.cos(theta) - vy * jnp.sin(theta)
        wy = vx * jnp.sin(theta) + vy * jnp.cos(theta)
        dtheta = omega * dt
        new_x = x + wx * dt
        new_y = y + wy * dt
        new_theta = theta + dtheta
        pos = jnp.array([new_x, new_y])
        return (new_x, new_y, new_theta), (pos, new_theta)
    
    init = (jnp.float32(START_X), jnp.float32(START_Y), jnp.float32(HEADING))
    _, (pos, hdg) = jax.lax.scan(step, init, jnp.arange(time_steps - 1))
    
    positions = jnp.concatenate([jnp.array([[START_X, START_Y]]), pos], axis=0)
    headings = jnp.concatenate([jnp.array([HEADING]), hdg], axis=0)
    
    # Clip to room bounds
    positions = jnp.clip(positions,
                         jnp.array([0.3, 0.3]),
                         jnp.array([ROOM_W - 0.3, ROOM_H - 0.3]))
    
    return positions, headings, vx, vy, omega


# =============================================================================
# Event & Label Generation
# =============================================================================
def generate_sample(key, time_steps=TIME_STEPS, dt=DT):
    """Generate one labelled event sample in the box.
    
    Returns:
      events:  (T, N_PIXELS)  values in {-1, 0, +1}
      labels:  (4,)  [vx_normalized, vy_normalized, omega_normalized, clearance_normalized]
      info:    dict
    """
    k_traj = jax.random.split(key, 2)[0]
    positions, headings, vx, vy, omega = generate_trajectory(k_traj, time_steps, dt)
    
    intensities, distances, hit_pts = cast_box_rays_batch(positions, headings)
    
    # Events
    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0,
              jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)
    
    # Labels: [vx, vy, omega, clearance]
    final_x = positions[-1, 0]
    final_y = positions[-1, 1]
    
    # Clearance = minimum distance to ANY wall at final timestep
    # (not just front wall — lateral motion + rotation means any wall could be nearest)
    min_clear = jnp.min(distances[-1])
    
    labels = jnp.array([
        vx / abs(VX_RANGE[1]),                    # vx in [-1, 1]
        vy / abs(VY_RANGE[1]),                     # vy in [-1, 1]
        omega / abs(OMEGA_RANGE[1]),                # omega in [-1, 1]
        jnp.tanh(min_clear / 2.0),                  # clearance, saturates at ~2m
    ])
    
    info = {
        'positions': positions,
        'headings': headings,
        'vx': vx,
        'vy': vy,
        'omega': omega,
        'intensities': intensities,
        'distances': distances,
        'clearance': min_clear,
    }
    return events, labels, info


def generate_batch(key, batch_size=BATCH_SIZE,
                   time_steps=TIME_STEPS, dt=DT):
    keys = jax.random.split(key, batch_size)
    events, labels, infos = jax.vmap(
        generate_sample, in_axes=(0, None, None)
    )(keys, time_steps, dt)
    info_list = [{k: v[i] for k, v in infos.items()} for i in range(batch_size)]
    return events, labels, info_list


# =============================================================================
# Visualization
# =============================================================================
def plot_box_sample(info, events, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    intens = np.array(info['intensities'])
    dists = np.array(info['distances'])
    ev = np.array(events)
    pos = np.array(info['positions'])
    T, N = ev.shape
    time_s = np.arange(T) * DT
    vx = float(info['vx'])
    vy = float(info['vy'])
    omega = float(info['omega'])
    clearance = float(info['clearance'])
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                            gridspec_kw={'height_ratios': [2, 1.2, 1.0]})
    
    # 1. Room view
    ax1 = axes[0]
    ax1.set_xlim(-0.5, ROOM_W + 0.5)
    ax1.set_ylim(-0.5, ROOM_H + 0.5)
    ax1.set_aspect('equal')
    
    room = Rectangle((0, 0), ROOM_W, ROOM_H, lw=2, ec='black', fc='#f0f0f0')
    ax1.add_patch(room)
    
    ax1.plot(pos[:, 0], pos[:, 1], '-', color='steelblue', lw=2, label='Trajectory')
    ax1.plot(pos[0, 0], pos[0, 1], 'o', color='limegreen', ms=10, label='Start')
    ax1.plot(pos[-1, 0], pos[-1, 1], 's', color='red', ms=10, label='End')
    
    # FOV
    fov_rad = np.radians(FOV_DEG)
    for step, c in [(0, 'limegreen'), (T-1, 'red')]:
        px, py = pos[step]
        for sign in (-1, 1):
            a = sign * fov_rad / 2
            ax1.plot([px, px + 1.5 * np.cos(a)], [py, py + 1.5 * np.sin(a)],
                     '--', color=c, alpha=0.5)
    
    ax1.set_title(f'Box Environment  |  vx={vx:.3f}  vy={vy:.3f}  ω={omega:.3f}\n'
                 f'clearance={clearance:.2f}m  |  Room {ROOM_W}×{ROOM_H}m',
                 fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    
    # 2. Event raster
    ax2 = axes[1]
    on_idx = np.where(ev > 0)
    off_idx = np.where(ev < 0)
    if len(on_idx[0]) > 0:
        ax2.scatter(on_idx[0], on_idx[1], c='tab:red', s=0.6, alpha=0.5, label='ON')
    if len(off_idx[0]) > 0:
        ax2.scatter(off_idx[0], off_idx[1], c='tab:blue', s=0.6, alpha=0.5, label='OFF')
    n_ev = np.sum(np.abs(ev))
    ax2.axhline(N//2, color='green', ls='--', lw=0.8, alpha=0.5, label='Heading')
    ax2.set_xlim(0, T)
    ax2.set_ylabel('Pixel')
    ax2.set_title(f'Event Raster  |  {n_ev} events ({100*n_ev/(T*N):.1f}%)', fontsize=10)
    ax2.legend(markerscale=8, loc='upper right', fontsize=7)
    
    # 3. Intensity
    ax3 = axes[2]
    ax3.imshow(intens.T, aspect='auto', cmap='gray',
               extent=[0, time_s[-1], N, 0], vmin=0, vmax=1)
    ax3.set_ylabel('Pixel')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Pixel Intensity', fontsize=10)
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📸 Saved to {save_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("  📦 The Box Environment — Curriculum Learning")
    print("=" * 60)
    print(f"  Room:          {ROOM_W}m × {ROOM_H}m (no obstacles)")
    print(f"  Start:         ({START_X}, {START_Y}), heading +x")
    print(f"  Pixels:        {N_PIXELS}, FOV: {FOV_DEG}°")
    print(f"  vx range:      {VX_RANGE} m/s")
    print(f"  vy range:      {VY_RANGE} m/s")
    print(f"  ω range:       {OMEGA_RANGE} rad/s")
    print(f"  Labels:        [vx, vy, omega, clearance]")
    print(f"  Threshold C:   {THRESHOLD}")
    print("=" * 60)
    
    key = jax.random.PRNGKey(SEED)
    
    import time as _time
    print("\n  ⚡ Generating single sample...")
    t0 = _time.time()
    events, labels, info = generate_sample(key)
    print(f"  Time: {_time.time()-t0:.3f}s (includes JIT)")
    
    n_ev = int(jnp.sum(jnp.abs(events)))
    on = float(jnp.mean(events > 0))
    off = float(jnp.mean(events < 0))
    print(f"  Events: {n_ev}/{N_PIXELS*TIME_STEPS} ({100*n_ev/(N_PIXELS*TIME_STEPS):.1f}%)")
    print(f"  ON: {on:.4f}, OFF: {off:.4f}")
    print(f"  vx={float(info['vx']):.3f}, vy={float(info['vy']):.3f}, ω={float(info['omega']):.3f}, cl={float(info['clearance']):.3f}m")
    print(f"  labels: vx={float(labels[0]):+.3f}, vy={float(labels[1]):+.3f}, ω={float(labels[2]):+.3f}, cl={float(labels[3]):+.3f}")
    
    print(f"\n  ⚡ Batch (B={BATCH_SIZE})...")
    t0 = _time.time()
    key2 = jax.random.split(key, 2)[0]
    ev_b, lb_b, _ = generate_batch(key2, BATCH_SIZE)
    print(f"  Time: {_time.time()-t0:.3f}s")
    
    for i in range(min(4, BATCH_SIZE)):
        l = lb_b[i]
        e = ev_b[i]
        ne = int(jnp.sum(jnp.abs(e)))
        print(f"    [{i}] vx={l[0]:+.3f}  vy={l[1]:+.3f}  ω={l[2]:+.3f}  cl={l[3]:+.3f}  events={ne}")
    
    # Correlations
    import numpy as np
    key3 = jax.random.split(key2, 2)[0]
    ev_lg, lb_lg, _ = generate_batch(key3, 512)
    spike_rates = np.array(jnp.mean(jnp.abs(ev_lg), axis=(1, 2)))
    vx_vals = np.array(lb_lg[:, 0])
    cl_vals = np.array(lb_lg[:, 1])
    r_vx = np.corrcoef(spike_rates, vx_vals)[0, 1]
    r_cl = np.corrcoef(spike_rates, cl_vals)[0, 1]
    om_vals = np.array(lb_lg[:, 2])
    vy_vals = np.array(lb_lg[:, 1])
    r_om = np.corrcoef(spike_rates, om_vals)[0, 1]
    r_vy = np.corrcoef(spike_rates, vy_vals)[0, 1]
    print(f"\n  🔬 Event rate correlations: r(vx)={r_vx:+.4f}, r(vy)={r_vy:+.4f}, r(ω)={r_om:+.4f}, r(cl)={r_cl:+.4f}")
    
    plot_box_sample(info, events,
                    save_path="/Users/lhooz/.openclaw/workspace/box_sample.png")
    
    print(f"\n  ✅ Done!")


if __name__ == "__main__":
    main()
