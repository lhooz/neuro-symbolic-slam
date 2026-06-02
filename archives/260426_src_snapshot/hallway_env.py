#!/usr/bin/env python3
"""
Infinite Hallway Environment — Monocular Optic Flow

A perfectly straight hallway, 3.0m wide, extending infinitely in x.
Robot starts centered (y=1.5m), heading along +x.
Only vx is randomized. Z is constant at 1.5m to each wall.

This eliminates monocular scale ambiguity (μ = v/Z):
  - Z = 1.5m (constant, both walls equidistant)
  - Optic flow on walls ∝ vx / Z = vx / 1.5
  - Temporal spike sequence directly encodes vx

Scene:
  - Wall at y=0 and y=3.0 (parallel to travel direction)
  - Multi-frequency continuous noise texture on both walls
  - 1D horizontal event sensor, 64 pixels, 180° FOV
  - No obstacles, no random room geometry

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HALLWAY_WIDTH = 3.0
HALLWAY_Y0 = 0.0
HALLWAY_Y1 = HALLWAY_WIDTH

N_PIXELS = 64
FOV_DEG = 180.0
DT = 0.02
TIME_STEPS = 100       # 2.0s at 50 Hz
BATCH_SIZE = 8
THRESHOLD = 0.015       # low threshold for rich optic flow

# Robot
VX_RANGE = (-0.8, 0.8)  # forward velocity (m/s)

# Texture (multi-frequency continuous)
TEX_FREQS = [2.0, 8.0, 20.0]
TEX_AMPS = [1.0, 0.5, 0.25]

# Rendering
# No distance dimming — Conservation of Radiance (ambient light = constant irradiance)
THRESHOLD = 0.015


# =============================================================================
# Wall Texture
# =============================================================================
def wall_texture(x_positions):
    """Multi-frequency continuous noise along x-axis.
    
    x_positions: (..., ) world x-coordinates on the wall surface
    returns: (..., ) intensity in [0.1, 0.9]
    """
    pattern = jnp.zeros_like(x_positions, dtype=jnp.float32)
    phases = [0.0, 1.0, 2.5]
    for freq, amp, phase in zip(TEX_FREQS, TEX_AMPS, phases):
        pattern = pattern + amp * jnp.sin(freq * x_positions + phase)
        pattern = pattern + amp * jnp.cos(freq * x_positions * 1.3 + phase * 0.7)
    max_amp = 2.0 * sum(TEX_AMPS)
    pattern = pattern / max_amp
    return jnp.clip(pattern * 0.4 + 0.5, 0.1, 0.9)


# =============================================================================
# Ray Casting (hallway only — 2 wall segments + virtual far wall)
# =============================================================================
def cast_hallway_rays(origin, heading, n_pixels=N_PIXELS):
    """Cast rays in a hallway. Returns (intensities, distances, hit_pts).
    
    The hallway has 2 walls: y=0 and y=HALLWAY_WIDTH.
    Rays heading upward (toward y=HALLWAY_WIDTH) hit the top wall.
    Rays heading downward (toward y=0) hit the bottom wall.
    
    For rays parallel to walls (near ±90° from heading),
    they'd never hit a wall — we return a very large distance and 0 intensity.
    """
    fov_rad = jnp.radians(FOV_DEG)
    # Angles relative to robot heading
    angles = heading + jnp.linspace(-fov_rad / 2, fov_rad / 2, n_pixels)
    
    # Ray direction
    dirs_x = jnp.cos(angles)
    dirs_y = jnp.sin(angles)
    
    ox, oy = origin[0], origin[1]
    
    # Distance to bottom wall (y=0): t_bot = -oy / dy (only if dy < 0)
    t_bot = jnp.where(dirs_y < -1e-6, -oy / dirs_y, 1e6)
    # Distance to top wall (y=HALLWAY_WIDTH): t_top = (HALLWAY_WIDTH - oy) / dy (only if dy > 0)
    t_top = jnp.where(dirs_y > 1e-6, (HALLWAY_WIDTH - oy) / dirs_y, 1e6)
    
    # Take nearest wall hit
    t = jnp.minimum(t_bot, t_top)
    t = jnp.where(t < 1e-5, 1e6, t)  # ignore hits behind/near robot
    
    # Hit points
    hit_x = ox + t * dirs_x
    hit_y = oy + t * dirs_y
    
    # Intensity from texture
    # Intensity from texture (no dimming — pure optic flow)
    intensities = wall_texture(hit_x)
    
    return intensities, t, jnp.stack([hit_x, hit_y], axis=-1)


def cast_hallway_rays_batch(positions, headings, n_pixels=N_PIXELS):
    """Vectorized over timesteps: (T,2) and (T,) → (T, N_PIXELS), (T, N_PIXELS)."""
    return jax.vmap(lambda p, h: cast_hallway_rays(p, h, n_pixels))(positions, headings)


# =============================================================================
# Trajectory Generation
# =============================================================================
def generate_trajectory(key, time_steps=TIME_STEPS, dt=DT):
    """Straight-line trajectory in the hallway.
    Robot starts at (0, hallway_width/2), heading +x.
    Only vx is randomized. vy=0, omega=0.
    """
    k1 = jax.random.split(key, 2)[0]
    vx = jax.random.uniform(k1, (), minval=VX_RANGE[0], maxval=VX_RANGE[1])
    
    y_center = HALLWAY_WIDTH / 2.0
    
    t = jnp.arange(time_steps, dtype=jnp.float32) * dt
    positions = jnp.stack([vx * t, jnp.full(time_steps, y_center)], axis=-1)
    headings = jnp.zeros(time_steps, dtype=jnp.float32)  # always heading +x
    
    return positions, headings, vx


# =============================================================================
# Event Generation
# =============================================================================
def generate_sample(key, time_steps=TIME_STEPS, dt=DT):
    """Generate one labelled event sample in the hallway.
    
    Returns:
      events:  (T, N_PIXELS)  values in {-1, 0, +1}
      labels:  (1,)  vx normalized to [-1, 1]
      info:    dict with trajectory + intensities
    """
    k_traj = jax.random.split(key, 2)[0]
    positions, headings, vx = generate_trajectory(k_traj, time_steps, dt)
    
    intensities, distances, hit_pts = cast_hallway_rays_batch(positions, headings)
    
    # Temporal contrast → events
    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0,
              jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)
    
    # Label: only vx
    labels = jnp.array([vx / abs(VX_RANGE[1])])
    
    info = {
        'positions': positions,
        'headings': headings,
        'vx': vx,
        'intensities': intensities,
        'distances': distances,
    }
    return events, labels, info


def generate_batch(key, batch_size=BATCH_SIZE,
                   time_steps=TIME_STEPS, dt=DT):
    """Generate a batch of hallway event samples."""
    keys = jax.random.split(key, batch_size)
    events, labels, infos = jax.vmap(
        generate_sample, in_axes=(0, None, None)
    )(keys, time_steps, dt)
    info_list = [{k: v[i] for k, v in infos.items()} for i in range(batch_size)]
    return events, labels, info_list


# =============================================================================
# Visualization
# =============================================================================
def plot_hallway_sample(info, events, save_path=None):
    """Visualize a hallway sample: top-down view + event raster + intensity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    intens = np.array(info['intensities'])
    dists = np.array(info['distances'])
    ev = np.array(events)
    pos = np.array(info['positions'])
    T, N = ev.shape
    time_s = np.arange(T) * DT
    vx = float(info['vx'])
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), 
                            gridspec_kw={'height_ratios': [2, 1.2, 1.0]})
    
    # 1. Top-down hallway view
    ax1 = axes[0]
    hw = HALLWAY_WIDTH
    # Show a 4m section of hallway
    x_start = max(0, pos[0, 0] - 1)
    x_end = pos[-1, 0] + 2
    ax1.set_xlim(x_start, x_end)
    ax1.set_ylim(-0.5, hw + 0.5)
    ax1.set_aspect('equal')
    
    # Walls
    ax1.axhline(0, color='black', lw=3)
    ax1.axhline(hw, color='black', lw=3)
    ax1.fill_between([x_start, x_end], -0.5, 0, color='gray', alpha=0.3)
    ax1.fill_between([x_start, x_end], hw, hw + 0.5, color='gray', alpha=0.3)
    
    # Texture preview on walls
    tex_x = np.linspace(x_start, x_end, 500)
    tex_y_bot = wall_texture(tex_x)
    tex_y_top = wall_texture(tex_x)
    ax1.plot(tex_x, tex_y_bot * 0.3, 'g-', lw=1, alpha=0.5)
    ax1.plot(tex_x, hw + tex_y_top * 0.3 - 0.15, 'g-', lw=1, alpha=0.5)
    
    # Trajectory
    ax1.plot(pos[:, 0], pos[:, 1], '-', color='steelblue', lw=2, label='Trajectory')
    ax1.plot(pos[0, 0], pos[0, 1], 'o', color='limegreen', ms=10, label='Start')
    ax1.plot(pos[-1, 0], pos[-1, 1], 's', color='red', ms=10, label='End')
    
    # FOV at start and end
    fov_rad = np.radians(FOV_DEG)
    for step, color, label in [(0, 'limegreen', 'Start FOV'), (T-1, 'red', 'End FOV')]:
        px, py = pos[step]
        for sign in (-1, 1):
            angle = sign * fov_rad / 2
            ax1.plot([px, px + 1.5 * np.cos(angle)], [py, py + 1.5 * np.sin(angle)],
                     '--', color=color, alpha=0.5)
    
    ax1.set_title(f'Hallway vx={vx:.3f} m/s  |  width={hw}m  |  Z={hw/2:.1f}m to each wall',
                 fontsize=11, fontweight='bold')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.legend(loc='upper right')
    
    # 2. Event raster
    ax2 = axes[1]
    on_idx = np.where(ev > 0)
    off_idx = np.where(ev < 0)
    if len(on_idx[0]) > 0:
        ax2.scatter(on_idx[0], on_idx[1], c='tab:red', s=0.6, alpha=0.5, label='ON (+1)')
    if len(off_idx[0]) > 0:
        ax2.scatter(off_idx[0], off_idx[1], c='tab:blue', s=0.6, alpha=0.5, label='OFF (−1)')
    n_ev = np.sum(np.abs(ev))
    ax2.axhline(N // 2, color='green', ls='--', lw=0.8, alpha=0.5)
    ax2.set_xlim(0, T)
    ax2.set_ylabel('Pixel')
    ax2.set_title(f'Event Raster  |  {n_ev} events ({100*n_ev/(T*N):.1f}%)  |  '
                  f'ON={np.sum(ev>0)} OFF={np.sum(ev<0)}', fontsize=10)
    ax2.legend(markerscale=8, loc='upper right', fontsize=7)
    
    # 3. Intensity image
    ax3 = axes[2]
    ax3.imshow(intens.T, aspect='auto', cmap='gray',
               extent=[0, time_s[-1], N, 0], vmin=0, vmax=1)
    ax3.set_ylabel('Pixel')
    ax3.set_title('Pixel Intensity', fontsize=10)
    ax3.set_xlabel('Time (s)')
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📸 Saved to {save_path}")
    plt.close(fig)


# =============================================================================
# Main (quick test)
# =============================================================================
def main():
    print("=" * 60)
    print("  🦊 Infinite Hallway — Optic Flow Environment")
    print("=" * 60)
    print(f"  Hallway width: {HALLWAY_WIDTH}m")
    print(f"  Z (wall dist): {HALLWAY_WIDTH/2:.1f}m (constant)")
    print(f"  Pixels:        {N_PIXELS}")
    print(f"  FOV:           {FOV_DEG}°")
    print(f"  vx range:      {VX_RANGE} m/s")
    print(f"  Threshold C:   {THRESHOLD}")
    print(f"  Dimming:       {'OFF' if DISABLE_DIMMING else 'ON'}")
    print("=" * 60)
    
    key = jax.random.PRNGKey(SEED)
    
    print("\n  ⚡ Generating single sample...")
    t0 = time.time()
    events, labels, info = generate_sample(key)
    print(f"  Time: {time.time()-t0:.3f}s (includes JIT)")
    
    n_ev = int(jnp.sum(jnp.abs(events)))
    on_frac = float(jnp.mean(events > 0))
    off_frac = float(jnp.mean(events < 0))
    print(f"  Events: {n_ev}/{N_PIXELS*TIME_STEPS} ({100*n_ev/(N_PIXELS*TIME_STEPS):.1f}%)")
    print(f"  ON: {on_frac:.4f}, OFF: {off_frac:.4f}")
    print(f"  vx: {float(info['vx']):.3f}, label: {float(labels[0]):.3f}")
    
    # Batch
    print(f"\n  ⚡ Batch (B={BATCH_SIZE})...")
    t0 = time.time()
    key2 = jax.random.split(key, 2)[0]
    ev_b, lb_b, _ = generate_batch(key2, BATCH_SIZE)
    print(f"  Time: {time.time()-t0:.3f}s")
    
    # Check event rate statistics
    print(f"\n  Event stats across batch:")
    for i in range(min(4, BATCH_SIZE)):
        e = ev_b[i]
        l = lb_b[i]
        ne = int(jnp.sum(jnp.abs(e)))
        on = float(jnp.mean(e > 0))
        off = float(jnp.mean(e < 0))
        print(f"    [{i}] vx={l[0]:+.3f}  events={ne}  ON={on:.3f}  OFF={off:.3f}")
    
    # Check correlation: does event rate encode vx?
    import numpy as np
    key3 = jax.random.split(key2, 2)[0]
    ev_large, lb_large, _ = generate_batch(key3, 256)
    spike_rates = np.array(jnp.mean(jnp.abs(ev_large), axis=(1, 2)))  # (256,)
    vx_vals = np.array(lb_large[:, 0])
    r_rate_vx = np.corrcoef(spike_rates, vx_vals)[0, 1]
    print(f"\n  🔬 Correlation: total event rate vs vx: r = {r_rate_vx:+.4f}")
    
    # Visualization
    print(f"\n  📊 Visualization...")
    plot_hallway_sample(info, events, 
                        save_path="/Users/lhooz/.openclaw/workspace/hallway_sample.png")
    
    print(f"\n  ✅ Done!")


if __name__ == "__main__":
    import time
    main()
