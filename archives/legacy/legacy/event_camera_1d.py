#!/usr/bin/env python3
"""
Synthetic 1D Event Camera Simulator for MAV State Estimation

Simulates a vertical 1D line sensor (64 pixels) mounted on a micro-aerial vehicle
observing a wall with sharp horizontal stripes. The drone's dynamic state affects
what the camera sees, producing spatially distinct event patterns for each state
component.

State vector: [vx, vy, pitch, pitch_rate]

Physics:
  A vertical 1D sensor at distance D from a wall with bar-code-like horizontal
  stripes. Each pixel i observes wall height:
      h(t, i) = y(t) + D(t) · tan(α_i + θ(t))
  where α_i is the pixel's angle from the optical axis and θ is pitch.

  Event patterns by state component:
    vy (vertical velocity):  Uniform edge shift across all pixels — simultaneous
                             events at stripe boundaries everywhere.
    vx (forward velocity):   Changes apparent stripe width via D → edges drift
                             at ~6× slower rate than vy, uniform spatially.
    pitch (attitude angle):  Perspective distortion — edges at sensor periphery
                             shift faster than center. Spatial gradient pattern.
    pitch_rate:              Continuous pitch change → sustained gradient events.

  The SNN learns to decode these spatial/temporal patterns into state estimates.

Output:
  - events: (B, T, N) dense array, values in {-1, 0, +1}
  - labels: (B, 4) normalized state [vx, vy, pitch₀, pitch_rate]

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import sparse as jsparse
import numpy as np
import time
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_PIXELS = 64
TIME_STEPS = 100           # 2.0 seconds at 50 Hz
DT = 0.02                  # timestep (seconds) — 50 Hz event sampling
BATCH_SIZE = 8
THRESHOLD = 0.12           # contrast threshold C

# Physics
WALL_DISTANCE_INIT = 3.0   # meters to wall at t=0
FOV_TOTAL_DEG = 60.0       # total vertical field of view
PIXEL_FOV = np.radians(FOV_TOTAL_DEG) / N_PIXELS  # rad per pixel
STRIPE_PERIOD = 0.3        # meters (wall stripe period — uniform everywhere)

# State parameter ranges
VX_RANGE = (-1.0, 1.0)             # m/s  (forward)
VY_RANGE = (-1.0, 1.0)             # m/s  (vertical)
PITCH_RANGE = (-0.17, 0.17)        # rad  (±10° initial pitch)
PITCH_RATE_RANGE = (-0.17, 0.17)   # rad/s (±10°/s)
PITCH_CLAMP = 0.52                  # rad  (±30° physical limit)
Y_OFFSET_RANGE = (-0.2, 0.2)       # m    (initial vertical offset)

SEED = 42


# ---------------------------------------------------------------------------
# Wall Texture — physically correct uniform stripes
# ---------------------------------------------------------------------------
def wall_intensity(h, x_wall):
    """
    Wall texture at vertical position h on the wall.

    A real wall: uniform horizontal stripes, same period everywhere.
    Like a barcode sticker. No depth dependence — the stripe pattern
    is painted on the wall, it doesn't change because the drone moves.

    The x_wall argument is kept for API compatibility but is ignored,
    because the wall's paint doesn't care where the drone is.

    h: vertical position on wall (meters), any shape (...)
    Returns: intensity in [0, 1]
    """
    # Sharp square-wave stripes: tanh(sin(...)) ≈ alternating 0 and 1
    stripe = jnp.tanh(20.0 * jnp.sin(2 * jnp.pi * h / STRIPE_PERIOD))
    # Map from [-1, 1] → [0.1, 0.9] (never fully black/white, realistic contrast)
    intensity = 0.4 * stripe + 0.5
    return jnp.clip(intensity, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Single-sample generation (vmappable)
# ---------------------------------------------------------------------------
def generate_sample(key, time_steps=TIME_STEPS, dt=DT):
    """
    Generate events + labels for one sample.

    Returns:
      events:      (time_steps, n_pixels) values in {-1, 0, +1}
      labels:      (4,) normalized state [vx, vy, pitch₀, pitch_rate]
      trajectory:  dict with per-timestep arrays for visualization
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # --- Sample state parameters ---
    vx = jax.random.uniform(k1, (), minval=VX_RANGE[0], maxval=VX_RANGE[1])
    vy = jax.random.uniform(k2, (), minval=VY_RANGE[0], maxval=VY_RANGE[1])
    pitch0 = jax.random.uniform(k3, (), minval=PITCH_RANGE[0], maxval=PITCH_RANGE[1])
    pitch_rate = jax.random.uniform(k4, (), minval=PITCH_RATE_RANGE[0], maxval=PITCH_RATE_RANGE[1])
    y0 = jax.random.uniform(k5, (), minval=Y_OFFSET_RANGE[0], maxval=Y_OFFSET_RANGE[1])

    # --- State trajectory (linear integration, pitch clamped) ---
    t = jnp.arange(time_steps, dtype=jnp.float32) * dt

    x_pos = vx * t                                              # (T,)
    y_pos = y0 + vy * t                                         # (T,)
    pitch = jnp.clip(pitch0 + pitch_rate * t, -PITCH_CLAMP, PITCH_CLAMP)  # (T,)
    wall_dist = jnp.maximum(WALL_DISTANCE_INIT - x_pos, 0.8)    # (T,)

    # --- Pixel → wall mapping ---
    pixel_indices = jnp.arange(N_PIXELS, dtype=jnp.float32)
    alpha = (pixel_indices - N_PIXELS / 2.0) * PIXEL_FOV        # (64,)

    # h(t, i) = y(t) + D(t) * tan(α_i + θ(t))
    h = y_pos[:, None] + wall_dist[:, None] * jnp.tan(alpha[None, :] + pitch[:, None])
    x_wall = jnp.broadcast_to(x_pos[:, None], (time_steps, N_PIXELS))

    # --- Intensity readings ---
    readings = wall_intensity(h, x_wall)                        # (T, 64)

    # --- Event generation (temporal contrast) ---
    prev_readings = jnp.concatenate([readings[:1], readings[:-1]], axis=0)
    delta = readings - prev_readings
    events = jnp.where(delta > THRESHOLD, 1.0,
              jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)

    # --- Normalized labels ---
    labels = jnp.array([
        vx / abs(VX_RANGE[1]),
        vy / abs(VY_RANGE[1]),
        pitch0 / abs(PITCH_RANGE[1]),
        pitch_rate / abs(PITCH_RATE_RANGE[1]),
    ])

    trajectory = {
        'x_pos': x_pos,
        'y_pos': y_pos,
        'pitch': pitch,
        'wall_dist': wall_dist,
        'readings': readings,
        'vx': vx,
        'vy': vy,
        'pitch_rate': pitch_rate,
    }

    return events, labels, trajectory


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------
def generate_batch(key, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, dt=DT):
    """
    Generate a batch of labeled event samples.

    Returns:
      events:        (B, T, N_PIXELS) dense, values in {-1, 0, +1}
      labels:        (B, 4) normalized state vectors
      trajectories:  list[B] dicts with per-timestep state arrays
    """
    keys = jax.random.split(key, batch_size)

    events, labels, trajectories = jax.vmap(
        generate_sample, in_axes=(0, None, None)
    )(keys, time_steps, dt)

    traj_list = [
        {k: v[i] for k, v in trajectories.items()}
        for i in range(batch_size)
    ]

    return events, labels, traj_list


def generate_batch_sparse(key, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, dt=DT):
    """Same as generate_batch but events returned as BCOO sparse."""
    events, labels, trajectories = generate_batch(key, batch_size, time_steps, dt)
    return jsparse.BCOO.fromdense(events), labels, trajectories


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_sample(events, trajectory, save_path=None, title=""):
    """Visualize one sample: intensity image, event raster, and pitch trace."""
    readings = trajectory['readings']
    pitch = np.array(trajectory['pitch'])
    vx, vy, pr = trajectory['vx'], trajectory['vy'], trajectory['pitch_rate']
    time_s = np.arange(events.shape[0]) * DT

    fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [1, 1.5, 0.8]})

    # 1. Pixel intensity image
    axes[0].imshow(np.array(readings).T, aspect='auto', cmap='gray',
                   extent=[0, time_s[-1], N_PIXELS, 0])
    axes[0].set_ylabel('Pixel')
    axes[0].set_title(
        f'{title} | vx={vx:.2f} m/s  vy={vy:.2f} m/s  '
        f'pitch₀={pitch[0]*180/np.pi:.1f}°  ω={pr*180/np.pi:.1f}°/s')

    # 2. Event raster
    ev = np.array(events)
    n_events = np.sum(np.abs(ev))
    t_idx, p_idx = np.where(ev != 0)
    if len(t_idx) > 0:
        vals = ev[t_idx, p_idx]
        axes[1].scatter(t_idx[vals > 0], p_idx[vals > 0],
                        c='tab:red', s=0.8, alpha=0.6, label='ON (+1)')
        axes[1].scatter(t_idx[vals < 0], p_idx[vals < 0],
                        c='tab:blue', s=0.8, alpha=0.6, label='OFF (−1)')
    axes[1].set_ylabel('Pixel')
    axes[1].set_xlim(0, events.shape[0])
    sparsity = 1 - n_events / np.prod(events.shape)
    axes[1].legend(markerscale=8, loc='upper right',
                   title=f'{n_events} events ({(1-sparsity)*100:.1f}%)')

    # 3. Pitch trajectory
    axes[2].plot(time_s, pitch * 180 / np.pi, 'g-', linewidth=1.5, label='Pitch (°)')
    axes[2].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angle (°)')
    axes[2].legend(loc='upper left')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  📸 Saved to {save_path}")
    plt.close(fig)


def plot_state_comparison(events_list, labels_list, save_path=None, n_samples=4):
    """
    Compare event patterns for samples with different dominant state components.
    Shows that each state variable creates a spatially distinct event signature.
    """
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 2.5 * n_samples))

    for i, (ev, lb) in enumerate(zip(events_list[:n_samples], labels_list[:n_samples])):
        ev_np = np.array(ev)
        t_idx, p_idx = np.where(ev_np != 0)
        if len(t_idx) > 0:
            vals = ev_np[t_idx, p_idx]
            axes[i].scatter(t_idx[vals > 0], p_idx[vals > 0],
                           c='tab:red', s=0.5, alpha=0.5)
            axes[i].scatter(t_idx[vals < 0], p_idx[vals < 0],
                           c='tab:blue', s=0.5, alpha=0.5)
        vx_n, vy_n, pitch_n, pr_n = lb
        axes[i].set_xlim(0, ev.shape[0])
        axes[i].set_ylabel('Pixel')
        sparsity = 1 - np.sum(np.abs(ev_np)) / np.prod(ev_np.shape)
        axes[i].set_title(
            f'vx={vx_n:+.2f}  vy={vy_n:+.2f}  pitch={pitch_n:+.2f}  ω={pr_n:+.2f}  '
            f'|  sparsity={sparsity:.1%}', fontsize=9)
        if i == n_samples - 1:
            axes[i].set_xlabel('Time step')

    fig.suptitle('Event patterns by state — note spatial structure differences',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📸 Saved comparison to {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  🦊 1D Event Camera — MAV State Estimation Simulator")
    print("=" * 60)
    print(f"  Pixels:       {N_PIXELS}")
    print(f"  Time steps:   {TIME_STEPS} ({TIME_STEPS * DT:.1f}s at {1/DT:.0f} Hz)")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Threshold C:  {THRESHOLD}")
    print(f"  Wall dist:    {WALL_DISTANCE_INIT}m, stripe period: {STRIPE_PERIOD}m")
    print(f"  Sensor FOV:   ±{FOV_TOTAL_DEG/2:.0f}°")
    print("-" * 60)
    print(f"  State vector: [vx, vy, pitch, pitch_rate]")
    print(f"  vx range:     {VX_RANGE} m/s")
    print(f"  vy range:     {VY_RANGE} m/s")
    print(f"  pitch₀ range: [{PITCH_RANGE[0]*180/np.pi:.0f}°, {PITCH_RANGE[1]*180/np.pi:.0f}°]")
    print(f"  ω range:      [{PITCH_RATE_RANGE[0]*180/np.pi:.0f}°/s, "
          f"{PITCH_RATE_RANGE[1]*180/np.pi:.0f}°/s]")
    print(f"  pitch clamp:  ±{PITCH_CLAMP*180/np.pi:.0f}°")
    print("=" * 60)

    key = jax.random.PRNGKey(SEED)

    # Generate batch
    print("\n  ⚡ Generating events...")
    events, labels, trajectories = generate_batch(key, BATCH_SIZE)

    n_events = jnp.sum(jnp.abs(events))
    total = np.prod(events.shape)
    sparsity = 1 - n_events / total
    print(f"  Shape:        {events.shape}")
    print(f"  Total events: {n_events.item()} / {total} ({n_events/total:.2%})")
    print(f"  Sparsity:     {sparsity:.2%}")
    print(f"  Labels shape: {labels.shape}")

    # Per-sample stats
    print(f"\n  Sample breakdown:")
    for i in range(min(4, BATCH_SIZE)):
        l = labels[i]
        n_ev = jnp.sum(jnp.abs(events[i])).item()
        on = jnp.sum(events[i] == 1.0).item()
        off = jnp.sum(events[i] == -1.0).item()
        print(f"    [{i}] vx={l[0]:+.2f} vy={l[1]:+.2f} "
              f"pitch={l[2]:+.2f} ω={l[3]:+.2f} | "
              f"{n_ev} events ({on} ON, {off} OFF) "
              f"sparsity={1-n_ev/(N_PIXELS*TIME_STEPS):.1%}")

    # Visualize first sample
    print(f"\n  📊 Plotting sample 0...")
    plot_sample(events[0], trajectories[0],
                save_path="/Users/lhooz/.openclaw/workspace/event_sample_mav.png",
                title="MAV Event Camera — Sample 0")

    # State comparison plot
    print(f"\n  📊 State comparison plot...")
    plot_state_comparison(events, labels,
                         save_path="/Users/lhooz/.openclaw/workspace/event_state_comparison.png")

    # Wall texture preview
    h_range = np.linspace(-0.5, 0.5, 200)
    x_range = np.linspace(0, 2, 300)
    H, X = np.meshgrid(h_range, x_range)
    texture = np.array(wall_intensity(jnp.array(H), jnp.array(X)))

    fig2, ax2 = plt.subplots(figsize=(14, 2))
    ax2.imshow(texture, aspect='auto', cmap='gray',
              extent=[h_range[0], h_range[-1], x_range[-1], x_range[0]])
    ax2.set_xlabel('Vertical position on wall (m)')
    ax2.set_ylabel('Wall depth (m)')
    ax2.set_title('Wall Texture — uniform horizontal stripes (same everywhere)')
    fig2.tight_layout()
    fig2.savefig("/Users/lhooz/.openclaw/workspace/wall_texture_mav.png", dpi=150)
    print(f"  📸 Saved wall texture to wall_texture_mav.png")
    plt.close(fig2)

    # Jit benchmark
    print(f"\n  ⚡ JIT benchmark...")
    k = random.PRNGKey(99)
    t0 = time.time()
    _ = generate_batch(k, 32, TIME_STEPS, DT)
    t1 = time.time()
    print(f"  Batch gen (B=32): {t1 - t0:.3f}s (first call, includes compile)")
    t0 = time.time()
    _ = generate_batch(k, 32, TIME_STEPS, DT)
    t1 = time.time()
    print(f"  Batch gen (B=32): {t1 - t0:.3f}s (compiled)")

    print("\n  ✅ Done!")


if __name__ == "__main__":
    main()
