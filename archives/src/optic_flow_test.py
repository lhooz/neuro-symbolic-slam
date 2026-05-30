#!/usr/bin/env python3
"""Optic Flow Verification Test

Generate a straight-line trajectory down a corridor and check
that the event raster shows the expected expansion pattern.

Expected: a continuous 'V' shaped pattern expanding from the 
focus of expansion (FOE) at the heading pixel.
"""

import jax
import jax.numpy as jnp
from event_camera_2d_nav import (
    ROOM_W, ROOM_H, N_PIXELS, FOV_DEG, DT, TIME_STEPS,
    THRESHOLD, N_OBSTACLES, compute_pixel_readings,
    obstacles_to_segments, generate_obstacles,
    generate_trajectory,
)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "/Users/lhooz/.openclaw/workspace/optic_flow_test.png"


def main():
    print("=" * 60)
    print("  🔬 Optic Flow Verification Test")
    print("=" * 60)

    key = jax.random.PRNGKey(42)

    # Generate a scene with obstacles
    k_obs, k_traj = jax.random.split(key, 2)
    obstacles = generate_obstacles(k_obs)
    segments = obstacles_to_segments(obstacles)

    # Build a STRAIGHT FORWARD trajectory
    # Start in the middle, heading along x-axis
    start_pos = jnp.array([2.0, 5.0])
    start_hdg = 0.0  # heading along +x
    vx, vy, omega = 0.6, 0.0, 0.0  # pure forward, no rotation

    # Manually build trajectory
    t_arr = jnp.arange(TIME_STEPS, dtype=jnp.float32) * DT
    positions = jnp.stack([start_pos[0] + vx * t_arr,
                           start_pos[1] + vy * t_arr], axis=-1)
    headings = jnp.full(TIME_STEPS, start_hdg)

    # Clip to room bounds
    positions = jnp.clip(positions,
                         jnp.array([0.5, 0.5]),
                         jnp.array([ROOM_W - 0.5, ROOM_H - 0.5]))

    print(f"  Trajectory: pure forward vx={vx}, vy={vy}, ω={omega}")
    print(f"  Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f}), heading={start_hdg:.1f}rad")
    print(f"  End:   ({positions[-1, 0]:.1f}, {positions[-1, 1]:.1f})")

    # Cast rays at every timestep
    readings = jax.vmap(lambda p, h: compute_pixel_readings(p, h, segments))(
        positions, headings)
    intensities = readings[0]  # (T, N)
    distances = readings[1]    # (T, N)

    # Temporal contrast → events
    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0,
              jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)

    intens_arr = np.array(intensities)
    ev_arr = np.array(events)
    n_events = int(np.sum(np.abs(ev_arr)))
    event_rate = 100 * n_events / (TIME_STEPS * N_PIXELS)
    
    print(f"  Total events: {n_events} / {TIME_STEPS * N_PIXELS} ({event_rate:.1f}%)")

    # Check for expansion pattern
    # For pure forward motion, events should concentrate at edges and show
    # expansion from the center (heading pixel)
    on_events = np.where(ev_arr > 0)
    off_events = np.where(ev_arr < 0)
    
    print(f"  ON events: {len(on_events[0])}, OFF events: {len(off_events[0])}")

    # Per-timestep event count to check temporal distribution
    ev_per_step = np.sum(np.abs(ev_arr), axis=1)
    print(f"  Events/timestep: min={ev_per_step.min()}, max={ev_per_step.max()}, "
          f"mean={ev_per_step.mean():.1f}")

    # Check pixel event distribution — should see more events at edges
    ev_per_pixel = np.sum(np.abs(ev_arr), axis=0)
    edge_events = np.sum(ev_per_pixel[:8]) + np.sum(ev_per_pixel[-8:])
    center_events = np.sum(ev_per_pixel[24:40])
    print(f"  Edge pixel events (0-7, 56-63): {edge_events}")
    print(f"  Center pixel events (24-39): {center_events}")
    print(f"  Edge/Center ratio: {edge_events/(center_events+1):.2f}")

    # Check for temporal expansion: do outer pixels get events LATER?
    # Compare first half vs second half timestep for edge vs center pixels
    first_half_ev = np.sum(np.abs(ev_arr[:50, :8]) + np.abs(ev_arr[:50, -8:]))
    second_half_ev = np.sum(np.abs(ev_arr[50:, :8]) + np.abs(ev_arr[50:, -8:]))
    print(f"  Edge events first half (t<1s): {first_half_ev}")
    print(f"  Edge events second half (t≥1s): {second_half_ev}")

    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [2, 1.2, 1.0, 0.8]})
    time_s = np.arange(TIME_STEPS) * DT

    # 1. Scene + trajectory
    ax1 = axes[0]
    ax1.set_xlim(-0.5, ROOM_W + 0.5)
    ax1.set_ylim(-0.5, ROOM_H + 0.5)
    ax1.set_aspect('equal')
    ax1.set_title(f'Optic Flow Test: Pure Forward vx={vx} m/s', fontsize=12, fontweight='bold')

    from matplotlib.patches import Rectangle
    room_rect = Rectangle((0, 0), ROOM_W, ROOM_H, linewidth=2,
                          edgecolor='black', facecolor='#f0f0f0')
    ax1.add_patch(room_rect)
    obs_np = np.array(obstacles)
    for o in obs_np:
        w, h = o[2] - o[0], o[3] - o[1]
        ax1.add_patch(Rectangle((o[0], o[1]), w, h,
                                facecolor='#555555', edgecolor='black', lw=1.2, alpha=0.85))

    pos_np = np.array(positions)
    ax1.plot(pos_np[:, 0], pos_np[:, 1], '-', color='steelblue', lw=2, label='Trajectory')
    ax1.plot(pos_np[0, 0], pos_np[0, 1], 'o', color='limegreen', ms=10, label='Start')
    ax1.plot(pos_np[-1, 0], pos_np[-1, 1], 's', color='red', ms=10, label='End')

    # Draw FOV at start and end
    fov_rad = np.radians(FOV_DEG)
    for step, label in [(0, 'Start'), (TIME_STEPS-1, 'End')]:
        px, py = pos_np[step]
        h = float(np.array(headings[step]))
        for sign in (-1, 1):
            angle = h + sign * fov_rad / 2
            ax1.plot([px, px + 2.5 * np.cos(angle)], [py, py + 2.5 * np.sin(angle)],
                     '--', color='green' if step == 0 else 'red', alpha=0.5)
    ax1.legend(loc='upper right')

    # 2. Event raster
    ax2 = axes[1]
    if len(on_events[0]) > 0:
        ax2.scatter(on_events[0], on_events[1], c='tab:red', s=0.8, alpha=0.5, label='ON (+1)')
    if len(off_events[0]) > 0:
        ax2.scatter(off_events[0], off_events[1], c='tab:blue', s=0.8, alpha=0.5, label='OFF (−1)')
    ax2.axhline(N_PIXELS // 2, color='green', ls='--', lw=0.8, alpha=0.5, label='Heading pixel')
    ax2.set_xlim(0, TIME_STEPS)
    ax2.set_ylabel('Pixel')
    ax2.set_title(f'Event Raster  |  {n_events} events ({event_rate:.1f}% active)', fontsize=10)
    ax2.legend(markerscale=8, loc='upper right', fontsize=7)

    # 3. Intensity image
    ax3 = axes[2]
    ax3.imshow(intens_arr.T, aspect='auto', cmap='gray',
               extent=[0, time_s[-1], N_PIXELS, 0], vmin=0, vmax=1)
    ax3.set_ylabel('Pixel')
    ax3.set_title('Pixel Intensity (2D texture × distance dimming)', fontsize=10)

    # 4. Event rate per pixel + temporal profile
    ax4 = axes[3]
    ax4.bar(range(N_PIXELS), ev_per_pixel, color='steelblue', alpha=0.7, width=1.0)
    ax4.axhline(np.mean(ev_per_pixel), color='red', ls='--', lw=0.8, label=f'Mean={np.mean(ev_per_pixel):.0f}')
    ax4.set_xlabel('Pixel index')
    ax4.set_ylabel('Total events')
    ax4.set_title(f'Event Distribution per Pixel (edge/center ratio: {edge_events/(center_events+1):.2f})', fontsize=10)
    ax4.legend()

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    print(f"\n  📸 Saved to {OUT}")

    # Verdict
    print("\n  📋 Verdict:")
    if event_rate < 5:
        print("  ❌ NEAR-BLIND — event rate too low (<5%). Walls are featureless.")
    elif event_rate < 15:
        print("  ⚠️  MARGINAL — some events but may not carry enough flow signal.")
    elif edge_events / (center_events + 1) < 1.0:
        print("  ⚠️  SUSPICIOUS — edge pixels should have MORE events than center during expansion.")
    else:
        print("  ✅ HEALTHY — good event rate with expansion pattern.")

    if second_half_ev > first_half_ev * 1.2:
        print("  ✅ EXPANSION DETECTED — edge events increase over time (looming).")
    else:
        print("  ⚠️  NO CLEAR EXPANSION — temporal distribution is flat.")


if __name__ == "__main__":
    main()
