#!/usr/bin/env python3
"""
Visualize ToF Fusion Environment

Generates:
  1. Top-down room view with trajectories + ToF rays
  2. Event camera raster (ON/OFF events)
  3. ToF distance trace

Author: Ada 🦊
"""

import sys
sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.collections import LineCollection

from src.sparse_forest import (
    generate_fixed_room_dataset,
    N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H, FOV_DEG,
)


def plot_tof_env(n_samples=4, seed=42):
    """Generate 3-panel visualization of ToF fusion environment."""

    key = jax.random.PRNGKey(seed)
    events, labels, tof_dists, positions, obstacles, segments, intensities = \
        generate_fixed_room_dataset(key, n_samples)

    # Convert to numpy immediately to avoid jax/matplotlib conflicts
    events = np.array(events)
    labels = np.array(labels)
    tof_dists = np.array(tof_dists)
    positions = np.array(positions)
    obstacles = np.array(obstacles)
    time_s = np.arange(TIME_STEPS) * DT

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))

    # ===============================================================
    # Panel 1: Top-down room view
    # ===============================================================
    ax = axes[0]
    ax.set_xlim(-0.3, ROOM_W + 0.3)
    ax.set_ylim(-0.3, ROOM_H + 0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)
    ax.set_title('Environment: Sparse Forest + ToF Laser', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25, linestyle='--')

    # Room boundary
    room_rect = Rectangle((0, 0), ROOM_W, ROOM_H,
                           linewidth=2.5, edgecolor='#333',
                           facecolor='#f8f8f5', alpha=0.5)
    ax.add_patch(room_rect)

    # Obstacles
    for i in range(len(obstacles)):
        o = obstacles[i]
        w, h = float(o[2] - o[0]), float(o[3] - o[1])
        rect = Rectangle((float(o[0]), float(o[1])), w, h,
                         facecolor='#5a5a5a', edgecolor='#222',
                         linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)

    # Trajectories + ToF rays
    for idx in range(n_samples):
        pos_xy = positions[idx]  # (T, 2)
        omega = float(labels[idx, 2]) * 0.5  # approximate heading
        vx, vy = float(labels[idx, 0]), float(labels[idx, 1])

        # Trajectory line (subsample for clarity)
        step = max(1, TIME_STEPS // 20)
        ax.plot(pos_xy[::step, 0], pos_xy[::step, 1], '-',
                color=colors[idx], alpha=0.65, lw=1.8,
                label=f'Traj {idx+1}', zorder=5)

        # Start marker (green)
        start = Circle((float(pos_xy[0, 0]), float(pos_xy[0, 1])),
                       0.12, color='#2ecc71', zorder=10)
        ax.add_patch(start)

        # End marker (red)
        end = Circle((float(pos_xy[-1, 0]), float(pos_xy[-1, 1])),
                    0.12, color='#e74c3c', zorder=10)
        ax.add_patch(end)

        # ToF ray at end position (single narrow beam, NOT a cone)
        heading = omega * time_s[-1]
        tof_dir = np.array([np.cos(heading), np.sin(heading)])
        tof_start = pos_xy[-1]
        tof_dist = float(tof_dists[idx, -1]) * 8.0  # denormalize (max 8m)

        # ToF laser is a single-point sensor — narrow beam (~2° divergence)
        # Draw as a thin line with a small circle at the end
        tof_end = tof_start + tof_dir * tof_dist
        ax.plot([float(tof_start[0]), float(tof_end[0])],
                [float(tof_start[1]), float(tof_end[1])],
                '-', color='#e74c3c', lw=2.5, alpha=0.85, zorder=12)

        # Small circle at ToF hit point
        hit_circle = Circle((float(tof_end[0]), float(tof_end[1])),
                          0.08, color='#e74c3c', zorder=15)
        ax.add_patch(hit_circle)

    # Event camera FOV annotation (side note)
    ax.text(0.02, 0.02,
            f'Camera FOV: {FOV_DEG}° (1D, {N_PIXELS}px)\nToF: Single forward ray, narrow beam (~2°)',
            transform=ax.transAxes, fontsize=7, va='bottom',
            color='#555', style='italic')

    # Legend
    handles = [plt.Line2D([0], [0], color=colors[i], lw=1.8,
                          label=f'Traj {i+1}') for i in range(n_samples)]
    handles.append(plt.Line2D([0], [0], color='#e74c3c', lw=2.5,
                               label='ToF ray (forward)'))
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)

    # ===============================================================
    # Panel 2: Event Camera Raster
    # ===============================================================
    ax = axes[1]
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Pixel (1D, 64px, 180° FOV)', fontsize=10)
    ax.set_title('Event Camera Output (ON=dot, OFF=circle)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, TIME_STEPS)
    ax.set_ylim(0, N_PIXELS)
    ax.grid(alpha=0.2, linestyle=':')

    half_fov = np.radians(FOV_DEG) / 2
    pixels = np.arange(N_PIXELS)
    angles = -half_fov + (2 * half_fov) * pixels / (N_PIXELS - 1)

    for idx in range(n_samples):
        ev_sample = events[idx]  # (T, 64)
        on_mask = ev_sample > 0
        off_mask = ev_sample < 0

        t_on, p_on = np.where(on_mask)
        t_off, p_off = np.where(off_mask)

        ax.scatter(t_on, p_on, c=[colors[idx]], s=12,
                   marker='o', alpha=0.5, label=f'Traj {idx+1}')
        ax.scatter(t_off, p_off, c=[colors[idx]], s=12,
                   marker='o', facecolors='none', edgecolors=colors[idx],
                   alpha=0.35, linewidths=1)

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # ===============================================================
    # Panel 3: ToF Distance Trace
    # ===============================================================
    ax = axes[2]
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Distance (m)', fontsize=10)
    ax.set_title('ToF Laser Rangefinder (Forward-Facing)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, time_s[-1])
    ax.set_ylim(0, 8.5)
    ax.grid(alpha=0.25, linestyle='--')

    for idx in range(n_samples):
        tof_m = tof_dists[idx] * 8.0  # denormalize to meters
        ax.plot(time_s, tof_m, '-', color=colors[idx], lw=2,
                alpha=0.8, label=f'Traj {idx+1}')

        # Mark final reading
        ax.scatter([time_s[-1]], [tof_m[-1]], color=colors[idx],
                  s=60, zorder=10, edgecolors='white', linewidths=1.5)

        # Shade "danger zone" (close obstacles)
        ax.fill_between(time_s, 0, 1.5, color='#e74c3c', alpha=0.08)
        ax.text(time_s[-1] - 0.8, 0.75, 'Danger\n<1.5m', fontsize=7,
               color='#e74c3c', alpha=0.7, va='center')

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # ===============================================================
    # Overall title
    # ===============================================================
    n_obs = len(obstacles)
    fig.suptitle(
        f'ToF Fusion Environment — Sparse Forest\n'
        f'Room: {ROOM_W}m × {ROOM_H}m | Obstacles: {n_obs} | '
        f'Samples: {n_samples} | '
        f'Camera: {N_PIXELS}px, {FOV_DEG}° FOV | ToF: 8m max range',
        fontsize=12, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    out_path = '/Users/lhooz/.openclaw/workspace/results/tof_env_visualization.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out_path}')

    # Print stats
    print(f'\n📊 Environment Stats:')
    print(f'  Room: {ROOM_W}m × {ROOM_H}m')
    print(f'  Obstacles: {n_obs}')
    print(f'  Samples: {n_samples}')
    print(f'  Trajectory length: {TIME_STEPS} steps = {time_s[-1]:.1f}s')
    print(f'  Events shape: {events.shape}')
    print(f'  ToF range: {tof_dists.min():.3f} – {tof_dists.max():.3f} (normalized)')
    print(f'  ToF range (m): {tof_dists.min()*8:.2f} – {tof_dists.max()*8:.2f}m')

    plt.close(fig)
    return out_path


if __name__ == '__main__':
    print('=' * 60)
    print('  🦊 ToF Fusion Environment Visualization')
    print('=' * 60)
    plot_tof_env(n_samples=4, seed=42)
    print('\n  ✅ Done!')
