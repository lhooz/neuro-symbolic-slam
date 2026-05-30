#!/usr/bin/env python3
"""
Visualize ToF Fusion — Fixed Room with ToF Laser

Simplified visualization showing:
  1. Top-down scene: room, obstacles, trajectories, ToF rays
  2. Event raster: ON/OFF events over time
  3. ToF trace: Z_tof readings over time

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib.collections as mc

from sparse_forest import (
    generate_fixed_room_dataset,
    N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H, FOV_DEG,
    VX_RANGE, VY_RANGE, OMEGA_RANGE,
)

SEED = 42


def plot_tof_fusion_simple(key, n_samples=4):
    """Simple 3-panel visualization of ToF fusion."""
    # Generate one fixed room with trajectories
    data = generate_fixed_room_dataset(key, n_samples)
    events = data[0]
    labels = data[1]
    tof_dists = data[2]
    obstacles = data[3]
    positions = data[4]

    # Use first N samples for visualization
    n_vis = min(n_samples, events.shape[0])
    vis_events = events[:n_vis]
    vis_labels = labels[:n_vis]
    vis_tof = tof_dists[:n_vis]
    vis_positions = positions[:n_vis]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_vis))
    time_s = np.arange(TIME_STEPS) * DT

    # ---- Panel 1: Top-down scene with ToF ----
    ax = axes[0]
    ax.set_xlim(-0.5, ROOM_W + 0.5)
    ax.set_ylim(-0.5, ROOM_H + 0.5)
    ax.set_aspect('equal')
    ax.set_title('Fixed Room with ToF Laser Rangefinder',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # Room boundary
    room_rect = Rectangle((0, 0), ROOM_W, ROOM_H,
                       linewidth=2, edgecolor='black',
                       facecolor='#f5f5f0', alpha=0.3)
    ax.add_patch(room_rect)

    # Obstacles
    for i in range(len(obstacles)):
        obs = obstacles[i]
        w, h = obs[2] - obs[0], obs[3] - obs[1]
        ax.add_patch(Rectangle((obs[0], obs[1]), w, h,
                               facecolor='#666666', edgecolor='black',
                               linewidth=1.2, alpha=0.85))

    # Trajectories + ToF rays
    for idx in range(n_samples):
        pos = np.array(positions[idx])  # (T, 2)
        omega = labels[idx, 2] * abs(OMEGA_RANGE[1])
        # Use constant heading (simplified)
        heading = omega * time_s[-1]  # Approximate final heading

        ax.plot(pos[:, 0], pos[:, 1], '-',
                color=colors[idx], alpha=0.5, lw=1.2,
                label=f'Traj {idx+1}')

        # Start marker
        ax.add_patch(plt.Circle(pos[0], 0.12, color=colors[idx], alpha=0.8))

        # End marker
        ax.add_patch(plt.Circle(pos[-1], 0.12, color='red', alpha=0.8))

        # ToF ray at end (forward-facing)
        tof_dir = np.array([np.cos(heading), np.sin(heading)])
        tof_start = pos[-1]
        tof_dist = np.array(tof_dists[idx, -1]) * 8.0  # Denormalize
        tof_end = tof_start + tof_dir * tof_dist

        ax.arrow(tof_start[0], tof_start[1],
                  tof_dir[0], tof_dir[1],
                  width=0.015, head_width=0.25,
                  head_length=tof_dist,
                  color='red', alpha=0.7, zorder=10)

    ax.legend(loc='upper right', fontsize=8, ncol=2)

    # ---- Panel 2: Event raster ----
    ax = axes[1]
    for idx in range(n_samples):
        ev = np.array(events[idx])
        on_idx = np.where(ev > 0)
        off_idx = np.where(ev < 0)
        t_on = on_idx[0] if len(on_idx[0]) > 0 else []
        t_off = off_idx[0] if len(off_idx[0]) > 0 else []
        p_on = on_idx[1] if len(on_idx) > 1 else []
        p_off = off_idx[1] if len(off_idx) > 1 else []

        ax.scatter(t_on, p_on, c=colors[idx], s=15,
                   marker='o', alpha=0.5, label=f'Traj {idx+1} ON')
        if len(t_off) > 0:
            ax.scatter(t_off, p_off, c=colors[idx], s=15,
                       marker='o', facecolors='white', edgecolors=colors[idx],
                       alpha=0.4, label=f'Traj {idx+1} OFF')

    ax.set_xlim(0, TIME_STEPS)
    ax.set_ylim(0, N_PIXELS)
    ax.set_title('Event Raster (1D Camera, ON/OFF Polarized)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Pixel')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper right', ncol=2)

    # ---- Panel 3: ToF trace ----
    ax = axes[2]
    for idx in range(n_samples):
        tof = np.array(tof_dists[idx])
        ax.plot(time_s, tof, '-', color=colors[idx], alpha=0.7, lw=2,
                label=f'Traj {idx+1}')

        # Highlight obstacles (low ToF)
        min_tof = np.min(tof)
        if min_tof < 0.3:
            obs_times = time_s[tof < 0.3]
            for ot in obs_times:
                ax.axvline(ot, color='red', linestyle='--',
                             alpha=0.2, lw=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z_tof (normalized [0,1])')
    ax.set_ylim(0, 1.05)
    ax.set_title('ToF Laser Rangefinder Readings',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # Overall title
    fig.suptitle('ToF Fusion: Fixed Room Environment\n'
                 f'Room: {ROOM_W}m×{ROOM_H}m, '
                 f'Obstacles: {obstacles.shape[0]}, '
                 f'Samples: {n_samples}\n'
                 f'Events: 1D camera ({N_PIXELS}px, {FOV_DEG}° FOV) | '
                 f'ToF: Forward-facing laser, max range 8m',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    path = '/Users/lhooz/.openclaw/workspace/tof_fusion_viz.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f'  📸 Saved to {path}')

    # Print stats
    print(f'\n  📊 Visualization Stats:')
    print(f'    Room: {obstacles.shape[0]} obstacles')
    print(f'    Trajectories: {n_samples}')
    print(f'    Events shape: {events.shape}')
    print(f'    ToF shape: {tof_dists.shape}')
    print(f'    Positions shape: {positions.shape}')
    print(f'    Mean Z_tof: {np.mean(tof_dists):.3f} (normalized)')
    print(f'    Min Z_tof: {np.min(tof_dists):.3f}')
    print(f'    Max Z_tof: {np.max(tof_dists):.3f}')


if __name__ == '__main__':
    print('=' * 70)
    print('  🦊 ToF Fusion Visualization — Simplified')
    print('=' * 70)

    key = jax.random.PRNGKey(SEED)
    plot_tof_fusion_simple(key, n_samples=4)

    print('\n  ✅ Done!')
