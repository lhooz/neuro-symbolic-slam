#!/usr/bin/env python3
"""
Simple ToF Environment Room Plot

Top-down view with obstacles, one trajectory, ToF ray.

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

from sparse_forest import (
    generate_fixed_room_dataset,
    N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H, FOV_DEG,
)

SEED = 42


def main():
    print('=' * 60)
    print('  🦊 ToF Fusion Environment — Simple Plot')
    print('=' * 60)

    key = jax.random.PRNGKey(SEED)
    events, labels, tof_dists, obstacles, segments, positions = \
        generate_fixed_room_dataset(key, 1)

    obs = np.array(obstacles)
    pos = np.array(positions[0])  # Use first sample's positions
    omega = labels[0, 2] * 0.5  # Approximate
    vx = labels[0, 0] * 0.8
    vy = labels[0, 1] * 0.3

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Room boundary
    room_rect = Rectangle((0, 0), ROOM_W, ROOM_H,
                       linewidth=2, edgecolor='black',
                       facecolor='#f5f5f0', alpha=0.3)
    ax.add_patch(room_rect)

    # Obstacles
    for i in range(len(obs)):
        o = obs[i]
        w, h = o[2] - o[0], o[3] - o[1]
        ax.add_patch(Rectangle((o[0], o[1]), w, h,
                               facecolor='#666666', edgecolor='black',
                               linewidth=1.2, alpha=0.85))

    # Trajectory (every 10th point)
    plot_pos = pos[::10]
    ax.plot(plot_pos[:, 0], plot_pos[:, 1], '-b-', alpha=0.7, lw=1.5,
            label='Trajectory')

    # Start marker
    ax.add_patch(plt.Circle(pos[0], 0.15, color='limegreen', alpha=0.9, zorder=10))

    # End marker
    ax.add_patch(plt.Circle(pos[-1], 0.15, color='red', alpha=0.9, zorder=10))

    # ToF ray at final position
    time_s = np.arange(TIME_STEPS) * DT
    heading = omega * time_s[-1]  # Approx heading
    tof_dir = np.array([np.cos(heading), np.sin(heading)])
    tof_start = pos[-1]
    tof_dist = np.array(tof_dists[0, -1]) * 8.0  # Denormalize
    tof_end = tof_start + tof_dir * tof_dist

    # Draw ToF ray
    ax.arrow(tof_start[0], tof_start[1],
              tof_dir[0], tof_dir[1],
              width=0.01, head_width=0.3,
              head_length=tof_dist,
              color='red', alpha=0.9, zorder=15,
              label='ToF Laser')

    # FOV cone at end
    half_fov = np.radians(FOV_DEG) / 2
    cone_angle1 = heading - half_fov
    cone_angle2 = heading + half_fov
    cone1 = tof_start + np.array([np.cos(cone_angle1), np.sin(cone_angle1)]) * 2.0
    cone2 = tof_start + np.array([np.cos(cone_angle2), np.sin(cone_angle2)]) * 2.0
    ax.fill([tof_start[0], cone1[0], cone2[0]],
              [tof_start[1], cone1[1], cone2[1]],
              'orange', alpha=0.15, edgecolor='red',
              label='1D Camera FOV')

    ax.set_xlim(-0.5, ROOM_W + 0.5)
    ax.set_ylim(-0.5, ROOM_H + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Fixed Room — ToF Fusion Environment\n'
                 f'Room: {ROOM_W}m×{ROOM_H}m, Obstacles: {len(obs)}\n'
                 f'ToF Range: 8m | Events: {N_PIXELS}px, {FOV_DEG}° FOV\n'
                 f'ω={omega:+.2f}, vx={vx:+.2f}, vy={vy:+.2f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = '/Users/lhooz/.openclaw/workspace/tof_fusion_env.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  📸 Saved to {path}')
    plt.close(fig)

    print(f'\n  📊 Stats:')
    print(f'    Room obstacles: {len(obs)}')
    print(f'    Trajectory length: {TIME_STEPS} timesteps')
    print(f'    Events per sample: {TIME_STEPS} × {N_PIXELS} = {TIME_STEPS * N_PIXELS}')
    print(f'    ToF final value: {tof_dists[0, -1]:.3f} (normalized)')
    print(f'    ToF final value (denorm): {tof_dist:.2f}m')


if __name__ == '__main__':
    main()

    print('\n  ✅ Done!')
