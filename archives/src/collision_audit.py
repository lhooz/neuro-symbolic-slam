#!/usr/bin/env python3
"""
Collision Audit — Old vs New Random Room Generator

Tests the OLD trajectory (pre-fix) for collision issues, and verifies
the NEW safe trajectory is clean.

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys
sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')
from event_camera_2d_nav import (
    generate_obstacles, generate_trajectory_safe, generate_sample,
    _generate_trajectory_inner, _is_clear, _trajectory_clear,
    ROOM_W, ROOM_H, ROBOT_MARGIN, N_OBSTACLES,
    TIME_STEPS, DT, N_PIXELS, THRESHOLD,
    ROBOT_MARGIN as MARGIN,
)

N_SAMPLES = 500
SAFE_RADIUS = 0.5
SEED = 42


def _generate_trajectory_old(key, time_steps=TIME_STEPS, dt=DT):
    """Replicate the OLD trajectory generator (with clipping, no obstacle check)."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    x0 = jax.random.uniform(k1, (), minval=MARGIN + 1, maxval=ROOM_W - MARGIN - 1)
    y0 = jax.random.uniform(k2, (), minval=MARGIN + 1, maxval=ROOM_H - MARGIN - 1)
    h0 = jax.random.uniform(k3, (), minval=0.0, maxval=2 * jnp.pi)
    vx = jax.random.uniform(k4, (), minval=-0.8, maxval=0.8)
    vy = jax.random.uniform(k5, (), minval=-0.3, maxval=0.3)
    omega = jax.random.uniform(jax.random.split(key, 6)[0], (), minval=-0.5, maxval=0.5)

    t = jnp.arange(time_steps, dtype=jnp.float32) * dt
    headings = (h0 + omega * t) % (2 * jnp.pi)
    cos_h, sin_h = jnp.cos(headings), jnp.sin(headings)
    wx = vx * cos_h - vy * sin_h
    wy = vx * sin_h + vy * cos_h

    dx = jnp.concatenate([jnp.zeros(1), jnp.cumsum(wx[:-1] * dt)])
    dy = jnp.concatenate([jnp.zeros(1), jnp.cumsum(wy[:-1] * dt)])
    raw_positions = jnp.stack([x0 + dx, y0 + dy], axis=-1)
    clipped = jnp.clip(raw_positions,
                       jnp.array([MARGIN, MARGIN]),
                       jnp.array([ROOM_W - MARGIN, ROOM_H - MARGIN]))
    return raw_positions, clipped, headings, float(vx), float(vy), float(omega), float(x0), float(y0)


def audit():
    print("=" * 60)
    print("  🔍 Collision Audit — Random Room Generator")
    print("=" * 60)
    print(f"  Samples:       {N_SAMPLES}")
    print(f"  Safe radius:   {SAFE_RADIUS}m")
    print(f"  Room:          {ROOM_W}×{ROOM_H}m")
    print(f"  Obstacles:     {N_OBSTACLES} random rects")
    print(f"  Robot margin:  {MARGIN}m")
    print("=" * 60)

    key = jax.random.PRNGKey(SEED)

    # === OLD GENERATOR STATS ===
    old_spawn_col = 0
    old_traj_pen = 0
    old_clipped = 0
    old_min_clears = []
    old_accepted_count = 0

    # === NEW GENERATOR STATS ===
    new_spawn_col = 0
    new_traj_pen = 0
    new_rejected = 0
    new_min_clears = []

    print(f"\n  Auditing {N_SAMPLES} samples (old vs new)...")

    for i in range(N_SAMPLES):
        key, k1, k2, k3 = jax.random.split(key, 4)
        obstacles = generate_obstacles(k1)

        # --- OLD generator ---
        raw_pos, clipped_pos, headings, vx, vy, omega, sx, sy = \
            _generate_trajectory_old(k2)

        # Spawn check
        spawn_ok = bool(jnp.all(_is_clear(
            jnp.float32(sx), jnp.float32(sy), obstacles, margin=SAFE_RADIUS)))
        if not spawn_ok:
            old_spawn_col += 1

        # Trajectory check (on RAW positions, before clipping)
        traj_ok = bool(jnp.all(_trajectory_clear(raw_pos, obstacles, margin=SAFE_RADIUS)))
        if not traj_ok:
            old_traj_pen += 1

        # Clipping check
        if not jnp.allclose(raw_pos, clipped_pos, atol=1e-6):
            old_clipped += 1

        # Min clearance (raw trajectory)
        dists = jax.vmap(lambda p: _min_clearance(p, obstacles))(raw_pos)
        old_min_clears.append(float(jnp.min(dists)))

        # --- NEW generator (safe) ---
        positions, headings2, vx2, vy2, omega2, accepted = \
            generate_trajectory_safe(k3, obstacles, TIME_STEPS, DT)

        if not bool(accepted):
            new_rejected += 1

        # Verify new is clean
        new_spawn = bool(jnp.all(_is_clear(positions[0, 0], positions[0, 1],
                                   obstacles, margin=SAFE_RADIUS)))
        if not new_spawn:
            new_spawn_col += 1
        new_traj = bool(jnp.all(_trajectory_clear(positions, obstacles, margin=SAFE_RADIUS)))
        if not new_traj:
            new_traj_pen += 1

        new_dists = jax.vmap(lambda p: _min_clearance(p, obstacles))(positions)
        new_min_clears.append(float(jnp.min(new_dists)))

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{N_SAMPLES}] OLD: spawn_col={old_spawn_col} "
                  f"traj_pen={old_traj_pen} clipped={old_clipped} | "
                  f"NEW: rejected={new_rejected} spawn_col={new_spawn_col} "
                  f"traj_pen={new_traj_pen}")

    # Results
    print(f"\n  {'='*60}")
    print(f"  📊 AUDIT RESULTS")
    print(f"  {'='*60}")

    print(f"\n  🔴 OLD GENERATOR (no collision checking):")
    print(f"     Spawn inside obstacle: {old_spawn_col}/{N_SAMPLES} "
          f"({100*old_spawn_col/N_SAMPLES:.1f}%)")
    print(f"     Trajectory penetration: {old_traj_pen}/{N_SAMPLES} "
          f"({100*old_traj_pen/N_SAMPLES:.1f}%)")
    print(f"     Position clipped (wall hit): {old_clipped}/{N_SAMPLES} "
          f"({100*old_clipped/N_SAMPLES:.1f}%)")
    old_mc = np.array(old_min_clears)
    print(f"     Mean min clearance: {old_mc.mean():.3f}m")
    print(f"     Worst clearance:    {old_mc.min():.3f}m")
    print(f"     P(Z < 0.3m):        {np.mean(old_mc < 0.3)*100:.1f}%")
    print(f"     P(Z < 0.5m):        {np.mean(old_mc < 0.5)*100:.1f}%")

    print(f"\n  🟢 NEW GENERATOR (rejection sampling):")
    print(f"     Rejection failures:  {new_rejected}/{N_SAMPLES} "
          f"({100*new_rejected/N_SAMPLES:.1f}%)")
    print(f"     Spawn inside obstacle: {new_spawn_col}/{N_SAMPLES}")
    print(f"     Trajectory penetration: {new_traj_pen}/{N_SAMPLES}")
    new_mc = np.array(new_min_clears)
    print(f"     Mean min clearance: {new_mc.mean():.3f}m")
    print(f"     Worst clearance:    {new_mc.min():.3f}m")

    # Verdict
    print(f"\n  {'='*60}")
    if old_traj_pen > N_SAMPLES * 0.1:
        print(f"  ❌ HYPOTHESIS CONFIRMED")
        print(f"     The old generator had {old_traj_pen}/{N_SAMPLES} "
              f"({100*old_traj_pen/N_SAMPLES:.1f}%) trajectory penetrations.")
        print(f"     Z→0 corruption was a significant gradient noise source.")
    if old_clipped > N_SAMPLES * 0.2:
        print(f"  ⚠️  CLIPPING ISSUE")
        print(f"     {old_clipped}/{N_SAMPLES} ({100*old_clipped/N_SAMPLES:.1f}%) "
              f"trajectories hit walls (position clamp artifact).")
    if new_spawn_col == 0 and new_traj_pen == 0:
        print(f"  ✅ FIX VERIFIED — New generator is collision-free")
    elif new_rejected > 0:
        print(f"  ⚠️  {new_rejected}/{N_SAMPLES} samples couldn't find safe trajectory "
              f"(max {20} resample attempts)")
    print(f"  {'='*60}")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, max(old_mc.max(), new_mc.max()) * 1.05, 60)
    axes[0].hist(old_mc, bins=bins, alpha=0.6, color='red', label='OLD (unsafe)', edgecolor='darkred')
    axes[0].hist(new_mc, bins=bins, alpha=0.6, color='green', label='NEW (safe)', edgecolor='darkgreen')
    axes[0].axvline(SAFE_RADIUS, color='black', ls='--', lw=2, label=f'Safe radius ({SAFE_RADIUS}m)')
    axes[0].axvline(0.3, color='orange', ls=':', lw=1.5, label='Danger (0.3m)')
    axes[0].set_xlabel('Min clearance (m)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Min Clearance Distribution: OLD vs NEW')
    axes[0].legend(fontsize=9)

    # CDF
    sorted_old = np.sort(old_mc)
    sorted_new = np.sort(new_mc)
    axes[1].plot(sorted_old, np.arange(len(sorted_old))/len(sorted_old),
                 'r-', lw=2, label='OLD (unsafe)')
    axes[1].plot(sorted_new, np.arange(len(sorted_new))/len(sorted_new),
                 'g-', lw=2, label='NEW (safe)')
    axes[1].axvline(SAFE_RADIUS, color='black', ls='--', lw=2, label=f'Safe radius')
    axes[1].set_xlabel('Min clearance (m)')
    axes[1].set_ylabel('CDF')
    axes[1].set_title('Cumulative Distribution')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f'Collision Audit — {N_SAMPLES} samples, {N_OBSTACLES} obstacles\n'
                 f'OLD: {old_traj_pen} penetrations, {old_clipped} clipped | '
                 f'NEW: {new_spawn_col + new_traj_pen} failures',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig('/Users/lhooz/.openclaw/workspace/collision_audit.png', dpi=150)
    print(f"\n  📸 Saved collision_audit.png")
    plt.close(fig)


def _min_clearance(pos, obstacles):
    """Min distance from pos to any obstacle surface or wall."""
    px, py = pos[0], pos[1]
    # Obstacle distances
    obs_d = jax.vmap(lambda r: _point_rect_dist_safe(px, py, r))(obstacles)
    min_obs = jnp.min(obs_d)
    # Wall distances
    wall_d = jnp.minimum(jnp.minimum(px, py),
                         jnp.minimum(ROOM_W - px, ROOM_H - py))
    return jnp.minimum(min_obs, wall_d)


def _point_rect_dist_safe(px, py, rect):
    """Distance from point to rectangle. Negative if inside."""
    cx = jnp.clip(px, rect[0], rect[2])
    cy = jnp.clip(py, rect[1], rect[3])
    outside = jnp.sqrt((px - cx)**2 + (py - cy)**2)
    inside = -jnp.minimum(
        jnp.minimum(px - rect[0], rect[2] - px),
        jnp.minimum(py - rect[1], rect[3] - py))
    inside_rect = (px >= rect[0]) & (px <= rect[2]) & (py >= rect[1]) & (py <= rect[3])
    return jnp.where(inside_rect, inside, outside)


if __name__ == "__main__":
    audit()
