#!/usr/bin/env python3
"""
SLAM Shadow Mapper — Event-VIO Proof of Concept

The SNN predicts [ω, clearance] from events.
IMU provides [vx, vy] (simulated by GT).
Fusion: Z_pred = |v_forward| × τ_pred

At each timestep, for each pixel:
  - Compute approach velocity: v_approach = dot(v_world, ray_dir)
  - τ = Z_min_pred / |v_forward|
  - Z_pixel = |v_approach| × τ
  - Project: point = position + ray_dir × Z_pixel

The accumulation of projected points over a trajectory
creates a "mental map" — the SNN's perception of the world.

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

import sys
sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')
from sparse_forest import (
    generate_sample, N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H, FOV_DEG,
    VX_RANGE, VY_RANGE, OMEGA_RANGE,
)
from snn_bio_vision import (
    run_snn, prepare_events, BETA, BETA_LI, V_TH,
)


def load_params(path="/Users/lhooz/.openclaw/workspace/bio_vision_params.npz"):
    data = np.load(path)
    return (jnp.array(data['W1']), jnp.array(data['W2']),
            jnp.array(data['W_li']), jnp.array(data['b_li']))


def predict_trajectory(events, params):
    """Run SNN on events, return predictions (T, 2)."""
    x_seq = prepare_events(events)
    U_seq = run_snn(x_seq, *params)
    # Average over loss window
    return jnp.mean(U_seq[-50:], axis=0)  # (2,)


def inverse_tanh(x, clip=0.99):
    """Inverse of tanh, with clipping for numerical stability."""
    x = np.clip(x, -clip, clip)
    return 0.5 * np.log((1 + x) / (1 - x))


def build_slam_map(info, cl_pred, vx, vy, omega,
                    n_rays=N_PIXELS, subsample=3):
    """Build SLAM point cloud from a single trajectory.
    
    Args:
        info: dict from generate_sample (positions, headings, obstacles, etc.)
        cl_pred: predicted clearance (scalar, in tanh space)
        vx, vy, omega: GT velocities
        n_rays: number of projection rays
        subsample: use every Nth timestep
    
    Returns:
        points: (N_points, 2) projected depth points
        errors: (N_points,) |Z_pred - Z_true| for each point
        trajectory: (T, 2) robot positions
    """
    positions = np.array(info['positions'])
    headings = np.array(info['headings'])
    obstacles = np.array(info['obstacles'])
    distances = np.array(info['distances'])
    
    # Convert predicted clearance to real distance
    Z_pred = inverse_tanh(float(cl_pred)) * 2.0
    Z_pred = max(Z_pred, 0.1)  # floor
    
    # Velocity in world frame (constant over trajectory)
    T = positions.shape[0]
    
    # Ray angles
    fov_rad = np.radians(FOV_DEG)
    ray_offsets = np.linspace(-fov_rad/2, fov_rad/2, n_rays)
    
    points = []
    errors = []
    
    for t in range(0, T, subsample):
        x, y = positions[t]
        theta = headings[t]
        
        # World-frame velocity
        wx = vx * np.cos(theta) - vy * np.sin(theta)
        wy = vx * np.sin(theta) + vy * np.cos(theta)
        v_world = np.array([wx, wy])
        heading_dir = np.array([np.cos(theta), np.sin(theta)])
        v_forward = np.dot(v_world, heading_dir)
        
        # Skip if barely moving (division by zero in τ)
        if abs(v_forward) < 0.05:
            continue
        
        # τ = Z_min / |v_forward|
        tau = Z_pred / abs(v_forward)
        
        # GT distance at this timestep
        dist_t = distances[t]
        
        for i, alpha in enumerate(ray_offsets):
            ray_angle = theta + alpha
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            
            # Approach velocity along this ray
            v_approach = np.dot(v_world, ray_dir)
            
            # Only project if approaching (v_approach > threshold)
            if v_approach < 0.05:
                continue
            
            # Predicted depth for this pixel: Z_pixel = v_approach × τ
            Z_pixel = v_approach * tau
            Z_pixel = max(Z_pixel, 0.1)
            
            # Project point
            px = x + Z_pixel * ray_dir[0]
            py = y + Z_pixel * ray_dir[1]
            
            # Clip to room bounds
            if px < 0 or px > ROOM_W or py < 0 or py > ROOM_H:
                continue
            
            # GT distance for error
            Z_true = dist_t[i] if i < len(dist_t) else dist_t[min(i, len(dist_t)-1)]
            error = abs(Z_pixel - Z_true)
            
            points.append([px, py])
            errors.append(error)
    
    return np.array(points), np.array(errors), positions


def plot_slam_overlay(points, errors, trajectory, obstacles,
                       title="SLAM Shadow Map", save_path=None):
    """Create the SLAM visualization: GT obstacles + SNN depth projections."""
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                            gridspec_kw={'width_ratios': [1.2, 1]})
    
    # === LEFT: SLAM Map ===
    ax1 = axes[0]
    ax1.set_xlim(-0.5, ROOM_W + 0.5)
    ax1.set_ylim(-0.5, ROOM_H + 0.5)
    ax1.set_aspect('equal')
    
    # Room boundary
    ax1.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H, lw=3, ec='black', fc='#fafafa'))
    
    # GT obstacles
    for o in obstacles:
        ax1.add_patch(Rectangle((o[0], o[1]), o[2]-o[0], o[3]-o[1],
                                fc='#cccccc', ec='black', lw=2, zorder=2))
    
    # SNN depth points (colored by error)
    if len(points) > 0:
        errors_clipped = np.clip(errors, 0, 3.0)
        scatter = ax1.scatter(points[:, 0], points[:, 1],
                             c=errors_clipped, cmap='RdYlGn_r',
                             s=3, alpha=0.6, vmin=0, vmax=3.0,
                             zorder=3, label='SNN depth')
    
    # Trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], '-', color='steelblue',
             lw=2, zorder=4, label='Trajectory')
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'o', color='limegreen',
             ms=10, zorder=5, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color='red',
             ms=10, zorder=5, label='End')
    
    # FOV at start and end
    fov_rad = np.radians(FOV_DEG)
    for step, c in [(0, 'limegreen'), (-1, 'red')]:
        px, py = trajectory[step]
        h = float(np.arctan2(
            trajectory[min(step+1, len(trajectory)-1), 1] - trajectory[step, 1],
            trajectory[min(step+1, len(trajectory)-1), 0] - trajectory[step, 0]))
        for sign in (-1, 1):
            a = h + sign * fov_rad / 2
            ax1.plot([px, px + 2*np.cos(a)], [py, py + 2*np.sin(a)],
                     '--', color=c, alpha=0.4, lw=1)
    
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
    
    # === RIGHT: Error Analysis ===
    ax2 = axes[1]
    if len(errors) > 0:
        ax2.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.median(errors), color='red', ls='--', lw=2,
                    label=f'Median: {np.median(errors):.2f}m')
        ax2.axvline(np.mean(errors), color='orange', ls='--', lw=2,
                    label=f'Mean: {np.mean(errors):.2f}m')
        pct_05 = np.percentile(errors, 5)
        pct_95 = np.percentile(errors, 95)
        ax2.axvspan(pct_05, pct_95, alpha=0.1, color='green',
                    label=f'90% CI: [{pct_05:.2f}, {pct_95:.2f}]m')
        ax2.legend(fontsize=9)
    ax2.set_xlabel('Depth Error (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('Projection Error Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, min(5, max(errors.max() * 1.1, 1) if len(errors) > 0 else 1))
    
    fig.suptitle('Event-VIO SLAM — SNN Shadow Map\n'
                 'Green dots = accurate depth, Red dots = large error',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📸 Saved to {save_path}")
    plt.close(fig)
    return fig


def plot_multi_trajectory(params, n_trajectories=6, seed=123):
    """Build and plot SLAM maps for multiple trajectories."""
    
    key = jax.random.PRNGKey(seed)
    all_points = []
    all_errors = []
    all_trajs = []
    all_obs = []
    summaries = []
    
    print(f"\n  🗺️  Building SLAM maps for {n_trajectories} trajectories...")
    
    for i in range(n_trajectories):
        key, subkey = jax.random.split(key)
        events, labels_4, info = generate_sample(subkey)
        
        # Extract labels
        omega_true = float(labels_4[2])
        cl_true = float(labels_4[3])
        vx = float(labels_4[0])
        vy = float(labels_4[1])
        
        # SNN prediction
        x_seq = prepare_events(events)
        U_seq = run_snn(x_seq, *params)
        pred = jnp.mean(U_seq[-50:], axis=0)
        omega_pred = float(pred[0])
        cl_pred = float(pred[1])
        
        # Build SLAM map
        points, errors, traj = build_slam_map(info, cl_pred, vx, vy, omega_true)
        
        all_points.append(points)
        all_errors.append(errors)
        all_trajs.append(traj)
        all_obs.append(np.array(info['obstacles']))
        
        # Stats
        Z_pred = inverse_tanh(cl_pred) * 2.0
        Z_true = float(jnp.min(info['distances'][-1]))
        n_pts = len(points)
        med_err = np.median(errors) if n_pts > 0 else 0
        
        summaries.append({
            'i': i, 'vx': vx, 'vy': vy, 'omega_true': omega_true,
            'omega_pred': omega_pred, 'cl_true': cl_true, 'cl_pred': cl_pred,
            'Z_true': Z_true, 'Z_pred': Z_pred,
            'n_points': n_pts, 'median_error': med_err,
            'obstacles': info['obstacles'],
        })
        
        print(f"    [{i}] vx={vx:+.2f} vy={vy:+.2f} ω={omega_true:+.2f} | "
              f"ω_pred={omega_pred:+.2f} cl_pred={cl_pred:+.2f} | "
              f"Z: {Z_true:.2f}→{Z_pred:.2f}m | "
              f"{n_pts} pts, med_err={med_err:.2f}m")
    
    # === Individual trajectory plots ===
    for i in range(n_trajectories):
        s = summaries[i]
        title = (f"Trajectory {i}: vx={s['vx']:+.2f} vy={s['vy']:+.2f} "
                f"ω={s['omega_true']:+.2f}\n"
                f"ω_pred={s['omega_pred']:+.2f}  "
                f"Z: {s['Z_true']:.2f}→{s['Z_pred']:.2f}m  "
                f"({s['n_points']} projections)")
        plot_slam_overlay(
            all_points[i], all_errors[i], all_trajs[i], all_obs[i],
            title=title,
            save_path=f"/Users/lhooz/.openclaw/workspace/slam_traj_{i}.png")
    
    # === Combined SLAM map ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_xlim(-0.5, ROOM_W + 0.5)
    ax.set_ylim(-0.5, ROOM_H + 0.5)
    ax.set_aspect('equal')
    ax.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H, lw=3, ec='black', fc='#fafafa'))
    
    # All trajectories
    colors_t = plt.cm.tab10(np.linspace(0, 1, n_trajectories))
    for i in range(n_trajectories):
        traj = all_trajs[i]
        ax.plot(traj[:, 0], traj[:, 1], '-', color=colors_t[i], lw=1.5, alpha=0.6)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors_t[i], ms=6)
    
    # All depth points
    combined_pts = np.vstack(all_points)
    combined_err = np.concatenate(all_errors)
    err_clipped = np.clip(combined_err, 0, 3.0)
    ax.scatter(combined_pts[:, 0], combined_pts[:, 1],
               c=err_clipped, cmap='RdYlGn_r', s=2, alpha=0.4,
               vmin=0, vmax=3.0, zorder=3)
    
    ax.set_title(f'Combined SLAM Map — {n_trajectories} trajectories\n'
                 f'{len(combined_pts)} depth projections  |  '
                 f'Median error: {np.median(combined_err):.2f}m',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    fig.tight_layout()
    fig.savefig('/Users/lhooz/.openclaw/workspace/slam_combined.png', dpi=150)
    print(f"\n  📸 Saved slam_combined.png")
    plt.close(fig)
    
    # Summary
    print(f"\n  {'='*60}")
    print(f"  📊 SLAM SUMMARY")
    print(f"  {'='*60}")
    total_pts = sum(s['n_points'] for s in summaries)
    all_err = np.concatenate(all_errors)
    print(f"  Trajectories:    {n_trajectories}")
    print(f"  Total points:    {total_pts}")
    print(f"  Median error:    {np.median(all_err):.2f}m")
    print(f"  Mean error:      {np.mean(all_err):.2f}m")
    print(f"  P90 error:       {np.percentile(all_err, 90):.2f}m")
    print(f"  P50 error:       {np.percentile(all_err, 50):.2f}m")
    print(f"  {'='*60}")
    
    return summaries


def main():
    print("=" * 60)
    print("  🗺️  SLAM Shadow Mapper — Event-VIO")
    print("=" * 60)
    
    # Load trained params
    try:
        params = load_params()
        print("  ✅ Loaded bio_vision_params.npz")
    except FileNotFoundError:
        print("  ❌ No trained params found!")
        print("  Run snn_bio_vision.py first.")
        return
    
    # Build SLAM maps
    summaries = plot_multi_trajectory(params, n_trajectories=6, seed=123)
    
    print(f"\n  ✅ SLAM mapping complete!")


if __name__ == "__main__":
    main()
