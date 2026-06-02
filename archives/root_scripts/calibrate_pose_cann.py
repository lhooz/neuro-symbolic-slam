#!/usr/bin/env python3
"""
Calibration Script: PoseCANN Open-Loop Isolation

Step 1: Disable Map CANN correction entirely (I_loop = 0 always).
Step 2: Sweep VEL_GAIN_XY and VEL_GAIN_TH to find values where
        open-loop dead-reckoning TRACKS the GT shape (ratio ≈ 1.0).
Step 3: Fix the confidence gate to be an absolute threshold.

Diagnostic:
  CANN/GT displacement ratio = 1.0 → perfect tracking
  CANN/GT displacement ratio = 0.0 → paralyzed (current gain too low)
  CANN/GT displacement ratio > 1.0 → overshooting (gain too high)

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
import time, sys, os

sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')

from src.sparse_forest import (
    generate_fixed_room_dataset,
    N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H, FOV_DEG,
    VX_RANGE, VY_RANGE, OMEGA_RANGE,
)
from src.snn_pose_cann import (
    PoseCANN,
    build_2d_cann_weights,
    build_1d_ring_weights,
    build_asymmetric_ring_weights,
    build_asymmetric_cann_weights_x,
    build_asymmetric_cann_weights_y,
    CANN_SIZE, RING_N,
)
from src.snn_vision_stdp import VisionSTDP


# ============================================================================
# CONFIG
# ============================================================================

BATCH_SIZE = 4
TIME_STEPS = 200          # 4 seconds
DRIFT_START = 1000        # Disable drift for open-loop test
N_SAMPLES_FOR_SIM = BATCH_SIZE

# Gain sweep range
GAIN_XY_MIN, GAIN_XY_MAX = 0.01, 0.5
GAIN_TH_MIN, GAIN_TH_MAX = 0.05, 0.5
N_GAIN_POINTS = 8


# ============================================================================
# POSE CANN WITH DISABLED MAP CORRECTION
# ============================================================================

class PoseCANNCalibrated(PoseCANN):
    """PoseCANN with forced zero map correction (open-loop isolation)."""

    def forward_openloop(self, kin_t):
        """One step with NO map correction."""
        return self.__call__(kin_t, map_correction=None)


def run_openloop_traj(pregenerated_data, vel_gain_xy, vel_gain_th,
                      n_samples=BATCH_SIZE, time_steps=TIME_STEPS,
                      verbose=True):
    """Run PoseCANN open-loop and return trajectories + diagnostics."""

    # ---- Use pre-generated data ----
    events, labels, tof_dists, positions, obstacles = pregenerated_data

    B, T, N = events.shape
    T = min(T, time_steps)
    events = events[:, :T, :]
    tof_dists = tof_dists[:, :T]
    positions = positions[:, :T, :]

    # ---- Headings ----
    dt_h = 3
    dx = positions[:, dt_h:, 0] - positions[:, :-dt_h, 0]
    dy = positions[:, dt_h:, 1] - positions[:, :-dt_h, 1]
    headings_raw = jnp.arctan2(dy, dx)
    pad = jnp.zeros((B, dt_h))
    headings = jnp.concatenate([pad, headings_raw], axis=1) % (2 * jnp.pi)

    # ---- Kinematics ----
    vx = labels[:, 0:1] * abs(VX_RANGE[1])
    vy = labels[:, 1:2] * abs(VY_RANGE[1])
    omega = labels[:, 2:3] * abs(OMEGA_RANGE[1])
    kin = jnp.stack([
        jnp.tile(vx, (1, T)),
        jnp.tile(vy, (1, T)),
        jnp.tile(omega, (1, T)),
    ], axis=2)

    pos_gt = positions[:, :, :2]
    th_gt = np.array(headings)

    # ---- Initialize PoseCANN with patched gains ----
    W_cann_mat = build_2d_cann_weights()
    W_ring_mat = build_1d_ring_weights()
    W_ring_asym_mat = build_asymmetric_ring_weights()
    W_cann_asym_x_mat = build_asymmetric_cann_weights_x()
    W_cann_asym_y_mat = build_asymmetric_cann_weights_y()

    # Patch module-level gains BEFORE instantiating PoseCANN
    import src.snn_pose_cann as spc
    import jax
    # Disable JIT to prevent caching of VEL_GAIN constants across gain sweeps
    jax.config.update('jax_disable_jit', True)
    original_xy = spc.VEL_GAIN_XY
    original_th = spc.VEL_GAIN_TH
    spc.VEL_GAIN_XY = vel_gain_xy
    spc.VEL_GAIN_TH = vel_gain_th

    pose_net = PoseCANN(random.PRNGKey(42),
                          W_cann_mat, W_ring_mat,
                          W_cann_asym_x_mat, W_cann_asym_y_mat, W_ring_asym_mat)
    pose_net.reset(B)
    pose_net.initialize_from_gt(jnp.array(pos_gt[:, 0]), headings[:, 0])

    # Restore original gains and JIT
    spc.VEL_GAIN_XY = original_xy
    spc.VEL_GAIN_TH = original_th
    jax.config.update('jax_disable_jit', False)

    # ---- Run open-loop ----
    x_e = np.zeros((B, T))
    y_e = np.zeros((B, T))
    th_e = np.zeros((B, T))

    for t in range(T):
        kin_t = kin[:, t, :]
        pose = pose_net(kin_t, map_correction=None)  # ZERO map correction
        x_e[:, t] = np.array(pose[:, 0])
        y_e[:, t] = np.array(pose[:, 1])
        th_e[:, t] = np.array(pose[:, 2])

    # ---- Compute diagnostics ----
    pos_err = np.sqrt(
        (x_e - pos_gt[:, :, 0])**2 + (y_e - pos_gt[:, :, 1])**2
    )

    th_err = np.abs(th_e - th_gt)
    th_err = np.minimum(th_err, 2*np.pi - th_err)

    # Displacement ratio
    gt_disp = np.sqrt(
        (pos_gt[:, -1, 0] - pos_gt[:, 0, 0])**2 +
        (pos_gt[:, -1, 1] - pos_gt[:, 0, 1])**2
    )
    cann_disp = np.sqrt(
        (x_e[:, -1] - x_e[:, 0])**2 +
        (y_e[:, -1] - y_e[:, 0])**2
    )
    disp_ratio = cann_disp / (gt_disp + 1e-9)

    # Mean step distance
    dx_e = np.diff(x_e, axis=1)
    dy_e = np.diff(y_e, axis=1)
    dx_gt = np.diff(pos_gt[:, :, 0], axis=1)
    dy_gt = np.diff(pos_gt[:, :, 1], axis=1)
    step_cann = np.sqrt(dx_e**2 + dy_e**2).mean()
    step_gt = np.sqrt(dx_gt**2 + dy_gt**2).mean()
    step_ratio = step_cann / (step_gt + 1e-9)

    if verbose:
        print(f'  gain_xy={vel_gain_xy:.3f} th={vel_gain_th:.3f} → '
              f'ratio={disp_ratio.mean():.3f} step_ratio={step_ratio:.3f} '
              f'err={pos_err.mean():.3f}m '
              f'angle_err={np.degrees(th_err.mean()):.1f}°')

    return {
        'x_e': x_e, 'y_e': y_e, 'th_e': th_e,
        'pos_gt': pos_gt, 'th_gt': th_gt,
        'pos_err': pos_err, 'th_err': th_err,
        'gt_disp': gt_disp, 'cann_disp': cann_disp,
        'disp_ratio': disp_ratio, 'step_ratio': step_ratio,
        'obstacles': np.array(obstacles),
        'vel_gain_xy': vel_gain_xy,
        'vel_gain_th': vel_gain_th,
        'B': B, 'T': T,
    }


# ============================================================================
# STEP 1: VELOCITY GAIN SWEEP
# ============================================================================

def sweep_gains(key):
    """Sweep VEL_GAIN_XY and VEL_GAIN_TH to find optimal open-loop tracking."""
    print("\n" + "="*65)
    print("  🔬 Step 1: Open-Loop Velocity Gain Sweep")
    print("="*65)

    # Generate ONE dataset and reuse for all gains (fair comparison)
    print(f"  Generating shared dataset (B={N_SAMPLES_FOR_SIM}, T={TIME_STEPS})...")
    pregen = generate_fixed_room_dataset(key, N_SAMPLES_FOR_SIM, time_steps=TIME_STEPS)
    print(f"  Dataset: GT bounds x=[{float(pregen[3][:,0,0].min()):.2f}, "
          f"{float(pregen[3][:,0,0].max()):.2f}]")

    gain_values = np.linspace(GAIN_XY_MIN, GAIN_XY_MAX, N_GAIN_POINTS)

    results_grid = []
    best = None
    best_score = float('inf')

    for g_xy in gain_values:
        res = run_openloop_traj(
            pregen, vel_gain_xy=g_xy, vel_gain_th=g_xy,
            n_samples=N_SAMPLES_FOR_SIM, time_steps=TIME_STEPS, verbose=True
        )
        results_grid.append(res)

        # Score = how close disp_ratio is to 1.0 AND low error
        score = abs(res['disp_ratio'].mean() - 1.0) + 0.5 * res['pos_err'].mean()
        if score < best_score:
            best_score = score
            best = res

    # Print summary
    print(f"\n📊 Gain Sweep Results (same dataset, different gains):")
    print(f"  {'gain_xy=th':>10} | {'disp_ratio':>12} | {'step_ratio':>11} | "
          f"{'pos_err':>8} | {'ang_err(°)':>10}")
    print(f"  {'-'*10}---{'-'*12}---{'-'*11}---{'-'*8}---{'-'*10}")
    for r in results_grid:
        g = r['vel_gain_xy']
        print(f"  {g:10.3f} | {r['disp_ratio'].mean():12.3f} | "
              f"{r['step_ratio']:11.3f} | {r['pos_err'].mean():8.3f}m | "
              f"{np.degrees(r['th_err'].mean()):10.1f}°")

    print(f"\n🏆 Best gain (rough): {best['vel_gain_xy']:.3f} "
          f"(disp_ratio={best['disp_ratio'].mean():.3f}, "
          f"err={best['pos_err'].mean():.3f}m)")

    return results_grid, best, pregen


# ============================================================================
# STEP 1b: FINE SWEEP AROUND BEST
# ============================================================================

def fine_sweep_2d(pregenerated_data, rough_best_xy):
    """Fine 2D sweep around the best gain found."""
    print("\n" + "="*65)
    print("  🔬 Step 1b: Fine 2D Sweep Around Best")
    print("="*65)

    # Grid around rough best
    n_fine = 5
    g_xy_lo = max(GAIN_XY_MIN, rough_best_xy * 0.3)
    g_xy_hi = min(GAIN_XY_MAX, rough_best_xy * 3.0)
    g_xy_range = np.linspace(g_xy_lo, g_xy_hi, n_fine)
    g_th_range = np.linspace(0.05, 0.6, n_fine)

    best = None
    best_score = float('inf')
    grid = np.zeros((n_fine, n_fine))

    for i, g_xy in enumerate(g_xy_range):
        for j, g_th in enumerate(g_th_range):
            res = run_openloop_traj(
                pregenerated_data, vel_gain_xy=g_xy, vel_gain_th=g_th,
                n_samples=N_SAMPLES_FOR_SIM, time_steps=TIME_STEPS,
                verbose=False
            )
            grid[i, j] = res['disp_ratio'].mean()
            score = abs(res['disp_ratio'].mean() - 1.0) + 0.3 * res['pos_err'].mean()
            if score < best_score:
                best_score = score
                best = res

    print(f"\n  2D Grid (displacement ratio):")
    print(f"        " + "  ".join([f"th={g_th:.2f}" for g_th in g_th_range]))
    for i, g_xy in enumerate(g_xy_range):
        row = "  ".join([f"{grid[i,j]:7.3f}" for j in range(n_fine)])
        print(f"  xy={g_xy:.3f}: {row}")

    print(f"\n🏆 Best fine: g_xy={best['vel_gain_xy']:.4f} g_th={best['vel_gain_th']:.4f}")
    print(f"    disp_ratio={best['disp_ratio'].mean():.3f} "
          f"step_ratio={best['step_ratio']:.3f}")
    print(f"    pos_err={best['pos_err'].mean():.3f}m "
          f"ang_err={np.degrees(best['th_err'].mean()):.1f}°")

    return best, grid


# ============================================================================
# STEP 2: FIX CONFIDENCE GATE
# ============================================================================

def design_confidence_gate():
    """Document the new absolute-threshold confidence gate design."""
    print("\n" + "="*65)
    print("  🔬 Step 2: Confidence Gate Redesign")
    print("="*65)

    # The old approach: normalized by max-recall → always saturates to 1.0
    # The new approach: absolute threshold on raw vision-memory overlap

    # New blend formula:
    #   raw_match = I_corr.sum()  (total recalled place-cell activation)
    #   vision_activity = vision_spikes.sum() (total active features)
    #   sparsity_norm = vision_activity / N_VISION  (0 to 1, how many features active)
    #
    # If sparsity_norm < SPARSITY_THRESHOLD → NOT a distinctive view → SILENT
    # If raw_match > MATCH_THRESHOLD → strong memory recall → ACTIVE
    #
    # SPARSITY_THRESHOLD = 0.05 (at least 5% of vision features active = distinctive)
    # MATCH_THRESHOLD = 0.10 (at least 10 total recalled activation = real match)

    print("""
Old (BROKEN):
  blend = clip(I_corr.sum() / max_recall * 10, 0, 1)
  → max_recall = vision_activity × 2.0 (theoretical max)
  → This normalizes OUT the magnitude, always gives blend ≈ 1.0
  → Map correction ALWAYS fires = teleportation

New (CORRECT):
  SPARSITY_THRESHOLD = 0.05   (at least 5% vision features firing = distinctive view)
  MATCH_THRESHOLD = 0.15      (at least 15% recalled activation = strong match)

  sparsity = vision_spikes.sum() / N_VISION          # how busy is the vision?
  match    = I_corr.sum() / N_PLACE                  # how strong is the recall?

  IF sparsity > SPARSITY_THRESHOLD AND match > MATCH_THRESHOLD:
      blend = clip((match - MATCH_THRESHOLD) / (0.5 - MATCH_THRESHOLD), 0, 1)
  ELSE:
      blend = 0.0   ← Map CANN stays COMPLETELY SILENT

This means:
  • Random/noisy views → sparse activity → NO correction
  • First-time novel views → sparse → NO correction (learning only)
  • Revisiting known location → dense match + high recall → STRONG ghost bump
""")

    return {
        'SPARSITY_THRESHOLD': 0.05,
        'MATCH_THRESHOLD': 0.15,
        'BLEND_SCALE': 10.0,
    }


# ============================================================================
# VISUALIZATION: PANEL 1 FIXED (Open-Loop with proper gains)
# ============================================================================

def plot_calibration_results(best_res, grid_2d, gate_params, save_path=None):
    """Generate a calibration report figure."""

    fig = plt.figure(figsize=(20, 10))

    # ---- Panel 1a: Open-loop trajectory with best gains ----
    ax1 = fig.add_subplot(2, 3, 1)
    B = min(3, best_res['B'])
    colors_traj = plt.cm.Blues(np.linspace(0.5, 0.9, B))
    colors_est = ['#E74C3C', '#E67E22', '#9B59B6']

    for i in range(B):
        ax1.plot(best_res['pos_gt'][i, ::4, 0],
                 best_res['pos_gt'][i, ::4, 1],
                 'o-', color=colors_traj[i], ms=4, lw=1.5, alpha=0.7)
        ax1.plot(best_res['x_e'][i, ::4],
                 best_res['y_e'][i, ::4],
                 's--', color=colors_est[i], ms=4, lw=2.0, alpha=0.85,
                 label=f'OL est {i}' if i == 0 else None)

    for o in best_res['obstacles']:
        w, h = float(o[2]-o[0]), float(o[3]-o[1])
        ax1.add_patch(Rectangle((float(o[0]), float(o[1])), w, h,
                                 facecolor='#888', edgecolor='#222', lw=1.0, alpha=0.8))
    ax1.set_xlim(-0.5, ROOM_W + 0.5)
    ax1.set_ylim(-0.5, ROOM_H + 0.5)
    ax1.set_aspect('equal')
    ax1.set_title('Panel 1a: Open-Loop Trajectory\n(GT ● vs OL ■)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2, linestyle='--')

    # ---- Panel 1b: Error over time ----
    ax2 = fig.add_subplot(2, 3, 2)
    t_arr = np.arange(best_res['T']) * DT
    for i in range(B):
        ax2.plot(t_arr, best_res['pos_err'][i],
                 color=colors_est[i], lw=1.5, alpha=0.7)
    ax2.plot(t_arr, best_res['pos_err'].mean(axis=0),
             color='#E74C3C', lw=3.0, ls='--', label='Mean error')
    ax2.set_title('Panel 1b: Open-Loop Position Error\n(meters vs time)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Error (m)')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2, linestyle='--')
    ax2.set_xlim(0, best_res['T'] * DT)
    ax2.set_ylim(bottom=0)

    # ---- Panel 1c: Displacement ratio bar chart ----
    ax3 = fig.add_subplot(2, 3, 3)
    ratios = best_res['disp_ratio']
    bars = ax3.bar(range(len(ratios)), ratios,
                   color=['#27AE60' if 0.5 < r < 1.5 else '#E74C3C' for r in ratios],
                   alpha=0.8, edgecolor='#222', lw=1.2)
    ax3.axhline(1.0, color='#27AE60', lw=2.0, ls='--', label='Ideal (1.0)')
    ax3.set_xticks(range(len(ratios)))
    ax3.set_xticklabels([f'Sample {i}' for i in range(len(ratios))], fontsize=8)
    ax3.set_ylabel('CANN/GT Displacement Ratio')
    ax3.set_title('Panel 1c: Tracking Ratio\n(1.0 = perfect)',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.2, linestyle='--', axis='y')
    for bar, ratio in zip(bars, ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{ratio:.2f}', ha='center', fontsize=9, fontweight='bold')

    # ---- Panel 2a: 2D gain heatmap ----
    ax4 = fig.add_subplot(2, 3, 4)
    g_xy_range = np.linspace(best_res['vel_gain_xy'] * 0.5,
                              best_res['vel_gain_xy'] * 2.0, 5)
    g_th_range = np.linspace(0.05, 0.4, 5)
    im = ax4.imshow(grid_2d, cmap='RdYlGn', aspect='auto', origin='lower',
                     vmin=0, vmax=max(2.0, grid_2d.max() * 1.1))
    ax4.set_xticks(range(5))
    ax4.set_xticklabels([f'{g:.2f}' for g in g_th_range], fontsize=8)
    ax4.set_yticks(range(5))
    ax4.set_yticklabels([f'{g:.3f}' for g in g_xy_range], fontsize=8)
    ax4.set_xlabel('VEL_GAIN_TH')
    ax4.set_ylabel('VEL_GAIN_XY')
    ax4.set_title('Panel 2a: Gain Sweep Heatmap\n(displacement ratio)',
                  fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax4, fraction=0.046, label='ratio')
    # Mark ideal contour
    ax4.contour(grid_2d, levels=[1.0], colors='white', linewidths=1.5,
                linestyles='--', alpha=0.8)

    # ---- Panel 2b: Heading comparison ----
    ax5 = fig.add_subplot(2, 3, 5)
    for i in range(min(2, B)):
        t_arr_h = np.arange(best_res['T']) * DT
        ax5.plot(t_arr_h, np.degrees(best_res['th_gt'][i]),
                 '-', color=colors_traj[i], lw=1.5, alpha=0.7, label=f'GT {i}')
        ax5.plot(t_arr_h, np.degrees(best_res['th_e'][i]),
                 '--', color=colors_est[i], lw=2.0, alpha=0.85, label=f'OL {i}')
    ax5.set_title('Panel 2b: Heading over Time\n(GT — vs OL --)',
                  fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Heading (°)')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.2, linestyle='--')

    # ---- Panel 2c: Confidence gate explanation ----
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    gate_text = (
        "CONFIDENCE GATE REDESIGN\n"
        "─" * 30 + "\n\n"
        "OLD (broken):\n"
        "  blend = clip(match/max_recall × 10, 0, 1)\n"
        "  → Always saturates to 1.0\n"
        "  → Ghost bump ALWAYS fires\n"
        "  → Teleportation!\n\n"
        "NEW (correct):\n"
        "  sparsity = vision.sum() / 256\n"
        "  match = I_corr.sum() / 1024\n\n"
        f"  SPARSITY_THRESHOLD = {gate_params['SPARSITY_THRESHOLD']}\n"
        "  (must have ≥5% vision features active)\n\n"
        f"  MATCH_THRESHOLD = {gate_params['MATCH_THRESHOLD']}\n"
        "  (must have ≥15% recall activation)\n\n"
        "  IF sparsity > threshold AND match > threshold:\n"
        "      blend = linear_ramp(match)\n"
        "  ELSE:\n"
        "      blend = 0.0 ← COMPLETELY SILENT\n\n"
        "Result:\n"
        "  • Novel view → sparse → no correction\n"
        "  • Revisiting familiar → dense + strong → ghost fires\n"
        "  • Only TRUE loop closures trigger"
    )
    ax6.text(0.05, 0.95, gate_text, transform=ax6.transAxes,
             fontsize=9, fontfamily='monospace',
             verticalalignment='top')
    ax6.set_title('Panel 2c: New Confidence Gate Design',
                  fontsize=11, fontweight='bold')

    # Summary stats
    fig.text(0.5, 0.01,
              f"Best gains: VEL_GAIN_XY={best_res['vel_gain_xy']:.4f}  "
              f"VEL_GAIN_TH={best_res['vel_gain_th']:.4f}  |  "
              f"Mean disp_ratio={best_res['disp_ratio'].mean():.3f}  "
              f"Mean pos_err={best_res['pos_err'].mean():.3f}m",
              ha='center', fontsize=10, style='italic', color='#333')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n  💾 Saved: {save_path}")

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*65)
    print("  🔬 PoseCANN Calibration: Open-Loop Isolation")
    print("="*65)

    key = random.PRNGKey(0xDEADC0DE)

    # Step 1: Rough gain sweep (uses same dataset for all gains)
    rough_results, best_rough, pregen_data = sweep_gains(key)

    # Step 1b: Fine 2D sweep
    best_fine, grid_2d = fine_sweep_2d(pregen_data, best_rough['vel_gain_xy'])

    # Step 2: Document new confidence gate
    gate_params = design_confidence_gate()

    # Visualize
    print("\n🎨 Generating calibration figure...")
    fig = plot_calibration_results(
        best_fine, grid_2d, gate_params,
        save_path='/Users/lhooz/.openclaw/workspace/pose_cann_calibration.png'
    )

    print(f"\n{'='*65}")
    print(f"  ✅ Calibration Complete")
    print(f"  Best VEL_GAIN_XY: {best_fine['vel_gain_xy']:.4f}")
    print(f"  Best VEL_GAIN_TH: {best_fine['vel_gain_th']:.4f}")
    print(f"  Displacement ratio: {best_fine['disp_ratio'].mean():.3f}")
    print(f"  Open-loop pos err: {best_fine['pos_err'].mean():.3f}m")
    print(f"{'='*65}")

    return best_fine, grid_2d, gate_params


if __name__ == '__main__':
    best_fine, grid_2d, gate_params = main()
