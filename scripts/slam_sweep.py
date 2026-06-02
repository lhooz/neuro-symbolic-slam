#!/usr/bin/env python3
"""
slam_sweep.py — Headless Parameter Sweep for SNN SLAM v7

Runs the simulation 50-100 times in parallel, tweaking biological hyperparameters
and finding the combination that minimizes Umeyama ATE.

Usage:
    python slam_sweep.py                    # grid search (all combos)
    python slam_sweep.py --optuna          # Bayesian optimization
    python slam_sweep.py --quick            # 20-trial fast scan
"""
import sys, os, time, json
ROOT = '/Users/lhooz/.openclaw/workspace'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src/stable1'))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # force CPU

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial
import itertools
from multiprocessing import Pool, cpu_count
import argparse

import src.stable1.snn_slam_system as S
import src.snn_place_cells as PC
import src.snn_pose_cann as PoseMod

# ─── PARAMETER SPACE ─────────────────────────────────────────────────────────

# Ultra-focused sweep: 2 most impactful parameters × 2 values
# Total: 2×2 = 4 combos × n_seeds = 8 trials (manageable)
PARAM_RANGES = {
    'GATING_STRENGTH': [2.0, 8.0],   # Low vs high texture amplification
    'MATURITY_GATE':   [0.50, 0.85], # Permissive vs strict loop closure gate
}

# ─── HEADLESS SIMULATION ─────────────────────────────────────────────────────

def run_headless_trial(params: dict, seed: int = 42, n_steps: int = 2000,
                       drift_start: int = 1000, drift_omega: float = 0.001) -> dict:
    """
    Run one headless trial of SNN SLAM and return ATE metrics.
    No matplotlib, no printing — pure numerics.

    Patches module-level constants in snn_place_cells and snn_pose_cann
    so the methods pick up the swept values.
    """
    # ── Patch module-level constants ──
    orig = {}
    for name, val in params.items():
        module = PC if hasattr(PC, name) else (PoseMod if hasattr(PoseMod, name) else None)
        if module and name in dir(module):
            orig[name] = (module, getattr(module, name))
            setattr(module, name, val)

    key = random.PRNGKey(seed)
    env = S.LiveEnvironment(key, chunk_size=n_steps + 100)

    system_ol = S.SNNSLAMSystem(random.PRNGKey(42), n_depth=S.N_DEPTH)
    system_cl = S.SNNSLAMSystem(random.PRNGKey(43), n_depth=S.N_DEPTH)
    system_ol.reset(1); system_cl.reset(1)

    _, _, _, pos0, th0, _ = env.step()
    system_ol.initialize_from_gt(jnp.array([pos0]), jnp.array([th0]))
    system_cl.initialize_from_gt(jnp.array([pos0]), jnp.array([th0]))

    gt_pos_hist = []
    imu_pos_hist = []
    ol_pos_hist = []
    cl_pos_hist = []
    gt_th_hist = []
    imu_th_hist = []

    x_imu, y_imu, th_imu = pos0[0], pos0[1], th0
    tof_angles = np.array([-np.pi/4, 0.0, np.pi/4])

    graph_poses = []
    graph_odom_edges = []
    node_tof_hits = []
    place_to_node = {}
    loop_closures = []
    loop_offsets_list = []
    loop_weights_list = []

    KEYFRAME_DIST = 0.15
    KEYFRAME_ANG = 0.20
    last_kf_cann = None

    step = 0
    while step < n_steps:
        ev_t, kin_t, tof_t, gt_pos, gt_th, intensity = env.step()
        ev_jax = jnp.array([ev_t])
        kin_jax = jnp.array([kin_t])
        tof_jax = jnp.array([tof_t])

        inject_drift = step >= drift_start

        if step > 0:
            bias = drift_omega if inject_drift else 0.0
            noise_std_dev = 0.005 if inject_drift else 0.0
            omega_b = kin_t[2] + bias + np.random.normal(0.0, noise_std_dev)
            vx_w = kin_t[0] * np.cos(th_imu) - kin_t[1] * np.sin(th_imu)
            vy_w = kin_t[0] * np.sin(th_imu) + kin_t[1] * np.cos(th_imu)
            x_imu += vx_w * S.DT
            y_imu += vy_w * S.DT
            th_imu = S.wrap_angle(th_imu + omega_b * S.DT)

        pose_ol, _, _ = system_ol.forward_step_open_loop(ev_jax, kin_jax, tof_jax, inject_drift=inject_drift)
        pose_cl, r_place, r_ring, is_confident, peak_idx_place, debug_gates = system_cl.forward_step(
            ev_jax, kin_jax, tof_jax, inject_drift=inject_drift)

        cx, cy, cth = float(pose_cl[0, 0]), float(pose_cl[0, 1]), float(pose_cl[0, 2])

        if last_kf_cann is None:
            last_kf_cann = (cx, cy, cth)

        kf_x, kf_y, kf_th = last_kf_cann
        dx = cx - kf_x
        dy = cy - kf_y
        local_dx = dx * np.cos(-kf_th) - dy * np.sin(-kf_th)
        local_dy = dx * np.sin(-kf_th) + dy * np.cos(-kf_th)
        local_dth = (cth - kf_th + np.pi) % (2*np.pi) - np.pi

        is_keyframe = False
        if len(graph_poses) == 0:
            is_keyframe = True
        else:
            dist = np.sqrt(dx**2 + dy**2)
            ang = np.abs(local_dth)
            if dist > KEYFRAME_DIST or ang > KEYFRAME_ANG:
                is_keyframe = True

        gt_pos_hist.append(gt_pos)
        gt_th_hist.append(gt_th)
        imu_pos_hist.append([x_imu, y_imu])
        imu_th_hist.append(th_imu)
        ol_pos_hist.append([float(pose_ol[0, 0]), float(pose_ol[0, 1])])
        cl_pos_hist.append([cx, cy])

        if is_keyframe:
            node_id = len(graph_poses)
            if node_id == 0:
                graph_poses.append([cx, cy, cth])
            else:
                graph_odom_edges.append([local_dx, local_dy, local_dth])
                graph_poses.append([cx, cy, cth])
            last_kf_cann = (cx, cy, cth)
            node_tof_hits.append(tof_t.copy())
            place_id = int(peak_idx_place[0])

            if place_id not in place_to_node:
                place_to_node[place_id] = []
            place_to_node[place_id].append(node_id)

            if is_confident[0]:
                matched_node = None
                min_dth = 999.0
                sm_th = params.get('RING_SELF_MATCH_THRESH', 0.001)
                p_th = params.get('PLAUSIBILITY_THRESH', 1.5)
                mat_th = params.get('MATURITY_GATE', 0.85)

                maturity = float(debug_gates['Maturity_Lvl'][0])
                if maturity < mat_th:
                    step += 1
                    continue

                if place_id in place_to_node:
                    for nid in place_to_node[place_id]:
                        if node_id - nid > 10:
                            _, _, nth = graph_poses[nid]
                            dth = abs(S.wrap_angle(cth - nth))
                            match_th = params.get('SELF_MATCH_THRESH', 0.001)
                            if dth < sm_th and dth < min_dth:
                                min_dth = dth
                                matched_node = nid

                if matched_node is not None:
                    conc_p = float(debug_gates['Conc_Place'][0])
                    conc_r = float(debug_gates['Conc_Ring'][0])
                    w_pos = (maturity * conc_p) * 0.20
                    w_th = (maturity * conc_r) * 0.15
                    loop_closures.append([matched_node, node_id])
                    loop_offsets_list.append([0.0, 0.0, 0.0])
                    loop_weights_list.append([w_pos, w_th])

        step += 1

    # ── Compute Umeyama ATE ──
    gt_arr = np.array(gt_pos_hist)
    ol_arr = np.array(ol_pos_hist)
    cl_arr = np.array(cl_pos_hist)
    min_len = min(len(gt_arr), len(ol_arr), len(cl_arr))
    gt_arr = gt_arr[:min_len]
    ol_arr = ol_arr[:min_len]
    cl_arr = cl_arr[:min_len]

    R_ol, t_ol = S.get_optimal_alignment_2d(ol_arr, gt_arr)
    ol_aligned = (R_ol @ ol_arr.T).T + t_ol
    R_cl, t_cl = S.get_optimal_alignment_2d(cl_arr, gt_arr)
    cl_aligned = (R_cl @ cl_arr.T).T + t_cl

    ate_ol = np.mean(np.sqrt((ol_aligned[:, 0] - gt_arr[:, 0])**2 +
                              (ol_aligned[:, 1] - gt_arr[:, 1])**2))
    ate_cl = np.mean(np.sqrt((cl_aligned[:, 0] - gt_arr[:, 0])**2 +
                              (cl_aligned[:, 1] - gt_arr[:, 1])**2))
    final_err_ol = float(np.sqrt((ol_aligned[-1, 0] - gt_arr[-1, 0])**2 +
                                   (ol_aligned[-1, 1] - gt_arr[-1, 1])**2))
    final_err_cl = float(np.sqrt((cl_aligned[-1, 0] - gt_arr[-1, 0])**2 +
                                   (cl_aligned[-1, 1] - gt_arr[-1, 1])**2))

    try:
        return {
            'params': params,
            'ate_ol': ate_ol,
            'ate_cl': ate_cl,
            'final_err_ol': final_err_ol,
            'final_err_cl': final_err_cl,
            'n_steps': min_len,
            'n_loop_closures': len(loop_closures),
        }
    finally:
        # ── Restore original module-level constants ──
        for name, (mod, original_val) in orig.items():
            setattr(mod, name, original_val)


# ─── GRID SEARCH ─────────────────────────────────────────────────────────────

def grid_search(param_ranges: dict, n_seeds: int = 3, n_steps: int = 2000,
                workers: int = None) -> list[dict]:
    """Exhaustively sweep all parameter combinations across n seeds."""

    # Generate all combinations
    keys = list(param_ranges.keys())
    values_lists = [param_ranges[k] for k in keys]
    combos = list(itertools.product(*values_lists))
    print(f"  🔬 Grid Search: {len(combos)} combos × {n_seeds} seeds = {len(combos)*n_seeds} trials")

    tasks = []
    for combo in combos:
        params = dict(zip(keys, combo))
        for seed in range(n_seeds):
            tasks.append((params, seed, n_steps))

    t0 = time.time()
    results = []

    if workers and workers > 1:
        with Pool(workers) as pool:
            for i, res in enumerate(pool.starmap(run_headless_trial, tasks)):
                results.append(res)
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(tasks) - i - 1) / rate / 60
                    print(f"  ⏳ {i+1}/{len(tasks)} done — {rate:.1f} trials/s — ETA {eta:.1f} min")
    else:
        for i, task in enumerate(tasks):
            res = run_headless_trial(*task)
            results.append(res)
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate / 60
                print(f"  ⏳ {i+1}/{len(tasks)} done — {rate:.1f} trials/s — ETA {eta:.1f} min")

    total_time = time.time() - t0
    print(f"\n  ✅ Grid search complete in {total_time/60:.1f} min")
    return results


# ─── OPTUNA BAYESIAN OPTIMIZATION ────────────────────────────────────────────

def optuna_search(param_ranges: dict, n_trials: int = 50, n_steps: int = 2000,
                  n_seeds: int = 2, workers: int = None) -> list[dict]:
    """Use Optuna to adaptively find the best parameters."""
    try:
        import optuna
    except ImportError:
        print("  ⚠️  Optuna not installed. Falling back to grid search.")
        return grid_search(param_ranges, n_seeds, n_steps, workers)

    print(f"  🧠 Optuna Bayesian Optimization: {n_trials} trials × {n_seeds} seeds")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial):
        params = {}
        for name, values in param_ranges.items():
            params[name] = trial.suggest_categorical(name, values)
        scores = []
        for seed in range(n_seeds):
            res = run_headless_trial(params, seed=seed, n_steps=n_steps)
            scores.append(res['ate_cl'])
        return np.mean(scores)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   n_jobs=workers or 1)

    best = study.best_trial
    print(f"\n  🏆 Best ATE_CL: {best.value:.4f}")
    print(f"  📋 Best params: {best.params}")
    return best


# ─── RESULTS REPORTER ────────────────────────────────────────────────────────

def report_results(results: list[dict], save_path: str = None):
    """Print leaderboard and save to JSON."""
    # Sort by closed-loop ATE
    results_sorted = sorted(results, key=lambda r: r['ate_cl'])

    print("\n" + "=" * 80)
    print("  🏆 SWEEP LEADERBOARD — Top 10 by Closed-Loop ATE")
    print("=" * 80)
    print(f"  {'Rank':>4}  {'ATE_CL':>8}  {'ATE_OL':>8}  {'Final_CL':>9}  {'LCs':>4}  {'Params'}")
    print("-" * 80)

    for i, r in enumerate(results_sorted[:10]):
        pstr = ', '.join(f"{k}={v}" for k, v in r['params'].items())
        print(f"  {i+1:>4}  {r['ate_cl']:>8.4f}  {r['ate_ol']:>8.4f}  "
              f"{r['final_err_cl']:>9.4f}  {r['n_loop_closures']:>4}  {pstr}")

    print("-" * 80)
    best = results_sorted[0]
    worst = results_sorted[-1]
    print(f"  Best:  ATE_CL={best['ate_cl']:.4f}, Final={best['final_err_cl']:.4f}m, LCs={best['n_loop_closures']}")
    print(f"  Worst: ATE_CL={worst['ate_cl']:.4f}, Final={worst['final_err_cl']:.4f}m, LCs={worst['n_loop_closures']}")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump({'results': results, 'best': results_sorted[0]}, f, indent=2, default=str)
        print(f"\n  💾 Full results saved: {save_path}")

    # Parameter importance (crude)
    print("\n  📊 Parameter Influence (mean ATE_CL per value):")
    param_names = list(PARAM_RANGES.keys())
    for pname in param_names:
        vals = PARAM_RANGES[pname]
        means = []
        for v in vals:
            subset = [r for r in results if r['params'].get(pname, None) == v]
            if subset:
                means.append((v, np.mean([r['ate_cl'] for r in subset])))
        if means:
            means.sort(key=lambda x: x[1])
            best_val = means[0][0]
            worst_val = means[-1][0]
            print(f"    {pname}: best={best_val} ({means[0][1]:.4f}) | "
                  f"worst={worst_val} ({means[-1][1]:.4f})")

    print("\n" + "=" * 80)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SNN SLAM v7 Parameter Sweep')
    parser.add_argument('--optuna', action='store_true', help='Use Bayesian optimization (Optuna)')
    parser.add_argument('--quick', action='store_true', help='Fast scan: fewer combos, 1 seed')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--n_seeds', type=int, default=3, help='Seeds per combo')
    parser.add_argument('--n_steps', type=int, default=2000, help='Sim steps per trial')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers (default: all CPUs)')
    parser.add_argument('--output', type=str, default='/Users/lhooz/.openclaw/workspace/sweep_results.json')
    args = parser.parse_args()

    if args.quick:
        # Fast scan: reduce to key params only, 1 seed
        PARAM_RANGES['GATING_STRENGTH'] = [2.0, 4.0, 8.0]
        PARAM_RANGES['MATURITY_GATE'] = [0.75, 0.85, 0.95]
        args.n_seeds = 1
        args.n_steps = 1500
        print("  ⚡ Quick mode: reduced parameter space, 1 seed, 1500 steps")

    workers = args.workers or max(1, cpu_count() - 1)
    print("=" * 70)
    print("  🦊 SNN SLAM v7 — Headless Parameter Sweep")
    print(f"  {'Optuna' if args.optuna else 'Grid Search'} | {args.n_seeds} seeds | "
          f"{args.n_steps} steps | {workers} workers")
    print("=" * 70)

    if args.optuna:
        best = optuna_search(PARAM_RANGES, n_trials=args.n_trials, n_steps=args.n_steps,
                             n_seeds=args.n_seeds, workers=workers)
        results = [best]
    else:
        results = grid_search(PARAM_RANGES, n_seeds=args.n_seeds, n_steps=args.n_steps,
                             workers=workers)

    report_results(results, save_path=args.output)


if __name__ == '__main__':
    main()
