#!/usr/bin/env python3
"""
slam_variance.py — Trajectory Variance Characterisation

Run SNN SLAM v7 with FIXED default parameters across many random seeds.
Measures how stable the architecture is across different room geometries.

This answers: "How much does ATE vary purely due to trajectory difficulty?"

Usage:
    python slam_variance.py              # 30 seeds, headless
    python slam_variance.py --seeds=10  # quick 10-seed scan
"""
import sys, os, time, json
ROOT = '/Users/lhooz/.openclaw/workspace'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src/stable1'))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import src.stable1.snn_slam_system as S


def run_trial(seed: int, n_steps: int = 2000, drift_start: int = 1000) -> dict:
    """Run one headless trial, return ATE metrics."""
    key = random.PRNGKey(seed)
    env = S.LiveEnvironment(key, chunk_size=n_steps + 100)

    system_ol = S.SNNSLAMSystem(random.PRNGKey(42), n_depth=S.N_DEPTH)
    system_cl = S.SNNSLAMSystem(random.PRNGKey(43), n_depth=S.N_DEPTH)
    system_ol.reset(1); system_cl.reset(1)

    _, _, _, pos0, th0, _ = env.step()
    system_ol.initialize_from_gt(jnp.array([pos0]), jnp.array([th0]))
    system_cl.initialize_from_gt(jnp.array([pos0]), jnp.array([th0]))

    gt_pos_hist, imu_pos_hist = [], []
    ol_pos_hist, cl_pos_hist = [], []
    gt_th_hist, imu_th_hist = [], []

    x_imu, y_imu, th_imu = pos0[0], pos0[1], th0
    graph_poses, graph_odom_edges = [], []
    node_tof_hits = []
    place_to_node = {}
    loop_closures = []

    KEYFRAME_DIST, KEYFRAME_ANG = 0.15, 0.20
    last_kf_cann = None

    step = 0
    while step < n_steps:
        ev_t, kin_t, tof_t, gt_pos, gt_th, _ = env.step()
        ev_j = jnp.array([ev_t]); kin_j = jnp.array([kin_t]); tof_j = jnp.array([tof_t])
        inject_drift = step >= drift_start

        if step > 0:
            omega_b = kin_t[2] + (S.DRIFT_OMEGA if inject_drift else 0.0)
            vx_w = kin_t[0] * np.cos(th_imu) - kin_t[1] * np.sin(th_imu)
            vy_w = kin_t[0] * np.sin(th_imu) + kin_t[1] * np.cos(th_imu)
            x_imu += vx_w * S.DT
            y_imu += vy_w * S.DT
            th_imu = S.wrap_angle(th_imu + omega_b * S.DT)

        pose_ol, _, _ = system_ol.forward_step_open_loop(ev_j, kin_j, tof_j, inject_drift=inject_drift)
        pose_cl, _, _, is_conf, peak_idx_place, debug_gates = system_cl.forward_step(
            ev_j, kin_j, tof_j, inject_drift=inject_drift)

        cx, cy, cth = float(pose_cl[0, 0]), float(pose_cl[0, 1]), float(pose_cl[0, 2])
        gt_pos_hist.append(gt_pos); gt_th_hist.append(gt_th)
        imu_pos_hist.append([x_imu, y_imu]); imu_th_hist.append(th_imu)
        ol_pos_hist.append([float(pose_ol[0, 0]), float(pose_ol[0, 1])])
        cl_pos_hist.append([cx, cy])

        # Keyframe + loop closure (matching stable1 defaults)
        if last_kf_cann is None: last_kf_cann = (cx, cy, cth)
        kf_x, kf_y, kf_th = last_kf_cann
        dx, dy = cx - kf_x, cy - kf_y
        local_dx = dx * np.cos(-kf_th) - dy * np.sin(-kf_th)
        local_dy = dx * np.sin(-kf_th) + dy * np.cos(-kf_th)
        local_dth = (cth - kf_th + np.pi) % (2*np.pi) - np.pi

        is_keyframe = (len(graph_poses) == 0 or
                       np.sqrt(dx**2 + dy**2) > KEYFRAME_DIST or
                       np.abs(local_dth) > KEYFRAME_ANG)

        if is_keyframe:
            nid = len(graph_poses)
            if nid > 0:
                graph_odom_edges.append([local_dx, local_dy, local_dth])
            graph_poses.append([cx, cy, cth])
            last_kf_cann = (cx, cy, cth)
            node_tof_hits.append(tof_t.copy())
            place_id = int(peak_idx_place[0])
            place_to_node.setdefault(place_id, []).append(nid)

            if is_conf[0]:
                matched_node = None
                min_dth = 999.0
                maturity = float(debug_gates['Maturity_Lvl'][0])
                if maturity < 0.85:
                    step += 1
                    continue
                for prev_nid in place_to_node.get(place_id, []):
                    if nid - prev_nid > 10:
                        _, _, nth = graph_poses[prev_nid]
                        dth = abs(S.wrap_angle(cth - nth))
                        if dth < 0.001 and dth < min_dth:  # default RING_SELF_MATCH_THRESH
                            min_dth = dth
                            matched_node = prev_nid
                if matched_node is not None:
                    conc_p = float(debug_gates['Conc_Place'][0])
                    conc_r = float(debug_gates['Conc_Ring'][0])
                    w_pos = (maturity * conc_p) * 0.20
                    w_th = (maturity * conc_r) * 0.15
                    loop_closures.append([matched_node, nid, w_pos, w_th])

        step += 1

    # Umeyama ATE
    gt = np.array(gt_pos_hist); ol = np.array(ol_pos_hist); cl = np.array(cl_pos_hist)
    mn = min(len(gt), len(ol), len(cl))
    gt, ol, cl = gt[:mn], ol[:mn], cl[:mn]

    R_ol, t_ol = S.get_optimal_alignment_2d(ol, gt)
    ol_aligned = (R_ol @ ol.T).T + t_ol
    R_cl, t_cl = S.get_optimal_alignment_2d(cl, gt)
    cl_aligned = (R_cl @ cl.T).T + t_cl

    ate_ol = np.mean(np.sqrt((ol_aligned[:, 0]-gt[:, 0])**2 + (ol_aligned[:, 1]-gt[:, 1])**2))
    ate_cl = np.mean(np.sqrt((cl_aligned[:, 0]-gt[:, 0])**2 + (cl_aligned[:, 1]-gt[:, 1])**2))
    final_ol = np.sqrt((ol_aligned[-1, 0]-gt[-1, 0])**2 + (ol_aligned[-1, 1]-gt[-1, 1])**2)
    final_cl = np.sqrt((cl_aligned[-1, 0]-gt[-1, 0])**2 + (cl_aligned[-1, 1]-gt[-1, 1])**2)
    imu_arr = np.array(imu_pos_hist[:mn])
    ate_imu = np.mean(np.sqrt((imu_arr[:, 0]-gt[:, 0])**2 + (imu_arr[:, 1]-gt[:, 1])**2))

    return {
        'seed': seed, 'n_steps': mn,
        'ate_imu': float(ate_imu),
        'ate_ol': float(ate_ol), 'final_ol': float(final_ol),
        'ate_cl': float(ate_cl), 'final_cl': float(final_cl),
        'n_loop_closures': len(loop_closures),
        'n_nodes': len(graph_poses),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=30, help='Number of random seeds')
    parser.add_argument('--steps', type=int, default=2000, help='Steps per trial')
    parser.add_argument('--output', type=str,
                       default='/Users/lhooz/.openclaw/workspace/variance_results.json')
    args = parser.parse_args()

    n = args.seeds
    t0 = time.time()
    results = []
    for i in range(n):
        seed = 42 + i * 111  # deterministic but varied seeds
        print(f"  Trial {i+1}/{n} (seed={seed})...", flush=True)
        r = run_trial(seed, n_steps=args.steps)
        results.append(r)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (n - i - 1) / rate / 60
        print(f"    ATE_IMU={r['ate_imu']:.4f}  ATE_OL={r['ate_ol']:.4f}  "
              f"ATE_CL={r['ate_cl']:.4f}  LCs={r['n_loop_closures']}  "
              f"Nodes={r['n_nodes']}  [ETA={eta:.1f}min]", flush=True)

    # Summary statistics
    ate_imu = [r['ate_imu'] for r in results]
    ate_ol = [r['ate_ol'] for r in results]
    ate_cl = [r['ate_cl'] for r in results]
    final_cl = [r['final_cl'] for r in results]
    lcs = [r['n_loop_closures'] for r in results]

    print(f"\n{'='*60}")
    print(f"  📊 TRAJECTORY VARIANCE CHARACTERISATION ({n} seeds)")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*60}")
    print(f"  {'ATE IMU (m)':<20} {np.mean(ate_imu):>8.4f} {np.std(ate_imu):>8.4f} "
          f"{np.min(ate_imu):>8.4f} {np.max(ate_imu):>8.4f}")
    print(f"  {'ATE Open-Loop (m)':<20} {np.mean(ate_ol):>8.4f} {np.std(ate_ol):>8.4f} "
          f"{np.min(ate_ol):>8.4f} {np.max(ate_ol):>8.4f}")
    print(f"  {'ATE Closed-Loop (m)':<20} {np.mean(ate_cl):>8.4f} {np.std(ate_cl):>8.4f} "
          f"{np.min(ate_cl):>8.4f} {np.max(ate_cl):>8.4f}")
    print(f"  {'Final Err CL (m)':<20} {np.mean(final_cl):>8.4f} {np.std(final_cl):>8.4f} "
          f"{np.min(final_cl):>8.4f} {np.max(final_cl):>8.4f}")
    print(f"  {'Loop Closures':<20} {np.mean(lcs):>8.1f} {np.std(lcs):>8.1f} "
          f"{np.min(lcs):>8d} {np.max(lcs):>8d}")
    print(f"{'='*60}")

    # Improvement over IMU
    improvement = (np.mean(ate_imu) - np.mean(ate_cl)) / np.mean(ate_imu) * 100
    print(f"  🦊 SNN CL improves over IMU by {improvement:.1f}% (mean ATE)")

    with open(args.output, 'w') as f:
        json.dump({'results': results, 'summary': {
            'n_seeds': n, 'n_steps': args.steps,
            'ate_imu_mean': float(np.mean(ate_imu)), 'ate_imu_std': float(np.std(ate_imu)),
            'ate_ol_mean': float(np.mean(ate_ol)), 'ate_ol_std': float(np.std(ate_ol)),
            'ate_cl_mean': float(np.mean(ate_cl)), 'ate_cl_std': float(np.std(ate_cl)),
            'final_cl_mean': float(np.mean(final_cl)), 'final_cl_std': float(np.std(final_cl)),
            'lcs_mean': float(np.mean(lcs)), 'lcs_std': float(np.std(lcs)),
            'improvement_pct': float(improvement),
        }}, f, indent=2, default=str)
    print(f"\n  💾 Results saved: {args.output}")
    print(f"  🕐 Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
