import os
#!/usr/bin/env python3
"""
slam_gate_monitor.py — Monitor SNN SLAM gate statistics over a long run.
Records all 7 gates + maturity + concentrations at each step for post-hoc analysis.
"""
import sys, os, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import src.snn_slam_system as S


def run_monitored_trial(seed: int, n_steps: int = 10000,
                        drift_start: int = 5000,
                        record_every: int = 10) -> dict:
    """
    Run one long trial, recording gate statistics over time.
    record_every: record every N steps to avoid huge output files.
    """
    key = random.PRNGKey(seed)
    env = S.LiveEnvironment(key, chunk_size=n_steps + 100)

    system_ol = S.SNNSLAMSystem(random.PRNGKey(42), n_depth=S.N_DEPTH)
    system_cl = S.SNNSLAMSystem(random.PRNGKey(43), n_depth=S.N_DEPTH)
    system_ol.reset(1); system_cl.reset(1)

    _, _, _, pos0, th0, _ = env.step()
    system_ol.initialize_pose(jnp.array([pos0]), jnp.array([th0]))
    system_cl.initialize_pose(jnp.array([pos0]), jnp.array([th0]))

    # Gate histories (sampled every `record_every` steps)
    gate_keys = [
        'G1_Distinctive', 'G2_Match', 'G3_AntiAlias',
        'G4_Plausible', 'G4b_HeadPlausible', 'G5_NotSelf',
        'G6_TemporalEMA', 'G7_Mature',
        'Maturity_Lvl', 'Conc_Place', 'Conc_Ring',
        'Raw_Vis_Act', 'Raw_Match', 'Jump_Dist',
        'Raw_Conf', 'Final_Conf',
    ]
    gate_hist = {k: [] for k in gate_keys}

    # Trajectory metrics
    gt_pos_hist, imu_pos_hist = [], []
    ol_pos_hist, cl_pos_hist = [], []
    step_hist = []

    x_imu, y_imu, th_imu = pos0[0], pos0[1], th0
    graph_poses, graph_odom_edges = [], []
    node_tof_hits = []
    place_to_node = {}
    loop_closures = []

    KEYFRAME_DIST, KEYFRAME_ANG = 0.15, 0.20
    last_kf_cann = None

    step = 0
    t0 = time.time()

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
        gt_pos_hist.append(gt_pos); ol_pos_hist.append([float(pose_ol[0, 0]), float(pose_ol[0, 1])])
        cl_pos_hist.append([cx, cy]); step_hist.append(step)

        # Record gates every N steps
        if step % record_every == 0:
            for k in gate_keys:
                if k not in debug_gates:
                    v = float('nan')
                else:
                    raw = debug_gates[k]
                    # Handle both scalar and 1D array (batch=1)
                    if hasattr(raw, 'ndim') and raw.ndim > 0:
                        v = float(raw.flatten()[0])
                    else:
                        v = float(raw)
                gate_hist[k].append(v)

        # Keyframe + LC logic (same as stable1)
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
                        if dth < 0.001 and dth < min_dth:
                            min_dth = dth
                            matched_node = prev_nid
                if matched_node is not None:
                    conc_p = float(debug_gates['Conc_Place'][0])
                    conc_r = float(debug_gates['Conc_Ring'][0])
                    w_pos = (maturity * conc_p) * 0.20
                    w_th = (maturity * conc_r) * 0.15
                    loop_closures.append([matched_node, nid, w_pos, w_th])

        step += 1
        if step % 500 == 0:
            elapsed = time.time() - t0
            rate = step / elapsed
            eta = (n_steps - step) / rate / 60
            print(f"  Step {step}/{n_steps} ({rate:.0f} steps/s, ETA={eta:.1f}min)", flush=True)

    # Umeyama ATE
    gt = np.array(gt_pos_hist); ol = np.array(ol_pos_hist); cl = np.array(cl_pos_hist)
    imu_arr = np.array(imu_pos_hist)  # (n_steps-1, 2)
    mn = min(len(gt), len(ol), len(cl), len(imu_arr))
    gt, ol, cl = gt[:mn], ol[:mn], cl[:mn]
    imu_arr = imu_arr[:mn]
    R_ol, t_ol = S.get_optimal_alignment_2d(ol, gt)
    ol_aligned = (R_ol @ ol.T).T + t_ol
    R_cl, t_cl = S.get_optimal_alignment_2d(cl, gt)
    cl_aligned = (R_cl @ cl.T).T + t_cl
    ate_imu = np.mean(np.sqrt((imu_arr[:, 0]-gt[:, 0])**2 + (imu_arr[:, 1]-gt[:, 1])**2))
    ate_ol = np.mean(np.sqrt((ol_aligned[:, 0]-gt[:, 0])**2 + (ol_aligned[:, 1]-gt[:, 1])**2))
    ate_cl = np.mean(np.sqrt((cl_aligned[:, 0]-gt[:, 0])**2 + (cl_aligned[:, 1]-gt[:, 1])**2))
    final_cl = float(np.sqrt((cl_aligned[-1, 0]-gt[-1, 0])**2 + (cl_aligned[-1, 1]-gt[-1, 1])**2))

    # Gate summary statistics
    gate_summary = {}
    for k, vals in gate_hist.items():
        arr = np.array(vals, dtype=float)
        gate_summary[k] = {
            'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
            'min': float(np.min(arr)), 'max': float(np.max(arr)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
        }

    # Per-500-step gate evolution
    step_arr = np.array(step_hist[::record_every])

    result = {
        'seed': seed, 'n_steps': n_steps, 'drift_start': drift_start,
        'ate_imu': float(ate_imu),
        'ate_ol': float(ate_ol), 'ate_cl': float(ate_cl),
        'final_cl': final_cl,
        'n_loop_closures': len(loop_closures),
        'n_nodes': len(graph_poses),
        'gate_summary': gate_summary,
        'step_hist': step_hist[::record_every],
    }
    # Attach sampled gate histories (downsampled)
    for k, vals in gate_hist.items():
        result[f'hist_{k}'] = vals

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--drift', type=int, default=5000)
    parser.add_argument('--record_every', type=int, default=10)
    parser.add_argument('--output', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gate_monitor.json'))
    args = parser.parse_args()

    print(f"Starting gate-monitored run: seed={args.seed}, {args.steps} steps", flush=True)
    t0 = time.time()
    r = run_monitored_trial(args.seed, args.steps, args.drift, args.record_every)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"ATE_IMU={r['ate_imu']:.4f}  ATE_OL={r['ate_ol']:.4f}  ATE_CL={r['ate_cl']:.4f}", flush=True)
    print(f"Final_CL={r['final_cl']:.4f}m  LCs={r['n_loop_closures']}  Nodes={r['n_nodes']}", flush=True)

    print("\nGate Summary (mean ± std):", flush=True)
    for k, s in r['gate_summary'].items():
        print(f"  {k:20s}: {s['mean']:.4f} ± {s['std']:.4f}  [min={s['min']:.4f}, max={s['max']:.4f}]", flush=True)

    with open(args.output, 'w') as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\n💾 Saved: {args.output}", flush=True)
