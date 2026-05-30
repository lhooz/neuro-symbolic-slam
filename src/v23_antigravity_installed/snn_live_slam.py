#!/usr/bin/env python3
"""
live_slam.py — Real-Time Execution for Neuromorphic SLAM Orchestrator
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import io
import time
import collections
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import jax
import jax.numpy as jnp
from jax import random

# ============================================================================
# 2. THE SYSTEM SCRIPT IMPORTS
# ============================================================================
# Assuming your main system file is named `snn_slam_system.py`. 
# We must import all the classes, helper functions, and configs used in run_live_slam.

from snn_slam_system import (
    # Classes
    LiveEnvironment,
    SNNSLAMSystem,
    SpikingOccupancyGrid,
    
    # Helper Functions
    wrap_angle,
    relax_graph,
    get_phase_correlation,
    get_optimal_alignment_2d,
    _draw_room,
    
    # Constants & Hyperparameters
    N_VISION,
    N_DEPTH,
    HDC_CONFIG,
    CANN_SIZES,
    RING_N,
    FOV_DEG,
    DT,
    DRIFT_OMEGA,
    LC_MATURITY,
    BASE_LC_FACTOR  # <--- ADD THIS HERE!
)

# ============================================================================
#  🚀 LIVE SLAM ORCHESTRATOR
# ============================================================================

def get_ray_indices(cx, cy, cth, tof_dists, tof_angles, res=0.10, grid_size=300, offset_m=10.0, max_rays=500): 
    hit_idx, free_idx = [], []
    MAX_VALID_RANGE = 7.4 
    
    for i in range(3):
        d = tof_dists[i]
        
        trace_dist = min(d, MAX_VALID_RANGE)
        for s in range(1, int(trace_dist / res)):
            fx = cx + (s * res) * np.cos(cth + tof_angles[i])
            fy = cy + (s * res) * np.sin(cth + tof_angles[i])
            fix, fiy = int((fx + offset_m) / res), int((fy + offset_m) / res)
            if 0 <= fix < grid_size and 0 <= fiy < grid_size:
                free_idx.append([fix, fiy])

        if d < MAX_VALID_RANGE:
            hx = cx + d * np.cos(cth + tof_angles[i])
            hy = cy + d * np.sin(cth + tof_angles[i])
            ix, iy = int((hx + offset_m) / res), int((hy + offset_m) / res)
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                hit_idx.append([ix, iy])

    # 🌟 JAX FIX: Pad with -1. This guarantees static array shapes for JIT.
    hit_pad = np.full((max_rays, 2), -1, dtype=np.int32)
    free_pad = np.full((max_rays, 2), -1, dtype=np.int32)
    
    n_hit = min(len(hit_idx), max_rays)
    if n_hit > 0: hit_pad[:n_hit] = np.array(hit_idx)[:n_hit]
    
    n_free = min(len(free_idx), max_rays)
    if n_free > 0: free_pad[:n_free] = np.array(free_idx)[:n_free]

    return hit_pad, free_pad

def run_live_slam(key):
    env = LiveEnvironment(key, chunk_size=30000)
    
    print(f"\n 🧠 Initializing Twin SNN SLAM Systems v3 (Vis={N_VISION}, ToF={N_DEPTH})...")
    system_ol = SNNSLAMSystem(random.PRNGKey(42), n_depth=N_DEPTH)
    system_cl = SNNSLAMSystem(random.PRNGKey(43), n_depth=N_DEPTH)
    system_ol.reset(1); system_cl.reset(1)

    _, _, _, pos0, th0, _ = env.step()
    system_ol.initialize_from_gt(jnp.array([pos0]), jnp.array([th0]))
    system_cl.initialize_from_gt(jnp.array([pos0]), jnp.array([th0]))

    history = collections.defaultdict(list)
    x_imu, y_imu, th_imu = pos0[0], pos0[1], th0
    
    live_drift_start = 1000
    
    # 🌟 V4.1: Graph Data Structures & Keyframing
    graph_poses = []
    graph_odom_edges = []
    node_tof_hits = []       
    loop_closures = []
    loop_offsets_list = []  # 🌟 NEW: Track Relative Transforms
    loop_weights_list = []  # 🌟 NEW: Track Confidence Springs
    
    # 🌟 NEW: The JAX-Native Memory Bank (Replaces the Python Dictionary)
    # Holds a 512-dim barcode for up to 15000 nodes.
    MAX_MAP_NODES = 15000  
    memory_bank = np.zeros((MAX_MAP_NODES, HDC_CONFIG["num_bits"]), dtype=np.float32)
    stdp_memory_bank = np.zeros((MAX_MAP_NODES, 256, N_VISION * 2), dtype=np.float32) # 🌟 NEW: Synaptic Snapshots
    
    MAX_LOOPS = 200
    KEYFRAME_DIST = 0.30     # Add node every xxcm
    KEYFRAME_ANG = 0.30      # Or every xx radians
    last_kf_cann = None      # Tracks the CANN state at the last keyframe
    lc_refractory_timer = 0

    # ---------------------------------------------------------
    # 🎨 SETUP LIVE PLOTTING (Upgraded to Overlapped UI)
    # ---------------------------------------------------------
    plt.ion()
    fig = plt.figure(figsize=(18, 10)) 
    
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1.5, 1.0, 1.0])
    ax_map = fig.add_subplot(gs[:2, 0]); ax_map.set_title("Phase 3: Real-Time Map (Live Umeyama Alignment)")
    
    gs_top_right = gs[0, 1].subgridspec(3, 3, height_ratios=[0.4, 0.4, 1.0])
    
    ax_place = fig.add_subplot(gs_top_right[0, :])
    ax_place.set_title("Place Cell Analog Activation (I_place)", fontsize=10)
    ax_place.set_yticks([])
    
    ax_grid_flat = fig.add_subplot(gs_top_right[1, :])
    ax_grid_flat.set_title("Spatial Grid Key (579-dim Flattened)", fontsize=10)
    ax_grid_flat.set_yticks([])
    
    ax_cann1 = fig.add_subplot(gs_top_right[2, 0])
    ax_cann2 = fig.add_subplot(gs_top_right[2, 1])
    ax_cann3 = fig.add_subplot(gs_top_right[2, 2])

    ax_1d = fig.add_subplot(gs[1, 1]); ax_1d.set_title("1D Heading: Memory (Purple) vs CANN Belief (Orange)")

    ax_err = fig.add_subplot(gs[2, :]); ax_err.set_title("Live Absolute Trajectory Error (ATE)")
    ax_err.set_xlabel("Time (s)"); ax_err.set_ylabel("Position Error (meters)")
    ax_err.grid(alpha=0.3, ls='--')
    
    line_err_imu, = ax_err.plot([], [], color='#E74C3C', lw=1.5, ls='--', label="Raw IMU Drift")
    line_err_ol, = ax_err.plot([], [], color='#E67E22', lw=1.5, ls='-.', label="Open-Loop ATE")
    line_err_cl, = ax_err.plot([], [], color='#27AE60', lw=2.5, label="Closed-Loop ATE")
    ax_err.legend(loc='upper left')
    
    _draw_room(ax_map, env.obstacles)
    
    sog = SpikingOccupancyGrid(map_size_m=30.0, res=0.10, offset_m=10.0)
    sog_state = sog.init_state()
    
    sog_img = ax_map.imshow(np.zeros((sog.grid_w, sog.grid_h)), 
                            cmap='magma', origin='lower', 
                            extent=[-sog.offset_m, 30.0 - sog.offset_m, -sog.offset_m, 30.0 - sog.offset_m], 
                            vmin=-0.2, vmax=1.0, alpha=0.8, zorder=2)
    
    fov_poly_gt = plt.Polygon(np.zeros((3, 2)), color='deepskyblue', alpha=0.10, zorder=1)
    ax_map.add_patch(fov_poly_gt)
    tof_rays_gt = [ax_map.plot([], [], color='blue', linestyle='--', lw=2.0, alpha=0.3, zorder=2)[0] for _ in range(3)]

    fov_poly = plt.Polygon(np.zeros((3, 2)), color='gold', alpha=0.15, zorder=3)
    ax_map.add_patch(fov_poly)
    tof_rays = [ax_map.plot([], [], 'r-', lw=2.5, alpha=0.4, zorder=4)[0] for _ in range(3)]
    
    gt_traj, = ax_map.plot([], [], 'b--', lw=1.5, alpha=0.3, label="Ground Truth", zorder=5)
    gt_head, = ax_map.plot([], [], 'bo', ms=6, alpha=0.5, zorder=6)
    
    live_traj, = ax_map.plot([], [], 'g-', lw=1.5, alpha=0.3, label="SNN Belief Trail", zorder=7)
    current_live_traj, = ax_map.plot([], [], color='#27AE60', lw=4.0, label="Current Belief", zorder=9)
    live_head, = ax_map.plot([], [], 'go', ms=10, zorder=10)
    # 🌟 NEW: Initialize the memory node scatter plot
    nodes_scatter, = ax_map.plot([], [], marker='.', color='silver', ms=5, alpha=0.2, label="Memory Nodes", zorder=8)
    # 🌟 NEW 1: Initialize the candidate nodes marker (Large hollow orange circles)
    candidates_scatter, = ax_map.plot([], [], 'o', markeredgecolor='orange', markerfacecolor='none', ms=12, markeredgewidth=2, label="HDC Candidates", zorder=9)
    ax_map.legend(loc='upper right', fontsize=8)

    learning_indicator = ax_map.text(0.02, 0.98, '👁️ Plasticity: ON', 
                                     transform=ax_map.transAxes, fontsize=12, fontweight='bold', color='lime',
                                     verticalalignment='top', 
                                     bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'), zorder=20)

    brain_img = ax_place.imshow(np.zeros((1, HDC_CONFIG["num_bits"])), cmap='magma', aspect='auto', vmin=0, vmax=1.0)
    grid_flat_img = ax_grid_flat.imshow(np.zeros((1, 579)), cmap='viridis', aspect='auto', vmin=0, vmax=1.0)
    
    cann1_img = ax_cann1.imshow(np.zeros((CANN_SIZES[0], CANN_SIZES[0])), origin='lower', cmap='cool', vmin=0, vmax=1.0)
    cann2_img = ax_cann2.imshow(np.zeros((CANN_SIZES[1], CANN_SIZES[1])), origin='lower', cmap='cool', vmin=0, vmax=1.0)
    cann3_img = ax_cann3.imshow(np.zeros((CANN_SIZES[2], CANN_SIZES[2])), origin='lower', cmap='cool', vmin=0, vmax=1.0)
    
    x_ring = np.arange(RING_N)
    line_ring_mem, = ax_1d.plot(x_ring, np.zeros(RING_N), color='#9B59B6', lw=3, label='Memory')
    line_ring_cann, = ax_1d.plot(x_ring, np.zeros(RING_N), color='#E67E22', lw=2, ls='--', label='CANN Belief')
    ax_1d.set_ylim(-0.1, 1.1); ax_1d.set_xticks([])
    ax_1d.legend(loc='upper right')

    plt.show(block=False)

    print("\n 🟢 LIVE SLAM RUNNING! Press Ctrl+C in terminal to stop and generate PNGs.\n")
    
    step = 0
    steps_since_last_lc = 0 
    tof_angles = np.array([-np.pi/4, 0.0, np.pi/4])
    t0 = time.time()

    ui_smooth_th = 0.0
    ui_smooth_t = np.zeros(2)
    history['live_offsets'] = [] 
    
    gif_filename = "snn_live_run.gif"
    print(f" 🎥 Recording live UI to {gif_filename} (Capturing every 15 steps)...")
    gif_writer = imageio.get_writer(gif_filename, fps=10) 

    # 🌟 NEW: Variables for HDC tracking and CLS Autopilot
    active_candidates = []
    surprise = 0.0
    
    try:
        while True:
            ev_t, kin_t, tof_t, gt_pos, gt_th, intensity = env.step()
            ev_jax, kin_jax, tof_jax = jnp.array([ev_t]), jnp.array([kin_t]), jnp.array([tof_t])
            
            steps_since_last_lc += 1 
            
            if lc_refractory_timer > 0:
                lc_refractory_timer -= 1

            inject_drift = step >= live_drift_start

            if step > 0:
                bias = DRIFT_OMEGA if inject_drift else 0.0
                noise_std_dev = 0.005 if inject_drift else 0.0
                random_noise = np.random.normal(0.0, noise_std_dev)
    
                omega_b = kin_t[2] + bias + random_noise
                
                # 🌟 THE FIX: 2nd-Order Midpoint Integration (RK2)
                # Calculates the heading exactly halfway through the timestep
                # to perfectly trace holonomic curves without Euler overshoot!
                th_mid = th_imu + (omega_b * DT) / 2.0
                
                vx_w = kin_t[0] * np.cos(th_mid) - kin_t[1] * np.sin(th_mid)
                vy_w = kin_t[0] * np.sin(th_mid) + kin_t[1] * np.cos(th_mid)
                
                x_imu += vx_w * DT
                y_imu += vy_w * DT
                th_imu = wrap_angle(th_imu + omega_b * DT)

            pose_ol, _, _ = system_ol.forward_step_open_loop(ev_jax, kin_jax, tof_jax, inject_drift=inject_drift)
            
            # 🌟 FIX 2: Use the unified 'surprise' variable!
            autopilot_on = surprise < 0.30
            pose_cl, r_place, r_ring, is_confident, peak_idx_place, debug_gates = system_cl.forward_step(
                ev_jax, kin_jax, tof_jax, inject_drift=inject_drift, autopilot_on=autopilot_on
            )
            
            # Update the Surprise Signal for the next frame
            surprise = 1.0 - float(debug_gates['Raw_Match'][0])
            
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
                if dist > KEYFRAME_DIST or ang > KEYFRAME_ANG or (is_confident[0] and lc_refractory_timer == 0):
                    is_keyframe = True

            # =================================================================
            # 🌟 START OF KEYFRAME LOGIC
            # =================================================================
            if is_keyframe:
                current_node_id = len(graph_poses)
                
                if current_node_id == 0:
                    graph_poses.append([cx, cy, cth])
                else:
                    graph_odom_edges.append([local_dx, local_dy, local_dth])
                    graph_poses.append([cx, cy, cth])
                
                last_kf_cann = (cx, cy, cth)
                node_tof_hits.append(tof_t.copy())

                recalled_place_id = int(peak_idx_place[0])
                
                # 🌟 THE FIX 1: Save the VISUAL Barcode to the database!
                vis_barcode_np = np.array(debug_gates['Visual_Barcode'][0])
                memory_bank[current_node_id] = vis_barcode_np
                
                # 🌟 NEW: Save the Synaptic Snapshot of the STDP working memory!
                stdp_weights_np = np.array(system_cl.vision_state.stdp_state.W[0])
                stdp_memory_bank[current_node_id] = stdp_weights_np

                lc_success = False

                # 🌟 THE LOOP CLOSURE ENGINE (Gated by Surprise!)
                if surprise >= 0.30 and lc_refractory_timer == 0:
                    matched_node = None
                    best_fitness_score = -999.0  # 🌟 FIX: Initialize the tracker here!
                    curr_tof = np.array(tof_t) 
                    is_cross_path = False
                    
                    valid_limit = current_node_id - 15
                    
                    if valid_limit > 0:
                        # 🌟 THE FIX 2: Query the database using what the robot SEES!
                        vis_barcode_np = np.array(debug_gates['Visual_Barcode'][0])
                        overlaps = np.dot(memory_bank[:valid_limit], vis_barcode_np)
                        valid_candidates = np.where(overlaps >= HDC_CONFIG["match_threshold"])[0]
                        
                        # 🌟 NEW 3: Save these IDs so the UI thread can plot them!
                        active_candidates = valid_candidates.tolist()

                        if len(valid_candidates) > 0:
                            valid_candidates = valid_candidates[np.argsort(overlaps[valid_candidates])[::-1]]
                            
                            for candidate_nid in valid_candidates:
                                nx, ny, nth = graph_poses[candidate_nid]
                                
                                # 1. Distance & Heading
                                spatial_dist = np.hypot(cx - nx, cy - ny)
                                dth = abs(wrap_angle(cth - nth))
                                
                                # 2. Local Geometry
                                curr_tof = np.array(tof_t) 
                                mem_tof = np.array(node_tof_hits[candidate_nid])
                                tof_diff = np.sum(np.abs(curr_tof - mem_tof))
                                
                                # =======================================================
                                # 🌟 THE UNIFIED PHYSICAL GATE (LOOSENED TOF)
                                # spatial_dist < 3.0m handles the identical hallway problem.
                                # tof_diff < 1.5m allows up to 50cm of average noise per laser!
                                # =======================================================
                                if dth < 0.30 and spatial_dist < 3.0 and tof_diff < 1.5: 
                                    
                                    # 🌟 THE LOCAL DISAMBIGUATION FITNESS SCORE
                                    vis_score = overlaps[candidate_nid] 
                                    fitness = vis_score - (spatial_dist * 2.0) - (tof_diff * 4.0)
                                    
                                    if fitness > best_fitness_score:
                                        best_fitness_score = fitness
                                        matched_node = int(candidate_nid)
                                    else:
                                        # print(f"   [Candidate {candidate_nid} Rejected]: ToF diff is {tof_diff:.2f}m (Limit: 0.6m).")
                                        pass
                                else:
                                    # print(f"   [Candidate {candidate_nid} Rejected]: Exceeds physical 180-FOV overlap limit (dTh: {dth:.2f} rad).")
                                    pass
                                    
                            if matched_node is None:
                                print(f"\n 🛡️ BOUNCER REJECTED: Found {len(valid_candidates)} visual matches, but ALL failed physical drift/FOV limits!")
                        else:
                            # best_overlap_score = np.max(overlaps) if len(overlaps) > 0 else 0.0
                            # print(f"\n 🛡️ BOUNCER REJECTED: Vision confident, but highest Visual Barcode overlap was {best_overlap_score:.1f}/{HDC_CONFIG['active_spikes_k']} spikes!")
                            pass

                    if matched_node is not None:
                        maturity = float(debug_gates['Maturity_Lvl'][0])
                        if maturity >= LC_MATURITY: 
                            matched_x, matched_y, matched_th = graph_poses[matched_node]
                            mem_tof = np.array(node_tof_hits[matched_node])
                            tof_diff = np.sum(np.abs(curr_tof - mem_tof))
                            
                            abort_lc = False
                            
                            if not abort_lc:
                                input_csnn = jnp.array(debug_gates["Debug_Input_CSNN"][0])
                                mem_csnn_ring = jnp.array(debug_gates["Debug_Mem_CSNN_Ring"][0])
                                
                                # 🌟 THE UPGRADE 1: Grab the current visual sparsity
                                vis_act = float(debug_gates['Raw_Vis_Act'][0])
                                
                                # 🛡️ THE SPARSITY GATE (Threshold set to 0.08)
                                if vis_act < 0.08:
                                    # Too sparse for FFT. Trust the memory node's base heading.
                                    pixel_shift = 0.0
                                    sub_pixel_th = 0.0
                                else:
                                    # Normal rich-texture Phase Correlation
                                    r_real = np.array(get_phase_correlation(input_csnn, mem_csnn_ring))
                                    N_VIS = len(input_csnn)
                                    N_PAD = len(r_real)  # 🌟 FIX 1: Get the true padded length (e.g., 512)
                                    
                                    search_radius = 64
                                    r_masked = np.zeros_like(r_real)
                                    r_masked[:search_radius + 1] = r_real[:search_radius + 1] 
                                    r_masked[-search_radius:] = r_real[-search_radius:] 
                                    
                                    peak_idx = np.argmax(r_masked)
                                    
                                    # 🌟 FIX 2: Modulo wrap using the padded length, not N_VIS
                                    y1 = r_real[(peak_idx - 1) % N_PAD]
                                    y2 = r_real[peak_idx] 
                                    y3 = r_real[(peak_idx + 1) % N_PAD]
                                    
                                    # 🌟 FIX 3: Subtract epsilon because the denominator is strictly negative
                                    sub_pixel_offset = (y1 - y3) / (2.0 * (y1 - 2.0 * y2 + y3) - 1e-8)
                                    
                                    # 🌟 FIX 4: Negative integer shifts must subtract the padded length
                                    shift_int = peak_idx if peak_idx <= search_radius else peak_idx - N_PAD
                                    
                                    pixel_shift = shift_int + sub_pixel_offset 
                                    pixel_ang_res = np.radians(FOV_DEG) / N_VISION
                                    
                                    sub_pixel_th = float(np.clip(-pixel_shift * pixel_ang_res, -0.30, 0.30))

                                true_burned_angle = float(debug_gates['Peak_Theta_Burn'][0])
                                vision_th = float(wrap_angle(true_burned_angle + sub_pixel_th))
                                lc_offset_th = float(wrap_angle(vision_th - matched_th))
                                r_idx = int(debug_gates['Peak_Ring'][0])

                                if not abort_lc and not is_cross_path:
                                    
                                    # =================================================================
                                    # 🛡️ DEFENSE LAYER 1: The Absolute Visual Gate (Your Idea)
                                    # Prevent violent graph snaps. If the total calculated shift is 
                                    # > 0.30 rad, it's too extreme to be safe.
                                    # =================================================================
                                    if abs(lc_offset_th) > 0.30:
                                        print(f"\n ⚠️ Aborted: Visual shift too extreme! lc_offset_th = {lc_offset_th:.2f} rad. Preventing violent graph snap.")
                                        matched_node = None
                                        abort_lc = True
                                        
                                    # =================================================================
                                    # 🛡️ DEFENSE LAYER 2: The Relative Reality Check
                                    # If the visual shift disagrees with the physical IMU by > 0.20 rads,
                                    # the camera is hallucinating on a repeating texture!
                                    # =================================================================
                                    if not abort_lc:
                                        dth_physical = float(wrap_angle(cth - matched_th))
                                        vision_contradiction = abs(float(wrap_angle(lc_offset_th - dth_physical)))
                                        
                                        if vision_contradiction > 0.20:  
                                            print(f"\n ⚠️ Aborted: Visual Hallucination! Vision claims offset {lc_offset_th:.2f} rad, but IMU claims {dth_physical:.2f} rad.")
                                            matched_node = None
                                            abort_lc = True

                                if not abort_lc and not is_cross_path:
                                    # 🌟 THE INVERSION FIX: Correct Physical Laser Mapping
                                    # tof_angles = [-45(Right), 0(Center), +45(Left)]
                                    # Therefore 0 is Right, 2 is Left!
                                    delta_R = curr_tof[0] - mem_tof[0]
                                    delta_L = curr_tof[2] - mem_tof[2]
                                    rotational_signature = delta_L - delta_R
                                    
                                    if abs(lc_offset_th) > 0.08:
                                        if lc_offset_th > 0 and rotational_signature < -0.10:
                                            print(f"\n ⚠️ Aborted: Directional Contradiction! Vision claims LEFT (+{lc_offset_th:.2f}), but lasers swept RIGHT ({rotational_signature:.2f}).")
                                            matched_node = None
                                            abort_lc = True
                                        elif lc_offset_th < 0 and rotational_signature > 0.10:
                                            print(f"\n ⚠️ Aborted: Directional Contradiction! Vision claims RIGHT ({lc_offset_th:.2f}), but lasers swept LEFT (+{rotational_signature:.2f}).")
                                            matched_node = None
                                            abort_lc = True

                            if not abort_lc:
                                if is_cross_path:
                                    lc_offset_x = 0.0
                                    lc_offset_y = 0.0
                                else:
                                    curr_tof = np.array(tof_t)
                                    mem_tof = np.array(node_tof_hits[matched_node])
                                        
                                    # ==================================================
                                    # 🌟 THE EGO-CENTRIC NORMAL FIX
                                    # Calculate surface normals directly from the 
                                    # local memory scan, bypassing the noisy global SOG!
                                    # ==================================================
                                    
                                    # 1. Convert pristine memory polar coordinates to Local Cartesian
                                    pts_x = mem_tof * np.cos(tof_angles)
                                    pts_y = mem_tof * np.sin(tof_angles)
                                    
                                    # 2. Find the wall vectors between the laser hits (Right->Center, Center->Left)
                                    v1_x, v1_y = pts_x[1] - pts_x[0], pts_y[1] - pts_y[0]
                                    v2_x, v2_y = pts_x[2] - pts_x[1], pts_y[2] - pts_y[1]
                                    
                                    # 3. Calculate normals by rotating wall vectors 90 degrees toward the robot (-y, x)
                                    n1_x, n1_y = -v1_y, v1_x
                                    n2_x, n2_y = -v2_y, v2_x
                                    
                                    # 4. Normalize the vectors
                                    norm1 = np.hypot(n1_x, n1_y) + 1e-8
                                    n1_x, n1_y = n1_x / norm1, n1_y / norm1
                                    
                                    norm2 = np.hypot(n2_x, n2_y) + 1e-8
                                    n2_x, n2_y = n2_x / norm2, n2_y / norm2
                                    
                                    # 5. Assign normals to the 3 rays (Center gets the average normal)
                                    normals_x = [n1_x, (n1_x + n2_x) / 2.0, n2_x]
                                    normals_y = [n1_y, (n1_y + n2_y) / 2.0, n2_y]
                                    
                                    # Re-normalize the center average
                                    norm_c = np.hypot(normals_x[1], normals_y[1]) + 1e-8
                                    normals_x[1] /= norm_c
                                    normals_y[1] /= norm_c

                                    A = np.stack([normals_x, normals_y], axis=1)
                                    P_mx = mem_tof * np.cos(tof_angles)
                                    P_my = mem_tof * np.sin(tof_angles)
                                    
                                    P_cx = curr_tof * np.cos(tof_angles + lc_offset_th)
                                    P_cy = curr_tof * np.sin(tof_angles + lc_offset_th)
                                    
                                    b = (P_mx - P_cx) * np.array(normals_x) + (P_my - P_cy) * np.array(normals_y)
                                        
                                    valid_mask = (curr_tof < 7.4) & (mem_tof < 7.4) & (np.abs(b) < 0.40)
                                        
                                    if np.sum(valid_mask) >= 2:
                                        A_valid = A[valid_mask]
                                        b_valid = b[valid_mask]
                                        
                                        AtA = A_valid.T @ A_valid
                                        det_AtA = np.linalg.det(AtA)
                                        
                                        # 🌟 Loosen matrix singularity check from 0.20 to 0.10
                                        if det_AtA < 0.10: 
                                            print(f"\n ⚠️ Aborted: Matrix Singularity! Det(AtA) = {det_AtA:.3f}. Geometry is too flat/ambiguous to solve.")
                                            matched_node = None
                                            abort_lc = True
                                            
                                        if not abort_lc:
                                            Atb = A_valid.T @ b_valid
                                            lambda_damp = 0.10  
                                            AtA_damped = AtA + lambda_damp * np.eye(2)
                                            offset_xy = np.linalg.solve(AtA_damped, Atb)
                                            
                                            lc_offset_x = float(np.clip(offset_xy[0], -0.75, 0.75))
                                            lc_offset_y = float(np.clip(offset_xy[1], -0.75, 0.75))
                                            
                                            # Verify the shift explains ALL in-range rays
                                            in_range = (curr_tof < 7.4) & (mem_tof < 7.4)
                                            explained_b = lc_offset_x * np.array(normals_x) + lc_offset_y * np.array(normals_y)
                                            residuals = np.abs(b - explained_b)
                                            
                                            # 🌟 Loosen the maximum allowed residual from 0.20 to 0.50 meters!
                                            if np.any(in_range & (residuals > 0.50)):
                                                print(f"\n ⚠️ Aborted: Perceptual Aliasing! Geometry shift couldn't explain all rays (Max residual: {np.max(residuals[in_range]):.2f}m).")
                                                matched_node = None
                                                abort_lc = True
                                    else:
                                        print(f"\n ⚠️ Aborted Loop Closure ({current_node_id} -> {matched_node}): Lasers out of range or residuals too high.")
                                        matched_node = None
                                        abort_lc = True

                            if not abort_lc:
                                conc_p   = float(debug_gates['Conc_Place'][0])
                                conc_r   = float(debug_gates['Conc_Ring'][0])
                                
                                if is_cross_path:
                                    w_pos = 0.0  
                                    lc_type_str = "1-DOF CROSS-PATH (Heading Only)"
                                else:
                                    w_pos = (maturity * conc_p) * 0.2 * BASE_LC_FACTOR
                                    lc_type_str = "3-DOF PARALLEL (Pos + Heading)"
                                    
                                w_th  = (maturity * conc_r) * 0.15 * BASE_LC_FACTOR
                                
                                print(f"\n\n 💥 LOOP CLOSURE SNAP [{lc_type_str}] (Node {current_node_id} -> Node {matched_node})!")
                                print(f"  ↳ CANN Belief : X={cx:.2f}m, Y={cy:.2f}m, Th={cth:.2f} rad")
                                print(f"  ↳ Memory Node : X={matched_x:.2f}m, Y={matched_y:.2f}m, Th={matched_th:.2f} rad")
                                print(f"  ↳ Raw ToF Now : R={curr_tof[0]:.3f}m, C={curr_tof[1]:.3f}m, L={curr_tof[2]:.3f}m")
                                print(f"  ↳ Raw ToF Mem : R={mem_tof[0]:.3f}m, C={mem_tof[1]:.3f}m, L={mem_tof[2]:.3f}m")
                                print(f"  ↳ Calc'd Shift: dX={lc_offset_x:.3f}m, dY={lc_offset_y:.3f}m, dTh={lc_offset_th:.3f} rad")
                                print(f"  ↳ Delta (Err) : dX={(cx-matched_x):.2f}m, dY={(cy-matched_y):.2f}m, dTh={wrap_angle(cth-matched_th):.2f} rad")
                                print(f"  ↳ Tension Wgt : W_Pos={w_pos:.2f}, W_Th={w_th:.2f}\n")
                                
                                current_csnn = np.array(debug_gates["Debug_Input_CSNN"][0])
                                current_stdp = np.array(debug_gates["Debug_Input_STDP"][0])
                                pixel_shift_int = int(-np.round(pixel_shift))
                                
                                aligned_csnn = np.zeros_like(current_csnn)
                                aligned_stdp = np.zeros_like(current_stdp)
                                fov_mask = np.zeros_like(current_csnn)
                                
                                if pixel_shift_int > 0:
                                    aligned_csnn[pixel_shift_int:] = current_csnn[:-pixel_shift_int]
                                    aligned_stdp[pixel_shift_int:] = current_stdp[:-pixel_shift_int]
                                    fov_mask[pixel_shift_int:] = 1.0
                                elif pixel_shift_int < 0:
                                    aligned_csnn[:pixel_shift_int] = current_csnn[-pixel_shift_int:]
                                    aligned_stdp[:pixel_shift_int] = current_stdp[-pixel_shift_int:]
                                    fov_mask[:pixel_shift_int] = 1.0
                                else:
                                    aligned_csnn = current_csnn.copy()
                                    aligned_stdp = current_stdp.copy()
                                    fov_mask[:] = 1.0
                                    
                                # ==================================================
                                # 🌟 THE FIX: Full HDC Barcode Memory Update
                                # Update all 16 active columns of the HDC memory simultaneously!
                                # ==================================================
                                recovered_barcode = np.array(debug_gates["Recovered_Spatial_Barcode"][0])
                                
                                system_cl.place_state = system_cl.place.apply_post_relaxation_update(
                                    system_cl.place_state,
                                    jnp.array([recovered_barcode]), 
                                    jnp.array([r_idx]),
                                    jnp.array([aligned_csnn]),
                                    jnp.array([aligned_stdp]),
                                    jnp.array([fov_mask]),
                                    ring_lr=0.05
                                )

                                # 🌟 THE CEREBELLUM FIX: Calculate the true hardware drift!
                                # Drift = Current IMU Heading - Corrected Visual Heading
                                accumulated_heading_error = float(wrap_angle(cth - vision_th))
                                time_elapsed_sec = steps_since_last_lc * DT
                                
                                # 🌟 THE UPGRADE 2: CEREBELLUM SPARSITY GATE
                                if time_elapsed_sec > 2.0:
                                    if vis_act >= 0.08:
                                        system_cl.calibrate_cerebellum(accumulated_heading_error, time_elapsed_sec)
                                    else:
                                        print(f"\n 🛡️ CEREBELLUM PROTECTED: Vision too sparse (Act: {vis_act:.2f}) to safely train drift bias!")
                                
                                steps_since_last_lc = 0
                                loop_closures.append([matched_node, current_node_id])
                                loop_offsets_list.append([lc_offset_x, lc_offset_y, lc_offset_th])
                                loop_weights_list.append([w_pos, w_th])

                                lc_success = True
                                lc_refractory_timer = 20

                # ==================================================
                # 🌟 THE ABORT PENALTY & AUTOMATIC PRUNING FIX
                # ==================================================
                if is_confident[0] and not lc_success:
                    lc_refractory_timer = 10 
                    
                    if dist < KEYFRAME_DIST and ang < KEYFRAME_ANG and current_node_id > 0:
                        print(f"  ✂️ PRUNING: Deleting aborted micro-node {current_node_id} to prevent Graph Spaghetti!")
                        graph_poses.pop()
                        graph_odom_edges.pop()
                        node_tof_hits.pop()
                        memory_bank[current_node_id] = 0.0
                        last_kf_cann = (float(graph_poses[-1][0]), float(graph_poses[-1][1]), float(graph_poses[-1][2]))
                        is_keyframe = False

            # ==================================================
            # 🌟 THE PHANTOM OPTIMIZER FIX
            # This must trigger EVERY time a keyframe is successfully added!
            # ==================================================
            if is_keyframe:
                MAX_NODES = 1000
                WINDOW_SIZE = 800
                
                N_total = len(graph_poses)
                start_idx = max(0, N_total - WINDOW_SIZE)
                
                active_nodes = graph_poses[start_idx:]
                num_active = len(active_nodes)
                active_odom = graph_odom_edges[start_idx:] if start_idx > 0 else graph_odom_edges
                    
                mapped_closures, mapped_offsets, mapped_weights = [], [], []
                frozen_anchors = []
                
                recent_lcs = loop_closures[-MAX_LOOPS:]
                recent_offs = loop_offsets_list[-MAX_LOOPS:]
                recent_wgts = loop_weights_list[-MAX_LOOPS:]
                
                for i in range(len(recent_lcs)):
                    target_id, current_id = recent_lcs[i]
                    if current_id < start_idx: 
                        continue 
                        
                    mapped_current = current_id - start_idx
                    if target_id >= start_idx:
                        mapped_target = target_id - start_idx
                    else:
                        mapped_target = num_active + len(frozen_anchors)
                        frozen_anchors.append(graph_poses[target_id])
                        
                    mapped_closures.append([mapped_target, mapped_current])
                    mapped_offsets.append(recent_offs[i])
                    mapped_weights.append(recent_wgts[i])
                    
                padded_poses = np.zeros((MAX_NODES, 3), dtype=np.float32)
                padded_poses[:num_active] = np.array(active_nodes)
                if frozen_anchors:
                    padded_poses[num_active:num_active+len(frozen_anchors)] = np.array(frozen_anchors)
                    
                padded_odom = np.zeros((MAX_NODES - 1, 3), dtype=np.float32)
                if num_active > 1:
                    padded_odom[:num_active-1] = np.array(active_odom)
                    
                odom_mask = np.zeros(MAX_NODES - 1, dtype=np.float32)
                if num_active > 1:
                    odom_mask[:num_active-1] = 1.0
                    
                is_frozen = np.zeros(MAX_NODES, dtype=bool)
                is_frozen[0] = True 
                if frozen_anchors:
                    is_frozen[num_active:num_active+len(frozen_anchors)] = True
                    
                lc_padded = np.zeros((MAX_LOOPS, 2), dtype=np.int32)
                lc_offsets_padded = np.zeros((MAX_LOOPS, 3), dtype=np.float32)
                lc_weights_padded = np.zeros((MAX_LOOPS, 2), dtype=np.float32)
                lc_mask = np.zeros(MAX_LOOPS, dtype=np.float32)
                
                num_lc = min(len(mapped_closures), MAX_LOOPS)
                if num_lc > 0:
                    lc_padded[:num_lc] = np.array(mapped_closures[-num_lc:])
                    lc_offsets_padded[:num_lc] = np.array(mapped_offsets[-num_lc:])
                    lc_weights_padded[:num_lc] = np.array(mapped_weights[-num_lc:])
                    lc_mask[:num_lc] = 1.0
                
                relaxed_p = relax_graph(
                    jnp.array(padded_poses), jnp.array(padded_odom), jnp.array(odom_mask),
                    jnp.array(lc_padded), jnp.array(lc_offsets_padded), 
                    jnp.array(lc_weights_padded), jnp.array(lc_mask),
                    jnp.array(is_frozen)
                )
                
                relaxed_active = np.array(relaxed_p[:num_active]).tolist()
                graph_poses[start_idx:] = relaxed_active
                
                corr_x, corr_y, corr_th = graph_poses[-1]
                
                # ==================================================
                # 🌟 THE FLAWLESS INTEGRATION FIX
                # Only destroy the continuous SNN membrane potentials if 
                # we ACTUALLY snapped to a new Loop Closure coordinate!
                # ==================================================
                if lc_success:
                    system_cl.initialize_from_gt(jnp.array([[corr_x, corr_y]]), jnp.array([corr_th]))
                    
                    # 🌟 NEW: MEMORY RECONSOLIDATION (The Synaptic Download)
                    recovered_stdp_weights = jnp.array([stdp_memory_bank[matched_node]])
                    system_cl.inject_stdp_memory(recovered_stdp_weights)
                
                last_kf_cann = (corr_x, corr_y, corr_th)

                # ==================================================
                # 🌟 FIXED: ONLY PLOT AND PRINT ON TRUE SNAP SUCCESS
                # ==================================================
                if lc_success and "Debug_Input_CSNN" in debug_gates:
                    input_csnn = np.array(debug_gates["Debug_Input_CSNN"][0])
                    input_stdp = np.array(debug_gates["Debug_Input_STDP"][0])
                    input_tof  = np.array(debug_gates["Debug_Input_ToF"][0]) 
                    
                    mem_csnn_place = np.array(debug_gates["Debug_Mem_CSNN"][0])
                    mem_stdp_place = np.array(debug_gates["Debug_Mem_STDP"][0])
                    mem_tof_place  = np.array(debug_gates["Debug_Mem_ToF"][0]) 
                    i_place = np.array(debug_gates["Debug_I_Place"][0])
                    
                    mem_csnn_ring = np.array(debug_gates["Debug_Mem_CSNN_Ring"][0])
                    mem_stdp_ring = np.array(debug_gates["Debug_Mem_STDP_Ring"][0])
                    mem_tof_ring  = np.array(debug_gates["Debug_Mem_ToF_Ring"][0]) 
                    i_ring  = np.array(debug_gates["Debug_I_Ring"][0]) 
                    
                    match_score = float(debug_gates["Raw_Match"][0])
                    
                    fig_debug, axs = plt.subplots(4, 2, figsize=(18, 13), gridspec_kw={'hspace': 0.5, 'wspace': 0.2})
                    
                    # ── LEFT COLUMN: PLACE CELL (WHERE) ──
                    axs[0, 0].plot(input_csnn, label="Camera Reality", color="#3498DB", lw=2)
                    axs[0, 0].plot(mem_csnn_place, label="Place Memory", color="#E67E22", linestyle="--", lw=2)
                    axs[0, 0].set_title(f"CSNN Place Anchor | Match: {match_score:.2f}", fontweight='bold')
                    axs[0, 0].legend(loc="upper right", fontsize=8); axs[0, 0].grid(alpha=0.3, linestyle="--")
                    
                    axs[1, 0].plot(input_tof, label="ToF Reality", color="#1ABC9C", lw=2)
                    axs[1, 0].plot(mem_tof_place, label="ToF Memory", color="#E74C3C", linestyle="--", lw=2)
                    axs[1, 0].set_title("ToF Depth Population Code (192 dims)", fontweight='bold')
                    axs[1, 0].legend(loc="upper right", fontsize=8); axs[1, 0].grid(alpha=0.3, linestyle="--")

                    axs[2, 0].plot(input_stdp, label="Camera Reality", color="#9B59B6", lw=2)
                    axs[2, 0].plot(mem_stdp_place, label="Place Memory", color="#2ECC71", linestyle="--", lw=2)
                    axs[2, 0].set_title("STDP Place Plasticity", fontweight='bold')
                    axs[2, 0].legend(loc="upper right", fontsize=8); axs[2, 0].grid(alpha=0.3, linestyle="--")
                    
                    axs[3, 0].plot(i_place, color="#E74C3C", lw=2)
                    
                    # 🌟 THE FIX: Plot the Top 16 Spikes of the HDC Barcode!
                    top_k_indices = np.argsort(i_place)[-HDC_CONFIG["active_spikes_k"]:]
                    for idx in top_k_indices:
                        axs[3, 0].axvline(idx, color='gold', linestyle='--', lw=1, alpha=0.5)
                        
                    axs[3, 0].set_title(f"HDC Spatial Barcode Activation (Top {HDC_CONFIG['active_spikes_k']} Spikes)", fontweight='bold')
                    axs[3, 0].grid(alpha=0.3, linestyle="--")

                    # ── RIGHT COLUMN: RING CELL (WHICH WAY) ──
                    axs[0, 1].plot(input_csnn, label="Camera Reality", color="#3498DB", lw=2)
                    axs[0, 1].plot(mem_csnn_ring, label="Ring Memory", color="#E67E22", linestyle="--", lw=2)
                    axs[0, 1].set_title(f"CSNN Ring Anchor (Conjunctive)", fontweight='bold')
                    axs[0, 1].legend(loc="upper right", fontsize=8); axs[0, 1].grid(alpha=0.3, linestyle="--")
                    
                    axs[1, 1].plot(input_tof, label="ToF Reality", color="#1ABC9C", lw=2)
                    axs[1, 1].plot(mem_tof_ring, label="Ring Memory", color="#E74C3C", linestyle="--", lw=2)
                    axs[1, 1].set_title("ToF Ring Geometry (Conjunctive)", fontweight='bold')
                    axs[1, 1].legend(loc="upper right", fontsize=8); axs[1, 1].grid(alpha=0.3, linestyle="--")

                    axs[2, 1].plot(input_stdp, label="Camera Reality", color="#9B59B6", lw=2)
                    axs[2, 1].plot(mem_stdp_ring, label="Ring Memory", color="#2ECC71", linestyle="--", lw=2)
                    axs[2, 1].set_title("STDP Ring Plasticity (Conjunctive)", fontweight='bold')
                    axs[2, 1].legend(loc="upper right", fontsize=8); axs[2, 1].grid(alpha=0.3, linestyle="--")
                    
                    axs[3, 1].plot(i_ring, color="#F39C12", lw=2)
                    peak_ring = np.argmax(i_ring)
                    axs[3, 1].axvline(peak_ring, color='blue', linestyle='--', lw=2, label=f"Winning Ring ({peak_ring})")
                    axs[3, 1].set_title("Heading Soma Activation", fontweight='bold')
                    axs[3, 1].legend(loc="upper right", fontsize=8); axs[3, 1].grid(alpha=0.3, linestyle="--")
                    
                    debug_filename = f"debug_csnn_step.png"
                    plt.savefig(debug_filename, bbox_inches='tight', dpi=100)
                    plt.close(fig_debug)
                    print(f"  ↳ 💾 Auto-saved multi-stream brain scan to {debug_filename}\n")

            # =================================================================
            # 🌟 END OF KEYFRAME LOGIC
            # =================================================================

            # --- Outside the keyframe block ---       
            if is_keyframe:
                local_dx, local_dy, local_dth = 0.0, 0.0, 0.0
                
            active_node_id = len(graph_poses) - 1
            history['frame_to_node'].append(active_node_id)
            history['live_offsets'].append((local_dx, local_dy, local_dth))
            
            # ... (The rest of the rendering/UI code remains identical down to the end of the while loop)
            
            smooth_cl_poses = []
            
            node_start_indices = [0] * len(graph_poses)
            for i, nid in enumerate(history['frame_to_node']):
                if i == 0 or history['frame_to_node'][i-1] != nid:
                    node_start_indices[nid] = i

            for i, nid in enumerate(history['frame_to_node']):
                rx, ry, rth = graph_poses[nid]
                
                ldx, ldy, ldth = history['live_offsets'][i]
                gdx_raw = ldx * np.cos(rth) - ldy * np.sin(rth)
                gdy_raw = ldx * np.sin(rth) + ldy * np.cos(rth)
                
                if nid + 1 < len(graph_poses):
                    nx, ny, nth = graph_poses[nid + 1]
                    
                    start_i = node_start_indices[nid]
                    end_i = node_start_indices[nid + 1]
                    steps = max(1, end_i - start_i)
                    alpha = (i - start_i) / steps
                    
                    edge_dx, edge_dy, edge_dth = graph_odom_edges[nid]
                    
                    tip_gdx = edge_dx * np.cos(rth) - edge_dy * np.sin(rth)
                    tip_gdy = edge_dx * np.sin(rth) + edge_dy * np.cos(rth)
                    
                    gap_x = nx - (rx + tip_gdx)
                    gap_y = ny - (ry + tip_gdy)
                    gap_th = wrap_angle(nth - (rth + edge_dth))
                    
                    final_x = rx + gdx_raw + (alpha * gap_x)
                    final_y = ry + gdy_raw + (alpha * gap_y)
                    final_th = wrap_angle(rth + ldth + (alpha * gap_th))
                    
                    smooth_cl_poses.append(np.array([final_x, final_y, final_th]))
                    
                else:
                    smooth_cl_poses.append(np.array([rx + gdx_raw, ry + gdy_raw, wrap_angle(rth + ldth)]))
                
            history['cl_pose'] = smooth_cl_poses
            history['raw_cann'].append(np.array(pose_cl[0]))
            
            rel_cx, rel_cy, rel_cth = smooth_cl_poses[-1]

            history['gt_pos'].append(gt_pos); history['gt_th'].append(gt_th)
            history['imu_pos'].append([x_imu, y_imu]); history['imu_th'].append(th_imu)
            history['ol_pose'].append(np.array(pose_ol[0]))
            history['conf'].append(float(is_confident[0]))
            
            history['pc_act'].append(np.array(r_place[0]))
            history['ring_mem_act'].append(np.array(r_ring[0])) 
            history['cann_act'].append(np.array(system_cl.pose.get_state_flat()[0]))
            history['ring_cann_act'].append(np.array(system_cl.pose.get_ring_activity()[0])) 
            history['intensities'].append(intensity)

            hit_idx, free_idx = get_ray_indices(
                rel_cx, rel_cy, rel_cth, 
                tof_t, tof_angles, res=sog.res, grid_size=sog.grid_w, offset_m=sog.offset_m
            )
            sog_state = sog.update(sog_state, jnp.array(hit_idx), jnp.array(free_idx))

            if step % 15 == 0:
                d = debug_gates
                    
                is_learning = abs(kin_t[2]) < system_cl.place.dynamic_saccade_thresh
                    
                if is_learning:
                    learning_indicator.set_text('[ PLASTICITY: ON ]')
                    learning_indicator.set_color('lime')
                else:
                    learning_indicator.set_text('[ PLASTICITY: OFF ]')
                    learning_indicator.set_color('#E74C3C') 

                # 🌟 THE UPGRADED PRINT LOGIC
                conf = bool(d['Final_Conf'][0])
                maturity = float(d['Maturity_Lvl'][0])
                vis_act = float(d['Raw_Vis_Act'][0])
                match_val = float(d['Raw_Match'][0])
                c_place = float(d['Conc_Place'][0])
                c_ring = float(d['Conc_Ring'][0])    # <--- ADD THIS
                tof_score = float(d['ToF_Score'][0])
                    
                # Determine the status code
                if surprise >= 0.30:
                    status = "🔴 SURPRISED"
                elif is_learning:
                    status = "🔵 AUTOPILOT"
                else:
                    status = "🟡 SACCADE"

                print(f"\r--- Step {step:04d} | Status: {status:10s} | Surprise: {surprise:.2f} | P_Match: {match_val:.2f} | ToFMatch: {tof_score:.2f} | P_Conc: {c_place:.2f} | R_Conc: {c_ring:.2f} | Mat: {maturity:.2f} ---        ", end="", flush=True)

                gt_arr = np.array(history['gt_pos'])
                gt_traj.set_data(gt_arr[:, 0], gt_arr[:, 1])
                gt_head.set_data([gt_pos[0]], [gt_pos[1]])
                
                cl_arr = np.array(history['cl_pose'])
                ol_arr = np.array(history['ol_pose'])
                imu_arr = np.array(history['imu_pos'])
                
                R_cl, t_cl = get_optimal_alignment_2d(cl_arr[:, :2], gt_arr)
                R_ol, t_ol = get_optimal_alignment_2d(ol_arr[:, :2], gt_arr)
                aligned_cl_path = (R_cl @ cl_arr[:, :2].T).T + t_cl
                aligned_ol_path = (R_ol @ ol_arr[:, :2].T).T + t_ol
                delta_theta = np.arctan2(R_cl[1, 0], R_cl[0, 0])

                # 🌟 TURNED OFF UMEYAMA ALIGNMENT
                # R_cl, t_cl = np.eye(2), np.zeros(2)
                # R_ol, t_ol = np.eye(2), np.zeros(2)
                # aligned_cl_path = cl_arr[:, :2]
                # aligned_ol_path = ol_arr[:, :2]
                # delta_theta = 0.0
                
                live_traj.set_data(aligned_cl_path[:, 0], aligned_cl_path[:, 1])

                # 🌟 NEW: Extract, align, and plot the memory nodes
                if len(graph_poses) > 0:
                    raw_nodes = np.array(graph_poses)
                    # Apply the exact same Umeyama transform as the trajectory
                    aligned_nodes = (R_cl @ raw_nodes[:, :2].T).T + t_cl
                    nodes_scatter.set_data(aligned_nodes[:, 0], aligned_nodes[:, 1])
                
                # 🌟 NEW 4: Extract and plot the active HDC Candidate nodes
                if len(active_candidates) > 0:
                    # Safely grab the [X, Y, Th] coordinates for every candidate ID
                    raw_cands = np.array([graph_poses[nid] for nid in active_candidates if nid < len(graph_poses)])
                    
                    if len(raw_cands) > 0:
                        # Apply the exact same map transform so they lock onto the black nodes
                        aligned_cands = (R_cl @ raw_cands[:, :2].T).T + t_cl
                        candidates_scatter.set_data(aligned_cands[:, 0], aligned_cands[:, 1])
                    else:
                        candidates_scatter.set_data([], [])
                        
                    # 🌟 THE FIX: Clear the list so the rings vanish on the NEXT UI frame!
                    active_candidates = []
                else:
                    candidates_scatter.set_data([], [])

                recent_cl = aligned_cl_path[-100:] 
                current_live_traj.set_data(recent_cl[:, 0], recent_cl[:, 1])
                
                aligned_cx, aligned_cy = aligned_cl_path[-1, 0], aligned_cl_path[-1, 1]
                aligned_cth = rel_cth + delta_theta
                live_head.set_data([aligned_cx], [aligned_cy])
                
                sog_img.set_data(np.array(sog_state.v_mem).T) 
                trans_data = mtransforms.Affine2D().rotate(delta_theta).translate(t_cl[0], t_cl[1]) + ax_map.transData
                sog_img.set_transform(trans_data)

                err_imu = np.sqrt(np.sum((imu_arr[:, :2] - gt_arr)**2, axis=1))
                err_ol = np.sqrt(np.sum((ol_arr[:, :2] - gt_arr)**2, axis=1))
                err_cl = np.sqrt(np.sum((cl_arr[:, :2] - gt_arr)**2, axis=1))
                
                t_axis = np.arange(len(err_cl)) * DT
                line_err_imu.set_data(t_axis, err_imu)
                line_err_ol.set_data(t_axis, err_ol)
                line_err_cl.set_data(t_axis, err_cl)
                
                if len(t_axis) > 0:
                    ax_err.set_xlim(0, max(10.0, t_axis[-1]))
                    ax_err.set_ylim(0, max(0.5, np.max(err_imu) * 1.1))
                
                R_fov = 30.0
                fov_rad = np.radians(FOV_DEG)
                gx, gy = gt_pos[0], gt_pos[1]
                
                for i in range(3):
                    gx_end = gx + tof_t[i] * np.cos(gt_th + tof_angles[i])
                    gy_end = gy + tof_t[i] * np.sin(gt_th + tof_angles[i])
                    tof_rays_gt[i].set_data([gx, gx_end], [gy, gy_end])
                    
                    rx_end = aligned_cx + tof_t[i] * np.cos(aligned_cth + tof_angles[i])
                    ry_end = aligned_cy + tof_t[i] * np.sin(aligned_cth + tof_angles[i])
                    tof_rays[i].set_data([aligned_cx, rx_end], [aligned_cy, ry_end])

                pt1_gt = [gx, gy]
                pt2_gt = [gx + R_fov * np.cos(gt_th + fov_rad/2), gy + R_fov * np.sin(gt_th + fov_rad/2)]
                pt3_gt = [gx + R_fov * np.cos(gt_th - fov_rad/2), gy + R_fov * np.sin(gt_th - fov_rad/2)]
                fov_poly_gt.set_xy([pt1_gt, pt2_gt, pt3_gt])

                pt1 = [aligned_cx, aligned_cy]
                pt2 = [aligned_cx + R_fov * np.cos(aligned_cth + fov_rad/2), aligned_cy + R_fov * np.sin(aligned_cth + fov_rad/2)]
                pt3 = [aligned_cx + R_fov * np.cos(aligned_cth - fov_rad/2), aligned_cy + R_fov * np.sin(aligned_cth - fov_rad/2)]
                fov_poly.set_xy([pt1, pt2, pt3])
                
                pc_act = np.array(r_place[0]).reshape(1, HDC_CONFIG["num_bits"])
                brain_img.set_data(pc_act / (pc_act.max() + 1e-8))
                
                flat_cann = history['cann_act'][-1]
                
                grid_flat_img.set_data(flat_cann.reshape(1, 579) / (flat_cann.max() + 1e-8))
                
                s1, s2 = CANN_SIZES[0]**2, CANN_SIZES[1]**2
                
                c1 = flat_cann[:s1].reshape(CANN_SIZES[0], CANN_SIZES[0])
                c2 = flat_cann[s1:s1+s2].reshape(CANN_SIZES[1], CANN_SIZES[1])
                c3 = flat_cann[s1+s2:].reshape(CANN_SIZES[2], CANN_SIZES[2])
                
                cann1_img.set_data(c1 / (c1.max() + 1e-8))
                cann2_img.set_data(c2 / (c2.max() + 1e-8))
                cann3_img.set_data(c3 / (c3.max() + 1e-8))
                
                ring_mem = history['ring_mem_act'][-1].flatten()
                line_ring_mem.set_ydata(ring_mem / (ring_mem.max() + 1e-8))

                ring_cann = history['ring_cann_act'][-1].flatten()
                line_ring_cann.set_ydata(ring_cann / (ring_cann.max() + 1e-8))
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=fig.dpi) 
                buf.seek(0)
                frame = imageio.v3.imread(buf, extension=".png") 
                gif_writer.append_data(frame)
                buf.close()
                
            step += 1

    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n 🛑 Halted! Simulated {step} steps in {elapsed:.1f}s.")
        
        gif_writer.close()
        print(f" 💾 Saved live animation to {gif_filename}")
        
        print(" 💾 Compiling logs and generating final PNG plots...")
        plt.ioff()
        plt.close(fig)

    min_len = min(len(history['gt_pos']), len(history['imu_pos']), 
                  len(history['ol_pose']), len(history['cl_pose']))
    
    gt_arr = np.array(history['gt_pos'][:min_len])
    imu_arr = np.array(history['imu_pos'][:min_len])
    ol_arr = np.array(history['ol_pose'][:min_len])
    cl_arr = np.array(history['cl_pose'][:min_len])
    th_gt = np.array(history['gt_th'][:min_len])
    th_imu = np.array(history['imu_th'][:min_len])
    
    history['conf'] = history['conf'][:min_len]
    step = min_len

    R_ol, t_ol = get_optimal_alignment_2d(ol_arr[:, :2], gt_arr)
    ol_arr_aligned = (R_ol @ ol_arr[:, :2].T).T + t_ol
    
    R_cl, t_cl = get_optimal_alignment_2d(cl_arr[:, :2], gt_arr)
    cl_arr_aligned = (R_cl @ cl_arr[:, :2].T).T + t_cl
    delta_th_cl = np.arctan2(R_cl[1, 0], R_cl[0, 0])
    
    final_map_pts = []
    for node_idx, (nx, ny, nth) in enumerate(graph_poses):
        hits = node_tof_hits[node_idx]
        for i in range(3):
            if hits[i] < 7.5:
                hx = nx + hits[i] * np.cos(nth + tof_angles[i])
                hy = ny + hits[i] * np.sin(nth + tof_angles[i])
                final_map_pts.append([hx, hy])
                
    map_pts = np.array(final_map_pts) if len(final_map_pts) > 0 else np.zeros((0, 2))
    if len(map_pts) > 0:
        map_pts = (R_cl @ map_pts.T).T + t_cl

    pos_err_imu = np.sqrt((imu_arr[:, 0] - gt_arr[:, 0])**2 + (imu_arr[:, 1] - gt_arr[:, 1])**2)
    pos_err_ol = np.sqrt((ol_arr_aligned[:, 0] - gt_arr[:, 0])**2 + (ol_arr_aligned[:, 1] - gt_arr[:, 1])**2)
    pos_err_cl = np.sqrt((cl_arr_aligned[:, 0] - gt_arr[:, 0])**2 + (cl_arr_aligned[:, 1] - gt_arr[:, 1])**2)

    def angle_err(a, b):
        diff = np.abs(a - b)
        return np.minimum(diff, 2*np.pi - diff)

    return {
        'B': 1, 'time_steps': step, 'drift_start': live_drift_start, 'obstacles': env.obstacles,
        'x_gt': gt_arr[None, :, 0], 'y_gt': gt_arr[None, :, 1], 'th_gt': th_gt[None, :],
        'x_imu': imu_arr[None, :, 0], 'y_imu': imu_arr[None, :, 1], 'th_imu': th_imu[None, :],
        'x_cl_raw': cl_arr[None, :, 0], 'y_cl_raw': cl_arr[None, :, 1], 'th_cl_raw': cl_arr[None, :, 2],
        'x_ol': ol_arr_aligned[None, :, 0], 'y_ol': ol_arr_aligned[None, :, 1], 'th_ol': ol_arr[None, :, 2],
        'x_cl': cl_arr_aligned[None, :, 0], 'y_cl': cl_arr_aligned[None, :, 1], 'th_cl': (cl_arr[:, 2] + delta_th_cl)[None, :],
        'pos_err_imu': pos_err_imu[None, :], 'pos_err_ol': pos_err_ol[None, :], 'pos_err_cl': pos_err_cl[None, :],
        'theta_err_imu': angle_err(th_imu, th_gt)[None, :], 'theta_err_ol': angle_err(ol_arr[:, 2], th_gt)[None, :], 'theta_err_cl': angle_err(cl_arr[:, 2], th_gt)[None, :],
        'loop_conf': np.array(history['conf'])[None, :],
        'pc_top_conf': np.zeros((1, step)), 'pc_x_decoded': np.zeros((1, step)), 'pc_y_decoded': np.zeros((1, step)),
        'sog_grid': np.array(sog_state.v_mem)
    }


if __name__ == '__main__':
    # We also need to import the visualizers from the system script for the main function
    from snn_slam_system import visualize_4panel, visualize_world_map
    import os

    print("=" * 65)
    print("  🦊  LIVE SNN SLAM System v3 — Continuous Exploration")
    print("=" * 65)

    seed_val = int(time.time() * 1000) % (2**31)
    key = random.PRNGKey(seed_val)
    print(f"  🎲 Generated New Random Room Seed: {seed_val}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    FIG_PATH = os.path.join(current_dir, "snn_slam_4panel.png")
    MAP_PATH = os.path.join(current_dir, "snn_slam_world_map.png")

    # Run the live SLAM
    results = run_live_slam(key)

    print(f"\n🎨 Generating final High-Res offline visualizations...")
    visualize_4panel(results, save_path=FIG_PATH)
    if 'sog_grid' in results:
        visualize_world_map(results, save_path=MAP_PATH)

    print(f"\n{'='*65}")
    print(f"  ✅ SYSTEM SHUTDOWN COMPLETE")
    print(f"{'='*65}")