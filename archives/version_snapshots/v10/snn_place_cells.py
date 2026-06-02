#!/usr/bin/env python3
"""
snn_place_cells.py — PlaceCellNetwork with BATCH-ISOLATED associative memory.
"""

import jax
from jax import random, nn as jnn, jit
import jax.numpy as jnp
from jax.lax import fori_loop
from functools import partial
from typing import NamedTuple

# ============================================================================
#  📦  CONSTANTS
# ============================================================================
N_CSNN        = 256    # Frozen CSNN semantic anchor
N_STDP        = 256    # Plastic STDP texture fingerprint
N_DEPTH       = 192    # ToF Gaussian RBF channels
MAP_SIZE      = 32    
N_PLACE       = MAP_SIZE * MAP_SIZE  
RING_N        = 64
ROOM_W        = 10.0  # meters
ROOM_H        = 10.0  # meters
MAP_AREA      = ROOM_W * ROOM_H       # 100 m²
SIGMA_M       = 0.50  # Gaussian sigma in METERS (world space)
GAIN          = 4.0   # excitatory Gaussian amplitude
CORRECTION_GAIN      = 2.0    # ghost bump amplitude
RING_CORRECTION_GAIN = 2.0

# 🌟 DELETED: CORRECTION_SMOOTH and RING_CORRECTION_SMOOTH

GATING_STRENGTH = 5.0  # STDP Multiplier power
SEQ_GATING_STRENGTH = 1.0  # 🌟 Sequence Memory Multiplier
HEADING_INERTIA_STRENGTH = 1.0  # 🌟 IMU Inertia for Ring Cells
LEARNING_CORE_THRESH  = 0.50   # Cuts off the Gaussian tails to prevent memory smearing
HOPFIELD_BETA         = 2.5    # Hopfield exponential sharpening (Inverse Temperature)

# gating thresholds
HEADING_THRESH        = 1.500
RING_SELF_MATCH_THRESH = 0.01  # Radians
PLAUSIBILITY_THRESH   = 1.500
SELF_MATCH_THRESH     = 0.01   # meters — restored: blocks "molasses effect" backward corrections
SPARSITY_THRESH       = 0.02   # match the average baseline magnitude of v_out vectors

TOFPOP_ANTI_ALIAS_POS = 0.50   # top-16 concentration threshold for anti-aliasing
TOFPOP_ANTI_ALIAS_HEA = 0.50   # top-4 concentration threshold for anti-aliasing
MATCH_THRESH          = 0.30   # Minimum fraction of theoretical max energy (0.0 - 1.0) for a valid match
MATURITY_GATE         = 0.60   # Hard security cutoff for loop closure readiness
MATCH_TOF             = 0.70   # TOF match threshold 

# EMA trace parameters for temporal binding
TRACE_ALPHA = 0.90   # slow decay: xx% carry-over, 1-xx% new input

def build_pc_preferred_locs(map_size=MAP_SIZE, room_w=ROOM_W, room_h=ROOM_H):
    gx = jnp.arange(map_size, dtype=jnp.float32)
    gy = jnp.arange(map_size, dtype=jnp.float32)
    gx_center = (gx + 0.5) * (room_w / map_size)
    gy_center = (gy + 0.5) * (room_h / map_size)

    xx, yy = jnp.meshgrid(gx_center, gy_center, indexing='xy')
    locs = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    return locs

def build_ring_preferred_th(ring_n=RING_N):
    return jnp.arange(ring_n, dtype=jnp.float32) * (2 * jnp.pi / ring_n)

# ============================================================================
#  📦  STATE STRUCTURE (CLEANED UP)
# ============================================================================
class PlaceCellState(NamedTuple):
    W_csnn_to_place: jnp.ndarray
    W_stdp_to_place: jnp.ndarray  
    W_tof_to_place: jnp.ndarray
    W_seq_to_place: jnp.ndarray
    
    W_csnn_to_ring: jnp.ndarray
    W_stdp_to_ring: jnp.ndarray    
    W_tof_to_ring: jnp.ndarray
    
    # 🌟 DELETED: I_correction_smooth_place & I_correction_smooth_ring
    
    trace_csnn: jnp.ndarray
    trace_stdp: jnp.ndarray        
    trace_tof: jnp.ndarray
    trace_place: jnp.ndarray
    trace_ring: jnp.ndarray
    confidence_ema: jnp.ndarray

# ============================================================================
#  🧠  PlaceCellNetwork
# ============================================================================
class PlaceCellNetwork:

    def __init__(self, n_csnn=N_CSNN, n_stdp=N_STDP, n_depth=N_DEPTH, fov_deg=60.0):
        self.n_csnn     = n_csnn
        self.n_stdp     = n_stdp
        self.n_depth    = n_depth
        self.n_place    = N_PLACE
        self.ring_n     = RING_N
        self.map_size   = MAP_SIZE
        
        # 🌟 Convert and store the true Field of View!
        self.fov_rad    = jnp.radians(fov_deg)
        
        # 🌟 SACCADIC MASKING MATH (Moved to init so the UI can read it!)
        self.pixel_ang_res = self.fov_rad / float(self.n_csnn) 
        self.max_blur_pixels = 4.1 
        self.dt = 0.05 
        self.dynamic_saccade_thresh = (self.max_blur_pixels * self.pixel_ang_res) / self.dt

        self.pc_preferred_locs = jnp.array(build_pc_preferred_locs())  
        self.ring_preferred_th  = build_ring_preferred_th()

    def init_state(self, B) -> PlaceCellState:
        return PlaceCellState(
            W_csnn_to_place = jnp.zeros((B, N_CSNN, self.n_place), dtype=jnp.float32),
            W_stdp_to_place = jnp.zeros((B, N_STDP, self.n_place), dtype=jnp.float32), 
            W_tof_to_place  = jnp.zeros((B, self.n_depth, self.n_place), dtype=jnp.float32),
            W_seq_to_place  = jnp.zeros((B, self.n_place, self.n_place), dtype=jnp.float32),
            
            W_csnn_to_ring  = jnp.zeros((B, self.n_place, N_CSNN, self.ring_n), dtype=jnp.float32),
            W_stdp_to_ring  = jnp.zeros((B, self.n_place, N_STDP, self.ring_n), dtype=jnp.float32),  
            W_tof_to_ring   = jnp.zeros((B, self.n_place, self.n_depth, self.ring_n), dtype=jnp.float32),
            
            trace_csnn  = jnp.zeros((B, N_CSNN), dtype=jnp.float32),
            trace_stdp  = jnp.zeros((B, N_STDP), dtype=jnp.float32), 
            trace_tof   = jnp.zeros((B, self.n_depth), dtype=jnp.float32),
            trace_place = jnp.zeros((B, self.n_place), dtype=jnp.float32),   
            trace_ring  = jnp.zeros((B, self.ring_n), dtype=jnp.float32),    
            confidence_ema = jnp.zeros((B,), dtype=jnp.float32)
        )

    def __call__(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, angular_vel=None, learn=True, confidence=None):
        B = vis_csnn.shape[0]

        # ── 🌟 DENDRITIC RECALL ──
        I_csnn_place = jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place)
        I_stdp_place = jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)
        I_tof_place  = jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)
        I_seq_place  = jnp.einsum('bp,bpq->bq', state.trace_place, state.W_seq_to_place)

        primary_senses = I_csnn_place + I_tof_place 
        raw_I_place = jnp.maximum(0.0, primary_senses * (1.0 + GATING_STRENGTH * I_stdp_place) * (1.0 + SEQ_GATING_STRENGTH * I_seq_place))
        
        norm_raw_place = raw_I_place / (jnp.max(raw_I_place, axis=1, keepdims=True) + 1e-8)
        I_place = jnp.power(norm_raw_place, HOPFIELD_BETA) * jnp.max(raw_I_place, axis=1, keepdims=True)

        joint_csnn_ring = jnp.einsum('bv,bpvr->bpr', vis_csnn, state.W_csnn_to_ring)
        joint_stdp_ring = jnp.einsum('bs,bpsr->bpr', vis_stdp, state.W_stdp_to_ring)
        joint_tof_ring  = jnp.einsum('bd,bpdr->bpr', tof_features, state.W_tof_to_ring)
        
        primary_ring_senses = joint_csnn_ring + joint_tof_ring
        
        joint_ring = jnp.maximum(0.0, 
            primary_ring_senses * (1.0 + GATING_STRENGTH * joint_stdp_ring) * (1.0 + HEADING_INERTIA_STRENGTH * ring_bump[:, None, :])
        )

        pose_bump_norm = pose_bump / (jnp.max(pose_bump, axis=1, keepdims=True) + 1e-8)
        raw_I_ring = jnp.einsum('bpr,bp->br', joint_ring, pose_bump_norm)
        
        norm_raw_ring = raw_I_ring / (jnp.max(raw_I_ring, axis=1, keepdims=True) + 1e-8)
        I_ring = jnp.power(norm_raw_ring, HOPFIELD_BETA) * jnp.max(raw_I_ring, axis=1, keepdims=True)

        # 🌟 DELETED: The temporal smoothing math block

        # ── Hebbian learning
        if learn:
            state, dW_cp, dW_sp, dW_dp, dW_cr, dW_sr, dW_dr, dW_seq = self._updateHebbian(state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, angular_vel, confidence)
            state = state._replace(
                W_csnn_to_place = jnp.clip(state.W_csnn_to_place + dW_cp, 0.0, 1.0),
                W_stdp_to_place = jnp.clip(state.W_stdp_to_place + dW_sp, 0.0, 1.0),
                W_tof_to_place  = jnp.clip(state.W_tof_to_place  + dW_dp, 0.0, 1.0),
                W_seq_to_place  = jnp.clip(state.W_seq_to_place  + dW_seq, 0.0, 1.0),
                W_csnn_to_ring  = jnp.clip(state.W_csnn_to_ring  + dW_cr, 0.0, 1.0),
                W_stdp_to_ring  = jnp.clip(state.W_stdp_to_ring  + dW_sr, 0.0, 1.0),
                W_tof_to_ring   = jnp.clip(state.W_tof_to_ring   + dW_dr, 0.0, 1.0)
            )

        # 🌟 UPDATED RETURN: Cleaned up tuple
        return state, (I_place, I_ring)

    def forward_mapping(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump=None, angular_vel=None, learn=True, confidence=None):
        return self(state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, angular_vel=angular_vel, learn=learn, confidence=confidence)

    def _updateHebbian(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, angular_vel=None, confidence=None):
        ETA_W = 0.05
        W_MAX = 1.0  
        LAMBDA_W = 0.025

        if confidence is not None:
            novelty = 1.0 - confidence
            eta_batch = ETA_W * novelty[:, None, None]
        else:
            eta_batch = ETA_W

        new_trace_csnn  = TRACE_ALPHA * state.trace_csnn  + (1 - TRACE_ALPHA) * vis_csnn
        new_trace_stdp  = TRACE_ALPHA * state.trace_stdp  + (1 - TRACE_ALPHA) * vis_stdp
        new_trace_tof   = TRACE_ALPHA * state.trace_tof   + (1 - TRACE_ALPHA) * tof_features
        new_trace_place = TRACE_ALPHA * state.trace_place + (1 - TRACE_ALPHA) * pose_bump
        new_trace_ring  = TRACE_ALPHA * state.trace_ring  + (1 - TRACE_ALPHA) * ring_bump

        if angular_vel is not None:
            # 🌟 USE THE NEW CLASS VARIABLE HERE
            is_stable = jnp.where(jnp.abs(angular_vel) < self.dynamic_saccade_thresh, 1.0, 0.0)
            eta_saccadic_2d = eta_batch * is_stable[:, None, None]
            eta_saccadic_3d = eta_batch * is_stable[:, None, None, None]
        else:
            eta_saccadic_2d = eta_batch
            eta_saccadic_3d = eta_batch

        # =================================================================
        # 🌟 MOVE THE MASKS UP HERE! (Before we use them)
        # =================================================================
        base_pose_mask = jnp.where(pose_bump > LEARNING_CORE_THRESH, 1.0, 0.0)
        mask_place = base_pose_mask[:, None, :] 
        mask_place_4d = base_pose_mask[:, :, None, None]
        mask_ring_4d  = jnp.where(ring_bump > LEARNING_CORE_THRESH, 1.0, 0.0)[:, None, None, :]
        mask_conj = mask_place_4d * mask_ring_4d

        # ── Place Cell Base Updates ──
        raw_dW_cp = eta_saccadic_2d * jnp.einsum('bv,bp->bvp', new_trace_csnn, new_trace_place)
        raw_dW_sp = eta_saccadic_2d * jnp.einsum('bs,bp->bsp', new_trace_stdp, new_trace_place)
        raw_dW_dp = eta_saccadic_2d * jnp.einsum('bd,bp->bdp', new_trace_tof, new_trace_place)
        raw_dW_seq = eta_batch * jnp.einsum('bi,bj->bij', state.trace_place, pose_bump) 
        
        # =================================================================
        # 🌟 THE MASTER LOCK & FLASH BULB PLASTICITY
        # Since the Ring Cell only gets ONE frame, we remove the 5% learning rate.
        # We use a 100% Delta Rule to instantly copy the pure shape of the input!
        # =================================================================
        synaptic_lock = jnp.where(jnp.sum(state.W_tof_to_ring, axis=2, keepdims=True) < 1.0, 1.0, 0.0)
        
        # Use the hard threshold mask (mask_conj) but WITHOUT the eta_batch penalty!
        # This acts as a 100% "Flash Bulb" for the exact winning Conjunctive Cell.
        flash_mask = mask_conj * synaptic_lock
        
        # 1. Ring Vision (Instant Delta Snapshot)
        vis_error = vis_csnn[:, None, :, None] - state.W_csnn_to_ring
        dW_cr = flash_mask * vis_error
        
        # 2. Ring STDP (Instant Delta Snapshot)
        stdp_error = vis_stdp[:, None, :, None] - state.W_stdp_to_ring
        dW_sr = flash_mask * stdp_error
        
        # 3. Ring ToF Geometry (Instant Delta Snapshot)
        tof_error = tof_features[:, None, :, None] - state.W_tof_to_ring
        dW_dr = flash_mask * tof_error

        # ── Place Cell Decays & Final Damping ──
        dec_cp = eta_batch * LAMBDA_W * new_trace_place[:, None, :] * state.W_csnn_to_place
        dec_sp = eta_batch * LAMBDA_W * new_trace_place[:, None, :] * state.W_stdp_to_place
        dec_dp = eta_batch * LAMBDA_W * new_trace_place[:, None, :] * state.W_tof_to_place
        dec_seq = eta_batch * LAMBDA_W * pose_bump[:, None, :] * state.W_seq_to_place 
        
        # 🌟 Only recalculate the Place Cell updates!
        dW_cp = (raw_dW_cp * (W_MAX - state.W_csnn_to_place) - dec_cp) * mask_place
        dW_sp = (raw_dW_sp * (W_MAX - state.W_stdp_to_place) - dec_sp) * mask_place
        dW_dp = (raw_dW_dp * (W_MAX - state.W_tof_to_place)  - dec_dp) * mask_place
        dW_seq = (raw_dW_seq * (W_MAX - state.W_seq_to_place) - dec_seq) * mask_place 

        state = state._replace(
            trace_csnn=new_trace_csnn, trace_stdp=new_trace_stdp, trace_tof=new_trace_tof,
            trace_place=new_trace_place, trace_ring=new_trace_ring
        )

        return state, dW_cp, dW_sp, dW_dp, dW_cr, dW_sr, dW_dr, dW_seq

    @partial(jax.jit, static_argnames=['self'])
    def apply_post_relaxation_update(self, state: PlaceCellState, p_idx, r_idx, aligned_csnn, aligned_stdp, fov_mask, ring_lr=0.05):
        """
        Phase 2 Plasticity: Applies a dopamine-gated, attention-masked Hebbian update
        to a specific Ring Cell memory AFTER the Orchestrator corrects the heading.
        """
        batch_idx = jnp.arange(aligned_csnn.shape[0])

        # 1. Grab the old, frozen memories
        old_w_csnn = state.W_csnn_to_ring[batch_idx, p_idx, :, r_idx]
        old_w_stdp = state.W_stdp_to_ring[batch_idx, p_idx, :, r_idx]

        # 2. Calculate the Deltas using the Attention Mask! 
        # (If mask == 0.0, the delta is 0.0, preventing catastrophic forgetting of the blind spot)
        dW_csnn = ring_lr * fov_mask * (aligned_csnn - old_w_csnn)
        dW_stdp = ring_lr * fov_mask * (aligned_stdp - old_w_stdp)

        # 3. Apply the updates safely using JAX
        new_W_csnn = state.W_csnn_to_ring.at[batch_idx, p_idx, :, r_idx].add(dW_csnn)
        new_W_stdp = state.W_stdp_to_ring.at[batch_idx, p_idx, :, r_idx].add(dW_stdp)

        return state._replace(
            W_csnn_to_ring = jnp.clip(new_W_csnn, 0.0, 1.0),
            W_stdp_to_ring = jnp.clip(new_W_stdp, 0.0, 1.0)
        )

    def compute_confidence_with_gates(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_xy, heading, ring_bump):
        B = vis_csnn.shape[0]

        I_csnn_place = jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place)
        I_stdp_place = jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)
        I_tof_place = jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)
        I_seq_place = jnp.einsum('bp,bpq->bq', state.trace_place, state.W_seq_to_place)
        
        primary_senses = I_csnn_place + I_tof_place 
        raw_I_place = jnp.maximum(0.0, primary_senses * (1.0 + GATING_STRENGTH * I_stdp_place) * (1.0 + SEQ_GATING_STRENGTH * I_seq_place))

        norm_raw_place = raw_I_place / (jnp.max(raw_I_place, axis=1, keepdims=True) + 1e-8)
        I_place = jnp.power(norm_raw_place, HOPFIELD_BETA) * jnp.max(raw_I_place, axis=1, keepdims=True)
 
        joint_csnn_ring = jnp.einsum('bv,bpvr->bpr', vis_csnn, state.W_csnn_to_ring)
        joint_stdp_ring = jnp.einsum('bs,bpsr->bpr', vis_stdp, state.W_stdp_to_ring)
        joint_tof_ring  = jnp.einsum('bd,bpdr->bpr', tof_features, state.W_tof_to_ring)
        
        primary_ring_senses = joint_csnn_ring + joint_tof_ring
        
        joint_ring = jnp.maximum(0.0, 
            primary_ring_senses * (1.0 + GATING_STRENGTH * joint_stdp_ring)
        )

        I_place_norm = I_place / (jnp.max(I_place, axis=1, keepdims=True) + 1e-8)
        raw_I_ring = jnp.einsum('bpr,bp->br', joint_ring, I_place_norm)
        
        norm_raw_ring = raw_I_ring / (jnp.max(raw_I_ring, axis=1, keepdims=True) + 1e-8)
        I_ring = jnp.power(norm_raw_ring, HOPFIELD_BETA) * jnp.max(raw_I_ring, axis=1, keepdims=True)

        recalled_place = jnp.max(I_place, axis=1)
        recalled_ring_max = jnp.max(I_ring, axis=1)

        vision_activity = jnp.mean(jnp.abs(vis_csnn), axis=1)
        is_distinctive = vision_activity > SPARSITY_THRESH

        raw_recalled_place = jnp.max(I_place, axis=1)
        
        max_csnn = jnp.sum(jnp.abs(vis_csnn), axis=1)
        max_stdp = jnp.sum(jnp.abs(vis_stdp), axis=1)
        max_tof  = jnp.sum(jnp.abs(tof_features), axis=1)
        max_seq  = jnp.sum(jnp.abs(state.trace_place), axis=1)
        
        max_possible_energy = (
            (max_csnn + max_tof) * (1.0 + GATING_STRENGTH * max_stdp) * (1.0 + SEQ_GATING_STRENGTH * max_seq)
        )
        
        recalled_place = raw_recalled_place / (max_possible_energy + 1e-8)

        has_learned_memory = recalled_place > 0.010 
        is_match = recalled_place > MATCH_THRESH

        peak_idx_place = jnp.argmax(I_place, axis=1)

        # =================================================================
        # 🌟 UPGRADED: CONJUNCTIVE GEOMETRIC GATE (True Cosine Similarity)
        # =================================================================
        peak_ring = jnp.argmax(I_ring, axis=1)
        batch_idx = jnp.arange(B)
        
        # Grab the exact 192-neuron memory vector for this specific heading
        winning_W_tof_ring = state.W_tof_to_ring[batch_idx, peak_idx_place, :, peak_ring]
        
        # 1. Calculate the exact Overlap (Dot Product)
        dot_product = jnp.sum(tof_features * winning_W_tof_ring, axis=1)
        
        # 2. Calculate the L2 Norms (Magnitudes) of both arrays
        norm_in = jnp.linalg.norm(tof_features, axis=1)
        norm_mem = jnp.linalg.norm(winning_W_tof_ring, axis=1)
        
        # 3. Cosine Similarity guarantees a strict 0.0 (No Match) to 1.0 (Perfect Match)
        tof_match_score = dot_product / (norm_in * norm_mem + 1e-8)
        
        is_tof_match = tof_match_score > MATCH_TOF

        top16_place = jax.lax.top_k(I_place, 16)[0].sum(axis=1)
        conc_place  = top16_place / (I_place.sum(axis=1) + 1e-8)

        top4_ring   = jax.lax.top_k(I_ring, 4)[0].sum(axis=1)
        conc_ring   = top4_ring / (I_ring.sum(axis=1) + 1e-8)

        is_place_anti_aliased = jnp.where(has_learned_memory, conc_place > TOFPOP_ANTI_ALIAS_POS, True) 
        is_ring_anti_aliased  = jnp.where(has_learned_memory, conc_ring > TOFPOP_ANTI_ALIAS_HEA, True) 
        is_anti_aliased = is_place_anti_aliased & is_ring_anti_aliased

        recalled_x = self.pc_preferred_locs[peak_idx_place, 0]
        recalled_y = self.pc_preferred_locs[peak_idx_place, 1]

        jump = jnp.sqrt((recalled_x - pose_xy[:, 0])**2 + (recalled_y - pose_xy[:, 1])**2)
        is_plausible = jump < PLAUSIBILITY_THRESH 

        winning_weights = jnp.take_along_axis(
            state.W_csnn_to_place, 
            peak_idx_place[:, None, None], 
            axis=2
        )[:, :, 0]
        
        top_synapses = jax.lax.top_k(winning_weights, 8)[0]
        memory_maturity = jnp.mean(top_synapses, axis=1)
        is_mature = memory_maturity > MATURITY_GATE

        angles = self.ring_preferred_th 
        p_ring_dist = I_ring / (I_ring.sum(axis=1, keepdims=True) + 1e-8)
        
        sin_sum = jnp.sum(jnp.sin(angles) * p_ring_dist, axis=1)
        cos_sum = jnp.sum(jnp.cos(angles) * p_ring_dist, axis=1)
        
        recalled_heading = jnp.arctan2(sin_sum, cos_sum) 
        heading_standard = (heading + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        angle_diff = jnp.abs(recalled_heading - heading_standard)
        angle_diff = jnp.where(angle_diff > jnp.pi, 2 * jnp.pi - angle_diff, angle_diff)
        
        is_heading_plausible = angle_diff < HEADING_THRESH

        is_pos_drifting = jump > SELF_MATCH_THRESH
        is_heading_drifting = angle_diff > RING_SELF_MATCH_THRESH
        is_not_self_matching = is_pos_drifting | is_heading_drifting

        is_confident_raw = (
            is_distinctive & is_match & is_anti_aliased & 
            is_plausible & is_heading_plausible & is_not_self_matching &
            is_mature & is_tof_match  # 🌟 Added is_tof_match here!
        )

        new_confidence_ema = 0.5 * state.confidence_ema + 0.5 * is_confident_raw.astype(jnp.float32)
        state = state._replace(confidence_ema=new_confidence_ema)
        
        is_temporally_consistent = new_confidence_ema > 0.80
        is_confident = is_confident_raw & is_temporally_consistent

        # =================================================================
        # 🌟 DEBUG EXTRACTION: Grab the raw memory weights for the winning cells
        # =================================================================
        # 1. Place Cell 2D Memory Slices
        winning_W_csnn = jnp.take_along_axis(state.W_csnn_to_place, peak_idx_place[:, None, None], axis=2)[:, :, 0]
        winning_W_stdp = jnp.take_along_axis(state.W_stdp_to_place, peak_idx_place[:, None, None], axis=2)[:, :, 0]
        winning_W_tof  = jnp.take_along_axis(state.W_tof_to_place, peak_idx_place[:, None, None], axis=2)[:, :, 0]

        # 2. Ring Cell 3D Conjunctive Memory Slices
        batch_idx = jnp.arange(B)
        peak_ring = jnp.argmax(I_ring, axis=1)
        # We index [Batch, Place_ID, Vision_Array, Ring_ID]
        winning_W_csnn_ring = state.W_csnn_to_ring[batch_idx, peak_idx_place, :, peak_ring]
        winning_W_stdp_ring = state.W_stdp_to_ring[batch_idx, peak_idx_place, :, peak_ring]
        winning_W_tof_ring  = state.W_tof_to_ring[batch_idx, peak_idx_place, :, peak_ring]

        debug_gates = {
            "Peak_Ring": peak_ring,
            "G1_Distinctive": is_distinctive,
            "G2_Match": is_match,
            "G2b_ToFMatch": is_tof_match, # 🌟 NEW: Track the geometry gate
            "ToF_Score": tof_match_score,
            "G3_AntiAlias": is_anti_aliased,
            "G4_Plausible": is_plausible,
            "G4b_HeadPlausible": is_heading_plausible,
            "G5_NotSelf": is_not_self_matching,
            "G6_TemporalEMA": new_confidence_ema,
            "G7_Mature": is_mature,
            "Maturity_Lvl": memory_maturity,
            "Raw_Conf": is_confident_raw,
            "Final_Conf": is_confident,
            "Conc_Place": conc_place,
            "Jump_Dist": jump,
            "Raw_Vis_Act": vision_activity,
            "Raw_Match": recalled_place,
            "Conc_Ring": conc_ring,
            "Debug_Input_CSNN": vis_csnn,
            "Debug_Input_STDP": vis_stdp,
            "Debug_Input_ToF": tof_features,
            "Debug_Input_Seq": state.trace_place,
            "Debug_Mem_CSNN": winning_W_csnn,
            "Debug_Mem_STDP": winning_W_stdp,
            "Debug_Mem_ToF": winning_W_tof,
            "Debug_Mem_CSNN_Ring": winning_W_csnn_ring,
            "Debug_Mem_STDP_Ring": winning_W_stdp_ring,
            "Debug_Mem_ToF_Ring": winning_W_tof_ring,
            "Debug_I_Place": I_place,
            "Debug_I_Ring": I_ring
        }

        return state, is_confident, peak_idx_place, debug_gates

    # 🌟 UPDATED: Seeds the Ground Truth into the Traces directly!
    def initialize_from_pose(self, state: PlaceCellState, pose_bump, ring_bump=None):
        new_trace_place = jnp.maximum(state.trace_place, pose_bump[:, :self.n_place])
        if ring_bump is not None:
            new_trace_ring = jnp.maximum(state.trace_ring, ring_bump)
            return state._replace(trace_place=new_trace_place, trace_ring=new_trace_ring)
        return state._replace(trace_place=new_trace_place)

    def decode_position(self, r_place):
        B = r_place.shape[0]
        grid = r_place.reshape((B, self.map_size, self.map_size))
        
        peak_idx_flat = jnp.argmax(r_place, axis=1)
        y_int = peak_idx_flat // self.map_size
        x_int = peak_idx_flat % self.map_size
        batch_idx = jnp.arange(B)
        
        x_left  = jnp.clip(x_int - 1, 0, self.map_size - 1)
        x_right = jnp.clip(x_int + 1, 0, self.map_size - 1)
        y_up    = jnp.clip(y_int - 1, 0, self.map_size - 1)
        y_down  = jnp.clip(y_int + 1, 0, self.map_size - 1)
        
        v_c  = grid[batch_idx, y_int, x_int]    
        v_xl = grid[batch_idx, y_int, x_left]   
        v_xr = grid[batch_idx, y_int, x_right]
        v_yu = grid[batch_idx, y_up, x_int]     
        v_yd = grid[batch_idx, y_down, x_int]
        
        denom_x = 2.0 * (v_xl - 2.0 * v_c + v_xr)
        delta_x = jnp.where((denom_x != 0) & (x_int > 0) & (x_int < self.map_size - 1),
                            (v_xl - v_xr) / denom_x, 
                            0.0)
                            
        denom_y = 2.0 * (v_yu - 2.0 * v_c + v_yd)
        delta_y = jnp.where((denom_y != 0) & (y_int > 0) & (y_int < self.map_size - 1),
                            (v_yu - v_yd) / denom_y, 
                            0.0)
                            
        x_cont = x_int.astype(jnp.float32) + delta_x
        y_cont = y_int.astype(jnp.float32) + delta_y
        
        room_w = ROOM_W  
        room_h = ROOM_H  
        
        x_dec = (x_cont + 0.5) * (room_w / self.map_size)
        y_dec = (y_cont + 0.5) * (room_h / self.map_size)
        
        return jnp.stack([x_dec, y_dec], axis=1)

    def decode_heading(self, r_ring):
        angles = self.ring_preferred_th
        p = r_ring / (r_ring.sum(axis=1, keepdims=True) + 1e-8)
        sin_sum = jnp.sum(jnp.sin(angles) * p, axis=1)
        cos_sum = jnp.sum(jnp.cos(angles) * p, axis=1)
        return jnp.arctan2(sin_sum, cos_sum) % (2 * jnp.pi)