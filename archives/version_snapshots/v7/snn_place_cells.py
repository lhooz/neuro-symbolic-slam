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
N_CSNN        = 128    # Frozen CSNN semantic anchor
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
CORRECTION_SMOOTH    = 0.20  # temporal smoothing of correction signal
RING_CORRECTION_SMOOTH = 0.20
GATING_STRENGTH = 4.0  # STDP Multiplier power
SEQ_GATING_STRENGTH = 1.5  # 🌟 NEW: Sequence Memory Multiplier
HEADING_INERTIA_STRENGTH = 1.5  # 🌟 NEW: IMU Inertia for Ring Cells

# SOTAv1 gating thresholds
TOFPOP_ANTI_ALIAS_POS = 0.70   # top-16 concentration threshold for anti-aliasing
TOFPOP_ANTI_ALIAS_HEA = 0.70   # top-4 concentration threshold for anti-aliasing
HEADING_THRESH        = 1.500
RING_SELF_MATCH_THRESH = 0.001  # Radians
PLAUSIBILITY_THRESH   = 1.500
SELF_MATCH_THRESH     = 0.001   # meters — restored: blocks "molasses effect" backward corrections

# EMA trace parameters for temporal binding (Fix #3)
# Event cameras are sparse/async — EMA "synaptic afterglow" binds vision→pose
TRACE_ALPHA = 0.50   # slow decay: xx% carry-over, 1-xx% new input

def build_pc_preferred_locs(map_size=MAP_SIZE, room_w=ROOM_W, room_h=ROOM_H):
    """Build (N_PLACE, 2) array of preferred (x,y) for each place cell."""
    gx = jnp.arange(map_size, dtype=jnp.float32)          # [0, 1, ..., 31]
    gy = jnp.arange(map_size, dtype=jnp.float32)
    gx_center = (gx + 0.5) * (room_w / map_size)            # [0.156, 0.469, ..., 9.844]
    gy_center = (gy + 0.5) * (room_h / map_size)

    xx, yy = jnp.meshgrid(gx_center, gy_center, indexing='xy')
    locs = jnp.stack([xx.ravel(), yy.ravel()], axis=1)     # (1024, 2)
    return locs

def build_ring_preferred_th(ring_n=RING_N):
    """Build (RING_N,) array of preferred heading angles in radians."""
    return jnp.arange(ring_n, dtype=jnp.float32) * (2 * jnp.pi / ring_n)

# ============================================================================
#  📦  STATE STRUCTURE (JAX FIX)
# ============================================================================
class PlaceCellState(NamedTuple):
    """Holds all dynamic state for the PlaceCellNetwork to enable pure functional execution."""
    W_csnn_to_place: jnp.ndarray
    W_stdp_to_place: jnp.ndarray   
    W_tof_to_place: jnp.ndarray
    W_seq_to_place: jnp.ndarray    # 🌟 NEW: Asymmetric Recurrent Sequence Memory
    
    W_csnn_to_ring: jnp.ndarray
    W_stdp_to_ring: jnp.ndarray    
    W_tof_to_ring: jnp.ndarray
    
    I_correction_smooth_place: jnp.ndarray
    I_correction_smooth_ring: jnp.ndarray
    
    trace_csnn: jnp.ndarray
    trace_stdp: jnp.ndarray        
    trace_tof: jnp.ndarray
    trace_place: jnp.ndarray
    trace_ring: jnp.ndarray
    confidence_ema: jnp.ndarray

# ============================================================================
#  🧠  PlaceCellNetwork (SOTA Multi-Compartment Pyramidal Architecture)
# ============================================================================
class PlaceCellNetwork:
    """BATCH-ISOLATED place + ring cell associative memory.
    
    Upgraded to Multi-Compartment Dendritic Segregation:
    - Basal Dendrites: Learn Vision features (128)
    - Apical Dendrites: Learn ToF depth context (8)
    - Soma: Multiplicative coincidence detection (I_vis * I_tof)
    """

    SPARSITY_THRESH  = 0.02  # match the average baseline magnitude of v_out vectors
    MATCH_THRESH     = 0.25  # Seed threshold

    def __init__(self, n_csnn=N_CSNN, n_stdp=N_STDP, n_depth=N_DEPTH):
        self.n_csnn     = n_csnn     # 128
        self.n_stdp     = n_stdp     # 256
        self.n_depth    = n_depth    # 192
        self.n_place    = N_PLACE    # 1024
        self.ring_n     = RING_N     # 64
        self.map_size   = MAP_SIZE   # 32

        self.pc_preferred_locs = jnp.array(build_pc_preferred_locs())  
        self.ring_preferred_th  = build_ring_preferred_th()

    def init_state(self, B) -> PlaceCellState:
        return PlaceCellState(
            W_csnn_to_place = jnp.zeros((B, N_CSNN, self.n_place), dtype=jnp.float32),
            W_stdp_to_place = jnp.zeros((B, N_STDP, self.n_place), dtype=jnp.float32), 
            W_tof_to_place  = jnp.zeros((B, self.n_depth, self.n_place), dtype=jnp.float32),
            W_seq_to_place  = jnp.zeros((B, self.n_place, self.n_place), dtype=jnp.float32),
            
            # 🌟 V4 UPGRADE: The Conjunctive 3D Tensor! Shape: (Batch, Place, Vision, Ring)
            W_csnn_to_ring  = jnp.zeros((B, self.n_place, N_CSNN, self.ring_n), dtype=jnp.float32),
            W_stdp_to_ring  = jnp.zeros((B, self.n_place, N_STDP, self.ring_n), dtype=jnp.float32),  
            W_tof_to_ring   = jnp.zeros((B, self.n_place, self.n_depth, self.ring_n), dtype=jnp.float32),
            
            I_correction_smooth_place = jnp.zeros((B, self.n_place), dtype=jnp.float32),  
            I_correction_smooth_ring  = jnp.zeros((B, self.ring_n), dtype=jnp.float32),   
            
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
        # Place Network
        I_csnn_place = jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place)
        I_stdp_place = jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)
        I_tof_place  = jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)
        I_seq_place  = jnp.einsum('bp,bpq->bq', state.trace_place, state.W_seq_to_place)

        # 🌟 THE FIX: Vision and ToF are Additive. STDP and Sequence are Multiplicative Modulators.
        primary_senses = I_csnn_place + I_tof_place 
        I_place = jnp.maximum(0.0, primary_senses * (1.0 + GATING_STRENGTH * I_stdp_place) * (1.0 + SEQ_GATING_STRENGTH * I_seq_place))

        # 🌟 V4 UPGRADE: Ring Network Conjunctive Recall
        joint_csnn_ring = jnp.einsum('bv,bpvr->bpr', vis_csnn, state.W_csnn_to_ring)
        joint_stdp_ring = jnp.einsum('bs,bpsr->bpr', vis_stdp, state.W_stdp_to_ring)
        joint_tof_ring  = jnp.einsum('bd,bpdr->bpr', tof_features, state.W_tof_to_ring)
        
        # 🌟 THE FIX: Vision and ToF are Additive for Ring Cells too!
        primary_ring_senses = joint_csnn_ring + joint_tof_ring
        
        # 🌟 THE INJECTION: Multiply by the IMU's Ring CANN
        joint_ring = jnp.maximum(0.0, 
            primary_ring_senses * 
            (1.0 + GATING_STRENGTH * joint_stdp_ring) * 
            (1.0 + HEADING_INERTIA_STRENGTH * ring_bump[:, None, :])
        )

        # 🌟 Multiply by the current IMU position (pose_bump) to extract the local compass!
        pose_bump_norm = pose_bump / (jnp.max(pose_bump, axis=1, keepdims=True) + 1e-8)
        I_ring = jnp.einsum('bpr,bp->br', joint_ring, pose_bump_norm)

        # ── Temporal smoothing
        new_smooth_place = CORRECTION_SMOOTH * state.I_correction_smooth_place + (1 - CORRECTION_SMOOTH) * I_place
        new_smooth_ring  = RING_CORRECTION_SMOOTH * state.I_correction_smooth_ring + (1 - RING_CORRECTION_SMOOTH) * I_ring

        state = state._replace(I_correction_smooth_place=new_smooth_place, I_correction_smooth_ring=new_smooth_ring)

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

        return state, (I_place, I_ring, new_smooth_place, new_smooth_ring)

    def forward_mapping(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump=None, angular_vel=None, learn=True, confidence=None):
        return self(state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, angular_vel=angular_vel, learn=learn, confidence=confidence)

    def _updateHebbian(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, angular_vel=None, confidence=None):
        ETA_W = 0.05
        W_MAX = 1.0  
        LAMBDA_W = 0.002

        if confidence is not None:
            novelty = 1.0 - confidence
            eta_batch = ETA_W * novelty[:, None, None]
        else:
            eta_batch = ETA_W

        FOV_RAD = jnp.radians(60.0)
        PIXEL_ANG_RES = FOV_RAD / 128.0 
        MAX_BLUR_PIXELS = 25.0 
        DT = 0.05 
        DYNAMIC_SACCADE_THRESH = (MAX_BLUR_PIXELS * PIXEL_ANG_RES) / DT

        new_trace_csnn  = TRACE_ALPHA * state.trace_csnn  + (1 - TRACE_ALPHA) * vis_csnn
        new_trace_stdp  = TRACE_ALPHA * state.trace_stdp  + (1 - TRACE_ALPHA) * vis_stdp
        new_trace_tof   = TRACE_ALPHA * state.trace_tof   + (1 - TRACE_ALPHA) * tof_features
        new_trace_place = TRACE_ALPHA * state.trace_place + (1 - TRACE_ALPHA) * pose_bump
        new_trace_ring  = TRACE_ALPHA * state.trace_ring  + (1 - TRACE_ALPHA) * ring_bump

        if angular_vel is not None:
            is_stable = jnp.where(jnp.abs(angular_vel) < DYNAMIC_SACCADE_THRESH, 1.0, 0.0)
            # Add dimensions to eta_saccadic depending on 2D vs 3D matrix targets
            eta_saccadic_2d = eta_batch * is_stable[:, None, None]
            eta_saccadic_3d = eta_batch * is_stable[:, None, None, None]
        else:
            eta_saccadic_2d = eta_batch
            eta_saccadic_3d = eta_batch

        # 🌟 PLACE CELLS: (Standard 2D)
        raw_dW_cp = eta_saccadic_2d * jnp.einsum('bv,bp->bvp', new_trace_csnn, new_trace_place)
        raw_dW_sp = eta_saccadic_2d * jnp.einsum('bs,bp->bsp', new_trace_stdp, new_trace_place)
        raw_dW_dp = eta_saccadic_2d * jnp.einsum('bd,bp->bdp', new_trace_tof, new_trace_place)
        raw_dW_seq = eta_batch * jnp.einsum('bi,bj->bij', state.trace_place, pose_bump) 
        
        # 🌟 V4 UPGRADE: RING CELLS (Coupled 3D Tensor)
        # Binds Place (bp) + Vision (bv) + Heading (br) into a single memory (bpvr)
        raw_dW_cr = eta_saccadic_3d * jnp.einsum('bp,bv,br->bpvr', new_trace_place, new_trace_csnn, new_trace_ring)
        raw_dW_sr = eta_saccadic_3d * jnp.einsum('bp,bs,br->bpsr', new_trace_place, new_trace_stdp, new_trace_ring)
        raw_dW_dr = eta_saccadic_3d * jnp.einsum('bp,bd,br->bpdr', new_trace_place, new_trace_tof, new_trace_ring)

        # Decay Place Cells
        dec_cp = eta_batch * LAMBDA_W * new_trace_place[:, None, :] * state.W_csnn_to_place
        dec_sp = eta_batch * LAMBDA_W * new_trace_place[:, None, :] * state.W_stdp_to_place
        dec_dp = eta_batch * LAMBDA_W * new_trace_place[:, None, :] * state.W_tof_to_place
        dec_seq = eta_batch * LAMBDA_W * pose_bump[:, None, :] * state.W_seq_to_place 
        
        # 🌟 V4 UPGRADE: Decay Ring Cells (Decay ONLY happens for the room you are currently standing in!)
        dec_cr = eta_batch * LAMBDA_W * new_trace_place[:, :, None, None] * new_trace_ring[:, None, None, :] * state.W_csnn_to_ring
        dec_sr = eta_batch * LAMBDA_W * new_trace_place[:, :, None, None] * new_trace_ring[:, None, None, :] * state.W_stdp_to_ring
        dec_dr = eta_batch * LAMBDA_W * new_trace_place[:, :, None, None] * new_trace_ring[:, None, None, :] * state.W_tof_to_ring

        # 🌟 SHARPENING FIX: Only burn memories into the absolute core of the bump!
        mask_place = jnp.where(pose_bump > 0.50, 1.0, 0.0)[:, None, :] 
        
        # 🌟 V4 UPGRADE: Conjunctive Masking (Only learn if BOTH pose and heading are active)
        mask_place_4d = jnp.where(pose_bump > 0.50, 1.0, 0.0)[:, :, None, None]
        mask_ring_4d  = jnp.where(ring_bump > 0.50, 1.0, 0.0)[:, None, None, :]
        mask_conj = mask_place_4d * mask_ring_4d

        # Masked Updates
        dW_cp = (raw_dW_cp * (W_MAX - state.W_csnn_to_place) - dec_cp) * mask_place
        dW_sp = (raw_dW_sp * (W_MAX - state.W_stdp_to_place) - dec_sp) * mask_place
        dW_dp = (raw_dW_dp * (W_MAX - state.W_tof_to_place)  - dec_dp) * mask_place
        dW_seq = (raw_dW_seq * (W_MAX - state.W_seq_to_place) - dec_seq) * mask_place 
        
        dW_cr = (raw_dW_cr * (W_MAX - state.W_csnn_to_ring)  - dec_cr) * mask_conj
        dW_sr = (raw_dW_sr * (W_MAX - state.W_stdp_to_ring)  - dec_sr) * mask_conj
        dW_dr = (raw_dW_dr * (W_MAX - state.W_tof_to_ring)   - dec_dr) * mask_conj

        state = state._replace(
            trace_csnn=new_trace_csnn, trace_stdp=new_trace_stdp, trace_tof=new_trace_tof,
            trace_place=new_trace_place, trace_ring=new_trace_ring
        )

        return state, dW_cp, dW_sp, dW_dp, dW_cr, dW_sr, dW_dr, dW_seq

    def compute_confidence_with_gates(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_xy, heading, ring_bump):
        B = vis_csnn.shape[0]

        # Place Network
        I_csnn_place = jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place)
        I_stdp_place = jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)
        I_tof_place = jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)
        I_seq_place = jnp.einsum('bp,bpq->bq', state.trace_place, state.W_seq_to_place)
        
        # 🌟 THE FIX: Vision and ToF are Additive. STDP and Sequence are Multiplicative Modulators.
        primary_senses = I_csnn_place + I_tof_place 
        I_place = jnp.maximum(0.0, primary_senses * (1.0 + GATING_STRENGTH * I_stdp_place) * (1.0 + SEQ_GATING_STRENGTH * I_seq_place))
        
        # 🌟 V4 UPGRADE: Ring Network Conjunctive Recall
        joint_csnn_ring = jnp.einsum('bv,bpvr->bpr', vis_csnn, state.W_csnn_to_ring)
        joint_stdp_ring = jnp.einsum('bs,bpsr->bpr', vis_stdp, state.W_stdp_to_ring)
        joint_tof_ring  = jnp.einsum('bd,bpdr->bpr', tof_features, state.W_tof_to_ring)
        
        # 🌟 THE FIX: Vision and ToF are Additive for Ring Cells too!
        primary_ring_senses = joint_csnn_ring + joint_tof_ring
        
        # 🌟 THE INJECTION: Multiply by the IMU's Ring CANN
        joint_ring = jnp.maximum(0.0, 
            primary_ring_senses * 
            (1.0 + GATING_STRENGTH * joint_stdp_ring) * 
            (1.0 + HEADING_INERTIA_STRENGTH * ring_bump[:, None, :])
        )

        # 🌟 Collapse the 3D tensor using the newly calculated visual I_place as the context key!
        I_place_norm = I_place / (jnp.max(I_place, axis=1, keepdims=True) + 1e-8)
        I_ring = jnp.einsum('bpr,bp->br', joint_ring, I_place_norm)

        recalled_place = jnp.max(I_place, axis=1)
        # 🌟 Updated to use max of the Ring activity
        recalled_ring_max = jnp.max(I_ring, axis=1)

        # Gate 1: Visual Distinctiveness
        vision_activity = jnp.mean(jnp.abs(vis_csnn), axis=1)
        is_distinctive = vision_activity > self.SPARSITY_THRESH

        raw_recalled_place = jnp.max(I_place, axis=1)
        
        # 🌟 THE TRUE NORMALIZATION FIX
        # Calculate the absolute maximum theoretical energy if the memory was a 100% perfect match
        max_csnn = jnp.sum(jnp.abs(vis_csnn), axis=1)
        max_stdp = jnp.sum(jnp.abs(vis_stdp), axis=1)
        max_tof  = jnp.sum(jnp.abs(tof_features), axis=1)
        max_seq  = jnp.sum(jnp.abs(state.trace_place), axis=1)
        
        # 🌟 MATCH THE ADDITIVE LOGIC FOR THE DENOMINATOR
        max_possible_energy = (
            (max_csnn + max_tof) * 
            (1.0 + GATING_STRENGTH * max_stdp) * 
            (1.0 + SEQ_GATING_STRENGTH * max_seq)
        )
        
        # Now, recalled_place is a pure percentage [0.0 to 1.0]!
        recalled_place = raw_recalled_place / (max_possible_energy + 1e-8) 

        # Gate 2: Match
        has_learned_memory = recalled_place > 0.010 
        is_match = recalled_place > self.MATCH_THRESH

        # Gate 3: Anti-Aliasing (Optimized with top_k)
        top16_place = jax.lax.top_k(I_place, 16)[0].sum(axis=1)
        conc_place  = top16_place / (I_place.sum(axis=1) + 1e-8)

        top4_ring   = jax.lax.top_k(I_ring, 4)[0].sum(axis=1)
        conc_ring   = top4_ring / (I_ring.sum(axis=1) + 1e-8)

        is_place_anti_aliased = jnp.where(has_learned_memory, conc_place > TOFPOP_ANTI_ALIAS_POS, True) 
        is_ring_anti_aliased  = jnp.where(has_learned_memory, conc_ring > TOFPOP_ANTI_ALIAS_HEA, True) 
        is_anti_aliased = is_place_anti_aliased & is_ring_anti_aliased

        # Gate 4: Plausibility (Position)
        peak_idx_place = jnp.argmax(I_place, axis=1)
        recalled_x = self.pc_preferred_locs[peak_idx_place, 0]
        recalled_y = self.pc_preferred_locs[peak_idx_place, 1]

        jump = jnp.sqrt((recalled_x - pose_xy[:, 0])**2 + (recalled_y - pose_xy[:, 1])**2)
        is_plausible = jump < PLAUSIBILITY_THRESH 

        # =================================================================
        # 🌟 NEW GATE 7: SYNAPTIC SATURATION (Memory Maturity)
        # =================================================================
        winning_weights = jnp.take_along_axis(
            state.W_csnn_to_place, 
            peak_idx_place[:, None, None], 
            axis=2
        )[:, :, 0]
        
        top_synapses = jax.lax.top_k(winning_weights, 8)[0]
        memory_maturity = jnp.mean(top_synapses, axis=1)
        is_mature = memory_maturity > 0.50 

        # =================================================================
        # 🌟 V4.8 FIX: Use Population Vector Coding (Centroid) for Heading
        # =================================================================
        angles = self.ring_preferred_th # Range: [0 ... 2pi]
        p_ring_dist = I_ring / (I_ring.sum(axis=1, keepdims=True) + 1e-8)
        
        # Calculate complex mean to get circular continuity
        sin_sum = jnp.sum(jnp.sin(angles) * p_ring_dist, axis=1)
        cos_sum = jnp.sum(jnp.cos(angles) * p_ring_dist, axis=1)
        
        # arctan2 standardizes the output to (-pi, pi] automatically
        recalled_heading = jnp.arctan2(sin_sum, cos_sum) 
        
        # Force current heading to the same (-pi, pi] range for comparison
        heading_standard = (heading + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        # Use circular subtraction for the plausibility gate
        angle_diff = jnp.abs(recalled_heading - heading_standard)
        angle_diff = jnp.where(angle_diff > jnp.pi, 2 * jnp.pi - angle_diff, angle_diff)
        
        is_heading_plausible = angle_diff < HEADING_THRESH

        # Gate 5: Self-Match
        is_pos_drifting = jump > SELF_MATCH_THRESH
        is_heading_drifting = angle_diff > RING_SELF_MATCH_THRESH
        is_not_self_matching = is_pos_drifting | is_heading_drifting

        is_confident_raw = (
            is_distinctive & is_match & is_anti_aliased & 
            is_plausible & is_heading_plausible & is_not_self_matching &
            is_mature
        )

        # Gate 6: Temporal Consistency
        new_confidence_ema = 0.5 * state.confidence_ema + 0.5 * is_confident_raw.astype(jnp.float32)
        state = state._replace(confidence_ema=new_confidence_ema)
        
        is_temporally_consistent = new_confidence_ema > 0.80
        is_confident = is_confident_raw & is_temporally_consistent

        debug_gates = {
            "G1_Distinctive": is_distinctive,
            "G2_Match": is_match,
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
            # 🌟 ADD THESE THREE NEW LINES:
            "Raw_Vis_Act": vision_activity,
            "Raw_Match": recalled_place,
            "Conc_Ring": conc_ring
        }

        return state, is_confident, peak_idx_place, debug_gates

    def initialize_from_pose(self, state: PlaceCellState, pose_bump, ring_bump=None):
        new_smooth_place = jnp.maximum(state.I_correction_smooth_place, pose_bump[:, :self.n_place])
        
        if ring_bump is not None:
            new_smooth_ring = jnp.maximum(state.I_correction_smooth_ring, ring_bump)
            return state._replace(I_correction_smooth_place=new_smooth_place, I_correction_smooth_ring=new_smooth_ring)
        
        return state._replace(I_correction_smooth_place=new_smooth_place)

    def decode_position(self, r_place):
        peak_idx = jnp.argmax(r_place, axis=1) 
        x_dec = self.pc_preferred_locs[peak_idx, 0]
        y_dec = self.pc_preferred_locs[peak_idx, 1]
        return jnp.stack([x_dec, y_dec], axis=1)

    def decode_heading(self, r_ring):
        angles = self.ring_preferred_th
        p = r_ring / (r_ring.sum(axis=1, keepdims=True) + 1e-8)
        sin_sum = jnp.sum(jnp.sin(angles) * p, axis=1)
        cos_sum = jnp.sum(jnp.cos(angles) * p, axis=1)
        return jnp.arctan2(sin_sum, cos_sum) % (2 * jnp.pi)

    def get_place_activity_flat(self, state: PlaceCellState):
        return state.I_correction_smooth_place

    def get_ring_activity_flat(self, state: PlaceCellState):
        return state.I_correction_smooth_ring