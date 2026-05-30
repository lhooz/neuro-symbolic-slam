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
N_POSE        = 579    # The continuous 2D Grid Cell dimension coming from the IMU

# 🚨 THE FIX: Scaled up to 512-dim with 16 active bits (True HDC Orthogonality!)
N_PLACE       = 128
RING_N        = 64
GAIN          = 4.0   
CORRECTION_GAIN      = 2.0    
RING_CORRECTION_GAIN = 2.0

GATING_STRENGTH = 8.0  
SEQ_GATING_STRENGTH = 1.0  
HEADING_INERTIA_STRENGTH = 1.0  
LEARNING_CORE_THRESH = 0.50  
HOPFIELD_BETA         = 2.5    

# gating thresholds
HEADING_THRESH        = 0.60
PLAUSIBILITY_THRESH   = 1.00

# 🌟 BUG FIX DELETED: RING_SELF_MATCH_THRESH (It was punishing perfect alignments!)

SPARSITY_THRESH       = 0.15   
TOFPOP_ANTI_ALIAS_POS = 0.50   
TOFPOP_ANTI_ALIAS_HEA = 0.50   
MATCH_THRESH          = 0.60   
MATURITY_GATE         = 0.60   
MATCH_TOF             = 0.70   # 🌟 BUG FIX: Lowered slightly to allow stable 2/3 ray matches

# EMA trace parameters for temporal binding
TRACE_ALPHA = 0.90   

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
    
    theta_burn: jnp.ndarray
    
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

    # 🌟 BUG FIX: Updated k_spikes default to 16 to match the 512-dim expansion
    def __init__(self, key, n_csnn=N_CSNN, n_stdp=N_STDP, n_depth=N_DEPTH, fov_deg=180.0, n_pose=N_POSE, n_place=N_PLACE, k_spikes=16):
        self.n_csnn     = n_csnn
        self.n_stdp     = n_stdp
        self.n_depth    = n_depth
        self.n_pose     = n_pose 
        self.n_place    = n_place 
        self.k_spikes   = k_spikes
        self.ring_n     = RING_N
        
        self.fov_rad    = jnp.radians(fov_deg)
        self.pixel_ang_res = self.fov_rad / float(self.n_csnn) 
        self.max_blur_pixels = 4.1 
        self.dt = 0.05 
        self.dynamic_saccade_thresh = (self.max_blur_pixels * self.pixel_ang_res) / self.dt

        self.ring_preferred_th  = build_ring_preferred_th()

        self.W_dg = jax.random.normal(key, (self.n_pose, self.n_place))

        k_vis_hash, key = random.split(key)
        self.W_vis_hash = jax.random.normal(k_vis_hash, (self.n_csnn, self.n_place))

    def get_place_barcode(self, pose_bump):
        """Converts the 579-dim continuous Grid Key into a sparse Barcode."""
        place_voltages = jnp.dot(pose_bump, self.W_dg)
        threshold = jax.lax.top_k(place_voltages, self.k_spikes)[0][:, -1:]
        return jnp.where(place_voltages >= threshold, 1.0, 0.0)

    def init_state(self, B) -> PlaceCellState:
        return PlaceCellState(
            W_csnn_to_place = jnp.zeros((B, N_CSNN, self.n_place), dtype=jnp.float32),
            W_stdp_to_place = jnp.zeros((B, N_STDP, self.n_place), dtype=jnp.float32), 
            W_tof_to_place  = jnp.zeros((B, self.n_depth, self.n_place), dtype=jnp.float32),
            W_seq_to_place  = jnp.zeros((B, self.n_place, self.n_place), dtype=jnp.float32),
            
            W_csnn_to_ring  = jnp.zeros((B, self.n_place, N_CSNN, self.ring_n), dtype=jnp.float32),
            W_stdp_to_ring  = jnp.zeros((B, self.n_place, N_STDP, self.ring_n), dtype=jnp.float32),  
            W_tof_to_ring   = jnp.zeros((B, self.n_place, self.n_depth, self.ring_n), dtype=jnp.float32),
            theta_burn      = jnp.zeros((B, self.n_place, self.ring_n), dtype=jnp.float32),
            
            trace_csnn  = jnp.zeros((B, N_CSNN), dtype=jnp.float32),
            trace_stdp  = jnp.zeros((B, N_STDP), dtype=jnp.float32), 
            trace_tof   = jnp.zeros((B, self.n_depth), dtype=jnp.float32),
            trace_place = jnp.zeros((B, self.n_place), dtype=jnp.float32),   
            trace_ring  = jnp.zeros((B, self.ring_n), dtype=jnp.float32),    
            confidence_ema = jnp.zeros((B,), dtype=jnp.float32)
        )

    def __call__(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, heading=None, angular_vel=None, learn=True, confidence=None):
        B = vis_csnn.shape[0]

        I_csnn_place = jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place)
        I_stdp_place = jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)
        I_tof_place  = jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)
        I_seq_place  = jnp.einsum('bp,bpq->bq', state.trace_place, state.W_seq_to_place)

        primary_senses = I_csnn_place * I_tof_place 
        raw_I_place = jnp.maximum(0.0, primary_senses * (1.0 + GATING_STRENGTH * I_stdp_place) * (1.0 + SEQ_GATING_STRENGTH * I_seq_place))
        
        norm_raw_place = raw_I_place / (jnp.max(raw_I_place, axis=1, keepdims=True) + 1e-8)
        I_place = jnp.power(norm_raw_place, HOPFIELD_BETA) * jnp.max(raw_I_place, axis=1, keepdims=True)

        joint_csnn_ring = jnp.einsum('bv,bpvr->bpr', vis_csnn, state.W_csnn_to_ring)
        joint_stdp_ring = jnp.einsum('bs,bpsr->bpr', vis_stdp, state.W_stdp_to_ring)
        joint_tof_ring  = jnp.einsum('bd,bpdr->bpr', tof_features, state.W_tof_to_ring)
        
        primary_ring_senses = joint_csnn_ring * joint_tof_ring
        joint_ring = jnp.maximum(0.0, 
            primary_ring_senses * (1.0 + GATING_STRENGTH * joint_stdp_ring) * (1.0 + HEADING_INERTIA_STRENGTH * ring_bump[:, None, :])
        )

        place_barcode = self.get_place_barcode(pose_bump)
        
        pose_bump_norm = place_barcode / (jnp.max(place_barcode, axis=1, keepdims=True) + 1e-8)
        raw_I_ring = jnp.einsum('bpr,bp->br', joint_ring, pose_bump_norm)
        
        norm_raw_ring = raw_I_ring / (jnp.max(raw_I_ring, axis=1, keepdims=True) + 1e-8)
        I_ring = jnp.power(norm_raw_ring, HOPFIELD_BETA) * jnp.max(raw_I_ring, axis=1, keepdims=True)

        if learn and heading is not None: 
            state, dW_cp, dW_sp, dW_dp, dW_cr, dW_sr, dW_dr, dW_seq, new_theta_burn = self._updateHebbian(state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, heading, angular_vel, confidence)
            state = state._replace(
                W_csnn_to_place = jnp.clip(state.W_csnn_to_place + dW_cp, 0.0, 1.0),
                W_stdp_to_place = jnp.clip(state.W_stdp_to_place + dW_sp, 0.0, 1.0),
                W_tof_to_place  = jnp.clip(state.W_tof_to_place  + dW_dp, 0.0, 1.0),
                W_seq_to_place  = jnp.clip(state.W_seq_to_place  + dW_seq, 0.0, 1.0),
                W_csnn_to_ring  = jnp.clip(state.W_csnn_to_ring  + dW_cr, 0.0, 1.0),
                W_stdp_to_ring  = jnp.clip(state.W_stdp_to_ring  + dW_sr, 0.0, 1.0),
                W_tof_to_ring   = jnp.clip(state.W_tof_to_ring   + dW_dr, 0.0, 1.0),
                theta_burn      = new_theta_burn 
            )

        return state, (I_place, I_ring)

    def forward_mapping(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump=None, heading=None, angular_vel=None, learn=True, confidence=None):
        return self(state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, heading=heading, angular_vel=angular_vel, learn=learn, confidence=confidence)

    def _updateHebbian(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, heading, angular_vel=None, confidence=None):
        ETA_W = 0.05
        W_MAX = 1.0  
        LAMBDA_W = 0.0025
        PASSIVE_LEAK = 0.001

        if confidence is not None:
            novelty = 1.0 - confidence
            eta_batch = ETA_W * novelty[:, None, None]
        else:
            eta_batch = ETA_W

        place_barcode = self.get_place_barcode(pose_bump)

        new_trace_place = TRACE_ALPHA * state.trace_place + (1 - TRACE_ALPHA) * place_barcode
        new_trace_ring  = TRACE_ALPHA * state.trace_ring  + (1 - TRACE_ALPHA) * ring_bump

        if angular_vel is not None:
            is_stable = jnp.where(jnp.abs(angular_vel) < self.dynamic_saccade_thresh, 1.0, 0.0)
            eta_saccadic_2d = eta_batch * is_stable[:, None, None]
            eta_saccadic_3d = eta_batch * is_stable[:, None, None, None]
        else:
            eta_saccadic_2d = eta_batch
            eta_saccadic_3d = eta_batch

        base_pose_mask = place_barcode
        mask_place = base_pose_mask[:, None, :] 
        mask_place_4d = base_pose_mask[:, :, None, None]
        
        peak_ring_val = jnp.max(ring_bump, axis=1, keepdims=True)
        mask_ring_4d  = jnp.where(ring_bump >= peak_ring_val * LEARNING_CORE_THRESH, 1.0, 0.0)[:, None, None, :]
        mask_conj = mask_place_4d * mask_ring_4d

        raw_dW_cp = eta_saccadic_2d * jnp.einsum('bv,bp->bvp', vis_csnn, place_barcode)
        raw_dW_sp = eta_saccadic_2d * jnp.einsum('bs,bp->bsp', vis_stdp, place_barcode)
        raw_dW_dp = eta_saccadic_2d * jnp.einsum('bd,bp->bdp', tof_features, place_barcode)
        raw_dW_seq = eta_batch * jnp.einsum('bi,bj->bij', state.trace_place, place_barcode)
        
        synaptic_lock = jnp.where(jnp.sum(state.W_tof_to_ring, axis=2, keepdims=True) < 1.0, 1.0, 0.0)
        
        joint_vis = jnp.einsum('bv,bpvr->bpr', vis_csnn, state.W_csnn_to_ring)
        active_place_mask = base_pose_mask[:, :, None]
        max_vis_match = jnp.max(joint_vis * active_place_mask, axis=(1, 2))
        
        max_energy = jnp.sum(jnp.abs(vis_csnn), axis=1)
        match_ratio = max_vis_match / (max_energy + 1e-8)
        
        is_novel = jnp.where(match_ratio < 0.75, 1.0, 0.0)
        novelty_mask = is_novel[:, None, None, None]

        flash_mask = mask_conj * synaptic_lock * novelty_mask
        
        flash_mask_3d = jnp.squeeze(flash_mask, axis=2)
        heading_expanded = heading[:, None, None]
        new_theta_burn = jnp.where(flash_mask_3d > 0.5, heading_expanded, state.theta_burn)

        vis_error = vis_csnn[:, None, :, None] - state.W_csnn_to_ring
        dW_cr = flash_mask * vis_error
        
        stdp_error = vis_stdp[:, None, :, None] - state.W_stdp_to_ring
        dW_sr = flash_mask * stdp_error
        
        tof_error = tof_features[:, None, :, None] - state.W_tof_to_ring
        dW_dr = flash_mask * tof_error

        # 🌟 BUG FIX: Included STDP multiplication in current_energy to correctly scale the spatial match gate
        current_energy = (jnp.sum(jnp.abs(vis_csnn), axis=1) * jnp.sum(jnp.abs(tof_features), axis=1)) * (1.0 + GATING_STRENGTH * jnp.sum(jnp.abs(vis_stdp), axis=1))
        
        best_mem_energy = jnp.max(
            (jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place) * 
             jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)) * 
            (1.0 + GATING_STRENGTH * jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)),
            axis=1
        )
        
        spatial_match_ratio = best_mem_energy / (current_energy + 1e-8)
        spatial_novelty_mask = jnp.where(spatial_match_ratio < 0.75, 1.0, 0.0)[:, None, None]

        dW_cp = (raw_dW_cp * (W_MAX - state.W_csnn_to_place) - (eta_batch * PASSIVE_LEAK * state.W_csnn_to_place)) * mask_place * spatial_novelty_mask
        dW_sp = (raw_dW_sp * (W_MAX - state.W_stdp_to_place) - (eta_batch * PASSIVE_LEAK * state.W_stdp_to_place)) * mask_place * spatial_novelty_mask
        dW_dp = (raw_dW_dp * (W_MAX - state.W_tof_to_place)  - (eta_batch * PASSIVE_LEAK * state.W_tof_to_place))  * mask_place * spatial_novelty_mask
        
        dec_seq = eta_batch * LAMBDA_W * place_barcode[:, None, :] * state.W_seq_to_place 
        dW_seq = (raw_dW_seq * (W_MAX - state.W_seq_to_place) - dec_seq) * mask_place

        state = state._replace(
            trace_place=new_trace_place, 
            trace_ring=new_trace_ring
        )

        return state, dW_cp, dW_sp, dW_dp, dW_cr, dW_sr, dW_dr, dW_seq, new_theta_burn

    @partial(jax.jit, static_argnames=['self'])
    def apply_post_relaxation_update(self, state: PlaceCellState, spatial_barcode, r_idx, aligned_csnn, aligned_stdp, fov_mask, ring_lr=0.05):
        batch_idx = jnp.arange(aligned_csnn.shape[0])

        aligned_csnn_exp = aligned_csnn[:, None, :] 
        aligned_stdp_exp = aligned_stdp[:, None, :] 
        fov_mask_exp = fov_mask[:, None, :]         

        old_w_csnn = state.W_csnn_to_ring[batch_idx, :, :, r_idx] 
        old_w_stdp = state.W_stdp_to_ring[batch_idx, :, :, r_idx]

        delta_csnn = aligned_csnn_exp - old_w_csnn
        delta_stdp = aligned_stdp_exp - old_w_stdp

        lr_up = 0.10    
        lr_down = 0.001 

        effective_lr_csnn = jnp.where(delta_csnn > 0, lr_up, lr_down)
        effective_lr_stdp = jnp.where(delta_stdp > 0, lr_up, lr_down)

        dW_csnn = effective_lr_csnn * fov_mask_exp * delta_csnn * spatial_barcode[:, :, None]
        dW_stdp = effective_lr_stdp * fov_mask_exp * delta_stdp * spatial_barcode[:, :, None]

        new_W_csnn = state.W_csnn_to_ring.at[batch_idx, :, :, r_idx].add(dW_csnn)
        new_W_stdp = state.W_stdp_to_ring.at[batch_idx, :, :, r_idx].add(dW_stdp)

        return state._replace(
            W_csnn_to_ring = jnp.clip(new_W_csnn, 0.0, 1.0),
            W_stdp_to_ring = jnp.clip(new_W_stdp, 0.0, 1.0)
        )

    def compute_confidence_with_gates(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, heading, ring_bump):
        B = vis_csnn.shape[0]
        live_barcode = self.get_place_barcode(pose_bump)

        vis_hash_voltages = jnp.dot(vis_csnn, self.W_vis_hash)
        vis_thresh = jax.lax.top_k(vis_hash_voltages, self.k_spikes)[0][:, -1:]
        visual_barcode = jnp.where(vis_hash_voltages >= vis_thresh, 1.0, 0.0)

        I_csnn_place = jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place)
        I_stdp_place = jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)
        I_tof_place  = jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)
        I_seq_place  = jnp.einsum('bp,bpq->bq', state.trace_place, state.W_seq_to_place)
        
        primary_senses = I_csnn_place * I_tof_place 
        pure_sensory_place = jnp.maximum(0.0, primary_senses * (1.0 + GATING_STRENGTH * I_stdp_place))
        
        norm_sensory_place = pure_sensory_place / (jnp.max(pure_sensory_place, axis=1, keepdims=True) + 1e-8)
        I_place_sensory = jnp.power(norm_sensory_place, HOPFIELD_BETA) * jnp.max(pure_sensory_place, axis=1, keepdims=True)

        raw_I_place_biased = pure_sensory_place * (1.0 + SEQ_GATING_STRENGTH * I_seq_place)
        norm_biased_place = raw_I_place_biased / (jnp.max(raw_I_place_biased, axis=1, keepdims=True) + 1e-8)
        I_place = jnp.power(norm_biased_place, HOPFIELD_BETA) * jnp.max(raw_I_place_biased, axis=1, keepdims=True)

        joint_csnn_ring = jnp.einsum('bv,bpvr->bpr', vis_csnn, state.W_csnn_to_ring)
        joint_stdp_ring = jnp.einsum('bs,bpsr->bpr', vis_stdp, state.W_stdp_to_ring)
        joint_tof_ring  = jnp.einsum('bd,bpdr->bpr', tof_features, state.W_tof_to_ring)
        
        primary_ring_senses = joint_csnn_ring * joint_tof_ring
        pure_sensory_ring = jnp.maximum(0.0, primary_ring_senses * (1.0 + GATING_STRENGTH * joint_stdp_ring))
        
        I_place_sensory_norm = I_place_sensory / (jnp.max(I_place_sensory, axis=1, keepdims=True) + 1e-8)
        raw_I_ring_sensory = jnp.einsum('bpr,bp->br', pure_sensory_ring, I_place_sensory_norm)
        
        norm_sensory_ring = raw_I_ring_sensory / (jnp.max(raw_I_ring_sensory, axis=1, keepdims=True) + 1e-8)
        I_ring_sensory = jnp.power(norm_sensory_ring, HOPFIELD_BETA) * jnp.max(raw_I_ring_sensory, axis=1, keepdims=True)

        joint_ring_biased = pure_sensory_ring * (1.0 + HEADING_INERTIA_STRENGTH * ring_bump[:, None, :])
        I_place_norm = I_place / (jnp.max(I_place, axis=1, keepdims=True) + 1e-8)
        raw_I_ring_biased = jnp.einsum('bpr,bp->br', joint_ring_biased, I_place_norm)
        
        norm_biased_ring = raw_I_ring_biased / (jnp.max(raw_I_ring_biased, axis=1, keepdims=True) + 1e-8)
        I_ring = jnp.power(norm_biased_ring, HOPFIELD_BETA) * jnp.max(raw_I_ring_biased, axis=1, keepdims=True)

        recalled_place = jnp.max(I_place_sensory, axis=1)

        top16_vis = jax.lax.top_k(jnp.abs(vis_csnn), 16)[0]
        vision_activity = jnp.mean(top16_vis, axis=1)
        is_distinctive = vision_activity > SPARSITY_THRESH

        max_csnn = jnp.sum(jnp.abs(vis_csnn), axis=1)
        max_stdp = jnp.sum(jnp.abs(vis_stdp), axis=1)
        max_tof  = jnp.sum(jnp.abs(tof_features), axis=1)
        
        max_sensory_energy = (max_csnn * max_tof) * (1.0 + GATING_STRENGTH * max_stdp)
        recalled_place = recalled_place / (max_sensory_energy + 1e-8)

        has_learned_memory = recalled_place > 0.010 
        is_match = recalled_place > MATCH_THRESH

        peak_idx_place = jnp.argmax(I_place, axis=1)
        peak_ring = jnp.argmax(I_ring, axis=1)
        batch_idx = jnp.arange(B)

        recovered_thresh = jax.lax.top_k(I_place, self.k_spikes)[0][:, -1:]
        recovered_spatial_barcode = jnp.where(I_place >= recovered_thresh, 1.0, 0.0)

        reconstructed_W_tof_ring = jnp.einsum('bp,bpdr->bdr', recovered_spatial_barcode, state.W_tof_to_ring) / self.k_spikes
        winning_W_tof_ring = reconstructed_W_tof_ring[batch_idx, :, peak_ring]
        
        dot_product = jnp.sum(tof_features * winning_W_tof_ring, axis=1)
        norm_in = jnp.linalg.norm(tof_features, axis=1)
        norm_mem = jnp.linalg.norm(winning_W_tof_ring, axis=1)
        tof_match_score = dot_product / (norm_in * norm_mem + 1e-8)
        
        is_tof_match = tof_match_score > MATCH_TOF

        VIRTUAL_DISTRACTOR_ENERGY = 0.1 

        place_norm = I_place_sensory / (jnp.max(I_place_sensory, axis=1, keepdims=True) + 1e-8)
        ring_norm  = I_ring_sensory / (jnp.max(I_ring_sensory, axis=1, keepdims=True) + 1e-8)

        top_k_place = jax.lax.top_k(place_norm, self.k_spikes)[0].sum(axis=1)
        conc_place = top_k_place / (place_norm.sum(axis=1) + VIRTUAL_DISTRACTOR_ENERGY)

        top8_ring   = jax.lax.top_k(ring_norm, 8)[0].sum(axis=1)
        conc_ring   = top8_ring / (ring_norm.sum(axis=1) + VIRTUAL_DISTRACTOR_ENERGY)

        is_place_anti_aliased = jnp.where(has_learned_memory, conc_place > TOFPOP_ANTI_ALIAS_POS, True)
        is_ring_anti_aliased  = jnp.where(has_learned_memory, conc_ring > TOFPOP_ANTI_ALIAS_HEA, True) 
        is_anti_aliased = is_place_anti_aliased & is_ring_anti_aliased

        winning_weights = jnp.take_along_axis(state.W_csnn_to_place, peak_idx_place[:, None, None], axis=2)[:, :, 0]
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

        # 🌟 BUG FIX: Removed `is_not_self_matching` from the gate array so perfect overlaps are celebrated!
        is_confident_raw = (
            is_distinctive & is_match & is_anti_aliased & 
            is_heading_plausible & is_mature & is_tof_match 
        )

        new_confidence_ema = 0.5 * state.confidence_ema + 0.5 * is_confident_raw.astype(jnp.float32)
        state = state._replace(confidence_ema=new_confidence_ema)
        
        is_temporally_consistent = new_confidence_ema > 0.80
        is_confident = is_confident_raw & is_temporally_consistent

        winning_W_csnn = jnp.einsum('bp,bvp->bv', recovered_spatial_barcode, state.W_csnn_to_place) / self.k_spikes
        winning_W_stdp = jnp.einsum('bp,bsp->bs', recovered_spatial_barcode, state.W_stdp_to_place) / self.k_spikes
        winning_W_tof  = jnp.einsum('bp,bdp->bd', recovered_spatial_barcode, state.W_tof_to_place) / self.k_spikes

        reconstructed_W_csnn_ring = jnp.einsum('bp,bpvr->bvr', recovered_spatial_barcode, state.W_csnn_to_ring) / self.k_spikes
        reconstructed_W_stdp_ring = jnp.einsum('bp,bpsr->bsr', recovered_spatial_barcode, state.W_stdp_to_ring) / self.k_spikes
        
        winning_W_csnn_ring = reconstructed_W_csnn_ring[batch_idx, :, peak_ring]
        winning_W_stdp_ring = reconstructed_W_stdp_ring[batch_idx, :, peak_ring]

        all_thetas = state.theta_burn[batch_idx, :, peak_ring]
        sin_thetas = jnp.sin(all_thetas)
        cos_thetas = jnp.cos(all_thetas)
        
        sum_sin = jnp.einsum('bp,bp->b', recovered_spatial_barcode, sin_thetas)
        sum_cos = jnp.einsum('bp,bp->b', recovered_spatial_barcode, cos_thetas)
        winning_theta_burn = jnp.arctan2(sum_sin, sum_cos)

        winning_W_csnn_ring_ui = winning_W_csnn_ring / (jnp.max(winning_W_csnn_ring, axis=1, keepdims=True) + 1e-8)
        winning_W_stdp_ring_ui = winning_W_stdp_ring / (jnp.max(winning_W_stdp_ring, axis=1, keepdims=True) + 1e-8)
        winning_W_tof_ring_ui  = winning_W_tof_ring / (jnp.max(winning_W_tof_ring, axis=1, keepdims=True) + 1e-8)

        debug_gates = {
            "Recovered_Spatial_Barcode": recovered_spatial_barcode, 
            "Visual_Barcode": visual_barcode,  
            "Live_Barcode": live_barcode, 
            "Peak_Ring": peak_ring,
            "Peak_Theta_Burn": winning_theta_burn, 
            "G1_Distinctive": is_distinctive,
            "G2_Match": is_match,
            "G2b_ToFMatch": is_tof_match, 
            "ToF_Score": tof_match_score,
            "G3_AntiAlias": is_anti_aliased,
            "G4b_HeadPlausible": is_heading_plausible,
            "G6_TemporalEMA": new_confidence_ema,
            "G7_Mature": is_mature,
            "Maturity_Lvl": memory_maturity,
            "Raw_Conf": is_confident_raw,
            "Final_Conf": is_confident,
            "Conc_Place": conc_place,
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
            "Debug_Mem_CSNN_Ring": winning_W_csnn_ring_ui,
            "Debug_Mem_STDP_Ring": winning_W_stdp_ring_ui,
            "Debug_Mem_ToF_Ring": winning_W_tof_ring_ui,
            "Debug_I_Place": I_place,
            "Debug_I_Ring": I_ring
        }

        return state, is_confident, peak_idx_place, debug_gates

    def initialize_from_pose(self, state: PlaceCellState, pose_bump, ring_bump=None):
        place_barcode = self.get_place_barcode(pose_bump)
        new_trace_place = jnp.maximum(state.trace_place, place_barcode)
        if ring_bump is not None:
            new_trace_ring = jnp.maximum(state.trace_ring, ring_bump)
            return state._replace(trace_place=new_trace_place, trace_ring=new_trace_ring)
        return state._replace(trace_place=new_trace_place)

    def decode_heading(self, r_ring):
        angles = self.ring_preferred_th
        p = r_ring / (r_ring.sum(axis=1, keepdims=True) + 1e-8)
        sin_sum = jnp.sum(jnp.sin(angles) * p, axis=1)
        cos_sum = jnp.sum(jnp.cos(angles) * p, axis=1)
        return jnp.arctan2(sin_sum, cos_sum) % (2 * jnp.pi)