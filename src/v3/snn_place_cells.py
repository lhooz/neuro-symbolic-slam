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
CORRECTION_SMOOTH    = 0.99  # temporal smoothing of correction signal
RING_CORRECTION_SMOOTH = 0.99
GATING_STRENGTH = 2.0  # STDP Multiplier power

# SOTAv1 gating thresholds
TOFPOP_ANTI_ALIAS_POS = 0.99   # top-16 concentration threshold for anti-aliasing
TOFPOP_ANTI_ALIAS_HEA = 0.99   # top-16 concentration threshold for anti-aliasing
HEADING_THRESH        = 1.50
PLAUSIBILITY_THRESH   = 3.00
SELF_MATCH_THRESH     = 0.30   # meters — restored: blocks "molasses effect" backward corrections

# EMA trace parameters for temporal binding (Fix #3)
# Event cameras are sparse/async — EMA "synaptic afterglow" binds vision→pose
TRACE_ALPHA = 0.90   # slow decay: 90% carry-over, 10% new input

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
    W_stdp_to_place: jnp.ndarray   # NEW: STDP Memory
    W_tof_to_place: jnp.ndarray
    
    W_csnn_to_ring: jnp.ndarray
    W_stdp_to_ring: jnp.ndarray    # NEW: STDP Memory
    W_tof_to_ring: jnp.ndarray
    
    I_correction_smooth_place: jnp.ndarray
    I_correction_smooth_ring: jnp.ndarray
    
    trace_csnn: jnp.ndarray
    trace_stdp: jnp.ndarray        # NEW: STDP Trace
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

    SPARSITY_THRESH  = 0.05  # match the average baseline magnitude of v_out vectors
    MATCH_THRESH     = 0.01  # Seed threshold

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
            W_stdp_to_place = jnp.zeros((B, N_STDP, self.n_place), dtype=jnp.float32), # NEW
            W_tof_to_place  = jnp.zeros((B, self.n_depth, self.n_place), dtype=jnp.float32),
            
            W_csnn_to_ring  = jnp.zeros((B, N_CSNN, self.ring_n), dtype=jnp.float32),
            W_stdp_to_ring  = jnp.zeros((B, N_STDP, self.ring_n), dtype=jnp.float32),  # NEW
            W_tof_to_ring   = jnp.zeros((B, self.n_depth, self.ring_n), dtype=jnp.float32),
            
            I_correction_smooth_place = jnp.zeros((B, self.n_place), dtype=jnp.float32),  
            I_correction_smooth_ring  = jnp.zeros((B, self.ring_n), dtype=jnp.float32),   
            
            trace_csnn  = jnp.zeros((B, N_CSNN), dtype=jnp.float32),
            trace_stdp  = jnp.zeros((B, N_STDP), dtype=jnp.float32), # NEW
            trace_tof   = jnp.zeros((B, self.n_depth), dtype=jnp.float32),
            trace_place = jnp.zeros((B, self.n_place), dtype=jnp.float32),   
            trace_ring  = jnp.zeros((B, self.ring_n), dtype=jnp.float32),    
            confidence_ema = jnp.zeros((B,), dtype=jnp.float32)
        )

    def __call__(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, learn=True, confidence=None):
        B = vis_csnn.shape[0]

        # ── 🌟 DENDRITIC RECALL ──
        # Place Network
        I_csnn_place = jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place)
        I_stdp_place = jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)
        I_tof_place  = jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)
        I_place = jnp.maximum(0.0, I_csnn_place * (1.0 + GATING_STRENGTH * I_stdp_place) * I_tof_place)

        # Ring Network
        I_csnn_ring = jnp.einsum('bv,bvr->br', vis_csnn, state.W_csnn_to_ring)
        I_stdp_ring = jnp.einsum('bs,bsr->br', vis_stdp, state.W_stdp_to_ring)
        I_tof_ring  = jnp.einsum('bd,bdr->br', tof_features, state.W_tof_to_ring)
        I_ring = jnp.maximum(0.0, I_csnn_ring * (1.0 + GATING_STRENGTH * I_stdp_ring) * I_tof_ring)

        # ── Temporal smoothing
        new_smooth_place = CORRECTION_SMOOTH * state.I_correction_smooth_place + (1 - CORRECTION_SMOOTH) * I_place
        new_smooth_ring  = RING_CORRECTION_SMOOTH * state.I_correction_smooth_ring + (1 - RING_CORRECTION_SMOOTH) * I_ring

        state = state._replace(
            I_correction_smooth_place=new_smooth_place,
            I_correction_smooth_ring=new_smooth_ring
        )

        # ── Hebbian learning
        if learn:
            state, dW_cp, dW_sp, dW_dp, dW_cr, dW_sr, dW_dr = self._updateHebbian(state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, confidence)
            state = state._replace(
                W_csnn_to_place = jnp.clip(state.W_csnn_to_place + dW_cp, 0.0, 1.0),
                W_stdp_to_place = jnp.clip(state.W_stdp_to_place + dW_sp, 0.0, 1.0),
                W_tof_to_place  = jnp.clip(state.W_tof_to_place  + dW_dp, 0.0, 1.0),
                W_csnn_to_ring  = jnp.clip(state.W_csnn_to_ring  + dW_cr, 0.0, 1.0),
                W_stdp_to_ring  = jnp.clip(state.W_stdp_to_ring  + dW_sr, 0.0, 1.0),
                W_tof_to_ring   = jnp.clip(state.W_tof_to_ring   + dW_dr, 0.0, 1.0)
            )

        return state, (I_place, I_ring, new_smooth_place, new_smooth_ring)

    def forward_mapping(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump=None, learn=True, confidence=None):
        return self(state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, learn=learn, confidence=confidence)

    def _updateHebbian(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump, confidence=None):
        ETA_W = 0.005 
        W_MAX = 1.0  
        LAMBDA_W = 0.02  

        if confidence is not None:
            novelty = 1.0 - confidence
            eta_batch = ETA_W * novelty[:, None, None]
        else:
            eta_batch = ETA_W

        # Update traces
        new_trace_csnn  = TRACE_ALPHA * state.trace_csnn  + (1 - TRACE_ALPHA) * vis_csnn
        new_trace_stdp  = TRACE_ALPHA * state.trace_stdp  + (1 - TRACE_ALPHA) * vis_stdp
        new_trace_tof   = TRACE_ALPHA * state.trace_tof   + (1 - TRACE_ALPHA) * tof_features
        new_trace_place = TRACE_ALPHA * state.trace_place + (1 - TRACE_ALPHA) * pose_bump
        new_trace_ring  = TRACE_ALPHA * state.trace_ring  + (1 - TRACE_ALPHA) * ring_bump

        # 🌟 INDEPENDENT LTP (Notice STDP learns 5x faster!)
        raw_dW_cp = eta_batch * jnp.einsum('bv,bp->bvp', new_trace_csnn, new_trace_place)
        raw_dW_sp = (eta_batch * 5.0) * jnp.einsum('bs,bp->bsp', new_trace_stdp, new_trace_place)
        raw_dW_dp = eta_batch * jnp.einsum('bd,bp->bdp', new_trace_tof, new_trace_place)
        
        raw_dW_cr = eta_batch * jnp.einsum('bv,br->bvr', new_trace_csnn, new_trace_ring)
        raw_dW_sr = (eta_batch * 5.0) * jnp.einsum('bs,br->bsr', new_trace_stdp, new_trace_ring)
        raw_dW_dr = eta_batch * jnp.einsum('bd,br->bdr', new_trace_tof, new_trace_ring)

        # Decay
        dec_cp = eta_batch * LAMBDA_W * new_trace_place[:, None, :] * state.W_csnn_to_place
        dec_sp = (eta_batch * 5.0) * LAMBDA_W * new_trace_place[:, None, :] * state.W_stdp_to_place
        dec_dp = eta_batch * LAMBDA_W * new_trace_place[:, None, :] * state.W_tof_to_place
        
        dec_cr = eta_batch * LAMBDA_W * new_trace_ring[:, None, :]  * state.W_csnn_to_ring
        dec_sr = (eta_batch * 5.0) * LAMBDA_W * new_trace_ring[:, None, :]  * state.W_stdp_to_ring
        dec_dr = eta_batch * LAMBDA_W * new_trace_ring[:, None, :]  * state.W_tof_to_ring

        # Structural Cages
        mask_place = jnp.where(pose_bump > 0.05, 1.0, 0.0)[:, None, :] 
        mask_ring  = jnp.where(ring_bump > 0.05, 1.0, 0.0)[:, None, :] 

        # Masked Updates
        dW_cp = (raw_dW_cp * (W_MAX - state.W_csnn_to_place) - dec_cp) * mask_place
        dW_sp = (raw_dW_sp * (W_MAX - state.W_stdp_to_place) - dec_sp) * mask_place
        dW_dp = (raw_dW_dp * (W_MAX - state.W_tof_to_place)  - dec_dp) * mask_place
        
        dW_cr = (raw_dW_cr * (W_MAX - state.W_csnn_to_ring)  - dec_cr) * mask_ring
        dW_sr = (raw_dW_sr * (W_MAX - state.W_stdp_to_ring)  - dec_sr) * mask_ring
        dW_dr = (raw_dW_dr * (W_MAX - state.W_tof_to_ring)   - dec_dr) * mask_ring

        state = state._replace(
            trace_csnn=new_trace_csnn,
            trace_stdp=new_trace_stdp,
            trace_tof=new_trace_tof,
            trace_place=new_trace_place,
            trace_ring=new_trace_ring
        )

        return state, dW_cp, dW_sp, dW_dp, dW_cr, dW_sr, dW_dr

    def compute_confidence_with_gates(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_xy, heading):
        B = vis_csnn.shape[0]

        # 1. Basal Anchor (CSNN)
        I_csnn_place = jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place)
        
        # 2. Apical Modulator (STDP Texture)
        I_stdp_place = jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)
        
        # 3. Apical Context (ToF Depth)
        I_tof_place = jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)
        
        # 🌟 THE MAGIC: Multiplicative Gating
        # If CSNN is 0, output is 0. If STDP is high, it multiplies the CSNN signal by up to (1 + GATING_STRENGTH)
        I_place = jnp.maximum(0.0, I_csnn_place * (1.0 + GATING_STRENGTH * I_stdp_place) * I_tof_place)
        
        # Repeat for Ring (Heading) network
        I_csnn_ring = jnp.einsum('bv,bvr->br', vis_csnn, state.W_csnn_to_ring)
        I_stdp_ring = jnp.einsum('bs,bsr->br', vis_stdp, state.W_stdp_to_ring)
        I_tof_ring  = jnp.einsum('bd,bdr->br', tof_features, state.W_tof_to_ring)
        I_ring = jnp.maximum(0.0, I_csnn_ring * (1.0 + GATING_STRENGTH * I_stdp_ring) * I_tof_ring)

        recalled_place = jnp.max(I_place, axis=1)
        recalled_ring  = jnp.max(I_ring, axis=1)

        # Gate 1: Visual Distinctiveness
        vision_activity = jnp.mean(jnp.abs(vis_csnn), axis=1)
        is_distinctive = vision_activity > self.SPARSITY_THRESH

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

        # Gate 4: Plausibility
        peak_idx_place = jnp.argmax(I_place, axis=1)
        recalled_x = self.pc_preferred_locs[peak_idx_place, 0]
        recalled_y = self.pc_preferred_locs[peak_idx_place, 1]

        jump = jnp.sqrt((recalled_x - pose_xy[:, 0])**2 + (recalled_y - pose_xy[:, 1])**2)
        is_plausible = jump < PLAUSIBILITY_THRESH 

        peak_idx_ring = jnp.argmax(I_ring, axis=1)
        recalled_heading = peak_idx_ring.astype(jnp.float32) * (2 * jnp.pi / RING_N)
        angle_diff = jnp.minimum(jnp.abs(recalled_heading - heading), 2 * jnp.pi - jnp.abs(recalled_heading - heading))
        is_heading_plausible = angle_diff < HEADING_THRESH

        # Gate 5: Self-Match
        is_not_self_matching = jump > SELF_MATCH_THRESH 

        is_confident_raw = (
            is_distinctive & is_match & is_anti_aliased & 
            is_plausible & is_heading_plausible & is_not_self_matching
        ) 

        # Gate 6: Temporal Consistency
        new_confidence_ema = 0.8 * state.confidence_ema + 0.2 * is_confident_raw.astype(jnp.float32)
        state = state._replace(confidence_ema=new_confidence_ema)
        
        is_temporally_consistent = new_confidence_ema > 0.15
        is_confident = is_confident_raw & is_temporally_consistent

        # Scale the blend smoothly based on the recalled voltage peak
        voltage_above_threshold = recalled_place - self.MATCH_THRESH
        blend = jnp.clip(voltage_above_threshold / 0.09, 0.0, 1.0)

        # 🐛 NEW: Compile debug metrics
        debug_gates = {
            "G1_Distinctive": is_distinctive,
            "G2_Match": is_match,
            "G3_AntiAlias": is_anti_aliased,
            "G4_Plausible": is_plausible,
            "G4b_HeadPlausible": is_heading_plausible,
            "G5_NotSelf": is_not_self_matching,
            "G6_TemporalEMA": new_confidence_ema,
            "Raw_Conf": is_confident_raw,
            "Final_Conf": is_confident,
            "Conc_Place": conc_place,
            "Jump_Dist": jump
        }

        return state, jnp.where(is_confident, blend, 0.0), debug_gates

    def compute_spatial_correction(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, pose_xy, heading):
        B = vis_csnn.shape[0]
        I_csnn_place = jnp.einsum('bv,bvp->bp', vis_csnn, state.W_csnn_to_place)
        I_stdp_place = jnp.einsum('bs,bsp->bp', vis_stdp, state.W_stdp_to_place)
        I_tof_place  = jnp.einsum('bd,bdp->bp', tof_features, state.W_tof_to_place)
        
        I_place = jnp.maximum(0.0, I_csnn_place * (1.0 + GATING_STRENGTH * I_stdp_place) * I_tof_place)

        peak_idx = jnp.argmax(I_place, axis=1) 
        recalled_x = self.pc_preferred_locs[peak_idx, 0]
        recalled_y = self.pc_preferred_locs[peak_idx, 1]

        recalled_gx = jnp.clip(jnp.round(recalled_x / (ROOM_W / MAP_SIZE)).astype(jnp.int32), 1, MAP_SIZE - 2)
        recalled_gy = jnp.clip(jnp.round(recalled_y / (ROOM_H / MAP_SIZE)).astype(jnp.int32), 1, MAP_SIZE - 2)

        gx = jnp.arange(MAP_SIZE, dtype=jnp.float32)[None, None, :]
        gy = jnp.arange(MAP_SIZE, dtype=jnp.float32)[None, :, None]
        d2 = (gx - recalled_gx[:, None, None])**2 + (gy - recalled_gy[:, None, None])**2
        
        sigma_mpc = SIGMA_M / (ROOM_W / MAP_SIZE) 
        ghost = GAIN * jnp.exp(-d2 / (2 * sigma_mpc**2)) 
        return CORRECTION_GAIN * ghost.reshape(B, N_PLACE) 

    def compute_ring_correction(self, state: PlaceCellState, vis_csnn, vis_stdp, tof_features, heading):
        B = vis_csnn.shape[0]
        I_csnn_ring = jnp.einsum('bv,bvr->br', vis_csnn, state.W_csnn_to_ring)
        I_stdp_ring = jnp.einsum('bs,bsr->br', vis_stdp, state.W_stdp_to_ring)
        I_tof_ring  = jnp.einsum('bd,bdr->br', tof_features, state.W_tof_to_ring)
        
        I_ring = jnp.maximum(0.0, I_csnn_ring * (1.0 + GATING_STRENGTH * I_stdp_ring) * I_tof_ring)

        peak_idx = jnp.argmax(I_ring, axis=1)
        recalled_th = peak_idx.astype(jnp.float32) * (2 * jnp.pi / RING_N) 

        th_1d = jnp.arange(RING_N, dtype=jnp.float32) * (2 * jnp.pi / RING_N) 
        d2_ring = ((th_1d[None, :] - recalled_th[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi)**2

        sigma_ring = 4.0 / RING_N * 2 * jnp.pi 
        ring_corr = jnp.exp(-d2_ring / (2 * sigma_ring**2)) 
        return RING_CORRECTION_GAIN * ring_corr

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