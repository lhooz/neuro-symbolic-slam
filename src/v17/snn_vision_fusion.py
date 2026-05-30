#!/usr/bin/env python3
"""
snn_vision_fusion.py — The Dual-Stream "Dreamer" Frontend Wrapper
Merges the frozen CSNN (Anchor) and the plastic STDP (Adaptation).
"""

import os
import jax
import jax.numpy as jnp
import flax.serialization
from typing import NamedTuple

# Import your two separate, modular brains
from snn_vision_csnn import CSNN_Base 
from snn_vision_stdp import VisionSTDP
from src.sparse_forest import N_PIXELS # Assuming you have this import

class FusionState(NamedTuple):
    time_surface: jnp.ndarray      # The shared pixel memory
    csnn_trace: jnp.ndarray        # Output memory of the CSNN
    stdp_trace: jnp.ndarray        # Output memory of the STDP

class DualStreamVisionCortex:
    def __init__(self, key, n_pixels, n_csnn_out=256, n_stdp_out=256):
        self.n_pixels = n_pixels
        self.n_csnn_out = n_csnn_out
        self.n_stdp_out = n_stdp_out
        
        # 1. Initialize Both Networks
        self.csnn = CSNN_Base() 
        self.stdp = VisionSTDP(key, n_input=n_pixels, n_hidden=n_stdp_out)
        
        # 🌟 FIX 1: Load the Frozen CSNN Weights
        weights_path = os.path.join(os.path.dirname(__file__), "frozen_csnn_weights.msgpack")
        if os.path.exists(weights_path):
            with open(weights_path, "rb") as f:
                self.frozen_params = flax.serialization.msgpack_restore(f.read())
            print("👁️ Dual-Stream Vision: Loaded Frozen CSNN Anchor successfully.")
        else:
            dummy_ev = jnp.zeros((1, n_pixels * 2)) # *2 for polarized Time Surface
            dummy_tof = jnp.zeros((1, 1))           # Adjust dummy_tof shape to match your CSNN input
            self.frozen_params = self.csnn.init(key, dummy_ev, dummy_tof)
            print(f"⚠️ WARNING: {weights_path} not found. Using random initialized weights.")

    def init_state(self, B) -> FusionState:
        # 🌟 THE FIX: Tell the STDP network to initialize its internal neurons!
        self.stdp.reset(B)

        return FusionState(
            time_surface=jnp.zeros((B, self.n_pixels * 2)), 
            csnn_trace=jnp.zeros((B, self.n_csnn_out)),
            stdp_trace=jnp.zeros((B, self.n_stdp_out))
        )

    def __call__(self, state: FusionState, ev_frame, tof_dist, learn=True):
        B = ev_frame.shape[0]
        
        # UPGRADE A: Generate the Shared Time Surface
        on_spike = jnp.clip(ev_frame, 0.0, 1.0)
        off_spike = jnp.clip(-ev_frame, 0.0, 1.0)
        ev_pol = jnp.concatenate([on_spike, off_spike], axis=1)
        
        # Decay old pixels, add new spikes (TS_DECAY = 0.8)
        new_time_surface = state.time_surface * 0.8 + ev_pol
        
        # THE SPLIT: Run both routes in parallel using the Time Surface
        csnn_raw_out = self.csnn.apply(self.frozen_params, new_time_surface, tof_dist)
        stdp_spikes, _ = self.stdp(new_time_surface, tof_dist, learn=learn)
        
        # UPGRADE B: Create Output Traces (Short-term memory)
        BETA_OUT = 0.95 
        new_csnn_trace = BETA_OUT * state.csnn_trace + (1.0 - BETA_OUT) * csnn_raw_out
        new_stdp_trace = BETA_OUT * state.stdp_trace + (1.0 - BETA_OUT) * stdp_spikes
        
        # 🌟 THE CSNN RECTIFIER
        new_csnn_trace_pos = jnp.maximum(0.0, new_csnn_trace)
        
        # =================================================================
        # 🌟 NEW: LATERAL EXCITATION (1D Spatial Blurring)
        # Allows live features to activate adjacent memory synapses!
        # =================================================================
        # 1. Use padding instead of rolling to prevent 180-degree teleportation
        csnn_left  = jnp.pad(new_csnn_trace_pos[:, :-1], ((0, 0), (1, 0)), mode='constant')
        csnn_right = jnp.pad(new_csnn_trace_pos[:, 1:],  ((0, 0), (0, 1)), mode='constant')
        
        # 2. Add the 20% shoulders, and divide by 1.40 to conserve energy
        csnn_blurred = (new_csnn_trace_pos + 0.20 * csnn_left + 0.20 * csnn_right) / 1.40
        
        # =================================================================
        # 🌟 STRICT STDP (Tiny Blur for Picket-Fence forgiveness)
        # =================================================================
        new_stdp_trace_clean = jnp.where(new_stdp_trace < 0.10, 0.0, new_stdp_trace)
        
        stdp_left  = jnp.pad(new_stdp_trace_clean[:, :-1], ((0, 0), (1, 0)), mode='constant')
        stdp_right = jnp.pad(new_stdp_trace_clean[:, 1:],  ((0, 0), (0, 1)), mode='constant')
        
        stdp_blurred = (new_stdp_trace_clean + 0.20 * stdp_left + 0.20 * stdp_right) / 1.40
        
        # THE MERGE: Normalize both independently
        norm_csnn = csnn_blurred / (jnp.linalg.norm(csnn_blurred, axis=-1, keepdims=True) + 1e-8)
        norm_stdp = stdp_blurred / (jnp.linalg.norm(stdp_blurred, axis=-1, keepdims=True) + 1e-8)

        # 🌟 FIX 2: Create the new state before returning
        new_state = FusionState(new_time_surface, new_csnn_trace, new_stdp_trace)

        # Output them as a tuple
        return new_state, (norm_csnn, norm_stdp)