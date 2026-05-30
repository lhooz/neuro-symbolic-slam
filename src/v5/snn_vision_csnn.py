#!/usr/bin/env python3
"""
snn_vision_csnn.py — Adaptive Visual Frontend 
(Upgraded: JIT-Compiled, Late-Fusion ToF, and LIF Sensory Persistence)

Split-Brain Architecture: The Twin's offline-trained feature extractor.
Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import flax.serialization
from typing import NamedTuple
import os
import sys

sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')

from src.sparse_forest import N_PIXELS

# ============================================================================
# 1. Surrogate Gradient
# ============================================================================
@jax.custom_vjp
def surrogate_spike(v, v_th=1.0):
    return jnp.where(v >= v_th, 1.0, 0.0)

def spike_fwd(v, v_th):
    return surrogate_spike(v, v_th), (v, v_th)

def spike_bwd(res, g):
    v, v_th = res
    alpha = 10.0 
    sg_grad = alpha / (1.0 + jnp.abs(alpha * (v - v_th)))**2
    return (g * sg_grad, None)

surrogate_spike.defvjp(spike_fwd, spike_bwd)

# ============================================================================
# 2. Frozen CSNN Base
# ============================================================================
class CSNN_Base(nn.Module):
    """
    Translates the 128-dim continuous Time Surface into a 128-dim feature vector.
    ToF is fused late to avoid redundant spatial convolutions!
    """
    @nn.compact
    def __call__(self, time_surface, x_tof):
        # 1. Vision Only
        batch_size = time_surface.shape[0]
        
        # The Fusion wrapper gives us a flat (B, 128) time_surface.
        # Convolution expects (Batch, Spatial_Dim, Channels) -> (B, 64, 2)
        x_vision = time_surface.reshape((batch_size, N_PIXELS, 2))

        # 1D Convolutions
        v1 = nn.Conv(features=16, kernel_size=(5,), strides=(2,), padding='SAME')(x_vision)
        s1 = surrogate_spike(v1, v_th=0.1)

        v2 = nn.Conv(features=32, kernel_size=(5,), strides=(2,), padding='SAME')(s1)
        s2 = surrogate_spike(v2, v_th=0.1)

        barcode = s2.reshape((batch_size, -1)) 

        # Late Fusion for ToF
        tof_flat = x_tof.reshape((batch_size, -1))
        fused_barcode = jnp.concatenate([barcode, tof_flat], axis=-1)

        v_out = nn.Dense(features=128)(fused_barcode)
        
        return v_out