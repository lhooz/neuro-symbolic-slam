#!/usr/bin/env python3
"""
train_vision_online.py — Phase 2: Live CSNN Vision Instinct Training
(Depth-Aware / GT-Miner Upgrade)

Author: Ada 🦊
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import flax.serialization
import optax
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sparse_forest import generate_sample, N_PIXELS

# ============================================================================
# 1. The Surrogate Gradient "Lie"
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
# 2. The Flax CSNN Base Model (UPGRADED: Aligned with Time Surface)
# ============================================================================
class CSNN_Base(nn.Module):
    @nn.compact
    def __call__(self, time_surface, x_tof):
        # time_surface: (Batch, N_PIXELS * 2)
        # x_tof: (Batch,) -> The normalized ToF distance
        batch_size = time_surface.shape[0]

        # Reshape the flat Time Surface for convolutions -> (B, 256, 2)
        x_vision = time_surface.reshape((batch_size, N_PIXELS, 2))

        # "Wake Up" Initialization for sparse event learning
        init_fn = nn.initializers.normal(stddev=0.15)

        v1 = nn.Conv(features=16, kernel_size=(5,), strides=(2,), padding='SAME', kernel_init=init_fn)(x_vision)
        s1 = surrogate_spike(v1, v_th=0.1)

        v2 = nn.Conv(features=32, kernel_size=(5,), strides=(2,), padding='SAME', kernel_init=init_fn)(s1)
        s2 = surrogate_spike(v2, v_th=0.1)

        barcode = s2.reshape((batch_size, -1)) 

        # Late Fusion for ToF
        tof_flat = x_tof.reshape((batch_size, -1))
        fused_barcode = jnp.concatenate([barcode, tof_flat], axis=-1)

        # 🌟 UPGRADE: 256 Features
        v_out = nn.Dense(features=256, kernel_init=init_fn)(fused_barcode)
        
        # 🌟 BIOLOGICAL UPGRADE: Enforce positive firing rates!
        return nn.relu(v_out)

def build_time_surfaces(events_seq):
    """Converts a sequence of raw events into polarized Time Surfaces."""
    on_spikes = jnp.clip(events_seq, 0.0, 1.0)
    off_spikes = jnp.clip(-events_seq, 0.0, 1.0)
    ev_pol = jnp.concatenate([on_spikes, off_spikes], axis=-1)

    def scan_fn(carry, x):
        new_ts = carry * 0.8 + x  # Matches TS_DECAY = 0.8 in FusionWrapper
        return new_ts, new_ts

    _, ts_seq = jax.lax.scan(scan_fn, jnp.zeros_like(ev_pol[0]), ev_pol)
    return ts_seq
# ============================================================================
# 3. Experience Replay: The Miner & The Pool (UPGRADED: Pairs for InfoNCE)
# ============================================================================
def mine_pairs_from_trajectory(events, tofs, pos_margin=3):
    """Mines only Anchors and Positives. Negatives are handled by the batch."""
    T = events.shape[0]
    anchors_ev, positives_ev = [], []
    anchors_tof, positives_tof = [], []
    
    for t in range(T - pos_margin - 1):
        a_ev, a_tof = events[t], tofs[t]
        
        pos_idx = t + np.random.randint(1, pos_margin + 1)
        p_ev, p_tof = events[pos_idx], tofs[pos_idx]
        
        anchors_ev.append(a_ev); positives_ev.append(p_ev)
        anchors_tof.append(a_tof); positives_tof.append(p_tof)
        
    return (np.array(anchors_ev), np.array(positives_ev),
            np.array(anchors_tof), np.array(positives_tof))

class ReplayBuffer:
    def __init__(self, capacity=15000):
        self.capacity = capacity
        self.A_pool_ev, self.P_pool_ev = [], []
        self.A_pool_tof, self.P_pool_tof = [], []
        
    def add(self, A_ev, P_ev, A_tof, P_tof):
        self.A_pool_ev.extend(list(A_ev)); self.P_pool_ev.extend(list(P_ev))
        self.A_pool_tof.extend(list(A_tof)); self.P_pool_tof.extend(list(P_tof))
        
        if len(self.A_pool_ev) > self.capacity:
            overflow = len(self.A_pool_ev) - self.capacity
            self.A_pool_ev = self.A_pool_ev[overflow:]; self.P_pool_ev = self.P_pool_ev[overflow:]
            self.A_pool_tof = self.A_pool_tof[overflow:]; self.P_pool_tof = self.P_pool_tof[overflow:]
            
    @property
    def size(self): return len(self.A_pool_ev)

    def sample(self, batch_size=128):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            jnp.array([self.A_pool_ev[i] for i in idx]), jnp.array([self.P_pool_ev[i] for i in idx]),
            jnp.array([self.A_pool_tof[i] for i in idx]), jnp.array([self.P_pool_tof[i] for i in idx])
        )

# ============================================================================
# 4. The Live Training Engine
# ============================================================================
def main():
    print("=" * 65)
    print(" 🧠 Phase 2: Live CSNN Vision Training (Depth-Aware)")
    print("=" * 65)

    # 1. Initialize the JAX PRNG, Model, and Optimizer
    key = random.PRNGKey(42)
    model = CSNN_Base()
    # 🌟 FIX: Double the size for the polarized Time Surface
    dummy_ts = jnp.zeros((1, N_PIXELS * 2))  
    dummy_tof = jnp.zeros((1,))
    
    # Always do a dummy initialization to set up the architectural structure
    key, subkey = random.split(key)
    params = model.init(subkey, dummy_ts, dummy_tof)

    # 🌟 NEW: Resume Training Logic (Safely loaded via from_bytes)
    save_path = os.path.join(os.path.dirname(__file__), "frozen_csnn_weights.msgpack")
    if os.path.exists(save_path):
        print(f"🔄 Found existing brain at {save_path}! Loading memories...")
        with open(save_path, "rb") as f:
            # FIX: Use the initialized params as the template!
            params = flax.serialization.from_bytes(params, f.read())
    else:
        print("🌱 No existing brain found. Spawning a new one from scratch...")

    # 🌟 THE FIX: Return to standard Adam! 
    # No weight decay, just gradient clipping and a slow, safe learning rate.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=3e-4) 
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, a_ev, p_ev, a_tof, p_tof):
        def loss_fn(p):
            out_A = model.apply(p, a_ev, a_tof)
            out_P = model.apply(p, p_ev, p_tof)
            
            # 1. Project to Hypersphere (Safely)
            norm_A = jnp.maximum(jnp.linalg.norm(out_A, axis=-1, keepdims=True), 1e-8)
            norm_P = jnp.maximum(jnp.linalg.norm(out_P, axis=-1, keepdims=True), 1e-8)
        
            out_A = out_A / norm_A
            out_P = out_P / norm_P
            
            # 2. Cosine Similarity Matrix (Clipped to prevent numerical instability)
            temperature = 0.1
            sim_matrix = jnp.clip(jnp.matmul(out_A, out_P.T) / temperature, -50.0, 50.0)
            
            # 3. Symmetric InfoNCE
            labels = jnp.arange(out_A.shape[0])
            loss_a = optax.softmax_cross_entropy_with_integer_labels(sim_matrix, labels)
            loss_p = optax.softmax_cross_entropy_with_integer_labels(sim_matrix.T, labels)
            
            # Average the two directions
            return jnp.mean(loss_a + loss_p) / 2.0
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params) 
        
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    pool = ReplayBuffer(capacity=15000)
    BATCH_SIZE = 128
    TRAIN_STEPS_PER_SIM = 20
    TOTAL_EPISODES = 1500

    start_time = time.time()
    
    for episode in range(1, TOTAL_EPISODES + 1):
        key, subkey = random.split(key)
        
        # generate_sample outputs info dict containing ToF and GT Positions
        events, labels, info = generate_sample(subkey, time_steps=300)
        
        # 🌟 NEW: Convert raw events into the Time Surface format
        time_surfaces = build_time_surfaces(events)
        # 🌟 THE FIX: Isolate the Center Ray (Index 1) for the Visual Cortex!
        tofs = info['tof'][:, 1] 
        
        positions = info['positions']
        
        # 🌟 FIX: Feed time_surfaces into the miner instead of raw events
        pairs = mine_pairs_from_trajectory(time_surfaces, tofs)
        pool.add(*pairs)
        
        if pool.size < BATCH_SIZE * 2:
            if episode % 10 == 0:
                print(f"🛌 Dreaming... Filling Pool ({pool.size}/{BATCH_SIZE * 2} required)")
            continue
            
        total_loss = 0.0
        for _ in range(TRAIN_STEPS_PER_SIM):
            # 🌟 FIX: Only sample and pass 4 variables (Anchors and Positives)
            a_e, p_e, a_t, p_t = pool.sample(BATCH_SIZE)
            params, opt_state, loss = train_step(params, opt_state, a_e, p_e, a_t, p_t)
            total_loss += loss
            
        avg_loss = total_loss / TRAIN_STEPS_PER_SIM
        
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            # 🌟 FIX: Updated print text to say InfoNCE
            print(f"🌍 Episode {episode:03d}/{TOTAL_EPISODES} | Pool Size: {pool.size:05d} | InfoNCE Loss: {avg_loss:8.4f} | Time: {elapsed:.1f}s")
            
            # 🌟 FIX: InfoNCE threshold adjusted to 0.01
            if avg_loss < 0.01:
                print("\n✨ InfoNCE Loss is incredibly low! The network has achieved perfect scale-invariant instincts.")
                break

    save_path = "frozen_csnn_weights.msgpack"
    with open(save_path, "wb") as f:
        f.write(flax.serialization.msgpack_serialize(params))
        
    print("=" * 65)
    print(f" ✅ Training Complete! Frozen visual instincts saved to:\n 📁 {os.path.abspath(save_path)}")
    print("=" * 65)

if __name__ == '__main__':
    main()