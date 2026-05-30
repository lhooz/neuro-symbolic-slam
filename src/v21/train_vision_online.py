#!/usr/bin/env python3
"""
train_vision_online.py — Phase 2: Live CSNN Vision Instinct Training
(Depth-Aware / GT-Miner Upgrade / Memory Leak Fixed)

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
# 1. The Surrogate Gradient
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
# 2. The Flax CSNN Base Model (Synchronized Image Scrambler Fix)
# ============================================================================
class CSNN_Base(nn.Module):
    @nn.compact
    def __call__(self, time_surface, x_tof):
        batch_size = time_surface.shape[0]

        # 🌟 THE IMAGE SCRAMBLER FIX: Must match the live inference module exactly!
        on_ts = time_surface[:, :N_PIXELS]
        off_ts = time_surface[:, N_PIXELS:]
        x_vision = jnp.stack([on_ts, off_ts], axis=-1)

        init_fn = nn.initializers.normal(stddev=0.15)

        v1 = nn.Conv(features=16, kernel_size=(5,), strides=(2,), padding='SAME', kernel_init=init_fn)(x_vision)
        s1 = surrogate_spike(v1, v_th=0.1)

        v2 = nn.Conv(features=32, kernel_size=(5,), strides=(2,), padding='SAME', kernel_init=init_fn)(s1)
        s2 = surrogate_spike(v2, v_th=0.1)

        barcode = s2.reshape((batch_size, -1)) 

        tof_flat = x_tof.reshape((batch_size, -1))
        fused_barcode = jnp.concatenate([barcode, tof_flat], axis=-1)

        v_out = nn.Dense(features=256, kernel_init=init_fn)(fused_barcode)
        
        # 🌟 THE INFONCE COLLAPSE FIX: Use softplus instead of relu!
        # Softplus guarantees biological positivity (no negative spikes) but maintains a smooth, 
        # non-zero gradient everywhere, preventing Division-by-Zero norm collapses.
        return nn.softplus(v_out)

def build_time_surfaces(events_seq):
    on_spikes = jnp.clip(events_seq, 0.0, 1.0)
    off_spikes = jnp.clip(-events_seq, 0.0, 1.0)
    ev_pol = jnp.concatenate([on_spikes, off_spikes], axis=-1)

    def scan_fn(carry, x):
        new_ts = carry * 0.8 + x
        return new_ts, new_ts

    _, ts_seq = jax.lax.scan(scan_fn, jnp.zeros_like(ev_pol[0]), ev_pol)
    return ts_seq

# ============================================================================
# 3. Experience Replay
# ============================================================================
def mine_pairs_from_trajectory(events, tofs, pos_margin=25): 
    # 🌟 Wider margin allows better scale/rotation invariance learning
    T = events.shape[0]
    num_pairs = T - pos_margin - 1
    
    anchors_ev = np.zeros((num_pairs, events.shape[1]), dtype=np.float32)
    positives_ev = np.zeros_like(anchors_ev)
    anchors_tof = np.zeros((num_pairs,), dtype=np.float32)
    positives_tof = np.zeros_like(anchors_tof)
    
    for i, t in enumerate(range(num_pairs)):
        pos_idx = t + np.random.randint(1, pos_margin + 1)
        
        # 🌟 THE LEAK FIX: Cast JAX DeviceArrays to standard NumPy immediately
        anchors_ev[i] = np.asarray(events[t])
        positives_ev[i] = np.asarray(events[pos_idx])
        anchors_tof[i] = np.asarray(tofs[t])
        positives_tof[i] = np.asarray(tofs[pos_idx])
        
    return anchors_ev, positives_ev, anchors_tof, positives_tof

class ReplayBuffer:
    # 🌟 THE LEAK FIX: Static Circular Buffer 
    def __init__(self, capacity=15000, event_dim=N_PIXELS*2):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.A_pool_ev = np.zeros((capacity, event_dim), dtype=np.float32)
        self.P_pool_ev = np.zeros((capacity, event_dim), dtype=np.float32)
        self.A_pool_tof = np.zeros((capacity,), dtype=np.float32)
        self.P_pool_tof = np.zeros((capacity,), dtype=np.float32)
        
    def add(self, A_ev, P_ev, A_tof, P_tof):
        batch_len = A_ev.shape[0]
        for i in range(batch_len):
            self.A_pool_ev[self.ptr] = A_ev[i]
            self.P_pool_ev[self.ptr] = P_ev[i]
            self.A_pool_tof[self.ptr] = A_tof[i]
            self.P_pool_tof[self.ptr] = P_tof[i]
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=128):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            jnp.array(self.A_pool_ev[idx]), jnp.array(self.P_pool_ev[idx]),
            jnp.array(self.A_pool_tof[idx]), jnp.array(self.P_pool_tof[idx])
        )

# ============================================================================
# 4. The Live Training Engine
# ============================================================================
def main():
    print("=" * 65)
    print(" 🧠 Phase 2: Live CSNN Vision Training (Depth-Aware)")
    print("=" * 65)

    key = random.PRNGKey(42)
    model = CSNN_Base()
    dummy_ts = jnp.zeros((1, N_PIXELS * 2))  
    dummy_tof = jnp.zeros((1,))
    
    key, subkey = random.split(key)
    params = model.init(subkey, dummy_ts, dummy_tof)

    save_path = os.path.join(os.path.dirname(__file__), "frozen_csnn_weights.msgpack")
    if os.path.exists(save_path):
        print(f"🔄 Found existing brain at {save_path}! Loading memories...")
        with open(save_path, "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())
    else:
        print("🌱 No existing brain found. Spawning a new one from scratch...")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-4, weight_decay=1e-4) 
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, a_ev, p_ev, a_tof, p_tof):
        def loss_fn(p):
            out_A = model.apply(p, a_ev, a_tof)
            out_P = model.apply(p, p_ev, p_tof)
            
            # 🌟 THE NaN FIX: Safe Euclidean Norm prevents jnp.linalg.norm crashes on 0.0
            norm_A = jnp.sqrt(jnp.sum(out_A**2, axis=-1, keepdims=True) + 1e-8)
            norm_P = jnp.sqrt(jnp.sum(out_P**2, axis=-1, keepdims=True) + 1e-8)
        
            out_A = out_A / norm_A
            out_P = out_P / norm_P
            
            temperature = 0.1
            sim_matrix = jnp.clip(jnp.matmul(out_A, out_P.T) / temperature, -50.0, 50.0)
            
            # 🌟 OPTAX FIX: Strictly define kwargs to prevent TypeError crashes on newer versions
            labels = jnp.arange(out_A.shape[0])
            loss_a = optax.softmax_cross_entropy_with_integer_labels(logits=sim_matrix, labels=labels)
            loss_p = optax.softmax_cross_entropy_with_integer_labels(logits=sim_matrix.T, labels=labels)
            
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
        events, labels, info = generate_sample(subkey, time_steps=300)
        
        time_surfaces = build_time_surfaces(events)
        
        # 🌟 THE TOF OVERPOWERING FIX: Normalize ToF to [0, 1]
        tofs = info['tof'][:, 1] / 8.0 
        
        pairs = mine_pairs_from_trajectory(time_surfaces, tofs)
        pool.add(*pairs)
        
        if pool.size < BATCH_SIZE * 2:
            if episode % 10 == 0:
                print(f"🛌 Dreaming... Filling Pool ({pool.size}/{BATCH_SIZE * 2} required)")
            continue
            
        total_loss = 0.0
        for _ in range(TRAIN_STEPS_PER_SIM):
            a_e, p_e, a_t, p_t = pool.sample(BATCH_SIZE)
            params, opt_state, loss = train_step(params, opt_state, a_e, p_e, a_t, p_t)
            total_loss += loss
            
        avg_loss = total_loss / TRAIN_STEPS_PER_SIM
        
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            print(f"🌍 Episode {episode:03d}/{TOTAL_EPISODES} | Pool Size: {pool.size:05d} | InfoNCE Loss: {avg_loss:8.4f} | Time: {elapsed:.1f}s")
            
            if avg_loss < 0.01:
                print("\n✨ InfoNCE Loss is incredibly low! The network has achieved perfect scale-invariant instincts.")
                break

    with open(save_path, "wb") as f:
        f.write(flax.serialization.msgpack_serialize(params))
        
    print("=" * 65)
    print(f" ✅ Training Complete! Frozen visual instincts saved to:\n 📁 {os.path.abspath(save_path)}")
    print("=" * 65)

if __name__ == '__main__':
    main()