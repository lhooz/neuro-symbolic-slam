#!/usr/bin/env python3
"""
snn_vision_stdp.py — Adaptive Visual Frontend via Unsupervised STDP

Split-Brain Architecture: The Twin's adaptive feature extractor.
* UPDATED WITH ULTIMATE DEBUG SUITE & GIF GENERATOR *
* UPGRADED: 100% Pure Functional JAX State *

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import NamedTuple

import sys
sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')

from src.sparse_forest import N_PIXELS, DT

# ============================================================================
# 1. STDP Hyperparameters
# ============================================================================

N_INPUT = N_PIXELS          
N_HIDDEN = 256              
N_TOF_CHANNELS = 4          

BETA_LIF = 0.85             
V_TH_STDP = 1.0             

DT_MS = DT * 1000           
TAU_STDP = 20.0             
A_PLUS = 0.01               
A_MINUS = 0.012             
W_MAX = 0.15                
W_MIN = 0.0                 
W_INIT_MEAN = 0.05          
W_INIT_STD = 0.02     
# 🌟 NEW: The Homeostatic Forgetting Rate!
# 1.0 = Never forget. 0.99 = Forget entirely in ~5 seconds.
W_DECAY_RATE = 0.99      

# 🌟 TRACE FIX: Lengthened so associations survive across frames!
TAU_TRACE = 100.0            
TRACE_DECAY = np.exp(-DT_MS / TAU_TRACE)

K_WTA = 12                  

THETA_INC = 0.50              
THETA_DECAY = 0.986          
THETA_MAX = 5.0              

# ============================================================================
# 2. Pure Functional State Tuple
# ============================================================================
class VisionSTDPState(NamedTuple):
    W: jnp.ndarray
    e_pre: jnp.ndarray
    e_post: jnp.ndarray
    v_hidden: jnp.ndarray
    v_th_adapt: jnp.ndarray
    spike_trace: jnp.ndarray  # Replaces infinite spike_count

# ============================================================================
# 3. Vision STDP Network Architecture
# ============================================================================
def lif_step(v, i_ext, beta=BETA_LIF):
    """Discrete LIF step (Charging and leaking only)."""
    return beta * v + i_ext

def localized_inhibition(spike, pool_size=16, k_per_pool=1):
    """Retinotopic Block-Local WTA."""
    B, N = spike.shape
    n_pools = N // pool_size
    
    spike_pools = spike.reshape((B, n_pools, pool_size))
    _, topk_idx = jax.lax.top_k(spike_pools, k_per_pool)
    
    batch_idx = jnp.arange(B, dtype=jnp.int32)[:, None, None]
    pool_idx = jnp.arange(n_pools, dtype=jnp.int32)[None, :, None]
    
    winner_mask = jnp.zeros((B, n_pools, pool_size), dtype=jnp.bool_)
    winner_mask = winner_mask.at[batch_idx, pool_idx, topk_idx].set(True)
    winner_mask_flat = winner_mask.reshape((B, N)).astype(jnp.float32)

    # 🌟 HALLUCINATION FIX: Ensure dead pools stay dead
    return jnp.where(spike > 0.0, winner_mask_flat, 0.0)

class STDPLayer:
    def __init__(self, n_pre, n_post, eta=A_PLUS, gamma=A_MINUS / A_PLUS, w_min=W_MIN, w_max=W_MAX):
        self.n_pre = n_pre
        self.n_post = n_post
        self.eta = eta
        self.gamma = gamma
        self.w_min = w_min
        self.w_max = w_max

        # Physical Retinotopy Mask
        n_pixels = n_pre // 2
        neurons_per_pixel = n_post / n_pixels
        neuron_idx = jnp.arange(n_post)
        pixel_idx = jnp.arange(n_pixels)
        
        center_pixels = neuron_idx / neurons_per_pixel
        dist = jnp.abs(center_pixels[:, None] - pixel_idx[None, :])
        dist_pol = jnp.tile(dist, (1, 2))
        
        RF_RADIUS = 8.0 
        self.mask = (dist_pol < RF_RADIUS).astype(jnp.float32)

    def init_weights(self, key):
        W = random.normal(key, (self.n_post, self.n_pre), dtype=jnp.float32) * W_INIT_STD + W_INIT_MEAN
        return jnp.clip(W, self.w_min, self.w_max) * self.mask

    def apply_batch(self, W, e_pre, e_post, spike_pre_all, spike_post_all):
        dW_batch = (self.eta * jnp.einsum('bp,bo->bop', e_pre, spike_post_all)
                     - self.eta * self.gamma * jnp.einsum('bo,bp->bop', e_post, spike_pre_all))

        new_e_pre = TRACE_DECAY * e_pre + spike_pre_all
        new_e_post = TRACE_DECAY * e_post + spike_post_all

        dW = dW_batch.mean(axis=0) 
        bound_factor = 1.0 - (W - self.w_min) / (self.w_max - self.w_min + 1e-8)
        dW_pos = dW * jnp.clip(bound_factor, 0.05, 1.0)
        dW_neg = dW * jnp.clip(1.0 - bound_factor, 0.05, 1.0)
        dW = jnp.where(dW >= 0, dW_pos, dW_neg)

        # 🌟 THE UPGRADE: Continuous Homeostatic Forgetting
        decayed_W = W * W_DECAY_RATE 

        # Add the learning updates on top of the decayed weights
        new_W = jnp.clip(decayed_W + dW, self.w_min, self.w_max) * self.mask
        
        return new_W, new_e_pre, new_e_post


class VisionSTDP:
    def __init__(self, key, n_input=N_INPUT, n_hidden=N_HIDDEN, k_wta=K_WTA, tof_channels=N_TOF_CHANNELS):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.k_wta = k_wta
        self.tof_channels = tof_channels
        self.n_polarized = 2 * n_input
        self.key = key
        
        self.stdp_layer = STDPLayer(self.n_polarized, n_hidden)

    def init_state(self, B) -> VisionSTDPState:
        """🌟 THE FIX: Returns a pure, mutable-free NamedTuple State"""
        W_init = self.stdp_layer.init_weights(self.key)
        
        return VisionSTDPState(
            W=jnp.tile(W_init[None, :, :], (B, 1, 1)), # Shape: (B, Hidden, Polarized)
            e_pre=jnp.zeros((B, self.n_polarized), dtype=jnp.float32),
            e_post=jnp.zeros((B, self.n_hidden), dtype=jnp.float32),
            v_hidden=jnp.zeros((B, self.n_hidden), dtype=jnp.float32),
            v_th_adapt=jnp.zeros((B, self.n_hidden), dtype=jnp.float32),
            spike_trace=jnp.zeros((B, self.n_hidden), dtype=jnp.float32)
        )

    def __call__(self, state: VisionSTDPState, time_surface, tof_dist, learn=True):
        # 🌟 Pure JAX Forward Pass (No self mutation)
        i_syn = jnp.einsum('bi,boi->bo', time_surface, state.W)
        i_ext = i_syn

        new_v_th_adapt = state.v_th_adapt * THETA_DECAY
        v_th_eff = V_TH_STDP + new_v_th_adapt 
        
        v_pre = lif_step(state.v_hidden, i_ext)
        spike_raw = jnp.maximum(v_pre - v_th_eff, 0.0)
        spike_out = localized_inhibition(spike_raw, pool_size=16, k_per_pool=1) 

        new_v_hidden = jnp.where(spike_out > 0.5, 0.0, v_pre)
        new_v_th_adapt = jnp.clip(new_v_th_adapt + THETA_INC * spike_out, 0.0, THETA_MAX)

        W_learned, e_pre_learned, e_post_learned = self.stdp_layer.apply_batch(state.W, state.e_pre, state.e_post, time_surface, spike_out)

        # 🌟 JAX-Safe Control Flow for Learning
        W_final = jnp.where(learn, W_learned, state.W)
        e_pre_final = jnp.where(learn, e_pre_learned, TRACE_DECAY * state.e_pre + time_surface)
        e_post_final = jnp.where(learn, e_post_learned, TRACE_DECAY * state.e_post + spike_out)

        # 🌟 THE INFINITE FEATURE LEAK FIX
        new_spike_trace = state.spike_trace * 0.95 + spike_out
        features = new_spike_trace / (1.0 + 1e-8) 

        new_state = VisionSTDPState(W_final, e_pre_final, e_post_final, new_v_hidden, new_v_th_adapt, new_spike_trace)
        return new_state, spike_out, features

# ============================================================================
# 4. The Ultimate Debug Suite: Sandbox Data & Evaluator
# ============================================================================

def generate_sandbox_moving_bar(n_samples=1, n_steps=200):
    events = np.zeros((n_samples, n_steps, N_INPUT))
    tof_dists = np.ones((n_samples, n_steps)) * 2.0 
    for t in range(n_steps):
        pos = int((np.sin(t / 15.0) + 1.0) / 2.0 * (N_INPUT - 6))
        events[:, t, pos:pos+5] = 1.0
    return events, tof_dists

def evaluate_vision_stdp(key, events, tof_dists, learn=True):
    B, T, _ = events.shape
    net = VisionSTDP(key, n_input=N_INPUT, n_hidden=N_HIDDEN)
    
    # 🌟 JAX FIX: Use the pure state object
    state = net.init_state(B)

    spike_history = np.zeros((T, B, N_HIDDEN))
    v_hidden_history = np.zeros((T, B, N_HIDDEN))
    weight_history = np.zeros((T, N_HIDDEN, 2 * N_INPUT))

    print(f"\n 🦊 Vision STDP Evaluation (B={B}, T={T}, learn={learn})")
    
    # 🌟 CRASH FIX: Initialize mock_time_surface before the loop!
    mock_time_surface = jnp.zeros((B, N_INPUT * 2))

    for t in range(T):
        ev_t = jnp.array(events[:, t, :])
        tof_t = jnp.array(tof_dists[:, t])

        on_spike = jnp.clip(ev_t, 0.0, 1.0)
        off_spike = jnp.clip(-ev_t, 0.0, 1.0)
        ev_pol = jnp.concatenate([on_spike, off_spike], axis=1)
        mock_time_surface = mock_time_surface * 0.8 + ev_pol

        # 🌟 NEW: Pure Function Execution
        state, spike, features = net(state, mock_time_surface, tof_t, learn=learn)
        
        spike_history[t] = np.array(spike)
        v_hidden_history[t] = np.array(state.v_hidden)
        weight_history[t] = np.array(state.W[0])

    spike_rate = spike_history.mean(axis=(0, 1)) 
    active_neurons = (spike_rate > 0.01).sum()
    W_change = np.abs(weight_history[-1] - weight_history[0]).mean()

    print(f"\n 📊 Feature Statistics:")
    print(f"    Active neurons (>1% rate): {active_neurons}/{N_HIDDEN} ({100*active_neurons/N_HIDDEN:.0f}%)")
    print(f"    Mean spike rate: {spike_rate.mean() / DT:.1f} spikes/s/neuron")
    print(f"    Mean |ΔW| over {T} steps: {W_change:.5f}")

    if W_change < 1e-4:
        print("    ⚠️ WARNING: STDP IS DEAD (No weight change detected).")
    else:
        print("    ✅ STDP Health: EXCELLENT (Active learning).")

    return {
        'events': events,
        'spike_history': spike_history,
        'v_hidden_history': v_hidden_history,
        'weight_history': weight_history,
        'spike_rate': spike_rate,
        'active_neurons': active_neurons,
    }

def create_stdp_debug_gif(results, batch_idx=0, save_path="stdp_debug.gif", step_skip=2):
    print(f"\n 🎬 Rendering STDP Diagnostic GIF to {save_path}...")
    events = results['events'][batch_idx]
    v_hidden = results['v_hidden_history'][:, batch_idx, :]
    spikes = results['spike_history'][:, batch_idx, :]
    weights = results['weight_history']
    T = events.shape[0]

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

    ax_in = fig.add_subplot(gs[0, 0])
    ax_in.set_title("Raw Event Camera Input (Moving Bar)", fontweight='bold')
    ax_in.set_xlim(0, N_INPUT)
    ax_in.set_ylim(-0.1, 1.1)
    line_in, = ax_in.plot([], [], 'k-', lw=2)

    ax_v = fig.add_subplot(gs[0, 1])
    ax_v.set_title("Membrane Potentials (Top 16 Neurons)", fontweight='bold')
    ax_v.set_xlim(0, 100) 
    ax_v.set_ylim(0, V_TH_STDP + THETA_MAX + 0.5) 
    ax_v.axhline(V_TH_STDP, color='r', linestyle='--', alpha=0.5)
    v_lines = [ax_v.plot([], [], lw=1.5, alpha=0.7)[0] for _ in range(16)]

    ax_spk = fig.add_subplot(gs[1, 0])
    ax_spk.set_title("Spike Raster (All 256 Neurons)", fontweight='bold')
    ax_spk.set_xlim(0, 100)
    ax_spk.set_ylim(0, N_HIDDEN)
    scatter_spk = ax_spk.scatter([], [], s=5, c='blue', marker='|')

    ax_w = fig.add_subplot(gs[1, 1])
    ax_w.set_title("STDP Receptive Fields", fontweight='bold')
    w_display = np.zeros((8, 8 * N_INPUT))
    img_w = ax_w.imshow(w_display, cmap='RdBu_r', vmin=W_MIN, vmax=W_MAX, aspect='auto')
    ax_w.axis('off')

    top_16_idx = np.argsort(results['spike_rate'])[::-1][:16]

    def update(frame):
        line_in.set_data(np.arange(N_INPUT), events[frame])

        start_f = max(0, frame - 100)
        x_data = np.arange(start_f, frame)
        for i, n_idx in enumerate(top_16_idx):
            v_lines[i].set_data(x_data, v_hidden[start_f:frame, n_idx])
        ax_v.set_xlim(start_f, start_f + 100)

        recent_spikes = spikes[start_f:frame]
        t_idx, n_idx = np.where(recent_spikes > 0.5)
        scatter_spk.set_offsets(np.c_[t_idx + start_f, n_idx])
        ax_spk.set_xlim(start_f, start_f + 100)

        W_frame = weights[frame, :64, :N_INPUT] 
        W_grid = W_frame.reshape(8, 8, N_INPUT).transpose(0, 1, 2).reshape(8, 8 * N_INPUT)
        img_w.set_data(W_grid)

        return line_in, scatter_spk, img_w, *v_lines

    frames = list(range(1, T, step_skip))
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(save_path, writer='pillow', fps=15, dpi=72)
    print(f" ✅ GIF saved successfully!")

if __name__ == '__main__':
    print("=" * 65)
    print(" 🦊 Vision STDP Ultimate Debug Suite")
    print("=" * 65)
    events, tof_dists = generate_sandbox_moving_bar(n_samples=1, n_steps=300)
    results = evaluate_vision_stdp(random.PRNGKey(42), events, tof_dists, learn=True)
    create_stdp_debug_gif(results, batch_idx=0, save_path="stdp_debug.gif", step_skip=2)