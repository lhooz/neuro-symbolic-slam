#!/usr/bin/env python3
"""
snn_vision_stdp.py — Adaptive Visual Frontend via Unsupervised STDP

Split-Brain Architecture: The Twin's adaptive feature extractor.
* UPDATED WITH ULTIMATE DEBUG SUITE & GIF GENERATOR *

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import time

import sys
sys.path.insert(0, '/Users/lhooz/.openclaw/workspace')

from src.sparse_forest import (
    N_PIXELS, TIME_STEPS, DT,
    FOV_DEG,
)

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

TAU_TRACE = 15.0            
import numpy as np
TRACE_DECAY = np.exp(-DT_MS / TAU_TRACE)

K_WTA = 12                  

THETA_INC = 0.50              
THETA_DECAY = 0.986          
THETA_MAX = 5.0              

TOF_GAIN = 0.1              
N_OUTPUT_FEATURES = N_HIDDEN


# ============================================================================
# 2. LIF Neuron with Lateral Inhibition
# ============================================================================

def lif_step(v, i_ext, beta=BETA_LIF):
    """Discrete LIF step (Charging and leaking only)."""
    # Voltage decays toward 0, external current pushes it up
    v_pre = beta * v + i_ext
    return v_pre

def localized_inhibition(spike, pool_size=16, k_per_pool=1):
    """Retinotopic Block-Local WTA. 
    Divides neurons into spatial cortical columns. Only the strongest 
    neuron in each physical column is allowed to fire."""
    B, N = spike.shape
    n_pools = N // pool_size
    
    # Reshape (B, 256) into (B, 16 blocks, 16 neurons)
    spike_pools = spike.reshape((B, n_pools, pool_size))
    
    # Find the local winner inside each physical block
    _, topk_idx = jax.lax.top_k(spike_pools, k_per_pool)
    
    batch_idx = jnp.arange(B, dtype=jnp.int32)[:, None, None]
    pool_idx = jnp.arange(n_pools, dtype=jnp.int32)[None, :, None]
    
    winner_mask = jnp.zeros((B, n_pools, pool_size), dtype=jnp.bool_)
    winner_mask = winner_mask.at[batch_idx, pool_idx, topk_idx].set(True)
    
    # Flatten back to (B, 256)
    return winner_mask.reshape((B, N)).astype(jnp.float32)


# ============================================================================
# 3. STDP Weight Update (Eligibility Trace Formulation)
# ============================================================================

class STDPLayer:
    def __init__(self, key, n_pre, n_post, eta=A_PLUS, gamma=A_MINUS / A_PLUS, w_min=W_MIN, w_max=W_MAX):
        self.n_pre = n_pre
        self.n_post = n_post
        self.eta = eta
        self.gamma = gamma
        self.w_min = w_min
        self.w_max = w_max

        # 🌟 NEW: Enforce Physical Retinotopy (Local Receptive Fields)
        # Map the 256 neurons to the 64 (128 polarized) pixels
        n_pixels = n_pre // 2
        neurons_per_pixel = n_post / n_pixels
        neuron_idx = jnp.arange(n_post)
        pixel_idx = jnp.arange(n_pixels)
        
        # Calculate physical distance from each neuron's center to each pixel
        center_pixels = neuron_idx / neurons_per_pixel
        dist = jnp.abs(center_pixels[:, None] - pixel_idx[None, :])
        
        # Replicate for ON and OFF channels
        dist_pol = jnp.tile(dist, (1, 2))
        
        # If a pixel is more than 8 pixels away from the neuron's center, cut the wire (mask = 0)
        RF_RADIUS = 8.0 
        self.mask = (dist_pol < RF_RADIUS).astype(jnp.float32)

        k1, k2 = random.split(key)
        self.W = random.normal(k1, (n_post, n_pre), dtype=jnp.float32) * W_INIT_STD + W_INIT_MEAN
        self.W = jnp.clip(self.W, w_min, w_max)
        
        # Apply the mask so initial weights outside the radius are strictly 0.0
        self.W = self.W * self.mask

    def reset_traces(self, B):
        # Traces must be 2D: (Batch, Neurons)
        self.e_pre = jnp.zeros((B, self.n_pre), dtype=jnp.float32)
        self.e_post = jnp.zeros((B, self.n_post), dtype=jnp.float32)

    def apply_batch(self, spike_pre_all, spike_post_all):
        # No need to broadcast anymore, they are already (B, N)
        dW_batch = (self.eta * jnp.einsum('bp,bo->bop', self.e_pre, spike_post_all)
                     - self.eta * self.gamma * jnp.einsum('bo,bp->bop', self.e_post, spike_pre_all))

        # Decay traces INDEPENDENTLY per batch element (Do NOT use mean!)
        self.e_pre = TRACE_DECAY * self.e_pre + spike_pre_all
        self.e_post = TRACE_DECAY * self.e_post + spike_post_all

        dW = dW_batch.mean(axis=0) 
        bound_factor = 1.0 - (self.W - self.w_min) / (self.w_max - self.w_min + 1e-8)
        dW_pos = dW * jnp.clip(bound_factor, 0.05, 1.0)
        dW_neg = dW * jnp.clip(1.0 - bound_factor, 0.05, 1.0)
        dW = jnp.where(dW >= 0, dW_pos, dW_neg)

        # 🌟 NEW: Re-apply the mask after STDP to guarantee synapses don't grow outside the radius
        self.W = jnp.clip(self.W + dW, self.w_min, self.w_max) * self.mask
        return self.W

# ============================================================================
# 4. Vision STDP Network (Full Frontend)
# ============================================================================

class VisionSTDP:
    def __init__(self, key, n_input=N_INPUT, n_hidden=N_HIDDEN, k_wta=K_WTA, tof_channels=N_TOF_CHANNELS):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.k_wta = k_wta
        self.tof_channels = tof_channels
        self.n_polarized = 2 * n_input

        k1, k2 = random.split(key)
        self.stdp = STDPLayer(k1, self.n_polarized, n_hidden)
        self._v_hidden = None
        self._spike_count = None

    def reset(self, B):
        self._v_hidden = jnp.zeros((B, self.n_hidden))
        self._v_th_adapt = jnp.zeros((B, self.n_hidden))
        self.stdp.reset_traces(B) # <--- Pass B here!
        self._spike_count = jnp.zeros((B, self.n_hidden))

    def polarize_events(self, ev_frame):
        on_spike = jnp.clip(ev_frame, 0.0, 1.0)
        off_spike = jnp.clip(-ev_frame, 0.0, 1.0)
        return jnp.concatenate([on_spike, off_spike], axis=1)

    def __call__(self, time_surface, tof_dist, learn=True):
        # Directly use the smooth Time Surface to inject continuous current!
        i_syn = time_surface @ self.stdp.W.T
        i_ext = i_syn

        self._v_th_adapt = self._v_th_adapt * THETA_DECAY
        v_th_eff = V_TH_STDP + self._v_th_adapt 
        
        # 1. Charge the membrane (LIF leak + current)
        v_pre = lif_step(self._v_hidden, i_ext)
        
        # 2. Who wants to spike? (Voltage exceeds adaptive threshold)
        spike_raw = jnp.maximum(v_pre - v_th_eff, 0.0)

        # 🌟 NEW: Use the Retinotopic Localized Inhibition
        # pool_size=16 means 256 neurons are split into 16 spatial blocks.
        # k_per_pool=1 means 1 winner per block (16 total winners, spread evenly)
        spike_out = localized_inhibition(spike_raw, pool_size=16, k_per_pool=1) 

        # 4. HARD RESET: Only the winners get their voltage drained to 0.0
        # Losers keep their current voltage for the next timestep
        self._v_hidden = jnp.where(spike_out > 0.5, 0.0, v_pre)

        self._v_th_adapt = jnp.clip(
            self._v_th_adapt + THETA_INC * spike_out,
            0.0, THETA_MAX
        )

        if learn:
            _ = self.stdp.apply_batch(time_surface, spike_out)

        self._spike_count = self._spike_count + spike_out
        features = self._spike_count / (1.0 + 1e-8) 

        return spike_out, features

    def get_weights(self):
        return self.W

# ============================================================================
# 5. The Ultimate Debug Suite: Sandbox Data & Evaluator
# ============================================================================

def generate_sandbox_moving_bar(n_samples=1, n_steps=200):
    """Generates a perfect, noise-free dataset of a moving line to test STDP."""
    events = np.zeros((n_samples, n_steps, N_INPUT))
    tof_dists = np.ones((n_samples, n_steps)) * 2.0 
    
    # Create a 5-pixel wide bar that moves left to right, then right to left
    for t in range(n_steps):
        pos = int((np.sin(t / 15.0) + 1.0) / 2.0 * (N_INPUT - 6))
        events[:, t, pos:pos+5] = 1.0  # ON events
    return events, tof_dists

def evaluate_vision_stdp(key, events, tof_dists, learn=True, stdp_verbose=True):
    """Run Vision STDP and record high-frequency state for debugging."""
    B, T, _ = events.shape
    net = VisionSTDP(key, n_input=N_INPUT, n_hidden=N_HIDDEN)
    net.reset(B)

    # High-Frequency Trackers
    spike_history = np.zeros((T, B, N_HIDDEN))
    v_hidden_history = np.zeros((T, B, N_HIDDEN))
    weight_history = np.zeros((T, N_HIDDEN, 2 * N_INPUT))

    print(f"\n 🦊 Vision STDP Evaluation (B={B}, T={T}, learn={learn})")
    print(f"    Input: {N_INPUT} pixels → {N_HIDDEN} feature neurons")
    print(f"    k-WTA: {K_WTA} winners/step | Homeostasis: Δθ={THETA_INC}")

    for t in range(T):
        ev_t = jnp.array(events[:, t, :])
        tof_t = jnp.array(tof_dists[:, t])

        # 🌟 FIX: Mimic the Fusion Wrapper's preprocessing
        on_spike = jnp.clip(ev_t, 0.0, 1.0)
        off_spike = jnp.clip(-ev_t, 0.0, 1.0)
        ev_pol = jnp.concatenate([on_spike, off_spike], axis=1)
        mock_time_surface = mock_time_surface * 0.8 + ev_pol

        # Pass the mock_time_surface instead of ev_t
        spike, features = net(mock_time_surface, tof_t, learn=learn)
        
        spike_history[t] = np.array(spike)
        v_hidden_history[t] = np.array(net._v_hidden)
        weight_history[t] = np.array(net.stdp.W)

    spike_rate = spike_history.mean(axis=(0, 1)) 
    active_neurons = (spike_rate > 0.01).sum()

    print(f"\n 📊 Feature Statistics:")
    print(f"    Active neurons (>1% rate): {active_neurons}/{N_HIDDEN} ({100*active_neurons/N_HIDDEN:.0f}%)")
    print(f"    Mean spike rate: {spike_rate.mean() / DT:.1f} spikes/s/neuron")
    print(f"    Weight range: [{float(net.stdp.W.min()):.4f}, {float(net.stdp.W.max()):.4f}]")

    W_change = np.abs(weight_history[-1] - weight_history[0]).mean()
    print(f"    Mean |ΔW| over {T} steps: {W_change:.5f}")
    
    if W_change < 1e-4:
        print("    ⚠️ WARNING: STDP IS DEAD (No weight change detected).")
    elif active_neurons < (N_HIDDEN * 0.1):
        print("    ⚠️ WARNING: DICTATOR BUGS DETECTED (Most neurons are dead). Increase THETA_INC.")
    else:
        print("    ✅ STDP Health: EXCELLENT (Active learning & good distribution).")

    return {
        'events': events,
        'spike_history': spike_history,
        'v_hidden_history': v_hidden_history,
        'weight_history': weight_history,
        'spike_rate': spike_rate,
        'active_neurons': active_neurons,
        'net': net,
    }

# ============================================================================
# 6. Diagnostic Visualization (GIF Generator)
# ============================================================================

def create_stdp_debug_gif(results, batch_idx=0, save_path="stdp_debug.gif", step_skip=2):
    """4-Panel animated GIF to watch the STDP brain learn in real-time."""
    print(f"\n 🎬 Rendering STDP Diagnostic GIF to {save_path}...")
    
    events = results['events'][batch_idx]
    v_hidden = results['v_hidden_history'][:, batch_idx, :]
    spikes = results['spike_history'][:, batch_idx, :]
    weights = results['weight_history']
    T = events.shape[0]

    # 👇 Reduced figsize from (16, 10) to (12, 8) to shrink file size
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

    # Panel 1: Input Events
    ax_in = fig.add_subplot(gs[0, 0])
    ax_in.set_title("Raw Event Camera Input (Moving Bar)", fontweight='bold')
    ax_in.set_xlim(0, N_INPUT)
    ax_in.set_ylim(-0.1, 1.1)
    line_in, = ax_in.plot([], [], 'k-', lw=2)

    # Panel 2: Membrane Potentials (Top 16 Neurons)
    ax_v = fig.add_subplot(gs[0, 1])
    ax_v.set_title("Membrane Potentials (Top 16 Neurons)", fontweight='bold')
    ax_v.set_xlim(0, 100) 
    
    # 👇 Scaled Y-Axis to fit the absolute maximum possible threshold (1.0 + 5.0)
    ax_v.set_ylim(0, V_TH_STDP + THETA_MAX + 0.5) 
    ax_v.axhline(V_TH_STDP, color='r', linestyle='--', alpha=0.5, label='Base Threshold')
    v_lines = [ax_v.plot([], [], lw=1.5, alpha=0.7)[0] for _ in range(16)]

    # Panel 3: Spike Raster
    ax_spk = fig.add_subplot(gs[1, 0])
    ax_spk.set_title("Spike Raster (All 256 Neurons)", fontweight='bold')
    ax_spk.set_xlim(0, 100)
    ax_spk.set_ylim(0, N_HIDDEN)
    scatter_spk = ax_spk.scatter([], [], s=5, c='blue', marker='|')

    # Panel 4: Receptive Fields
    ax_w = fig.add_subplot(gs[1, 1])
    ax_w.set_title("STDP Receptive Fields (Live Weight Updates)", fontweight='bold')
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
        t_idx = t_idx + start_f
        scatter_spk.set_offsets(np.c_[t_idx, n_idx])
        ax_spk.set_xlim(start_f, start_f + 100)

        W_frame = weights[frame, :64, :N_INPUT] 
        W_grid = W_frame.reshape(8, 8, N_INPUT).transpose(0, 1, 2).reshape(8, 8 * N_INPUT)
        img_w.set_data(W_grid)

        return line_in, scatter_spk, img_w, *v_lines

    frames = list(range(1, T, step_skip))
    # 👇 Added dpi=72 to drastically reduce GIF file size
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(save_path, writer='pillow', fps=15, dpi=72)
    print(f" ✅ GIF saved successfully!")

# ============================================================================
# 7. Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print(" 🦊 Vision STDP Ultimate Debug Suite")
    print("=" * 65)

    # Run the noise-free Sandbox test to guarantee STDP is mathematically working
    print("\n[PHASE 1] Generating Noise-Free Sandbox Data (Moving Bar)...")
    events, tof_dists = generate_sandbox_moving_bar(n_samples=1, n_steps=300)
    
    results = evaluate_vision_stdp(random.PRNGKey(42), events, tof_dists, learn=True)
    
    create_stdp_debug_gif(results, batch_idx=0, save_path="/Users/lhooz/.openclaw/workspace/results/stdp_debug.gif", step_skip=2)

    print(f"\n ✅ All STDP Debugging Complete! Open stdp_debug.gif to watch the brain learn.")