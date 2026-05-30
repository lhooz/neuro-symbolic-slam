#!/usr/bin/env python3
"""
SNN Training on 2D Navigation Event Camera Data

Wires event_camera_2d_nav.py into the SNN pipeline.
Predicts [vx, vy, omega, min_clearance] from 1D event streams.

Architecture:
  Events (B,T,64) → Polarity Split (B,T,128) → LIF Layer 1 (128→128)
  → LIF Layer 2 (128→64) → LI Readout (64→4, temporal integration) → (B,4)

  The LI readout preserves temporal ordering: U_t = β_li * U_{t-1} + S_t @ W_li + b_li.
  Loss applied over last 50 timesteps (skip burn-in). Raw voltage, no tanh.

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

from event_camera_2d_nav import (
    generate_batch, N_PIXELS, TIME_STEPS, DT,
    N_OBSTACLES, ROOM_W, ROOM_H,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-3
N_EPOCHS = 300
TRAIN_BATCH = 64
EVAL_BATCH = 64
HIDDEN1 = 128
HIDDEN2 = 64
GRAD_CLIP_NORM = 1.0
STATE_DIM = 4
INPUT_DIM = 2 * N_PIXELS  # 128 (ON/OFF polarity split)

# SNN hyperparameters
BETA = 0.85
BETA_LI = 0.95        # Leaky Integrator readout decay (higher = longer memory)
LOSS_WINDOW = 50      # Use last N timesteps for loss (skip burn-in)
V_TH = 1.0
ALPHA_SURR = 2.0

SEED = 42


# =============================================================================
# 1. SURROGATE GRADIENT
# =============================================================================
@jax.custom_vjp
def spike_fn(x):
    return jnp.heaviside(x, 0.0)

def spike_fn_fwd(x):
    return spike_fn(x), x

def spike_fn_bwd(res, g):
    x = res
    grad = ALPHA_SURR / (1.0 + jnp.abs(ALPHA_SURR * x)) ** 2
    return (g * grad,)

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)


# =============================================================================
# 2. LIF NEURON DYNAMICS
# =============================================================================
def lif_step(state, x_t, W, beta=BETA, v_th=V_TH):
    U_prev, S_prev = state
    I_t = jnp.dot(x_t, W)
    U_t = beta * U_prev + I_t - (S_prev * v_th)
    S_t = spike_fn(U_t - v_th)
    return (U_t, S_t), S_t


def run_snn_layer(x_seq, W, beta=BETA, v_th=V_TH):
    batch = x_seq.shape[1]
    hidden = W.shape[1]
    def step(state, x_t):
        return lif_step(state, x_t, W, beta, v_th)
    _, out = jax.lax.scan(step,
                          (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden))),
                          x_seq)
    return out  # (T, B, hidden)


# =============================================================================
# 3. LEAKY INTEGRATOR (LI) READOUT
# =============================================================================
def run_li_readout(spike_seq, W_li, b_li, beta_li=BETA_LI):
    """
    Non-spiking Leaky Integrator readout over a spike sequence.
    Preserves temporal ordering — BPTT gradients flow through the scan.

    spike_seq: (T, B, HIDDEN2) — spike outputs from last LIF layer
    W_li:      (HIDDEN2, STATE_DIM) — readout weights
    b_li:      (STATE_DIM,) — trainable bias
    returns:   (T, B, STATE_DIM) — full voltage trace (no tanh, no clipping)

    Dynamics (no threshold, no spiking, no reset):
        U_t = beta_li * U_{t-1} + S_t @ W_li + b_li
    """
    def li_step(U_prev, s_t):
        U_t = beta_li * U_prev + jnp.dot(s_t, W_li) + b_li
        return U_t, U_t

    batch = spike_seq.shape[1]
    U_init = jnp.zeros((batch, W_li.shape[1]))
    _, U_seq = jax.lax.scan(li_step, U_init, spike_seq)
    return U_seq  # (T, B, STATE_DIM) — full trace for windowed loss


# =============================================================================
# 4. FULL NETWORK
# =============================================================================
def run_snn(x_seq, W1, W2, W_li, b_li, beta=BETA, beta_li=BETA_LI, v_th=V_TH):
    h = run_snn_layer(x_seq, W1, beta, v_th)
    h = run_snn_layer(h, W2, beta, v_th)
    # LI readout: raw voltage trace, no tanh, no bounding
    U_seq = run_li_readout(h, W_li, b_li, beta_li)
    return U_seq  # (T, B, STATE_DIM)


# =============================================================================
# 4. PREPROCESSING
# =============================================================================
def prepare_events(events):
    """Split events into ON/OFF polarity channels. (B,T,64) → (T,B,128)"""
    on = jnp.maximum(events, 0.0)
    off = jnp.maximum(-events, 0.0)
    polarized = jnp.concatenate([on, off], axis=-1)
    return jnp.transpose(polarized, (1, 0, 2))


# =============================================================================
# 5. ADAM OPTIMIZER
# =============================================================================
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = jax.tree_util.tree_map(jnp.zeros_like, params)
        self.v = jax.tree_util.tree_map(jnp.zeros_like, params)

    def step(self, params, grads):
        self.t += 1
        self.m = jax.tree_util.tree_map(
            lambda m, g: self.b1 * m + (1 - self.b1) * g, self.m, grads)
        self.v = jax.tree_util.tree_map(
            lambda v, g: self.b2 * v + (1 - self.b2) * g ** 2, self.v, grads)
        mh = jax.tree_util.tree_map(lambda m: m / (1 - self.b1 ** self.t), self.m)
        vh = jax.tree_util.tree_map(lambda v: v / (1 - self.b2 ** self.t), self.v)
        return jax.tree_util.tree_map(
            lambda p, m, v: p - self.lr * m / (jnp.sqrt(v) + self.eps),
            params, mh, vh)


# =============================================================================
# 6. TRAINING
# =============================================================================
@jax.value_and_grad
def loss_fn(params, x_seq, labels):
    U_seq = run_snn(x_seq, *params)
    # Windowed loss: skip burn-in, average MSE over last LOSS_WINDOW timesteps
    # U_seq: (T, B, STATE_DIM), labels: (B, STATE_DIM)
    U_window = U_seq[-LOSS_WINDOW:]                        # (W, B, STATE_DIM)
    # labels broadcast to (W, B, STATE_DIM) — same label for all timesteps
    loss = jnp.mean((U_window - labels[jnp.newaxis, :, :]) ** 2)
    return loss


def clipped_update(params, grads, optimizer):
    total_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)
    ))
    clip_factor = jnp.minimum(1.0, GRAD_CLIP_NORM / (total_norm + 1e-8))
    clipped = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    return optimizer.step(params, clipped)


def init_params(key):
    k1, k2, k3 = jax.random.split(key, 3)
    W1 = jax.random.normal(k1, (INPUT_DIM, HIDDEN1)) * jnp.sqrt(2.0 / INPUT_DIM) * 7.0   # neuromorphic scaling
    W2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * jnp.sqrt(2.0 / HIDDEN1) * 1.0       # neuromorphic scaling
    W_li = jax.random.normal(k3, (HIDDEN2, STATE_DIM)) * 0.1   # 10x scale
    b_li = jnp.zeros((STATE_DIM,))                                # trainable bias
    return (W1, W2, W_li, b_li)


def train():
    print("=" * 60)
    print("  🦊 SNN 2D Navigation — Training")
    print("=" * 60)
    print(f"  Environment:   {ROOM_W}m × {ROOM_H}m room, {N_OBSTACLES} obstacles")
    print(f"  Architecture:  {INPUT_DIM} → {HIDDEN1} → {HIDDEN2} → LI({STATE_DIM})")
    print(f"  Timesteps:     {TIME_STEPS} ({TIME_STEPS * DT:.1f}s)")
    print(f"  Epochs:        {N_EPOCHS}")
    print(f"  Train batch:   {TRAIN_BATCH}")
    print(f"  Eval batch:    {EVAL_BATCH}")
    print(f"  LR:            {LEARNING_RATE}")
    print(f"  β_LIF:         {BETA}, β_LI: {BETA_LI}, V_th: {V_TH}")
    print(f"  Readout:       LI (stateful, no spike, no reset, no tanh)")
    print(f"  Loss window:   last {LOSS_WINDOW} timesteps (skip burn-in)")
    print(f"  W init:        W1×7, W2×1 (neuromorphic scaling)")
    print(f"  W_li init:     0.1 (10x scale), trainable bias")
    print(f"  Labels:        [vx, vy, omega, min_clearance]")
    print("=" * 60)

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    optimizer = Adam(params, lr=LEARNING_RATE)

    # JIT compile
    print("\n  🔨 Compiling (this will also JIT the env generator)...")
    t0 = time.time()
    dummy_x = jnp.zeros((TIME_STEPS, 4, INPUT_DIM))
    dummy_l = jnp.zeros((4, STATE_DIM))
    loss_fn(params, dummy_x, dummy_l)
    print(f"  Loss compile:  {time.time() - t0:.2f}s")

    # Also warm up the env generator
    print("  Warming up env generator...")
    t0 = time.time()
    key, k0 = jax.random.split(key)
    generate_batch(k0, 4)
    print(f"  Env compile:   {time.time() - t0:.2f}s")

    history = {'train_loss': [], 'eval_loss': []}
    best_eval = float('inf')
    best_params = params
    label_names = ['vx (m/s)', 'vy (m/s)', 'ω (rad/s)', 'clearance']
    label_scales = jnp.array([0.8, 0.3, 0.5, 1.0])  # for real-unit RMSE

    print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Eval':>8}  "
          f"{'RMSE vx':>8}  {'RMSE vy':>8}  {'RMSE ω':>8}  {'RMSE cl':>8}")

    for epoch in range(1, N_EPOCHS + 1):
        # Train
        key, subkey = jax.random.split(key)
        events, labels, _ = generate_batch(subkey, TRAIN_BATCH)
        x_seq = prepare_events(events)

        loss, grads = loss_fn(params, x_seq, labels)
        params = clipped_update(params, grads, optimizer)
        history['train_loss'].append(float(loss))

        # Evaluate
        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS:
            key, subkey = jax.random.split(key)
            ev_eval, lb_eval, _ = generate_batch(subkey, EVAL_BATCH)
            x_eval = prepare_events(ev_eval)

            preds = run_snn(x_eval, *params)
            # Use mean of last window for evaluation
            pred_eval = jnp.mean(preds[-LOSS_WINDOW:], axis=0)  # (B, STATE_DIM)
            eval_mse = jnp.mean((pred_eval - lb_eval) ** 2)

            rmse = jnp.sqrt(jnp.mean((pred_eval - lb_eval) ** 2, axis=0))
            rmse_real = rmse * label_scales

            history['eval_loss'].append(float(eval_mse))
            history['preds'] = np.array(pred_eval)
            history['labels'] = np.array(lb_eval)

            print(f"  {epoch:>5d}  {float(loss):>8.4f}  {float(eval_mse):>8.4f}  "
                  f"{rmse_real[0]:>7.3f}  {rmse_real[1]:>7.3f}  "
                  f"{rmse_real[2]:>7.3f}  {rmse_real[3]:>7.3f}")

            if float(eval_mse) < best_eval:
                best_eval = float(eval_mse)
                best_params = jax.tree_util.tree_map(jnp.copy, params)

    # Final eval
    print(f"\n  📊 Final evaluation (best model)...")
    key, subkey = jax.random.split(key)
    ev_f, lb_f, _ = generate_batch(subkey, EVAL_BATCH)
    x_f = prepare_events(ev_f)
    preds_f = run_snn(x_f, *best_params)
    # Mean over last window for final evaluation
    pred_avg = jnp.mean(preds_f[-LOSS_WINDOW:], axis=0)

    final_mse = float(jnp.mean((pred_avg - lb_f) ** 2))
    rmse_f = jnp.sqrt(jnp.mean((pred_avg - lb_f) ** 2, axis=0))
    rmse_real = rmse_f * label_scales

    print(f"\n  Best eval loss: {best_eval:.4f}")
    print(f"  Final MSE:      {final_mse:.4f}")
    print(f"  RMSE (real units):")
    for name, val in zip(label_names, rmse_real):
        print(f"    {name:>16s}: {val:.4f}")

    # Per-label correlation
    print(f"\n  Correlations (predicted vs true):")
    for name, pred_col, true_col in zip(label_names,
                                         np.array(pred_avg).T,
                                         np.array(lb_f).T):
        r = np.corrcoef(pred_col, true_col)[0, 1]
        print(f"    {name:>16s}: r = {r:+.3f}")

    # Sample predictions
    print(f"\n  📋 Predictions (normalized [-1,1]):")
    print(f"  {'':>4}  {'vx':>5} {'vy':>5} {'ω':>5} {'cl':>5}  |  "
          f"{'vx̂':>6} {'vŷ':>6} {'ω̂':>6} {'cl̂':>6}")
    for i in range(min(8, EVAL_BATCH)):
        t, p = lb_f[i], pred_avg[i]
        print(f"  [{i:>2}]  {t[0]:>+5.2f} {t[1]:>+5.2f} {t[2]:>+5.2f} {t[3]:>+5.2f}  |  "
              f"{p[0]:>+6.2f} {p[1]:>+6.2f} {p[2]:>+6.2f} {p[3]:>+6.2f}")

    # Save
    _plot_training(history,
                   "/Users/lhooz/.openclaw/workspace/snn_2d_training_curve.png")
    _save_params(best_params,
                 "/Users/lhooz/.openclaw/workspace/snn_2d_params.npz")

    print(f"\n  ✅ Training complete!")
    return best_params


# =============================================================================
# VISUALIZATION
# =============================================================================
def _plot_training(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    ax1 = axes[0]
    ax1.plot(history['train_loss'], alpha=0.4, label='Train')
    n_eval = len(history['eval_loss'])
    eval_epochs = [i * 10 if i * 10 < len(history['train_loss'])
                   else len(history['train_loss']) - 1 for i in range(n_eval)]
    ax1.plot(eval_epochs, history['eval_loss'],
             'o-', color='tab:orange', ms=3, label='Eval')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(alpha=0.3)

    # True vs predicted scatter
    ax2 = axes[1]
    if 'preds' in history and 'labels' in history:
        names = ['vx', 'vy', 'ω', 'clearance']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for c_idx in range(4):
            p = history['preds'][:, c_idx]
            l = history['labels'][:, c_idx]
            ax2.scatter(l, p, alpha=0.4, s=10, c=colors[c_idx], label=names[c_idx])
        lim = 1.2
        ax2.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, label='y=x')
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_xlabel('True')
        ax2.set_ylabel('Predicted')
        ax2.set_title('True vs Predicted (by label)')
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

    # Per-label residual distributions
    ax3 = axes[2]
    if 'preds' in history and 'labels' in history:
        residual = history['preds'] - history['labels']
        names = ['vx', 'vy', 'ω', 'clearance']
        bp = ax3.boxplot([residual[:, i] for i in range(4)],
                         labels=names, patch_artist=True)
        box_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax3.axhline(0, color='gray', lw=0.5, ls='--')
        ax3.set_ylabel('Residual (pred − true)')
        ax3.set_title('Prediction Errors by Label')
        ax3.grid(alpha=0.3)

    fig.suptitle('SNN 2D Navigation Training', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  📸 Saved to {path}")
    plt.close(fig)


def _save_params(params, path):
    W1, W2, W_li, b_li = params
    np.savez(path, W1=np.array(W1), W2=np.array(W2),
             W_li=np.array(W_li), b_li=np.array(b_li))
    print(f"  💾 Saved params to {path}")


if __name__ == "__main__":
    train()
