#!/usr/bin/env python3
"""
Event Camera → SNN State Estimation Pipeline

Integrates events from event_camera_1d.py with a Spiking Neural Network
to estimate the MAV dynamic state vector [vx, vy, pitch, pitch_rate].

Architecture:
  Events (B,T,64) → Polarity Split (B,T,128) → LIF Layer 1 (128→128)
  → LIF Layer 2 (128→64) → Spike Count (B,64) → Linear Readout (B,4)

Training:
  - Loss: MSE between predicted and true normalized state
  - Optimizer: Adam
  - BPTT via JAX autodiff + jax.lax.scan
  - Readout: tanh-bounded linear layer on accumulated spike counts

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

from event_camera_1d import (
    generate_batch, N_PIXELS, TIME_STEPS, DT,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-3
N_EPOCHS = 300
TRAIN_BATCH = 128
EVAL_BATCH = 64
HIDDEN1 = 128
HIDDEN2 = 64
GRAD_CLIP_NORM = 1.0
STATE_DIM = 4
INPUT_DIM = 2 * N_PIXELS  # 128 (ON/OFF polarity split)

# SNN hyperparameters
BETA = 0.85          # membrane potential decay (slightly lower = faster dynamics)
V_TH = 1.0           # spike threshold
ALPHA_SURR = 2.0     # surrogate gradient steepness


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
# 3. FULL NETWORK
# =============================================================================
def run_snn(x_seq, W1, W2, W_read, b_read, beta=BETA, v_th=V_TH):
    h = run_snn_layer(x_seq, W1, beta, v_th)       # (T, B, H1)
    h = run_snn_layer(h, W2, beta, v_th)            # (T, B, H2)
    spike_count = jnp.sum(h, axis=0)                 # (B, H2)
    pred = jnp.tanh(jnp.dot(spike_count, W_read) + b_read)  # (B, 4)
    return pred


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
    pred = run_snn(x_seq, *params)
    return jnp.mean((pred - labels) ** 2)


def clipped_update(params, grads, optimizer):
    """Apply gradient clipping before the optimizer step."""
    # Clip by global norm
    total_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)
    ))
    clip_factor = jnp.minimum(1.0, GRAD_CLIP_NORM / (total_norm + 1e-8))
    clipped = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    return optimizer.step(params, clipped)


def init_params(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    W1 = jax.random.normal(k1, (INPUT_DIM, HIDDEN1)) * jnp.sqrt(2.0 / INPUT_DIM)
    W2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * jnp.sqrt(2.0 / HIDDEN1)
    W_out = jax.random.normal(k3, (HIDDEN2, STATE_DIM)) * 0.01
    b_out = jnp.zeros(STATE_DIM)
    return (W1, W2, W_out, b_out)


def train():
    print("=" * 60)
    print("  🦊 SNN State Estimation — Training")
    print("=" * 60)
    print(f"  Architecture:  {INPUT_DIM} → {HIDDEN1} → {HIDDEN2} → {STATE_DIM}")
    print(f"  Timesteps:     {TIME_STEPS} ({TIME_STEPS * DT:.1f}s)")
    print(f"  Epochs:        {N_EPOCHS}")
    print(f"  Train batch:   {TRAIN_BATCH}")
    print(f"  Eval batch:    {EVAL_BATCH}")
    print(f"  LR:            {LEARNING_RATE}")
    print(f"  β (decay):     {BETA}, V_th: {V_TH}")
    print("=" * 60)

    key = jax.random.PRNGKey(0)
    params = init_params(key)
    optimizer = Adam(params, lr=LEARNING_RATE)

    # JIT compile
    print("\n  🔨 Compiling...")
    t0 = time.time()
    dummy_x = jnp.zeros((TIME_STEPS, 4, INPUT_DIM))
    dummy_l = jnp.zeros((4, STATE_DIM))
    loss_fn(params, dummy_x, dummy_l)
    print(f"  Compile: {time.time() - t0:.2f}s")

    history = {'train_loss': [], 'eval_loss': []}
    best_eval = float('inf')
    best_params = params

    print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Eval':>8}  "
          f"{'RMSE vx':>8}  {'RMSE vy':>8}  {'RMSE p':>8}  {'RMSE ω':>8}")

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
            eval_mse = jnp.mean((preds - lb_eval) ** 2)

            rmse = jnp.sqrt(jnp.mean((preds - lb_eval) ** 2, axis=0))
            scales = jnp.array([1.0, 1.0, 0.17, 0.17])
            rmse_real = rmse * scales

            history['eval_loss'].append(float(eval_mse))
            history['preds'] = np.array(preds)
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

    final_mse = float(jnp.mean((preds_f - lb_f) ** 2))
    rmse_f = jnp.sqrt(jnp.mean((preds_f - lb_f) ** 2, axis=0))
    scales = jnp.array([1.0, 1.0, 0.17, 0.17])
    rmse_real = rmse_f * scales

    print(f"\n  Best eval loss: {best_eval:.4f}")
    print(f"  Final MSE:      {final_mse:.4f}")
    print(f"  RMSE (real units):")
    for name, val in zip(['vx (m/s)', 'vy (m/s)', 'pitch (rad)', 'ω (rad/s)'], rmse_real):
        print(f"    {name:>16s}: {val:.4f}")

    # Sample predictions
    print(f"\n  📋 Predictions (normalized [-1,1]):")
    print(f"  {'':>4}  {'vx':>5} {'vy':>5} {'pit':>5} {'ω':>5}  |  "
          f"{'vx̂':>6} {'vŷ':>6} {'pit̂':>6} {'ω̂':>6}")
    for i in range(min(8, EVAL_BATCH)):
        t, p = lb_f[i], preds_f[i]
        print(f"  [{i:>2}]  {t[0]:>+5.2f} {t[1]:>+5.2f} {t[2]:>+5.2f} {t[3]:>+5.2f}  |  "
              f"{p[0]:>+6.2f} {p[1]:>+6.2f} {p[2]:>+6.2f} {p[3]:>+6.2f}")

    # Save
    _plot_training(history, "/Users/lhooz/.openclaw/workspace/snn_training_curve.png")
    _save_params(best_params, "/Users/lhooz/.openclaw/workspace/snn_params.npz")

    print(f"\n  ✅ Training complete!")
    return best_params


# =============================================================================
# VISUALIZATION
# =============================================================================
def _plot_training(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(history['train_loss'], alpha=0.4, label='Train')
    n_eval = len(history['eval_loss'])
    eval_epochs = [i * 10 if i * 10 < len(history['train_loss']) else len(history['train_loss']) - 1
                   for i in range(n_eval)]
    ax1.plot(eval_epochs, history['eval_loss'],
             'o-', color='tab:orange', ms=3, label='Eval')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(alpha=0.3)

    # True vs predicted scatter
    if 'preds' in history and 'labels' in history:
        p = history['preds'].flatten()
        l = history['labels'].flatten()
        ax2.scatter(l, p, alpha=0.3, s=8, c='steelblue')
        lim = 1.2
        ax2.plot([-lim, lim], [-lim, lim], 'r--', lw=0.8, label='y=x')
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_xlabel('True (normalized)')
        ax2.set_ylabel('Predicted (normalized)')
        ax2.set_title('True vs Predicted')
        ax2.legend()
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  📸 Saved to {path}")
    plt.close(fig)


def _save_params(params, path):
    W1, W2, W_out, b_out = params
    np.savez(path, W1=np.array(W1), W2=np.array(W2),
             W_out=np.array(W_out), b_out=np.array(b_out))
    print(f"  💾 Saved params to {path}")


if __name__ == "__main__":
    train()
