#!/usr/bin/env python3
"""
SNN vx-Only in the Infinite Hallway

Full SNN: Events (128) → LIF(128) → LIF(64) → LI(1)
Hallway: 3m wide, no obstacles, Z=1.5m constant, only vx randomized.

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

from hallway_env import (
    generate_batch, N_PIXELS, TIME_STEPS, DT,
    HALLWAY_WIDTH, THRESHOLD,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-3
N_EPOCHS = 500
TRAIN_BATCH = 64
EVAL_BATCH = 256
HIDDEN1 = 128
HIDDEN2 = 64
STATE_DIM = 1
INPUT_DIM = 2 * N_PIXELS

BETA = 0.85
BETA_LI = 0.95
V_TH = 1.0
ALPHA_SURR = 2.0
LOSS_WINDOW = 50
GRAD_CLIP_NORM = 1.0
VX_SCALE = 0.8
SEED = 42

# Neuromorphic weight scaling
W1_MULT = 7.0
W2_MULT = 1.0


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
# 2. LIF NEURON (soft reset by subtraction)
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
    return out


# =============================================================================
# 3. LEAKY INTEGRATOR READOUT (gradient-normalized)
# =============================================================================
def run_li_readout(spike_seq, W_li, b_li, beta_li=BETA_LI):
    T = spike_seq.shape[0]
    norm = 1.0 / T
    def li_step(U_prev, s_t):
        U_t = beta_li * U_prev + jnp.dot(s_t, W_li) * norm + b_li
        return U_t, U_t
    batch = spike_seq.shape[1]
    U_init = jnp.zeros((batch, W_li.shape[1]))
    _, U_seq = jax.lax.scan(li_step, U_init, spike_seq)
    return U_seq


# =============================================================================
# 4. FULL NETWORK
# =============================================================================
def run_snn(x_seq, W1, W2, W_li, b_li, beta=BETA, beta_li=BETA_LI, v_th=V_TH):
    h = run_snn_layer(x_seq, W1, beta, v_th)
    h = run_snn_layer(h, W2, beta, v_th)
    U_seq = run_li_readout(h, W_li, b_li, beta_li)
    return U_seq


# =============================================================================
# 5. PREPROCESSING
# =============================================================================
def prepare_events(events):
    on = jnp.maximum(events, 0.0)
    off = jnp.maximum(-events, 0.0)
    polarized = jnp.concatenate([on, off], axis=-1)
    return jnp.transpose(polarized, (1, 0, 2))


# =============================================================================
# 6. ADAM
# =============================================================================
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = jax.tree_util.tree_map(jnp.zeros_like, params)
        self.v = jax.tree_util.tree_map(jnp.zeros_like, params)
    def step(self, params, grads):
        self.t += 1
        self.m = jax.tree_util.tree_map(lambda m, g: self.b1*m+(1-self.b1)*g, self.m, grads)
        self.v = jax.tree_util.tree_map(lambda v, g: self.b2*v+(1-self.b2)*g**2, self.v, grads)
        mh = jax.tree_util.tree_map(lambda m: m/(1-self.b1**self.t), self.m)
        vh = jax.tree_util.tree_map(lambda v: v/(1-self.b2**self.t), self.v)
        return jax.tree_util.tree_map(lambda p,m,v: p-self.lr*m/(jnp.sqrt(v)+self.eps), params, mh, vh)


# =============================================================================
# 7. TRAINING
# =============================================================================
@jax.value_and_grad
def loss_fn(params, x_seq, labels):
    U_seq = run_snn(x_seq, *params)
    U_window = U_seq[-LOSS_WINDOW:]
    return jnp.mean((U_window - labels[jnp.newaxis, :, :]) ** 2)


def clipped_update(params, grads, optimizer):
    total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    clip = jnp.minimum(1.0, GRAD_CLIP_NORM / (total_norm + 1e-8))
    clipped = jax.tree_util.tree_map(lambda g: g * clip, grads)
    return optimizer.step(params, clipped)


def init_params(key):
    k1, k2, k3 = jax.random.split(key, 3)
    W1 = jax.random.normal(k1, (INPUT_DIM, HIDDEN1)) * jnp.sqrt(2.0/INPUT_DIM) * W1_MULT
    W2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * jnp.sqrt(2.0/HIDDEN1) * W2_MULT
    W_li = jax.random.normal(k3, (HIDDEN2, STATE_DIM)) * 0.1
    b_li = jnp.zeros((STATE_DIM,))
    return (W1, W2, W_li, b_li)


def train():
    print("=" * 60)
    print("  🏗️ SNN vx-Only — Infinite Hallway (Z=1.5m constant)")
    print("=" * 60)
    print(f"  Hallway:       {HALLWAY_WIDTH}m wide, no obstacles")
    print(f"  Z (wall dist): {HALLWAY_WIDTH/2:.1f}m (CONSTANT)")
    print(f"  Architecture:  {INPUT_DIM} → LIF({HIDDEN1}) → LIF({HIDDEN2}) → LI({STATE_DIM})")
    print(f"  Timesteps:     {TIME_STEPS} ({TIME_STEPS*DT:.1f}s)")
    print(f"  Epochs:        {N_EPOCHS}")
    print(f"  Train batch:   {TRAIN_BATCH} (fresh vx each sample)")
    print(f"  Eval batch:    {EVAL_BATCH}")
    print(f"  LR:            {LEARNING_RATE}")
    print(f"  β_LIF={BETA}, β_LI={BETA_LI}, V_th={V_TH}, α_surr={ALPHA_SURR}")
    print(f"  W init:        W1×{W1_MULT}, W2×{W2_MULT}")
    print(f"  LI input:      normalized 1/T")
    print(f"  Loss window:   last {LOSS_WINDOW} timesteps")
    print(f"  Threshold C:   {THRESHOLD}")
    print("=" * 60)

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    opt = Adam(params, lr=LEARNING_RATE)

    # JIT compile
    print("\n  🔨 Compiling...")
    t0 = time.time()
    dummy_x = jnp.zeros((TIME_STEPS, 4, INPUT_DIM))
    dummy_l = jnp.zeros((4, 1))
    loss_fn(params, dummy_x, dummy_l)
    print(f"  Loss compile:  {time.time()-t0:.2f}s")

    print("  Warming up env...")
    t0 = time.time()
    key, k0 = jax.random.split(key)
    generate_batch(k0, 4)
    print(f"  Env compile:   {time.time()-t0:.2f}s")

    best_eval = float('inf')
    best_r = 0
    best_params = params

    print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Eval':>8}  {'|∇|':>6}  {'RMSE vx':>8}  {'r(vx)':>8}")

    for epoch in range(1, N_EPOCHS + 1):
        key, subkey = jax.random.split(key)
        events, labels, _ = generate_batch(subkey, TRAIN_BATCH)
        x_seq = prepare_events(events)

        loss, grads = loss_fn(params, x_seq, labels)
        gn = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        params = clipped_update(params, grads, opt)

        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS:
            key, subkey = jax.random.split(key)
            ev2, lb2, _ = generate_batch(subkey, EVAL_BATCH)
            x2 = prepare_events(ev2)
            U2 = run_snn(x2, *params)
            pred = jnp.mean(U2[-LOSS_WINDOW:], axis=0)
            mse = float(jnp.mean((pred - lb2) ** 2))
            rmse = float(jnp.sqrt(mse) * VX_SCALE)
            r = np.corrcoef(np.array(pred)[:, 0], np.array(lb2)[:, 0])[0, 1]

            marker = " ★" if mse < best_eval else ""
            if mse < best_eval:
                best_eval = mse
                best_r = r
                best_params = jax.tree_util.tree_map(jnp.copy, params)

            print(f"  {epoch:>5d}  {float(loss):>8.4f}  {mse:>8.4f}  "
                  f"{float(gn):>5.1f}  {rmse:>7.3f}  {r:>+7.3f}{marker}")

    # Final eval
    print(f"\n  📊 Final evaluation (best model)...")
    key, subkey = jax.random.split(key)
    ev_f, lb_f, _ = generate_batch(subkey, EVAL_BATCH)
    x_f = prepare_events(ev_f)
    U_f = run_snn(x_f, *best_params)
    pred_f = jnp.mean(U_f[-LOSS_WINDOW:], axis=0)
    r_f = np.corrcoef(np.array(pred_f)[:, 0], np.array(lb_f)[:, 0])[0, 1]
    mse_f = float(jnp.mean((pred_f - lb_f) ** 2))

    print(f"\n  Best eval MSE: {best_eval:.4f}")
    print(f"  Final MSE:     {mse_f:.4f}")
    print(f"  RMSE vx (m/s): {float(jnp.sqrt(mse_f)) * VX_SCALE:.4f}")
    print(f"  Correlation:   r = {r_f:+.4f}")

    print(f"\n  📋 Sample predictions:")
    print(f"  {'':>4}  {'vx true':>8}  {'vx pred':>8}  {'err':>8}")
    for i in range(min(12, EVAL_BATCH)):
        t, p = lb_f[i, 0], pred_f[i, 0]
        print(f"  [{i:>2}]  {t:>+8.3f}  {p:>+8.3f}  {float(p-t):>+8.3f}")

    print(f"\n  {'='*60}")
    if abs(r_f) > 0.5:
        print(f"  ✅ PASS — SNN learns vx in hallway! (r={r_f:+.3f})")
        print(f"  Monocular scale ambiguity eliminated.")
        print(f"  Ready to add vy, ω, clearance.")
    elif abs(r_f) > 0.3:
        print(f"  ⚠️  MARGINAL — signal present but weak (r={r_f:+.3f})")
    else:
        print(f"  ❌ FAIL — vx still unlearnable (r={r_f:+.3f})")
    print(f"  {'='*60}")

    # Save
    W1, W2, W_li, b_li = best_params
    np.savez("/Users/lhooz/.openclaw/workspace/hallway_vx_params.npz",
             W1=np.array(W1), W2=np.array(W2),
             W_li=np.array(W_li), b_li=np.array(b_li))
    print(f"  💾 Saved params")

    # Plot
    _plot_training(epoch, best_eval, best_r, 
                   "/Users/lhooz/.openclaw/workspace/hallway_vx_curve.png")
    print(f"  ✅ Done!")
    return best_params, r_f


def _plot_training(n_epochs, best_mse, best_r, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.text(0.5, 0.6, f'Hallway vx-Only\nr = {best_r:+.4f}\nMSE = {best_mse:.4f}\n{n_epochs} epochs',
            ha='center', va='center', fontsize=16, fontweight='bold',
            transform=ax.transAxes,
            color='green' if abs(best_r) > 0.5 else 'red')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.suptitle('SNN Infinite Hallway — vx Prediction', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  📸 Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    train()
