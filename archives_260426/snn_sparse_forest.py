#!/usr/bin/env python3
"""
SNN 4 DoF — Sparse Forest

Full SNN: Events(128) → LIF(128) → LIF(64) → LI(4)
Environment: 10×10m sparse forest, 2-3 obstacles (max 1×1m)
Labels: [vx, vy, omega, clearance]

Proven architecture from box curriculum:
  - Soft reset LIF neurons
  - Leaky integrator readout with 1/T gradient normalization
  - Windowed loss (last 50 timesteps)
  - Adam optimizer with gradient clipping
  - No distance dimming (Conservation of Radiance)
  - Hard rejection: zero collisions in training data

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

from sparse_forest import (
    generate_batch, N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H, THRESHOLD,
    VX_RANGE, VY_RANGE, OMEGA_RANGE,
    SAFE_MARGIN,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-3
N_EPOCHS = 500
TRAIN_BATCH = 32           # smaller batch (hard rejection is slower)
EVAL_BATCH = 128
HIDDEN1 = 128
HIDDEN2 = 64
STATE_DIM = 4
INPUT_DIM = 2 * N_PIXELS

BETA = 0.85
BETA_LI = 0.95
V_TH = 1.0
ALPHA_SURR = 2.0
LOSS_WINDOW = 50
GRAD_CLIP_NORM = 1.0

SEED = 42
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
    batch, hidden = x_seq.shape[1], W.shape[1]
    _, out = jax.lax.scan(
        lambda s, x: lif_step(s, x, W, beta, v_th),
        (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden))),
        x_seq)
    return out


# =============================================================================
# 3. LI READOUT (gradient-normalized)
# =============================================================================
def run_li_readout(spike_seq, W_li, b_li, beta_li=BETA_LI):
    T = spike_seq.shape[0]
    norm = 1.0 / T
    _, U_seq = jax.lax.scan(
        lambda U, s: (beta_li * U + jnp.dot(s, W_li) * norm + b_li,
                       beta_li * U + jnp.dot(s, W_li) * norm + b_li),
        jnp.zeros((spike_seq.shape[1], W_li.shape[1])),
        spike_seq)
    return U_seq


# =============================================================================
# 4. FULL NETWORK
# =============================================================================
def run_snn(x_seq, W1, W2, W_li, b_li):
    h = run_snn_layer(x_seq, W1)
    h = run_snn_layer(h, W2)
    return run_li_readout(h, W_li, b_li)


# =============================================================================
# 5. PREPROCESSING
# =============================================================================
def prepare_events(events):
    on = jnp.maximum(events, 0.0)
    off = jnp.maximum(-events, 0.0)
    return jnp.transpose(jnp.concatenate([on, off], axis=-1), (1, 0, 2))


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
        return jax.tree_util.tree_map(
            lambda p, m, v: p-self.lr*m/(jnp.sqrt(v)+self.eps), params, mh, vh)


# =============================================================================
# 7. TRAINING
# =============================================================================
@jax.value_and_grad
def loss_fn(params, x_seq, labels):
    U_seq = run_snn(x_seq, *params)
    U_window = U_seq[-LOSS_WINDOW:]
    return jnp.mean((U_window - labels[jnp.newaxis, :, :]) ** 2)


def clipped_update(params, grads, opt):
    gn = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    clip = jnp.minimum(1.0, GRAD_CLIP_NORM / (gn + 1e-8))
    return opt.step(params, jax.tree_util.tree_map(lambda g: g * clip, grads))


def init_params(key):
    k1, k2, k3 = jax.random.split(key, 3)
    W1 = jax.random.normal(k1, (INPUT_DIM, HIDDEN1)) * jnp.sqrt(2.0/INPUT_DIM) * W1_MULT
    W2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * jnp.sqrt(2.0/HIDDEN1) * W2_MULT
    W_li = jax.random.normal(k3, (HIDDEN2, STATE_DIM)) * 0.1
    b_li = jnp.zeros((STATE_DIM,))
    return (W1, W2, W_li, b_li)


def train():
    label_names = ['vx', 'vy', 'omega', 'clearance']
    label_scales = jnp.array([abs(VX_RANGE[1]), abs(VY_RANGE[1]),
                               abs(OMEGA_RANGE[1]), 2.0])

    print("=" * 70)
    print("  🌲 SNN 4 DoF — Sparse Forest (Collision-Free)")
    print("=" * 70)
    print(f"  Room:          {ROOM_W}×{ROOM_H}m, 2-3 obstacles (≤1×1m)")
    print(f"  Safe margin:   {SAFE_MARGIN}m (hard rejection)")
    print(f"  Architecture:  {INPUT_DIM} → LIF({HIDDEN1}) → LIF({HIDDEN2}) → LI({STATE_DIM})")
    print(f"  Timesteps:     {TIME_STEPS} ({TIME_STEPS*DT:.1f}s)")
    print(f"  Epochs:        {N_EPOCHS}")
    print(f"  Train batch:   {TRAIN_BATCH} (hard rejection, slower)")
    print(f"  Eval batch:    {EVAL_BATCH}")
    print(f"  LR:            {LEARNING_RATE}")
    print(f"  β_LIF={BETA}, β_LI={BETA_LI}, V_th={V_TH}")
    print(f"  Labels:        [{', '.join(label_names)}]")
    print(f"  Dimming:       OFF (Conservation of Radiance)")
    print("=" * 70)

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    opt = Adam(params, lr=LEARNING_RATE)

    # JIT compile loss
    print("\n  🔨 Compiling loss function...")
    t0 = time.time()
    loss_fn(params, jnp.zeros((TIME_STEPS, 4, INPUT_DIM)), jnp.zeros((4, STATE_DIM)))
    print(f"  Compile: {time.time()-t0:.2f}s")

    print("  Warming up environment...")
    t0 = time.time()
    key, k0 = jax.random.split(key)
    generate_batch(k0, 4)
    print(f"  Env warmup: {time.time()-t0:.2f}s")

    best_eval = float('inf')
    best_params = params

    hdr = f"  {'Epoch':>5}  {'Train':>8}  {'Eval':>8}  {'|∇|':>6}  "
    hdr += "  ".join(f"{'r_'+n:>8}" for n in label_names)
    hdr += f"  {'RMSE cl':>8}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    epoch_time_sum = 0
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        key, subkey = jax.random.split(key)
        events, labels, _ = generate_batch(subkey, TRAIN_BATCH)
        x_seq = prepare_events(events)

        loss, grads = loss_fn(params, x_seq, labels)
        gn = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        params = clipped_update(params, grads, opt)
        epoch_time = time.time() - t0
        epoch_time_sum += epoch_time

        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS:
            key, subkey = jax.random.split(key)
            ev2, lb2, _ = generate_batch(subkey, EVAL_BATCH)
            x2 = prepare_events(ev2)
            U2 = run_snn(x2, *params)
            pred = jnp.mean(U2[-LOSS_WINDOW:], axis=0)
            mse = float(jnp.mean((pred - lb2) ** 2))
            rmse = jnp.sqrt(jnp.mean((pred - lb2) ** 2, axis=0)) * label_scales

            corr_strs = []
            for c in range(STATE_DIM):
                r = np.corrcoef(np.array(pred)[:, c], np.array(lb2)[:, c])[0, 1]
                corr_strs.append(f"{r:>+7.3f}")

            marker = " ★" if mse < best_eval else ""
            if mse < best_eval:
                best_eval = mse
                best_params = jax.tree_util.tree_map(jnp.copy, params)

            avg_t = epoch_time_sum / epoch
            parts = [f"  {epoch:>5d}  {float(loss):>8.4f}  {mse:>8.4f}  "
                     f"{float(gn):>5.1f}"]
            for cs in corr_strs:
                parts.append(f"{cs:>8}")
            parts.append(f"{float(rmse[3]):>8.3f}{marker}")
            print("".join(parts))

    # Final evaluation
    print(f"\n  📊 Final evaluation (best model)...")
    key, subkey = jax.random.split(key)
    ev_f, lb_f, _ = generate_batch(subkey, EVAL_BATCH)
    x_f = prepare_events(ev_f)
    U_f = run_snn(x_f, *best_params)
    pred_f = jnp.mean(U_f[-LOSS_WINDOW:], axis=0)

    final_mse = float(jnp.mean((pred_f - lb_f) ** 2))
    rmse_f = jnp.sqrt(jnp.mean((pred_f - lb_f) ** 2, axis=0)) * label_scales

    print(f"\n  Best eval MSE: {best_eval:.4f}")
    print(f"  Final MSE:     {final_mse:.4f}")
    print(f"  RMSE (real units):")
    for name, val in zip(label_names, rmse_f):
        print(f"    {name:>16s}: {val:.4f}")

    print(f"\n  Correlations:")
    rs = []
    for name, pred_col, true_col in zip(label_names,
                                         np.array(pred_f).T,
                                         np.array(lb_f).T):
        r = np.corrcoef(pred_col, true_col)[0, 1]
        rs.append(r)
        print(f"    {name:>16s}: r = {r:+.4f}")

    print(f"\n  📋 Sample predictions:")
    print(f"  {'':>4}  {'vx_t':>6} {'vx_p':>6}  {'vy_t':>6} {'vy_p':>6}  "
          f"{'ω_t':>6} {'ω_p':>6}  {'cl_t':>6} {'cl_p':>6}")
    for i in range(min(12, EVAL_BATCH)):
        t, p = lb_f[i], pred_f[i]
        print(f"  [{i:>2}]  {t[0]:>+5.2f} {p[0]:>+5.2f}  {t[1]:>+5.2f} {p[1]:>+5.2f}  "
              f"{t[2]:>+5.2f} {p[2]:>+5.2f}  {t[3]:>+5.2f} {p[3]:>+5.2f}")

    r_vx, r_vy, r_om, r_cl = rs
    print(f"\n  {'='*70}")
    thresholds = [0.5, 0.3, 0.5, 0.3]
    names_short = ['vx', 'vy', 'ω', 'cl']
    ok = [abs(r) > th for r, th in zip(rs, thresholds)]
    n_ok = sum(ok)
    summary = ', '.join(f"{n}({r:+.3f})" for n, r in zip(names_short, rs))
    if n_ok == 4:
        print(f"  ✅ PASS — all 4 DoF learned in Sparse Forest!")
    elif n_ok >= 2:
        print(f"  ⚠️  PARTIAL — {n_ok}/4 learned:")
    else:
        print(f"  ❌ FAIL — only {n_ok}/4 learned:")
    print(f"     {summary}")
    print(f"  {'='*70}")

    # Save
    W1, W2, W_li, b_li = best_params
    np.savez("/Users/lhooz/.openclaw/workspace/sparse_forest_params.npz",
             W1=np.array(W1), W2=np.array(W2),
             W_li=np.array(W_li), b_li=np.array(b_li))
    print(f"  💾 Saved params")

    # Plot
    _plot(pred_f, lb_f, best_eval,
           "/Users/lhooz/.openclaw/workspace/sparse_forest_curve.png")
    print(f"  ✅ Done!")
    return best_params, rs


def _plot(preds, labels, best_mse, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    names = ['vx', 'vy', 'omega', 'clearance']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    ax0 = axes[0]
    ax0.text(0.5, 0.5, f'Sparse Forest\n4 DoF + Obstacles\nBest MSE: {best_mse:.4f}',
             ha='center', va='center', fontsize=14, fontweight='bold',
             transform=ax0.transAxes)
    ax0.axis('off')

    for idx, (ax, name, color) in enumerate(zip(axes[1:], names, colors)):
        p = np.array(preds)[:, idx]
        l = np.array(labels)[:, idx]
        r = np.corrcoef(p, l)[0, 1]
        ax.scatter(l, p, alpha=0.5, s=15, c=color)
        lim = max(abs(l).max(), abs(p).max()) * 1.2
        lim = max(lim, 0.3)
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('True'); ax.set_ylabel('Predicted')
        ax.set_title(f'{name} (r={r:+.3f})', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

    fig.suptitle('SNN Sparse Forest — [vx, vy, ω, clearance]', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  📸 Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    train()
