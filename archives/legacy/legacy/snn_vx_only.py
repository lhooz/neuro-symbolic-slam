#!/usr/bin/env python3
"""
Two-stage diagnostic:
  Stage 1: Events → LI only (no LIF) with gradient fix
  Stage 2: Full SNN with wider surrogate gradient + soft reset (if Stage 1 fails)

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

from event_camera_2d_nav import (
    generate_batch, N_PIXELS, TIME_STEPS, DT,
    N_OBSTACLES, ROOM_W, ROOM_H, THRESHOLD,
    TEX_FREQS, TEX_AMPS,
)

# ---------------------------------------------------------------------------
# Shared Config
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-3
N_EPOCHS = 500
TRAIN_BATCH = 64
EVAL_BATCH = 256
HIDDEN1 = 128
HIDDEN2 = 64
STATE_DIM = 1
INPUT_DIM = 2 * N_PIXELS

BETA_LIF = 0.85
BETA_LI = 0.95
V_TH = 1.0
LOSS_WINDOW = 50
GRAD_CLIP_NORM = 1.0
VX_SCALE = 0.8
SEED = 42

# Surrogate gradient widths to compare
ALPHA_WIDE = 0.5    # wide (smooth, far-from-threshold gradients flow)
ALPHA_NARROW = 2.0  # narrow (original, steep cutoff)


# =============================================================================
# SURROGATE GRADIENT (parametric)
# =============================================================================
def make_spike_fn(alpha):
    """Create spike_fn with configurable surrogate gradient width.
    
    f(x) = heaviside(x)
    f'(x) = alpha / (1 + |alpha*x|)^2
    
    alpha=0.5: wide (gradients flow for |x| < ~2)
    alpha=2.0: narrow (gradients flow only for |x| < ~0.5)
    """
    @jax.custom_vjp
    def spike_fn(x):
        return jnp.heaviside(x, 0.0)

    def fwd(x):
        return spike_fn(x), x

    def bwd(res, g):
        x = res
        grad = alpha / (1.0 + jnp.abs(alpha * x)) ** 2
        return (g * grad,)

    spike_fn.defvjp(fwd, bwd)
    return spike_fn


# =============================================================================
# LIF NEURON (soft reset by subtraction)
# =============================================================================
def make_lif_step(spike_fn, beta=BETA_LIF, v_th=V_TH):
    """LIF with soft reset: U_t = β*U_prev + I_t - S_prev*V_th
    
    Soft reset preserves sub-threshold integration and temporal rhythm.
    Already implemented — this just makes it parametric over spike_fn.
    """
    def lif_step(state, x_t, W):
        U_prev, S_prev = state
        I_t = jnp.dot(x_t, W)
        U_t = beta * U_prev + I_t - (S_prev * v_th)
        S_t = spike_fn(U_t - v_th)
        return (U_t, S_t), S_t
    return lif_step


def run_snn_layer(x_seq, W, lif_step_fn):
    batch = x_seq.shape[1]
    hidden = W.shape[1]
    def step(state, x_t):
        return lif_step_fn(state, x_t, W)
    _, out = jax.lax.scan(step,
                          (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden))),
                          x_seq)
    return out


# =============================================================================
# LEAKY INTEGRATOR READOUT
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
# ADAM
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
# PREPROCESSING
# =============================================================================
def prepare_events(events):
    on = jnp.maximum(events, 0.0)
    off = jnp.maximum(-events, 0.0)
    polarized = jnp.concatenate([on, off], axis=-1)
    return jnp.transpose(polarized, (1, 0, 2))


def extract_vx_label(labels):
    return labels[:, :1]


def clipped_update(params, grads, optimizer):
    total_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)
    ))
    clip_factor = jnp.minimum(1.0, GRAD_CLIP_NORM / (total_norm + 1e-8))
    clipped = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    return optimizer.step(params, clipped)


# =============================================================================
# STAGE 1: Events → LI only (no LIF)
# =============================================================================
def run_stage1():
    print("\n" + "=" * 60)
    print("  🔬 STAGE 1: Events → LI Readout (no LIF layers)")
    print("=" * 60)
    print("  Purpose: Can a linear temporal model learn vx at all?")
    print("=" * 60)

    key = jax.random.PRNGKey(SEED)

    # Direct model: x_seq → LI → vx
    def run_model(x_seq, W_li, b_li):
        return run_li_readout(x_seq, W_li, b_li, BETA_LI)

    def init(key):
        k1 = jax.random.split(key, 2)[0]
        W_li = jax.random.normal(k1, (INPUT_DIM, STATE_DIM)) * 0.1
        b_li = jnp.zeros((STATE_DIM,))
        return (W_li, b_li)

    @jax.value_and_grad
    def loss_fn(params, x_seq, labels):
        U_seq = run_model(x_seq, *params)
        U_window = U_seq[-LOSS_WINDOW:]
        return jnp.mean((U_window - labels[jnp.newaxis, :, :]) ** 2)

    params = init(key)
    opt = Adam(params, lr=LEARNING_RATE)

    # Warmup
    dummy_x = jnp.zeros((TIME_STEPS, 4, INPUT_DIM))
    dummy_l = jnp.zeros((4, 1))
    loss_fn(params, dummy_x, dummy_l)
    key, k0 = jax.random.split(key)
    generate_batch(k0, 4, disable_dimming=True)

    best_r = 0
    best_mse = float('inf')
    best_params = params

    print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Eval':>8}  {'|∇|':>6}  {'r(vx)':>8}")

    for epoch in range(1, N_EPOCHS + 1):
        key, subkey = jax.random.split(key)
        ev, lb, _ = generate_batch(subkey, TRAIN_BATCH, disable_dimming=True)
        x = prepare_events(ev)
        vx = extract_vx_label(lb)

        loss, grads = loss_fn(params, x, vx)
        gn = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        params = clipped_update(params, grads, opt)

        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS:
            key, subkey = jax.random.split(key)
            ev2, lb2, _ = generate_batch(subkey, EVAL_BATCH, disable_dimming=True)
            x2 = prepare_events(ev2); vx2 = extract_vx_label(lb2)
            U = run_model(x2, *params)
            pred = jnp.mean(U[-LOSS_WINDOW:], axis=0)
            mse = float(jnp.mean((pred - vx2) ** 2))
            r = np.corrcoef(np.array(pred)[:, 0], np.array(vx2)[:, 0])[0, 1]

            if mse < best_mse:
                best_mse = mse; best_r = r
                best_params = jax.tree_util.tree_map(jnp.copy, params)

            print(f"  {epoch:>5d}  {float(loss):>8.4f}  {mse:>8.4f}  "
                  f"{float(gn):>5.1f}  {r:>+7.3f}")

    # Final eval
    key, subkey = jax.random.split(key)
    ev_f, lb_f, _ = generate_batch(subkey, EVAL_BATCH, disable_dimming=True)
    x_f = prepare_events(ev_f); vx_f = extract_vx_label(lb_f)
    U_f = run_model(x_f, *best_params)
    pred_f = jnp.mean(U_f[-LOSS_WINDOW:], axis=0)
    r_f = np.corrcoef(np.array(pred_f)[:, 0], np.array(vx_f)[:, 0])[0, 1]

    print(f"\n  Best eval MSE: {best_mse:.4f}")
    print(f"  Correlation:   r = {r_f:+.4f}")

    if abs(r_f) > 0.5:
        print(f"  ✅ PASS — Linear temporal model works!")
        return True, r_f
    elif abs(r_f) > 0.3:
        print(f"  ⚠️  MARGINAL (r={r_f:+.3f})")
        return False, r_f
    else:
        print(f"  ❌ FAIL — Events alone don't carry vx (r={r_f:+.3f})")
        return False, r_f


# =============================================================================
# STAGE 2: Full SNN with upgraded LIF
# =============================================================================
def run_stage2():
    print("\n" + "=" * 60)
    print("  🧠 STAGE 2: Full SNN — Wider Surrogate + Soft Reset")
    print("=" * 60)
    print(f"  Surrogate alpha: {ALPHA_WIDE} (was {ALPHA_NARROW})")
    print(f"  LIF reset: soft (subtract V_th, preserve sub-threshold)")
    print(f"  Architecture: {INPUT_DIM} → LIF({HIDDEN1}) → LIF({HIDDEN2}) → LI({STATE_DIM})")
    print(f"  β_LIF={BETA_LIF}, β_LI={BETA_LI}, V_th={V_TH}")
    print(f"  W init: W1×7, W2×1")
    print(f"  LI input: normalized 1/T")
    print("=" * 60)

    spike_fn = make_spike_fn(ALPHA_WIDE)
    lif_step_fn = make_lif_step(spike_fn, BETA_LIF, V_TH)

    key = jax.random.PRNGKey(SEED)

    def run_snn(x_seq, W1, W2, W_li, b_li):
        h = run_snn_layer(x_seq, W1, lif_step_fn)
        h = run_snn_layer(h, W2, lif_step_fn)
        return run_li_readout(h, W_li, b_li, BETA_LI)

    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        W1 = jax.random.normal(k1, (INPUT_DIM, HIDDEN1)) * jnp.sqrt(2.0 / INPUT_DIM) * 7.0
        W2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * jnp.sqrt(2.0 / HIDDEN1) * 1.0
        W_li = jax.random.normal(k3, (HIDDEN2, STATE_DIM)) * 0.1
        b_li = jnp.zeros((STATE_DIM,))
        return (W1, W2, W_li, b_li)

    @jax.value_and_grad
    def loss_fn(params, x_seq, labels):
        U_seq = run_snn(x_seq, *params)
        U_window = U_seq[-LOSS_WINDOW:]
        return jnp.mean((U_window - labels[jnp.newaxis, :, :]) ** 2)

    params = init(key)
    opt = Adam(params, lr=LEARNING_RATE)

    # Warmup
    dummy_x = jnp.zeros((TIME_STEPS, 4, INPUT_DIM))
    dummy_l = jnp.zeros((4, 1))
    loss_fn(params, dummy_x, dummy_l)
    key, k0 = jax.random.split(key)
    generate_batch(k0, 4, disable_dimming=True)

    best_r = 0
    best_mse = float('inf')
    best_params = params

    print(f"\n  {'Epoch':>5}  {'Train':>8}  {'Eval':>8}  {'|∇|':>6}  {'r(vx)':>8}")

    for epoch in range(1, N_EPOCHS + 1):
        key, subkey = jax.random.split(key)
        ev, lb, _ = generate_batch(subkey, TRAIN_BATCH, disable_dimming=True)
        x = prepare_events(ev)
        vx = extract_vx_label(lb)

        loss, grads = loss_fn(params, x, vx)
        gn = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        params = clipped_update(params, grads, opt)

        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS:
            key, subkey = jax.random.split(key)
            ev2, lb2, _ = generate_batch(subkey, EVAL_BATCH, disable_dimming=True)
            x2 = prepare_events(ev2); vx2 = extract_vx_label(lb2)
            U = run_snn(x2, *params)
            pred = jnp.mean(U[-LOSS_WINDOW:], axis=0)
            mse = float(jnp.mean((pred - vx2) ** 2))
            r = np.corrcoef(np.array(pred)[:, 0], np.array(vx2)[:, 0])[0, 1]

            marker = " ★" if mse < best_mse else ""
            if mse < best_mse:
                best_mse = mse; best_r = r
                best_params = jax.tree_util.tree_map(jnp.copy, params)

            print(f"  {epoch:>5d}  {float(loss):>8.4f}  {mse:>8.4f}  "
                  f"{float(gn):>5.1f}  {r:>+7.3f}{marker}")

    # Final eval
    key, subkey = jax.random.split(key)
    ev_f, lb_f, _ = generate_batch(subkey, EVAL_BATCH, disable_dimming=True)
    x_f = prepare_events(ev_f); vx_f = extract_vx_label(lb_f)
    U_f = run_snn(x_f, *best_params)
    pred_f = jnp.mean(U_f[-LOSS_WINDOW:], axis=0)
    r_f = np.corrcoef(np.array(pred_f)[:, 0], np.array(vx_f)[:, 0])[0, 1]

    print(f"\n  📊 Final evaluation:")
    print(f"  Best eval MSE: {best_mse:.4f}")
    print(f"  RMSE vx (m/s): {float(jnp.sqrt(best_mse)) * VX_SCALE:.4f}")
    print(f"  Correlation:   r = {r_f:+.4f}")

    print(f"\n  📋 Sample predictions:")
    print(f"  {'':>4}  {'vx true':>8}  {'vx pred':>8}  {'err':>8}")
    for i in range(min(12, EVAL_BATCH)):
        t, p = vx_f[i, 0], pred_f[i, 0]
        print(f"  [{i:>2}]  {t:>+8.3f}  {p:>+8.3f}  {float(p-t):>+8.3f}")

    print(f"\n  {'='*60}")
    if abs(r_f) > 0.5:
        print(f"  ✅ PASS — Full SNN with wider surrogate learns vx!")
        print(f"  Ready to re-enable all 4 labels.")
    elif abs(r_f) > 0.3:
        print(f"  ⚠️  MARGINAL (r={r_f:+.3f}) — promising, needs more tuning")
    else:
        print(f"  ❌ FAIL (r={r_f:+.3f})")
    print(f"  {'='*60}")

    # Save
    W1, W2, W_li, b_li = best_params
    np.savez("/Users/lhooz/.openclaw/workspace/snn_vx_only_params.npz",
             W1=np.array(W1), W2=np.array(W2),
             W_li=np.array(W_li), b_li=np.array(b_li))
    print(f"  💾 Saved params")

    return abs(r_f) > 0.3, r_f


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t_total = time.time()

    # Stage 1: Baseline (Events → LI)
    stage1_pass, r1 = run_stage1()

    if stage1_pass:
        print(f"\n  🎉 Stage 1 PASSED (r={r1:+.3f})")
        print(f"  Temporal linear model works. LIF layers are the bottleneck.")
    else:
        print(f"\n  💀 Stage 1 FAILED (r={r1:+.3f})")
        print(f"  Events alone don't carry enough vx signal.")
        print(f"  Proceeding to Stage 2 anyway with upgraded LIF layers...")

    # Stage 2: Full SNN with wider surrogate gradient
    stage2_pass, r2 = run_stage2()

    print(f"\n\n{'='*60}")
    print(f"  📋 FINAL REPORT")
    print(f"{'='*60}")
    print(f"  Stage 1 (Events→LI):        r = {r1:+.4f}  {'✅' if stage1_pass else '❌'}")
    print(f"  Stage 2 (Full SNN, α={ALPHA_WIDE}): r = {r2:+.4f}  {'✅' if stage2_pass else '❌'}")
    print(f"  Total time: {time.time()-t_total:.0f}s")
    print(f"{'='*60}")
