#!/usr/bin/env python3
"""
Multimodal SNN — Event-VIO with Neuromodulatory IMU Fusion

Architecture:
  Visual pathway:  Events(128) → W_vis → LIF(128) → W2 → LIF(64)
  IMU pathway:     [vx, vy] → W_imu → Linear(128) — non-spiking current injection
  Fusion:          U_t = β*U_prev + W_vis @ S + W_imu @ imu  (current injection)
  Readout:         LI(2) → [ω, clearance]

The IMU signal acts as a smooth modulatory bias on the LIF membrane,
exactly like haltere feedback in fly visual neurons.
No input concatenation — sparse spikes and dense IMU stay separate.

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
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-3
N_EPOCHS = 500
TRAIN_BATCH = 32
EVAL_BATCH = 128
HIDDEN1 = 128
HIDDEN2 = 64
STATE_DIM = 2           # [omega, clearance]
EVENT_DIM = 2 * N_PIXELS   # 128 (ON + OFF polarized)
IMU_DIM = 2              # [vx, vy]

BETA = 0.85
BETA_LI = 0.95
V_TH = 1.0
ALPHA_SURR = 2.0
LOSS_WINDOW = 50
GRAD_CLIP_NORM = 1.0

SEED = 42
W1_MULT = 7.0
W2_MULT = 1.0
W_IMU_INIT = 1.0       # stronger IMU projection (must compete with visual ×7)


# =============================================================================
# 1. SURROGATE GRADIENT
# =============================================================================
@jax.custom_vjp
def spike_fn(x):
    return jnp.heaviside(x, 0.0)

def spike_fn_fwd(x):
    return spike_fn(x), x

def spike_fn_bwd(res, g):
    grad = ALPHA_SURR / (1.0 + jnp.abs(ALPHA_SURR * res)) ** 2
    return (g * grad,)

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)


# =============================================================================
# 2. LIF with Neuromodulatory Current Injection
# =============================================================================
def lif_fused_step(state, x_t, W_vis, imu_current, beta=BETA, v_th=V_TH):
    """LIF neuron with visual spikes + IMU current injection.
    
    U_t = β * U_prev + W_vis @ x_t + imu_current - S_prev * v_th
                    ^^^^^^^^   ^^^^^^^^^^^^
                    visual      neuromodulatory
                    spikes      IMU bias
    
    The IMU current is constant across timesteps (constant-velocity flight).
    """
    U_prev, S_prev = state
    I_visual = jnp.dot(x_t, W_vis)
    U_t = beta * U_prev + I_visual + imu_current - (S_prev * v_th)
    S_t = spike_fn(U_t - v_th)
    return (U_t, S_t), S_t


def run_visual_layer(x_seq, W_vis, W_imu, imu_vec, beta=BETA, v_th=V_TH):
    """Run fused LIF layer: visual spikes + IMU current injection.
    
    Args:
        x_seq:     (T, B, EVENT_DIM) visual spike sequence
        W_vis:     (EVENT_DIM, HIDDEN1) visual weight matrix
        W_imu:     (IMU_DIM, HIDDEN1) IMU projection matrix
        imu_vec:   (B, IMU_DIM) IMU readings (constant over T)
    Returns:
        spikes:    (T, B, HIDDEN1)
    """
    # Project IMU → hidden dim (computed once, broadcast across T)
    imu_current = jnp.dot(imu_vec, W_imu)  # (B, HIDDEN1)
    
    batch, hidden = x_seq.shape[1], W_vis.shape[1]
    _, spikes = jax.lax.scan(
        lambda s, x: lif_fused_step(s, x, W_vis, imu_current, beta, v_th),
        (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden))),
        x_seq)
    return spikes


def run_snn_layer_pure(x_seq, W, beta=BETA, v_th=V_TH):
    """Standard LIF layer (no IMU fusion, for 2nd layer)."""
    batch, hidden = x_seq.shape[1], W.shape[1]
    _, out = jax.lax.scan(
        lambda s, x: lif_fused_step(s, x, W, jnp.zeros((batch, hidden)), beta, v_th),
        (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden))),
        x_seq)
    return out


def run_imu_fused_layer(x_seq, W, W_imu, imu_vec, beta=BETA, v_th=V_TH):
    """LIF layer with IMU current injection (for 2nd layer fusion)."""
    imu_current = jnp.dot(imu_vec, W_imu)
    batch, hidden = x_seq.shape[1], W.shape[1]
    _, out = jax.lax.scan(
        lambda s, x: lif_fused_step(s, x, W, imu_current, beta, v_th),
        (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden))),
        x_seq)
    return out


# =============================================================================
# 3. LI READOUT
# =============================================================================
def run_li_readout(spikes, W_li, b_li, beta_li=BETA_LI):
    T = spikes.shape[0]
    norm = 1.0 / T
    _, U = jax.lax.scan(
        lambda U, s: (beta_li * U + jnp.dot(s, W_li) * norm + b_li,
                       beta_li * U + jnp.dot(s, W_li) * norm + b_li),
        jnp.zeros((spikes.shape[1], W_li.shape[1])),
        spikes)
    return U


# =============================================================================
# 4. FULL NETWORK
# =============================================================================
def run_snn(x_seq, imu_vec, W_vis, W_imu, W2, W_imu2, W_li, b_li):
    """Multimodal SNN: events + IMU → [omega, clearance].
    IMU injected at both LIF layers.
    """
    h = run_visual_layer(x_seq, W_vis, W_imu, imu_vec)
    h = run_imu_fused_layer(h, W2, W_imu2, imu_vec)
    return run_li_readout(h, W_li, b_li)


# =============================================================================
# 5. PREPROCESSING
# =============================================================================
def prepare_events(events):
    on = jnp.maximum(events, 0.0)
    off = jnp.maximum(-events, 0.0)
    return jnp.transpose(jnp.concatenate([on, off], axis=-1), (1, 0, 2))


def prepare_batch(events, labels_4):
    """Prepare events + extract IMU and target labels."""
    x_seq = prepare_events(events)
    imu = labels_4[:, [0, 1]]    # [vx, vy] normalized
    targets = labels_4[:, [2, 3]] # [omega, clearance]
    return x_seq, imu, targets


# =============================================================================
# 6. ADAM
# =============================================================================
class Adam:
    def __init__(self, params, lr=1e-3):
        self.lr, self.b1, self.b2, self.eps = lr, 0.9, 0.999, 1e-8
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
def loss_fn(params, x_seq, imu_vec, targets):
    W_vis, W_imu, W2, W_imu2, W_li, b_li = params
    U_seq = run_snn(x_seq, imu_vec, W_vis, W_imu, W2, W_imu2, W_li, b_li)
    U_window = U_seq[-LOSS_WINDOW:]
    return jnp.mean((U_window - targets[jnp.newaxis, :, :]) ** 2)


def clipped_update(params, grads, opt):
    gn = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    clip = jnp.minimum(1.0, GRAD_CLIP_NORM / (gn + 1e-8))
    return opt.step(params, jax.tree_util.tree_map(lambda g: g * clip, grads))


def init_params(key):
    keys = jax.random.split(key, 6)
    W_vis = jax.random.normal(keys[0], (EVENT_DIM, HIDDEN1)) * jnp.sqrt(2.0/EVENT_DIM) * W1_MULT
    W_imu = jax.random.normal(keys[1], (IMU_DIM, HIDDEN1)) * W_IMU_INIT
    W2 = jax.random.normal(keys[2], (HIDDEN1, HIDDEN2)) * jnp.sqrt(2.0/HIDDEN1) * W2_MULT
    W_imu2 = jax.random.normal(keys[3], (IMU_DIM, HIDDEN2)) * W_IMU_INIT
    W_li = jax.random.normal(keys[4], (HIDDEN2, STATE_DIM)) * 0.1
    b_li = jnp.zeros((STATE_DIM,))
    return (W_vis, W_imu, W2, W_imu2, W_li, b_li)


def train():
    label_names = ['omega', 'clearance']

    print("=" * 70)
    print("  🦟 Multimodal SNN — Neuromodulatory IMU Fusion")
    print("=" * 70)
    print(f"  Visual pathway: Events({EVENT_DIM}) → W_vis → LIF({HIDDEN1}) → W2 → LIF({HIDDEN2})")
    print(f"  IMU pathway:    [vx, vy]({IMU_DIM}) → W_imu → L1({HIDDEN1}) + W_imu2 → L2({HIDDEN2})")
    print(f"  Fusion:         U += W_vis @ Spikes + W_imu @ imu  (both layers)")
    print(f"  W_imu init:     {W_IMU_INIT} (×3 stronger than before)")
    print(f"  Readout:        LI({STATE_DIM}) → [{', '.join(label_names)}]")
    print(f"  Environment:    Sparse Forest (collision-free)")
    print(f"  Epochs:         {N_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"  Dimming:        OFF")
    print("=" * 70)

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    opt = Adam(params, lr=LEARNING_RATE)

    print("\n  🔨 Compiling...")
    t0 = time.time()
    dummy_x = jnp.zeros((TIME_STEPS, 4, EVENT_DIM))
    dummy_imu = jnp.zeros((4, IMU_DIM))
    dummy_t = jnp.zeros((4, STATE_DIM))
    loss_fn(params, dummy_x, dummy_imu, dummy_t)
    print(f"  Compile: {time.time()-t0:.2f}s")

    print("  Warming up env...")
    t0 = time.time()
    key, k0 = jax.random.split(key)
    generate_batch(k0, 4)
    print(f"  Env: {time.time()-t0:.2f}s")

    best_eval = float('inf')
    best_params = params

    hdr = f"  {'Epoch':>5}  {'Train':>8}  {'Eval':>8}  {'|∇|':>6}"
    for n in label_names:
        hdr += f"  {'r_'+n:>8}"
    hdr += f"  {'RMSE cl':>8}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(1, N_EPOCHS + 1):
        key, subkey = jax.random.split(key)
        events, labels_4, _ = generate_batch(subkey, TRAIN_BATCH)
        x_seq, imu, targets = prepare_batch(events, labels_4)

        loss, grads = loss_fn(params, x_seq, imu, targets)
        gn = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        params = clipped_update(params, grads, opt)

        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS:
            key, subkey = jax.random.split(key)
            ev2, lb2_4, _ = generate_batch(subkey, EVAL_BATCH)
            x2, imu2, tgt2 = prepare_batch(ev2, lb2_4)
            W_vis, W_imu, W2, W_imu2, W_li, b_li = params
            U2 = run_snn(x2, imu2, W_vis, W_imu, W2, W_imu2, W_li, b_li)
            pred = jnp.mean(U2[-50:], axis=0)
            mse = float(jnp.mean((pred - tgt2) ** 2))

            r_om = np.corrcoef(np.array(pred)[:, 0], np.array(tgt2)[:, 0])[0, 1]
            r_cl = np.corrcoef(np.array(pred)[:, 1], np.array(tgt2)[:, 1])[0, 1]
            rmse_cl = float(jnp.sqrt(jnp.mean((pred[:, 1] - tgt2[:, 1])**2)) * 2.0)

            marker = " ★" if mse < best_eval else ""
            if mse < best_eval:
                best_eval = mse
                best_params = jax.tree_util.tree_map(jnp.copy, params)

            print(f"  {epoch:>5d}  {float(loss):>8.4f}  {mse:>8.4f}  "
                  f"{float(gn):>5.1f}  {r_om:>+7.3f}  {r_cl:>+7.3f}  "
                  f"{rmse_cl:>8.3f}{marker}")

    # Final evaluation
    print(f"\n  📊 Final evaluation (best model)...")
    key, subkey = jax.random.split(key)
    ev_f, lb_f_4, info_f = generate_batch(subkey, EVAL_BATCH)
    x_f, imu_f, tgt_f = prepare_batch(ev_f, lb_f_4)
    W_vis, W_imu, W2, W_imu2, W_li, b_li = best_params
    U_f = run_snn(x_f, imu_f, W_vis, W_imu, W2, W_imu2, W_li, b_li)
    pred_f = jnp.mean(U_f[-50:], axis=0)

    rmse_om = float(jnp.sqrt(jnp.mean((pred_f[:, 0] - tgt_f[:, 0])**2)) * abs(OMEGA_RANGE[1]))
    rmse_cl = float(jnp.sqrt(jnp.mean((pred_f[:, 1] - tgt_f[:, 1])**2)) * 2.0)

    rs = []
    for name, pred_col, true_col in zip(label_names,
                                         np.array(pred_f).T,
                                         np.array(tgt_f).T):
        r = np.corrcoef(pred_col, true_col)[0, 1]
        rs.append(r)
        print(f"    {name:>16s}: r = {r:+.4f}")

    print(f"\n  📋 Sample predictions:")
    print(f"  {'':>4}  {'ω_t':>6} {'ω_p':>6}  {'cl_t':>6} {'cl_p':>6}  {'vx':>6}  {'vy':>6}")
    for i in range(min(12, EVAL_BATCH)):
        t, p = tgt_f[i], pred_f[i]
        t4 = lb_f_4[i]
        print(f"  [{i:>2}]  {t[0]:>+5.2f} {p[0]:>+5.2f}  {t[1]:>+5.2f} {p[1]:>+5.2f}  "
              f"{t4[0]:>+5.2f} {t4[1]:>+5.2f}")

    r_om, r_cl = rs
    print(f"\n  {'='*70}")
    om_ok = abs(r_om) > 0.5
    cl_ok = abs(r_cl) > 0.3
    if om_ok and cl_ok:
        print(f"  ✅ PASS — ω (r={r_om:+.3f}) + clearance (r={r_cl:+.3f})")
        print(f"  🧠 Multimodal fusion works! IMU unlocks visual decomposition.")
        print(f"  🗺️  Ready for SLAM Shadow Mapper.")
    elif om_ok or cl_ok:
        ok_name = 'ω' if om_ok else 'clearance'
        ok_r = r_om if om_ok else r_cl
        print(f"  ⚠️  PARTIAL — {ok_name} (r={ok_r:+.3f}) learned")
    else:
        print(f"  ❌ FAIL — ω (r={r_om:+.3f}), clearance (r={r_cl:+.3f})")
    print(f"  {'='*70}")

    # Save
    np.savez("/Users/lhooz/.openclaw/workspace/multimodal_params.npz",
             W_vis=np.array(W_vis), W_imu=np.array(W_imu),
             W2=np.array(W2), W_imu2=np.array(W_imu2),
             W_li=np.array(W_li), b_li=np.array(b_li))
    print(f"  💾 Saved params")

    # Plot
    _plot(pred_f, tgt_f, best_eval,
           "/Users/lhooz/.openclaw/workspace/multimodal_curve.png")
    print(f"  ✅ Done!")
    return best_params, rs


def _plot(preds, labels, best_mse, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = ['omega', 'clearance']
    colors = ['tab:green', 'tab:red']
    for idx, (ax, name, color) in enumerate(zip(axes, names, colors)):
        p = np.array(preds)[:, idx]
        l = np.array(labels)[:, idx]
        r = np.corrcoef(p, l)[0, 1]
        ax.scatter(l, p, alpha=0.5, s=15, c=color)
        lim = max(abs(l).max(), abs(p).max()) * 1.2
        lim = max(lim, 0.3)
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('True'); ax.set_ylabel('Predicted')
        ax.set_title(f'{name} (r={r:+.3f})', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    fig.suptitle(f'Multimodal SNN — Events + IMU Fusion  |  MSE: {best_mse:.4f}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  📸 Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    train()
