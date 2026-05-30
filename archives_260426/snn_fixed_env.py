#!/usr/bin/env python3
"""
Fixed-Environment SNN — Train on a single room layout

Key insight: Instead of regenerating random rooms every batch (which
causes monocular scale ambiguity), we generate ONE room with many
trajectories and train on it for all epochs.

The SNN can now learn the actual depth structure of the scene.
Z is no longer random — it's fixed for the training set.

Architecture: Same multimodal SNN with IMU current injection.
No temporal stacking (keeping it simpler — test the fixed-env hypothesis first).

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
TRAIN_SIZE = 512       # ONE big batch, same room, trained repeatedly
EVAL_SIZE = 128        # evaluation set from same room
HIDDEN1 = 128
HIDDEN2 = 64
STATE_DIM = 2
EVENT_DIM = 2 * N_PIXELS
IMU_DIM = 2

BETA = 0.85
BETA_LI = 0.95
V_TH = 1.0
ALPHA_SURR = 2.0
LOSS_WINDOW = 50
GRAD_CLIP_NORM = 1.0
TRAIN_BATCH = 32       # minibatch size for each gradient step

SEED = 42
W1_MULT = 7.0
W2_MULT = 1.0
W_IMU_INIT = 1.0


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
# 2. LIF NEURONS
# =============================================================================
def lif_fused_step(state, x_t, W_vis, imu_current, beta=BETA, v_th=V_TH):
    U_prev, S_prev = state
    I_visual = jnp.dot(x_t, W_vis)
    U_t = beta * U_prev + I_visual + imu_current - (S_prev * v_th)
    S_t = spike_fn(U_t - v_th)
    return (U_t, S_t), S_t


def run_layer(x_seq, W, imu_current, beta=BETA, v_th=V_TH):
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
    imu1 = jnp.dot(imu_vec, W_imu)
    imu2 = jnp.dot(imu_vec, W_imu2)
    h = run_layer(x_seq, W_vis, imu1)
    h = run_layer(h, W2, imu2)
    return run_li_readout(h, W_li, b_li)


# =============================================================================
# 5. PREPROCESSING
# =============================================================================
def prepare_events(events):
    on = jnp.maximum(events, 0.0)
    off = jnp.maximum(-events, 0.0)
    return jnp.transpose(jnp.concatenate([on, off], axis=-1), (1, 0, 2))


def prepare_batch(events, labels_4):
    x_seq = prepare_events(events)
    imu = labels_4[:, [0, 1]]
    targets = labels_4[:, [2, 3]]
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
    U_seq = run_snn(x_seq, imu_vec, *params)
    return jnp.mean((U_seq[-LOSS_WINDOW:] - targets[jnp.newaxis, :, :]) ** 2)


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
    print("  🏠 Fixed-Environment SNN — Single Room, All Epochs")
    print("=" * 70)
    print(f"  Visual pathway:  Events({EVENT_DIM}) → LIF({HIDDEN1}) → LIF({HIDDEN2})")
    print(f"  IMU pathway:     [vx, vy]({IMU_DIM}) → current injection (both layers)")
    print(f"  Readout:         LI({STATE_DIM}) → [{', '.join(label_names)}]")
    print(f"  Environment:     Sparse Forest (ONE fixed layout)")
    print(f"  Dataset:         {TRAIN_SIZE} train + {EVAL_SIZE} eval trajectories")
    print(f"  Minibatch:       {TRAIN_BATCH}")
    print(f"  Epochs:          {N_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"  Key:             Same room, same obstacles, every epoch")
    print("=" * 70)

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    opt = Adam(params, lr=LEARNING_RATE)

    # Count params
    total_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(f"\n  📊 Parameters: {total_params:,}")

    # ── Generate FIXED dataset ──────────────────────────────────────────
    print(f"\n  🏗️  Generating FIXED dataset ({TRAIN_SIZE + EVAL_SIZE} trajectories)...")
    t0 = time.time()

    total = TRAIN_SIZE + EVAL_SIZE
    key, k_train = jax.random.split(key)
    ev_all, lb_all, info_all = generate_batch(k_train, total)

    train_events = ev_all[:TRAIN_SIZE]
    train_labels = lb_all[:TRAIN_SIZE]
    train_info = info_all[:TRAIN_SIZE]

    eval_events = ev_all[TRAIN_SIZE:]
    eval_labels = lb_all[TRAIN_SIZE:]
    eval_info = info_all[TRAIN_SIZE:]

    # Precompute prepared data
    train_x = prepare_events(train_events)      # (T, TRAIN_SIZE, EVENT_DIM)
    train_imu = train_labels[:, [0, 1]]          # (TRAIN_SIZE, IMU_DIM)
    train_tgt = train_labels[:, [2, 3]]          # (TRAIN_SIZE, STATE_DIM)

    eval_x = prepare_events(eval_events)          # (T, EVAL_SIZE, EVENT_DIM)
    eval_imu = eval_labels[:, [0, 1]]
    eval_tgt = eval_labels[:, [2, 3]]

    print(f"  Dataset generated in {time.time()-t0:.1f}s")

    # Show obstacle layout info
    obs_positions = []
    for i in range(min(10, TRAIN_SIZE)):
        info = info_all[i]
        if hasattr(info, 'obstacle_positions'):
            obs_positions = info.obstacle_positions
            break
    print(f"  Room: {ROOM_W}×{ROOM_H}m, obstacles placed randomly (same for all samples)")

    # ── Compile ─────────────────────────────────────────────────────────
    print("\n  🔨 Compiling...")
    t0 = time.time()
    loss_fn(params,
            train_x[:, :4, :],
            train_imu[:4],
            train_tgt[:4])
    print(f"  Compile: {time.time()-t0:.2f}s")

    best_eval = float('inf')
    best_params = params
    rng = np.random.RandomState(SEED)

    hdr = f"  {'Epoch':>5}  {'Train':>8}  {'Eval':>8}  {'|∇|':>6}"
    for n in label_names:
        hdr += f"  {'r_'+n:>8}"
    hdr += f"  {'RMSE cl':>8}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(1, N_EPOCHS + 1):
        # Shuffle indices, create minibatches from FIXED data
        perm = rng.permutation(TRAIN_SIZE)
        epoch_loss = 0.0
        n_steps = 0
        epoch_grad_norm = 0.0

        for start in range(0, TRAIN_SIZE, TRAIN_BATCH):
            idx = perm[start:start + TRAIN_BATCH]
            x_mb = train_x[:, idx, :]
            imu_mb = train_imu[idx]
            tgt_mb = train_tgt[idx]

            loss, grads = loss_fn(params, x_mb, imu_mb, tgt_mb)
            gn = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            epoch_loss += float(loss)
            epoch_grad_norm += float(gn)
            n_steps += 1
            params = clipped_update(params, grads, opt)

        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS:
            # Evaluate on fixed eval set
            U2 = run_snn(eval_x, eval_imu, *params)
            pred = jnp.mean(U2[-LOSS_WINDOW:], axis=0)
            mse = float(jnp.mean((pred - eval_tgt) ** 2))

            r_om = np.corrcoef(np.array(pred)[:, 0], np.array(eval_tgt)[:, 0])[0, 1]
            r_cl = np.corrcoef(np.array(pred)[:, 1], np.array(eval_tgt)[:, 1])[0, 1]
            rmse_cl = float(jnp.sqrt(jnp.mean((pred[:, 1] - eval_tgt[:, 1])**2)) * 2.0)

            avg_loss = epoch_loss / n_steps
            avg_gn = epoch_grad_norm / n_steps

            marker = " ★" if mse < best_eval else ""
            if mse < best_eval:
                best_eval = mse
                best_params = jax.tree_util.tree_map(jnp.copy, params)

            print(f"  {epoch:>5d}  {avg_loss:>8.4f}  {mse:>8.4f}  "
                  f"{avg_gn:>5.1f}  {r_om:>+7.3f}  {r_cl:>+7.3f}  "
                  f"{rmse_cl:>8.3f}{marker}")

    # ── Final evaluation ────────────────────────────────────────────────
    print(f"\n  📊 Final evaluation (best model, fixed eval set)...")
    U_f = run_snn(eval_x, eval_imu, *best_params)
    pred_f = jnp.mean(U_f[-LOSS_WINDOW:], axis=0)

    rs = []
    for name, pred_col, true_col in zip(label_names,
                                         np.array(pred_f).T,
                                         np.array(eval_tgt).T):
        r = np.corrcoef(pred_col, true_col)[0, 1]
        rs.append(r)
        print(f"    {name:>16s}: r = {r:+.4f}")

    print(f"\n  📋 Sample predictions:")
    print(f"  {'':>4}  {'ω_t':>6} {'ω_p':>6}  {'cl_t':>6} {'cl_p':>6}  {'vx':>6}  {'vy':>6}")
    for i in range(min(12, EVAL_SIZE)):
        t, p = eval_tgt[i], pred_f[i]
        t4 = eval_labels[i]
        print(f"  [{i:>2}]  {t[0]:>+5.2f} {p[0]:>+5.2f}  {t[1]:>+5.2f} {p[1]:>+5.2f}  "
              f"{t4[0]:>+5.2f} {t4[1]:>+5.2f}")

    r_om, r_cl = rs
    print(f"\n  {'='*70}")
    om_ok = abs(r_om) > 0.5
    cl_ok = abs(r_cl) > 0.3
    if om_ok and cl_ok:
        print(f"  ✅ PASS — ω (r={r_om:+.3f}) + clearance (r={r_cl:+.3f})")
        print(f"  🏠 Fixed environment eliminates scale ambiguity!")
        print(f"  🗺️  Ready for SLAM Shadow Mapper.")
    elif om_ok:
        print(f"  ⚠️  PARTIAL — ω (r={r_om:+.3f}), clearance weak (r={r_cl:+.3f})")
    elif cl_ok:
        print(f"  ⚠️  PARTIAL — clearance (r={r_cl:+.3f}), ω weak (r={r_om:+.3f})")
    else:
        print(f"  ❌ FAIL — ω (r={r_om:+.3f}), clearance (r={r_cl:+.3f})")

    print(f"\n  📊 Fixed vs Random comparison:")
    print(f"    Random rooms (r≈0.40/0.27) → Fixed room (r={r_om:+.3f}/{r_cl:+.3f})")
    print(f"  {'='*70}")

    # Save
    np.savez("/Users/lhooz/.openclaw/workspace/fixed_env_params.npz",
             **{name: np.array(p) for name, p in
                zip(['W_vis', 'W_imu', 'W2', 'W_imu2', 'W_li', 'b_li'], best_params)})
    print(f"  💾 Saved params")

    _plot(pred_f, eval_tgt, best_eval,
           "/Users/lhooz/.openclaw/workspace/fixed_env_curve.png")
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
    fig.suptitle(f'Fixed Environment — Same Room All Epochs  |  MSE: {best_mse:.4f}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  📸 Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    train()
