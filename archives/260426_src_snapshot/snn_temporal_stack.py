#!/usr/bin/env python3
"""
Multimodal SNN with Temporal Stacking — Event-VIO

Architecture:
  Visual pathway:  Events(t-N...t) stacked → [128×N] → W_vis → LIF(128) → LIF(64)
  IMU pathway:     [vx, vy] → W_imu → current injection at both LIF layers
  Fusion:         U += W_vis @ stacked_events + W_imu @ imu
  Readout:        LI(2) → [ω, clearance]

The rolling buffer (N=2, = 40ms) gives the first hidden layer
true Δx/Δt gradients — a digital Hassenstein-Reichardt detector.
The SNN can now compute spatial-temporal cross-correlations
to separate depth-dependent from depth-independent optic flow.

Author: Ada 🦊
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

from sparse_forest import (
    generate_batch, generate_fixed_room_dataset,
    N_PIXELS, TIME_STEPS, DT,
    ROOM_W, ROOM_H, THRESHOLD,
    VX_RANGE, VY_RANGE, OMEGA_RANGE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-3
N_EPOCHS = 100
TRAIN_BATCH = 32
EVAL_BATCH = 128
FIXED_TRAIN_SAMPLES = 640   # pre-generate once, reuse all 100 epochs
HIDDEN1 = 128
HIDDEN2 = 64
STATE_DIM = 2           # [omega, clearance]
EVENT_DIM = 2 * N_PIXELS   # 128
IMU_DIM = 2              # [vx, vy]
TEMPORAL_N = 2           # rolling buffer size (2 × 20ms = 40ms)
STACKED_EVENT_DIM = EVENT_DIM * TEMPORAL_N  # 256

BETA = 0.85
BETA_LI = 0.95
V_TH = 1.0
ALPHA_SURR = 2.0
LOSS_WINDOW = 50
GRAD_CLIP_NORM = 1.0

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
# 2. ROLLING BUFFER (delay lines)
# =============================================================================
def init_buffer(batch_size):
    """Initialize rolling buffer with zeros. Shape: (N, B, EVENT_DIM)."""
    return jnp.zeros((TEMPORAL_N, batch_size, EVENT_DIM), dtype=jnp.float32)


def shift_buffer(buffer, new_events):
    """Shift buffer left and append new events. Returns updated buffer.
    
    buffer: (N, B, EVENT_DIM) — the rolling window of past events
    new_events: (B, EVENT_DIM) — current timestep events
    
    Returns: (N, B, EVENT_DIM)
    """
    # Shift: drop oldest, keep middle, append newest
    shifted = jnp.concatenate([buffer[1:], new_events[None, :, :]], axis=0)
    return shifted


def flatten_buffer(buffer):
    """Flatten (N, B, EVENT_DIM) → (B, N * EVENT_DIM) for the weight matrix.
    
    This creates the spatiotemporal input where the network can learn
    delay-line cross-correlations — the digital equivalent of the
    Hassenstein-Reichardt elementary motion detector.
    """
    # (N, B, D) → (B, N, D) → (B, N*D)
    return jnp.reshape(jnp.transpose(buffer, (1, 0, 2)),
                       (buffer.shape[1], buffer.shape[0] * buffer.shape[2]))


# =============================================================================
# 3. LIF with current injection
# =============================================================================
def lif_step(state, x_t, W, imu_current, beta=BETA, v_th=V_TH):
    U_prev, S_prev = state
    I = jnp.dot(x_t, W) + imu_current
    U_t = beta * U_prev + I - (S_prev * v_th)
    S_t = spike_fn(U_t - v_th)
    return (U_t, S_t), S_t


# =============================================================================
# 4. FIRST LAYER — Visual with temporal stacking + IMU fusion
# =============================================================================
def run_visual_layer_with_buffer(event_seq, W_vis, W_imu, imu_vec,
                                   beta=BETA, v_th=V_TH):
    """LIF layer with rolling buffer and IMU current injection.
    
    Args:
        event_seq: (T, B, EVENT_DIM) — raw events, one timestep per scan step
        W_vis:     (STACKED_EVENT_DIM, HIDDEN1) — visual weight matrix
        W_imu:     (IMU_DIM, HIDDEN1) — IMU projection
        imu_vec:   (B, IMU_DIM) — IMU readings (constant)
    
    The scan state is:
        - buffer: (N, B, EVENT_DIM) — rolling event window
        - (U, S): (B, HIDDEN1) each — LIF membrane + spikes
    """
    batch, _ = event_seq.shape[1], event_seq.shape[2]
    hidden = W_vis.shape[1]
    
    imu_current = jnp.dot(imu_vec, W_imu)  # (B, HIDDEN1)
    
    def step(carry, x_t):
        buffer, (U_prev, S_prev) = carry
        # Shift buffer and append new events
        new_buffer = shift_buffer(buffer, x_t)
        # Flatten buffer → stacked spatiotemporal input
        stacked = flatten_buffer(new_buffer)  # (B, N*D)
        # LIF update with visual + IMU input
        I_visual = jnp.dot(stacked, W_vis)
        U_t = beta * U_prev + I_visual + imu_current - (S_prev * v_th)
        S_t = spike_fn(U_t - v_th)
        return (new_buffer, (U_t, S_t)), S_t
    
    buffer_init = init_buffer(batch)
    _, spikes = jax.lax.scan(
        step,
        (buffer_init, (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden)))),
        event_seq)
    return spikes


# =============================================================================
# 5. SECOND LAYER — standard LIF with IMU fusion
# =============================================================================
def run_imu_fused_layer(spikes, W, W_imu, imu_vec, beta=BETA, v_th=V_TH):
    imu_current = jnp.dot(imu_vec, W_imu)
    batch, hidden = spikes.shape[1], W.shape[1]
    _, out = jax.lax.scan(
        lambda s, x: lif_step(s, x, W, imu_current, beta, v_th),
        (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden))),
        spikes)
    return out


# =============================================================================
# 6. LI READOUT
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
# 7. FULL NETWORK
# =============================================================================
def run_snn(event_seq, imu_vec, W_vis, W_imu, W2, W_imu2, W_li, b_li):
    """Multimodal SNN with temporal stacking."""
    # Layer 1: events with rolling buffer + IMU injection
    h = run_visual_layer_with_buffer(event_seq, W_vis, W_imu, imu_vec)
    # Layer 2: spikes + IMU injection
    h = run_imu_fused_layer(h, W2, W_imu2, imu_vec)
    # Readout
    return run_li_readout(h, W_li, b_li)


# =============================================================================
# 8. PREPROCESSING
# =============================================================================
def prepare_events(events):
    """Polarize events: (B, T, N) → (T, B, 2*N) with ON/OFF channels."""
    on = jnp.maximum(events, 0.0)
    off = jnp.maximum(-events, 0.0)
    return jnp.transpose(jnp.concatenate([on, off], axis=-1), (1, 0, 2))


def prepare_batch(events, labels_4):
    x_seq = prepare_events(events)
    imu = labels_4[:, [0, 1]]
    targets = labels_4[:, [2, 3]]
    return x_seq, imu, targets


# =============================================================================
# 9. ADAM
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
# 10. TRAINING
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
    W_vis = (jax.random.normal(keys[0], (STACKED_EVENT_DIM, HIDDEN1))
             * jnp.sqrt(2.0 / STACKED_EVENT_DIM) * W1_MULT)
    W_imu = jax.random.normal(keys[1], (IMU_DIM, HIDDEN1)) * W_IMU_INIT
    W2 = (jax.random.normal(keys[2], (HIDDEN1, HIDDEN2))
           * jnp.sqrt(2.0 / HIDDEN1) * W2_MULT)
    W_imu2 = jax.random.normal(keys[3], (IMU_DIM, HIDDEN2)) * W_IMU_INIT
    W_li = jax.random.normal(keys[4], (HIDDEN2, STATE_DIM)) * 0.1
    b_li = jnp.zeros((STATE_DIM,))
    return (W_vis, W_imu, W2, W_imu2, W_li, b_li)


def train():
    label_names = ['omega', 'clearance']

    print("=" * 70)
    print("  ⚡ Multimodal SNN — Temporal Stacking (Hassenstein-Reichardt)")
    print("=" * 70)
    print(f"  Rolling buffer:  N={TEMPORAL_N} ({TEMPORAL_N*DT*1000:.0f}ms delay lines)")
    print(f"  Stacked input:   {EVENT_DIM}×{TEMPORAL_N} = {STACKED_EVENT_DIM}")
    print(f"  Visual pathway:  [{STACKED_EVENT_DIM}] → W_vis → LIF({HIDDEN1}) → LIF({HIDDEN2})")
    print(f"  IMU pathway:     [{IMU_DIM}] → W_imu → current injection (both layers)")
    print(f"  Readout:         LI({STATE_DIM}) → [{', '.join(label_names)}]")
    print(f"  Environment:     Sparse Forest (ONE fixed room, different trajectories)")
    print(f"  Epochs:          {N_EPOCHS} (reduced from 500 for faster iteration), LR: {LEARNING_RATE}")
    print(f"  Buffer fills:    epochs ~{TIME_STEPS//TEMPORAL_N} before full context")
    print(f"  Dataset:         FIXED room — {FIXED_TRAIN_SAMPLES} train + {EVAL_BATCH} eval trajectories")
    print("=" * 70)

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    opt = Adam(params, lr=LEARNING_RATE)

    # Count params
    total_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(f"\n  📊 Parameters: {total_params:,}")
    for name, p in zip(['W_vis', 'W_imu', 'W2', 'W_imu2', 'W_li', 'b_li'], params):
        print(f"    {name:>8s}: {p.shape}")

    print("\n  🔨 Compiling...")
    t0 = time.time()
    loss_fn(params,
            jnp.zeros((TIME_STEPS, 4, EVENT_DIM)),
            jnp.zeros((4, IMU_DIM)),
            jnp.zeros((4, STATE_DIM)))
    print(f"  Compile: {time.time()-t0:.2f}s")

    # Pre-generate ONE fixed room — train and eval both use the SAME obstacles
    print(f"\n  🏠 Generating FIXED room with {FIXED_TRAIN_SAMPLES} train + {EVAL_BATCH} eval trajectories...")
    t0 = time.time()
    key, data_key = jax.random.split(key)
    data_key, train_key, eval_key = jax.random.split(data_key, 3)

    # Generate all trajectories in the same room
    fixed_events, fixed_labels, fixed_tof, fixed_positions, obstacles, segments, *extra = \
        generate_fixed_room_dataset(train_key, FIXED_TRAIN_SAMPLES + EVAL_BATCH)

    train_events = fixed_events[:FIXED_TRAIN_SAMPLES]
    train_labels = fixed_labels[:FIXED_TRAIN_SAMPLES]
    eval_events = fixed_events[FIXED_TRAIN_SAMPLES:]
    eval_labels = fixed_labels[FIXED_TRAIN_SAMPLES:]

    print(f"  Room obstacles: {obstacles.shape[0]}")
    print(f"  Train samples:  {train_events.shape}")
    print(f"  Eval samples:   {eval_events.shape}")
    print(f"  Total time:     {time.time()-t0:.1f}s")

    # Pre-prepare all batches
    n_train_batches = FIXED_TRAIN_SAMPLES // TRAIN_BATCH
    train_x = prepare_events(train_events)    # (T, TRAIN_N, EVENT_DIM)
    train_imu = train_labels[:, [0, 1]]       # (TRAIN_N, 2)
    train_targets = train_labels[:, [2, 3]]   # (TRAIN_N, 2)

    eval_x = prepare_events(eval_events)
    eval_imu = eval_labels[:, [0, 1]]
    eval_targets = eval_labels[:, [2, 3]]

    best_eval = float('inf')
    best_params = params

    hdr = f"  {'Epoch':>5}  {'Train':>8}  {'Eval':>8}  {'|∇|':>6}"
    for n in label_names:
        hdr += f"  {'r_'+n:>8}"
    hdr += f"  {'RMSE cl':>8}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(1, N_EPOCHS + 1):
        # Train on FIXED room — shuffle batch order per epoch
        perm = jax.random.permutation(key, n_train_batches)
        key, _ = jax.random.split(key)
        for bi in perm:
            start = int(bi) * TRAIN_BATCH
            end = start + TRAIN_BATCH
            x_seq = train_x[:, start:end, :]
            imu = train_imu[start:end]
            targets = train_targets[start:end]

            loss, grads = loss_fn(params, x_seq, imu, targets)
            gn = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            params = clipped_update(params, grads, opt)

        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS:
            U2 = run_snn(eval_x, eval_imu, *params)
            pred = jnp.mean(U2[-LOSS_WINDOW:], axis=0)
            mse = float(jnp.mean((pred - eval_targets) ** 2))

            r_om = np.corrcoef(np.array(pred)[:, 0], np.array(eval_targets)[:, 0])[0, 1]
            r_cl = np.corrcoef(np.array(pred)[:, 1], np.array(eval_targets)[:, 1])[0, 1]
            rmse_cl = float(jnp.sqrt(jnp.mean((pred[:, 1] - eval_targets[:, 1])**2)) * 2.0)

            marker = " ★" if mse < best_eval else ""
            if mse < best_eval:
                best_eval = mse
                best_params = jax.tree_util.tree_map(jnp.copy, params)

            print(f"  {epoch:>5d}  {float(loss):>8.4f}  {mse:>8.4f}  "
                  f"{float(gn):>5.1f}  {r_om:>+7.3f}  {r_cl:>+7.3f}  "
                  f"{rmse_cl:>8.3f}{marker}")

    # Final evaluation — same room, different trajectories
    print(f"\n  📊 Final evaluation (same room, {eval_events.shape[0]} unseen trajectories)...")
    U_f = run_snn(eval_x, eval_imu, *best_params)
    pred_f = jnp.mean(U_f[-LOSS_WINDOW:], axis=0)

    rs = []
    for name, pred_col, true_col in zip(label_names,
                                         np.array(pred_f).T,
                                         np.array(eval_targets).T):
        r = np.corrcoef(pred_col, true_col)[0, 1]
        rs.append(r)
        print(f"    {name:>16s}: r = {r:+.4f}")

    print(f"\n  📋 Sample predictions:")
    print(f"  {'':>4}  {'ω_t':>6} {'ω_p':>6}  {'cl_t':>6} {'cl_p':>6}  {'vx':>6}  {'vy':>6}")
    for i in range(min(12, eval_events.shape[0])):
        t, p = eval_targets[i], pred_f[i]
        t4 = eval_labels[i]
        print(f"  [{i:>2}]  {t[0]:>+5.2f} {p[0]:>+5.2f}  {t[1]:>+5.2f} {p[1]:>+5.2f}  "
              f"{t4[0]:>+5.2f} {t4[1]:>+5.2f}")

    r_om, r_cl = rs
    print(f"\n  {'='*70}")
    om_ok = abs(r_om) > 0.5
    cl_ok = abs(r_cl) > 0.3
    if om_ok and cl_ok:
        print(f"  ✅ PASS — ω (r={r_om:+.3f}) + clearance (r={r_cl:+.3f})")
        print(f"  🧠 Temporal stacking + IMU fusion = flow decomposition!")
    elif om_ok:
        print(f"  ⚠️  PARTIAL — ω (r={r_om:+.3f}), clearance weak (r={r_cl:+.3f})")
    elif cl_ok:
        print(f"  ⚠️  PARTIAL — clearance (r={r_cl:+.3f}), ω weak (r={r_om:+.3f})")
    else:
        print(f"  ❌ FAIL — ω (r={r_om:+.3f}), clearance (r={r_cl:+.3f})")

    # Comparison with no-stacking baseline
    print(f"\n  📊 Temporal Stacking Impact:")
    print(f"    No stacking (r=0.40/0.27) → N={TEMPORAL_N} (r={r_om:+.3f}/{r_cl:+.3f})")
    print(f"  {'='*70}")

    # Save
    np.savez("/Users/lhooz/.openclaw/workspace/temporal_stack_params.npz",
             **{name: np.array(p) for name, p in
                zip(['W_vis', 'W_imu', 'W2', 'W_imu2', 'W_li', 'b_li'], best_params)})
    print(f"  💾 Saved params")

    _plot(pred_f, eval_targets, best_eval,
           "/Users/lhooz/.openclaw/workspace/temporal_stack_curve.png")
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
    fig.suptitle(f'Temporal Stacking (N={TEMPORAL_N}) — Events + IMU  |  MSE: {best_mse:.4f}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  📸 Saved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    train()
