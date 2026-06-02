#!/usr/bin/env python3
"""Calibrate SNN firing rates by sweeping weight initialization multipliers.
No training — just forward passes to find healthy firing regime.

Targets:
  Layer 1: ~15-25% firing rate
  Layer 2: ~10% firing rate
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from event_camera_2d_nav import generate_batch, N_PIXELS, TIME_STEPS, DT
from snn_2d_nav import (
    run_snn_layer, run_li_readout, prepare_events,
    BETA, BETA_LI, V_TH, HIDDEN1, HIDDEN2, STATE_DIM, INPUT_DIM,
)

N_OBSTACLES = 6
ROOM_W, ROOM_H = 10.0, 10.0

def calibrate(w1_mult, w2_mult, key, batch=64):
    """Run one forward pass, return firing rates for both layers."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    W1 = jax.random.normal(k1, (INPUT_DIM, HIDDEN1)) * jnp.sqrt(2.0 / INPUT_DIM) * w1_mult
    W2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * jnp.sqrt(2.0 / HIDDEN1) * w2_mult
    W_li = jax.random.normal(k3, (HIDDEN2, STATE_DIM)) * 0.1
    b_li = jnp.zeros((STATE_DIM,))

    events, labels, _ = generate_batch(k4, batch)
    x_seq = prepare_events(events)

    h1 = run_snn_layer(x_seq, W1, BETA, V_TH)  # (T, B, H1)
    h2 = run_snn_layer(h1, W2, BETA, V_TH)       # (T, B, H2)

    rate1 = float(jnp.mean(h1)) * 100  # percentage
    rate2 = float(jnp.mean(h2)) * 100

    return rate1, rate2


def main():
    print("=" * 60)
    print("  🔬 SNN Firing Rate Calibration")
    print("=" * 60)
    print(f"  Input dim: {INPUT_DIM}, H1: {HIDDEN1}, H2: {HIDDEN2}")
    print(f"  β: {BETA}, V_th: {V_TH}, T: {TIME_STEPS}")
    print(f"  Base W1 scale: {float(jnp.sqrt(2.0/INPUT_DIM)):.4f}")
    print(f"  Base W2 scale: {float(jnp.sqrt(2.0/HIDDEN1)):.4f}")
    print("=" * 60)

    key = jax.random.PRNGKey(42)

    # Warm up JIT
    print("\n  🔨 Compiling...")
    calibrate(1.0, 1.0, key, batch=4)
    print("  Done.\n")

    # Coarse sweep
    print(f"  {'W1×':>6} {'W2×':>6}  {'L1 rate':>8} {'L2 rate':>8}  {'Verdict':>12}")
    print(f"  {'─'*6} {'─'*6}  {'─'*8} {'─'*8}  {'─'*12}")

    configs = []
    for w1m in [4, 5, 6, 7, 8]:
        for w2m in [1, 1.5, 2, 2.5, 3]:
            key, subkey = jax.random.split(key)
            r1, r2 = calibrate(w1m, w2m, subkey)
            configs.append((w1m, w2m, r1, r2))

            # Quick verdict
            l1_ok = 15 <= r1 <= 25
            l2_ok = 5 <= r2 <= 15
            if l1_ok and l2_ok:
                v = "✅ GOLDILOCKS"
            elif l1_ok or r2 > 0.5:
                v = "🔍 close"
            elif r1 < 2 and r2 < 1:
                v = "💀 dead"
            elif r1 > 50:
                v = "🔥 saturated"
            else:
                v = ""
            print(f"  {w1m:>6g} {w2m:>6g}  {r1:>7.2f}% {r2:>7.2f}%  {v}")

    # Find best configs (L1 near 15-25%, L2 near 10%)
    print(f"\n  🏆 Best configs (closest to L1≈20%, L2≈10%):")
    scored = sorted(configs, key=lambda c: (c[2] - 20)**2 + (c[3] - 10)**2)
    for w1m, w2m, r1, r2 in scored[:5]:
        loss = (r1 - 20)**2 + (r2 - 10)**2
        print(f"    W1×{w1m:>3g}  W2×{w2m:>3g}  →  L1={r1:.2f}%  L2={r2:.2f}%  (score={loss:.1f})")


if __name__ == "__main__":
    main()
