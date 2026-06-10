#!/usr/bin/env python3
"""
Targeted stress test for SNN SLAM v7:
1. SOG voltage saturation (does beta=0.9995 allow smearing on hover?)
2. MAX_NODES limit (push to 1001 nodes and observe crash)
"""
import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

os.environ['MPLBACKEND'] = 'Agg'

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.sparse_forest import generate_fixed_room_dataset
from src.snn_vision_fusion import DualStreamVisionCortex
from src.snn_pose_cann import PoseCANN
from src.snn_place_cells import PlaceCellNetwork
import src.snn_slam_system as stable_slam
LiveEnvironment = stable_slam.LiveEnvironment
SpikingOccupancyGrid = stable_slam.SpikingOccupancyGrid
ToFPopulationCoder = stable_slam.ToFPopulationCoder
relax_graph = stable_slam.relax_graph
DRIFT_START = stable_slam.DRIFT_START
DRIFT_OMEGA = stable_slam.DRIFT_OMEGA
# KEYFRAME_* are local to run_live_slam; use hardcoded values from stable1
KEYFRAME_DIST = 0.15   # meters
KEYFRAME_ANG = 0.20    # radians

print("=" * 60)
print("  🧪 SNN SLAM v7 Stress Test")
print("=" * 60)

# ─── TEST 1: SOG SMERING ───────────────────────────────────────
print("\n📍 TEST 1: SOG Voltage Saturation on Hover")
print("-" * 40)
sog = SpikingOccupancyGrid(map_size_m=30.0, res=0.10, offset_m=10.0)
state = sog.init_state()

# Simulate hovering: ToF beams hit the same cells repeatedly
# Place fake hits at cell (50, 50) and free space around it
center_hit = jnp.array([[50, 50]])
ring_free = []
for dx in range(-5, 6):
    for dy in range(-5, 6):
        if abs(dx) >= 3 or abs(dy) >= 3:
            ring_free.append([50+dx, 50+dy])
ring_free = jnp.array(ring_free)

max_voltages = []
for step in range(500):
    state = sog.update(state, center_hit, ring_free)
    max_v = float(jnp.max(state.v_mem))
    max_voltages.append(max_v)
    if step % 100 == 0:
        print(f"  Step {step:4d}: V_max = {max_v:.4f}  (V_th={sog.v_th})")

print(f"\n  RESULT: V_max after 500 hover steps = {max_voltages[-1]:.4f}")
if max_voltages[-1] > sog.v_th * 0.9:
    print(f"  ⚠️  SOG NOT SATURATING — voltage keeps climbing!")
    print(f"      This confirms: no V_MAX cap = potential wall thickening on long hover.")
else:
    print(f"  ✅ SOG voltage stable")

plt.figure(figsize=(10, 4))
plt.plot(max_voltages, 'b-', lw=2)
plt.axhline(sog.v_th, color='r', ls='--', label=f'V_th={sog.v_th}')
plt.axhline(sog.v_th * sog.beta, color='orange', ls=':', label=f'V_leak≈{sog.v_th * sog.beta:.3f}')
plt.xlabel('Hover Steps'); plt.ylabel('Max Membrane Voltage (V)')
plt.title('SOG Voltage vs Hover Time (No V_MAX Cap)')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(ROOT, 'sog_voltage_test.png'), dpi=150)
print(f"  💾 Saved: {os.path.join(ROOT, 'sog_voltage_test.png')}")

# ─── TEST 2: NODE ACCUMULATION ────────────────────────────────
print("\n📍 TEST 2: Node Accumulation Rate (How fast do we hit 1000?)")
print("-" * 40)
key = random.PRNGKey(42)
env = LiveEnvironment(key, chunk_size=10000)  # large chunk

node_positions = []
node_count = 0
prev_pos = None

# Simulate node creation logic
for step in range(0, min(5000, len(env.ev)), 1):
    cx = env.kin[step, 0]
    cy = env.kin[step, 1]
    cth = env.kin[step, 2]
    
    if prev_pos is not None:
        dist = np.sqrt((cx - prev_pos[0])**2 + (cy - prev_pos[1])**2)
        dth = abs(cth - prev_pos[2])
        if dist > KEYFRAME_DIST or dth > KEYFRAME_ANG:
            node_count += 1
            node_positions.append((step, cx, cy, cth, dist, dth))
            prev_pos = (cx, cy, cth)
    else:
        node_count = 1
        prev_pos = (cx, cy, cth)
        node_positions.append((step, cx, cy, cth, 0.0, 0.0))
    
    if step % 500 == 0 and step > 0:
        print(f"  Step {step:4d}: {node_count} nodes created")

total_sim_time = 5000 * 0.02  # seconds
print(f"\n  Simulated {total_sim_time:.0f}s ({5000} steps @ 50Hz)")
print(f"  Total nodes created: {node_count}")
print(f"  Avg speed: {total_sim_time/node_count:.1f}s per node")
print(f"  At this rate, hitting MAX_NODES=1000 would take: {1000 * total_sim_time/node_count / 60:.1f} min")
print(f"\n  RESULT: {'⚠️  WILL HIT 1000-NODE LIMIT in long runs' if node_count < 1000 else '✅ Within limits for 2000-step run'}")

# ─── TEST 3: Graph relaxation with 800 active + 200 frozen ───
print("\n📍 TEST 3: Graph Relaxation with Frozen Nodes")
print("-" * 40)
# Create a simulated graph with 800 active + 200 frozen nodes
N = 1000
t_final = 50  # seconds
poses = np.zeros((N, 3), dtype=np.float32)
poses[:, 0] = np.arange(N) * 0.15  # 15cm spacing
is_frozen = np.zeros(N, dtype=bool)
is_frozen[200:] = True  # First 200 are frozen anchors

odom_edges = np.zeros((N-1, 3), dtype=np.float32)
odom_mask = np.ones(N-1, dtype=np.float32)

loop_closures = np.array([[50, 950], [100, 900]], dtype=np.int64)
loop_offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
loop_weights = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
loop_mask = np.array([1.0, 1.0], dtype=np.float32)

print(f"  Running relax_graph with {N} nodes ({N-200} active, 200 frozen)...")
t0 = time.time()
result = relax_graph(poses, odom_edges, odom_mask, loop_closures, loop_offsets, 
                      loop_weights, loop_mask, is_frozen, iterations=3000)
elapsed = time.time() - t0
print(f"  ✅ Graph relaxation completed in {elapsed:.2f}s")
print(f"  Final pose[0]: x={result[0,0]:.3f}, y={result[0,1]:.3f}, th={result[0,2]:.3f}")
print(f"  Final pose[999]: x={result[999,0]:.3f}, y={result[999,1]:.3f}, th={result[999,2]:.3f}")

print("\n" + "=" * 60)
print("  ✅ ALL STRESS TESTS COMPLETE")
print("=" * 60)
