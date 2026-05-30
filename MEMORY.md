# MEMORY.md — Ada's Long-Term Memory 🦊

## About Hao
- Scientific researcher building neuromorphic perception for flapping-wing MAVs (HornetRL)
- 30g micro aerial vehicle target, neuromorphic hardware
- Appreciates concise, direct communication — no fluff
- Timezone: Asia/Shanghai

## Project: Event-VIO SNN
- **Status:** ToF fusion breakthrough — clearance r=+0.380 (+78% jump)
- **Core insight:** Monocular 1D camera + ToF laser = sufficient for navigation
- **Architecture proven:** Box curriculum r>0.95 when Z is constant
- **Blocker resolved:** ToF laser rangefinder provides Z, breaking scale ambiguity

## Key Technical Decisions (Permanent)
1. DT=0.02s immutable (flight controller constraint)
2. No distance dimming (Conservation of Radiance veto)
3. No MLP/rate-coding — continuous-time SNN dynamics only
4. No input concatenation for IMU — neuromodulatory current injection only
5. Hard rejection sampling for collision-free data
6. Fresh random batches every epoch (scene-invariant learning)

## Architecture
```
Events → ON/OFF Polarize → W_vis → LIF(128) → LIF(64) → LI Readout → [ω, clearance]
                                     ↑
IMU [vx, vy, Z_tof] → W_kin → current injection
```
- LIF: β=0.85, V_th=1.0, surrogate α=2.0
- LI: β=0.95, 1/T normalization, trainable bias

## Curriculum Results
| Stage | Environment | ω r | clearance r |
|---|---|---|---|
| Box 4-DoF | 5×5m constant Z | +0.98 | +0.77 |
| Sparse Forest + IMU | 10×10m random Z | +0.40 | +0.27 |
| Sparse Forest + IMU + N=2 buffer | random Z | +0.41 | +0.26 |
| Sparse Forest + IMU + N=2 (fixed room) | fixed Z | +0.595 | +0.214 |
| **Sparse Forest + ToF fusion (fixed room)** | **fixed Z** | **+0.602** | **+0.380** |

## Project: SNN SLAM System (snn_slam_system.py)
- **Status:** INTEGRATED — 35-45% error reduction vs open-loop IMU
- **Architecture:** VisionSTDP + PoseCANN + MapCANN + Hebbian Memory
- **Core breakthrough:** Ghost bump competes with drifted IMU bump inside CANN Mexican Hat

### Architecture
```
events + ToF → VisionSTDP (256 features, k-WTA, adaptive thresholds)
                              ↓
vision_spikes (256) → W_vis_to_map (Hebbian, 256×1024)
                              ↓  [Phase B: Inference]
I_loop = vision_spikes · W_vis_to_map → confidence-gated ghost bump
                              ↓
IMU [vx,vy,ω] + I_loop → PoseCANN (Mexican Hat resolves competition)
                              ↓  [Phase C: Odometry]
pose_bump (1024) ←→ DoG recurrent dynamics → [x̂, ŷ, θ̂]
                              ↓  [Phase D: Mapping]
ΔW = η·(vision_spikes ⊗ pose_bump) − λ·W_vis_to_map
```

### Key Results
| Run | OL final err | CL final err | Improvement |
|---|---|---|---|
| B=3, T=200, drift@t=60 | 6.94m | 3.97m | 38.9% |
| B=4, T=200, drift@t=80 | 10.24m | 7.10m | 35.8% |

### Active Files
- `src/snn_slam_system.py` — Master orchestrator (NEW)

## Project: SNN-SLAM Digital Twin (snn_slam_twin.py)
- **Status:** WORKING — 0.086m position error, 1.9° heading error
- **Core breakthrough:** Holonomic heading fix + body-frame to global-frame velocity rotation
- **Architecture:** 2D CANN (32×32) + 1D Ring Attractor (64 neurons), DoG weights
- **Key insight:** For holonomic motion, true_heading = world_displacement - body_velocity_direction

### Final Architecture
```
Events → ON/OFF → LIF → hidden features
                              ↓
[vx, vy, ω] → velocity_injection → CANN shift + Ring rotation
                              ↓
CANN (x,y bump) ← DoG recurrent → position readout (circular mean)
Ring (θ bump) ← DoG recurrent → heading readout (circular mean)
```

### Key Parameters
- VEL_GAIN_XY = 0.05, VEL_GAIN_TH = 0.15
- SENS_GAIN_XY = 0.05, SENS_GAIN_TH = 0.02 (both disabled)
- CANN: A_exc=0.5, A_inh=0.125 (area-balanced), self-conn ×0.5
- Ring: RING_A_EXC=1.0, RING_A_INH=0.50, σ=2/4 neurons

### Active Files
- `src/snn_slam_twin.py` — Digital twin (NEW)

## Project: SNN SLAM v2 — Gaussian Population Coding (ACTIVE)
- **Status:** Running (2026-04-04) — B-spline trajectories, time-varying kinematics
- **B-spline trajectories:** scipy splprep/splev, opposing-quadrant control points (8 pts, cubic)
- **Kinematics (FIXED 2026-04-04):**
  - labels shape (T, 4) — per-timestep vx, vy, omega, clearance (was: (4,) mean constants)
  - splev evaluates directly on dense t_u grids (was: polygon via np.interp on control points)
  - omega time-varying from finite-diff of GT heading (was: jnp.tile of mean omega)
  - body-frame vx/vy from world velocity finite-diff + R(-theta) rotation (holonomic heading fix)
- **Results (1000 steps, 3 batch, drift @ t=80):**
  - IMU: mean=3.07m, final=5.48m, angular=4.9°
  - SNN OL/CL: mean=3.16m, final=5.58m, angular=76° (angular divergence on long loops)
- **Architecture:** VisionSTDP (256) ⊗ ToF Gaussian RBF (8 ch) → 2048 fused → W(2048×1024) Hebbian → place cells
- **SOTA Gates:** SPARSITY + MATCH(dual) + ANTI-ALIASING(dual) + PLAUSIBILITY + SELF-MATCH

### Active Files (v2)
- `src/snn_slam_system.py` — Master orchestrator (v2 with population coding)
- `src/snn_place_cells.py` — PlaceCellNetwork (refactored from snn_map_cann.py)
- `src/snn_vision_stdp.py` — VisionSTDP frontend (unchanged)
- `src/snn_pose_cann.py` — PoseCANN + `estimate_position()` method (added this session)

## Project: SNN SLAM v7 (stable1 — current)
- **Status:** ACTIVE — Sweep running (May 7, 2026)
- **Key changes from v6:**
  - Dual-key loop closure: place cells AND ring cells must agree
  - 3D conjunctive ring tensor: W[place × vision × heading] symmetry breaking
  - is_frozen masking for sliding window marginalization in graph relaxation
  - SOG (Spiking Occupancy Grid): 2D LIF sheet for neuromorphic mapping
  - Umeyama live alignment (SVD, centroid-preserving)
  - Maturity gate hard cutoff (≥0.85)
  - V_MAX ceiling added to SOG (was: no cap)
  - DRIFT_OMEGA now has random walk noise component
  - Tighter ring heading gate: 0.30 rad (was: 0.60)
- **Files:** src/stable1/ is the latest (effectively v7)
  - `snn_slam_system.py` — orchestrator
  - `snn_place_cells.py` — PlaceCellNetwork
  - `snn_pose_cann.py` — PoseCANN
  - `sparse_forest.py` — environment
- **Stress test results:**
  - SOG voltage oscillates between 0–0.7V (V_th=1.0) — no saturation on hover
  - Node creation: ~1 node/sec → 1000-node limit hit at ~16 min
  - Graph relaxation: 1000 nodes in 0.09s (JAX XLA)
- **Sweep running:** `slam_sweep.py` — 8-trial grid (GATING_STRENGTH × MATURITY_GATE)
  - ~4.5 min/trial → ~36 min total. Running in background.

## Active Files
- `sparse_forest.py` — Environment + ToF sensor + fixed room dataset
- `snn_slam_system.py` — Master orchestrator v2 (VisionSTDP + PoseCANN + PlaceCellNetwork)
- `snn_vision_stdp.py` — VisionSTDP frontend
- `snn_pose_cann.py` — PoseCANN (2D CANN + 1D Ring)
- `snn_place_cells.py` — PlaceCellNetwork (refactored, formerly snn_map_cann.py)
- `snn_slam_twin.py` — Digital twin standalone
- `slam_sweep.py` — Headless parameter sweep engine (NEW)
- `run_slam.py` — Wrapper for stable1 with path fix (NEW)

## Environment Setup
- Python venv: `source /Users/lhooz/.openclaw/workspace/.venv/bin/activate`
- JAX + numpy installed in venv
- Available models: zai/glm-5-turbo (default), zai/glm-4.6v

## Known Issues
- Training jobs >30min get SIGTERM'd — mitigate with 100 epochs
- matplotlib + jax.numpy array truth-value conflict breaks Rectangle/Circle patches
- Kimi 2.5 not configured (needs API key + models.json entry)

## Lessons Learned
- ALWAYS read PROJECT_SUMMARY.md before touching files
- Legacy files exist — newest ≠ current
- Context loss between sessions — daily logging essential
- matplotlib needs float() conversion for jax arrays in patch constructors
- sparse_forest labels shape (B, 4) = [vx_n, vy_n, omega_n, clearance_n] is CONSTANT per trajectory
  (each batch has one constant omega, NOT time-varying!) — critical for heading integration
- Cross-contamination of imports between `src/` and subfolders like `src/stable1/` can be avoided by using strictly relative/sibling local imports in subfolders.
- Integrating raw binary spikes directly into continuous SNN traces during cold-start leads to perpetual zero values; use normalized EMA features instead.
