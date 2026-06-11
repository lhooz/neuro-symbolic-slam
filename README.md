# neuro-symbolic-slam: Split-Brain Neuro-Symbolic Spiking SLAM

| **Closed-Loop Spiking SLAM** | **Unsupervised STDP Feature Extraction** |
| :---: | :---: |
| <img src="results/snn_live_run.gif" width="375"> | <img src="results/stdp_debug.gif" width="375"> |
| *Real-time neuro-symbolic SLAM. Tracks 3-DOF robot coordinates using grid-cell spiking attractors (CANN) and closes loops via dynamic graph optimization.* | *Online unsupervised Spike-Timing-Dependent Plasticity (STDP) under active-dependent Synaptic Scaling, learning stable visual receptive fields directly from high-frequency event streams.* |

**neuro-symbolic-slam** is a JAX-accelerated, biologically plausible **Neuro-Symbolic Spiking SLAM** system for neuromorphic robotics. It unifies high-frequency event-driven visual processing, spiking continuous attractor network dynamics, and Hebbian plasticity to track 3-DOF robot poses, construct topological spatial maps, and close loops with industrial-grade robustness.

Key features include:
* **Split-Brain Vision Frontend:** Combines a fixed convolutional spiking neural network (CSNN) for instant edge-extraction with a plastic, self-organizing STDP frontend that learns custom receptive fields on event time-surfaces.
* **Continuous Attractor Dynamics (CANN):** Implements a 2D grid-cell continuous bump attractor for spatial path-integration and a 1D head-direction ring attractor to track headings in continuous time without Euler overshoot (via RK2 midpoint integration).
* **Complementary Filter Gravity Correction:** Integrates an on-board Complementary Filter fusing proper thoracic acceleration (subject to natural flapping vibrations) and gyroscope yaw rates to estimate absolute gravity pitch ($\theta_{\text{accel}} = \text{atan2}(a_x, a_z) + 1.0$), injecting a Gaussian corrective current into the 1D Ring Attractor to eliminate heading drift.
* **Activity-Dependent Synaptic Scaling:** Leverages a biologically grounded L1 weight scaling rule alongside an Asymmetric Instar update rule (Fast Learn, Slow Forget) to keep synaptic weights completely stable during idle periods and prevent catastrophic forgetting.
* **10-Tier Loop Closure Defense Pipeline:** Uses multi-sensory confidence gating, hyperdimensional visual barcodes, SeqSLAM-style sequence verification, geometric ICP (iterative closest point) validation on Time-of-Flight (ToF) rays, and a cerebellum-corrected heading sanity check.
* **Robust Graph Relaxation:** Integrates a spring-mass network graph optimizer equipped with **Dynamic Covariance Scaling (DCS)** outlier rejection, allowing the graph to seamlessly ignore false matches while permanently locking valid loops.
* **Pure Functional JAX Architecture:** Designed from the ground up using pure functional programming in JAX, allowing zero-mutation state propagation, high-frequency execution, and compilation to GPU/TPU accelerators.

### 📂 Project Structure

```text
neuro-symbolic-slam/            <-- Repository Root
├── src/                        <-- Core Neural Components
│   ├── snn_slam_system.py      # Split-Brain system orchestrator (Perception/Inference/Odo/Mapping)
│   ├── snn_live_slam.py        # Live SLAM loop coordinator & loop-closure gating pipeline
│   ├── snn_place_cells.py      # Place cell mapping, Hebbian memory bank & surprise computation
│   ├── snn_pose_cann.py        # 2D grid-cell and 1D head-direction ring attractor networks
│   ├── snn_vision_stdp.py      # Unsupervised STDP layer with active-dependent Synaptic Scaling
│   ├── snn_vision_fusion.py    # Spatiotemporal fusion of polarized CSNN and STDP visual channels
│   ├── snn_vision_csnn.py      # Fixed convolutional spiking neural network edge-extractor
│   ├── sparse_forest.py        # Differentiable virtual arena environment & virtual sensor rendering
│   ├── train_vision_online.py  # Online unsupervised STDP vision training script
│   └── frozen_csnn_weights.msgpack  # Pre-trained sensory CSNN weights (essential resource)
├── scripts/                    <-- Diagnostic & Utility Scripts
│   ├── run_slam.py             # Main entrypoint to execute closed-loop SLAM simulation
│   ├── slam_gate_monitor.py    # Live loop-closure gate diagnostics
│   ├── slam_sweep.py           # Hyperparameter sweep runner
│   ├── slam_variance.py        # Variance analysis across runs
│   └── stress_test.py          # System stress-test harness
├── results/                    <-- Output Media (GIFs tracked; bulk data files excluded)
│   ├── snn_live_run.gif        # 30-second live SLAM loop-closure animation highlight
│   └── stdp_debug.gif          # Visual diagnostic animation showing unsupervised STDP learning
└── PROJECT_SUMMARY.md          # Technical project summary
```

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.10+ installed. Clone the repository and install the dependencies:

```bash
git clone https://github.com/lhooz/neuro-symbolic-slam.git
cd neuro-symbolic-slam
pip install -r requirements.txt
```

*(Note: Requires `jax`, `jaxlib`, `numpy`, `matplotlib`, and `msgpack`)*

### 2. Running the System
To run the live SLAM system with the full 4-panel real-time visualization:

```bash
python scripts/run_slam.py
```

---

## 🎨 Under the Hood: Neuro-Symbolic Loop Gating

When the robot moves, physical sensors accumulate tracking errors. To resolve this, the system calculates a **Surprise** signal by checking the overlap between the visual reality (sensory input) and the position-based place cell expectation (CANN attractor belief):

$$\text{Surprise} = 1.0 - \text{Raw}_{\text{Match}}$$

This surprise signal controls two critical neuro-symbolic pathways:
* **Loop Closure Activation ($\text{Surprise} \ge 0.30$):** Under moderate sensory mismatch, the loop closure engine is triggered to query past visual barcodes and execute multi-stage defense gates.
* **Autopilot Learning Freeze ($\text{Surprise} \ge 0.60$):** Under extreme visual discrepancy, unsupervised STDP learning is frozen (autopilot off) to protect the pre-existing visual memory from catastrophic forgetting.

When the loop closure engine is activated ($\ge 0.30$), it initiates the following multi-stage verification pipeline to align the graph:

```mermaid
graph TD
    A[CANN/IMU Drift] -->|Surprise >= 0.30| B[HDC Barcode Retrieval]
    B -->|Match Overlap >= 6| C[SeqSLAM Sequence Verification]
    C -->|5-Frame Coherence| D[Geometric ICP Validation]
    D -->|ToF Ray Match >= 0.25m| E[Cerebellum Heading Gate]
    E -->|Angle Diff < 0.35rad| F[Graph Optimization]
    F -->|DCS Spring Relaxation| G[Drift Corrected!]
```
