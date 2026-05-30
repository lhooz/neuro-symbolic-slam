# snn-slam: Split-Brain Neuro-Symbolic Spiking SLAM

| **Closed-Loop Spiking SLAM** | **Unsupervised STDP Feature Extraction** |
| :---: | :---: |
| <img src="snn_slam_realtime.gif" width="375"> | <img src="stdp_debug.gif" width="375"> |
| *Real-time neuro-symbolic SLAM. Tracks 3-DOF robot coordinates using grid-cell spiking attractors (CANN) and closes loops via dynamic graph optimization.* | *Online unsupervised Spike-Timing-Dependent Plasticity (STDP) under active-dependent Synaptic Scaling, learning stable visual receptive fields directly from high-frequency event streams.* |

**snn-slam** is a JAX-accelerated, biologically plausible **Neuro-Symbolic Spiking SLAM** system for neuromorphic robotics. It unifies high-frequency event-driven visual processing, spiking continuous attractor network dynamics, and Hebbian plasticity to track 3-DOF robot poses, construct topological spatial maps, and close loops with industrial-grade robustness.

Key features include:
* **Split-Brain Vision Frontend:** Combines a fixed convolutional spiking neural network (CSNN) for instant edge-extraction with a plastic, self-organizing STDP frontend that learns custom receptive fields on event time-surfaces.
* **Continuous Attractor Dynamics (CANN):** Implements a 2D grid-cell continuous bump attractor for spatial path-integration and a 1D head-direction ring attractor to track headings in continuous time without Euler overshoot (via RK2 midpoint integration).
* **Activity-Dependent Synaptic Scaling:** Leverages a biologically grounded L1 weight scaling rule alongside an Asymmetric Instar update rule (Fast Learn, Slow Forget) to keep synaptic weights completely stable during idle periods and prevent catastrophic forgetting.
* **10-Tier Loop Closure Defense Pipeline:** Uses multi-sensory confidence gating, hyperdimensional visual barcodes, SeqSLAM-style sequence verification, geometric ICP (iterative closest point) validation on Time-of-Flight (ToF) rays, and a cerebellum-corrected heading sanity check.
* **Robust Graph Relaxation:** Integrates a spring-mass network graph optimizer equipped with **Dynamic Covariance Scaling (DCS)** outlier rejection, allowing the graph to seamlessly ignore false matches while permanently locking valid loops.
* **Pure Functional JAX Architecture:** Designed from the ground up using pure functional programming in JAX, allowing zero-mutation state propagation, high-frequency execution, and compilation to GPU/TPU accelerators.

### 📂 Project Structure

```text
snn-slam/                       <-- Repository Root
├── src/                        <-- Core Neural Components
│   ├── snn_live_slam.py        # Live SLAM loop coordinator & loop-closure gating pipeline
│   ├── snn_slam_system.py      # Split-Brain system orchestrator (Perception/Inference/Odo/Mapping)
│   ├── snn_place_cells.py      # Place cell mapping, Hebbian memory bank & surprise computation
│   ├── snn_pose_cann.py        # 2D grid-cell and 1D head-direction ring attractor networks
│   ├── snn_vision_stdp.py      # Unsupervised STDP layer with active-dependent Synaptic Scaling
│   ├── snn_vision_fusion.py    # Spatiotemporal fusion of polarized CSNN and STDP visual channels
│   ├── snn_vision_csnn.py      # Fixed convolutional spiking neural network edge-extractor
│   └── sparse_forest.py        # Differentiable virtual arena environment & virtual sensor rendering
├── run_slam.py                 # Main entrypoint to execute closed-loop SLAM simulation
├── slam_gate_monitor.py        # Debug suite analyzing the loop closure gating pipeline activation
└── slam_sweep.py               # Hyperparameter sweep coordinator analyzing loop-closure success rates
```

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.10+ installed. Clone the repository and install the dependencies:

```bash
git clone https://github.com/lhooz/snn-slam.git
cd snn-slam
pip install -r requirements.txt
```

*(Note: Requires `jax`, `jaxlib`, `numpy`, `matplotlib`, and `msgpack`)*

### 2. Running the System
To run the live SLAM system with the full 4-panel real-time visualization:

```bash
python run_slam.py
```

To run a diagnostic sweep or inspect individual gate behaviors across frames:

```bash
python slam_gate_monitor.py
```

---

## 🎨 Under the Hood: Neuro-Symbolic Loop Gating

When the robot moves, physical sensors drift. To solve this, your system calculates a **Surprise** signal by checking the overlap between the visual reality and the position-based place cell expectation:

$$\text{Surprise} = 1.0 - \text{Raw\_Match}$$

When surprise exceeds the threshold ($\ge 0.30$), the loop closure engine initiates the multi-stage defense gate to align coordinates:

```mermaid
graph TD
    A[CANN/IMU Drift] -->|Surprise >= 0.30| B[HDC Barcode Retrieval]
    B -->|Match Overlap >= 6| C[SeqSLAM Sequence Verification]
    C -->|5-Frame Coherence| D[Geometric ICP Validation]
    D -->|ToF Ray Match >= 0.25m| E[Cerebellum Heading Gate]
    E -->|Angle Diff < 0.35rad| F[Graph Optimization]
    F -->|DCS Spring Relaxation| G[Drift Corrected!]
```
