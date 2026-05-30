"""
snn_place_cells.py — PlaceCellNetwork with BATCH-ISOLATED associative memory.

KEY FIX (Batch Contamination Bug):
Each trajectory in the batch now has its OWN associative memory matrix.
Previously, all batch elements shared a single (2048, 1024) weight matrix,
causing catastrophic interference: Hebbian updates from one trajectory
corrupted the memory associations for all others.

Changes:
- W_fused_to_place: (2048, 1024) → (B, 2048, 1024) — one matrix per trajectory
- W_fused_to_ring:  (2048, 64)  → (B, 2048, 64)   — one matrix per trajectory
- _smoothed_recall_I: now (B, N_PLACE) per trajectory (not shared scalar)
- _smoothed_correction_I: now (B, N_PLACE) per trajectory
- Einsum: 'bf,fp->bp' → 'bf,bfp->bp'; 'bf,fr->br' → 'bf,bfr->br'
- Hebbian delta: no more .mean(axis=0) — each trajectory updates its own weights
"""

from jax import random, nn as jnn, jit
import jax.numpy as jnp
from jax.lax import fori_loop
from functools import partial

# ============================================================================
#  📦  CONSTANTS
# ============================================================================
N_VISION      = 256   # VisionSTDP output features (ON + OFF polarize)
N_DEPTH       = 8     # ToF Gaussian RBF channels
N_FUSED       = N_VISION * N_DEPTH   # 2048 = outer-product fused features
MAP_SIZE      = 32    # 2D CANN grid size (32×32 = 1024 place cells)
N_PLACE       = MAP_SIZE * MAP_SIZE   # 1024
RING_N        = 64    # 1D ring attractor for heading (64 neurons)
ROOM_W        = 10.0  # meters
ROOM_H        = 10.0  # meters
MAP_AREA      = ROOM_W * ROOM_H       # 100 m²
CELL_AREA     = MAP_AREA / N_PLACE    # ≈ 0.0977 m² per place cell
SIGMA_M       = 0.50  # Gaussian sigma in METERS (world space)
GAIN          = 4.0   # excitatory Gaussian amplitude
TOF_SIGMA = 0.05 # ToF Gaussian RBF width in meters (Hardware precision)
VEL_GAIN_XY   = 0.05  # body vel → CANN shift (unchanged — odometry)
VEL_GAIN_TH   = 0.15  # omega → ring shift
SENS_GAIN_XY  = 0.00  # disabled
SENS_GAIN_TH  = 0.00  # disabled
CORRECTION_GAIN     = 0.05    # ghost bump amplitude (was 0.01 → now 1.0 for strong correction)
RING_CORRECTION_GAIN = 1.0
CORRECTION_SMOOTH   = 0.90  # temporal smoothing of correction signal
RING_CORRECTION_SMOOTH = 0.90  # same for ring
TOFPOP_SIGMA = 0.20 # RBF width for ToF Gaussian population coder (Neural width)
INNER_INHIB   = 0.0   # inner-ring inhibition within a place field (disabled)

# SOTAv1 gating thresholds
TOFPOP_MATCH_THRESH  = 0.50   # ToF MATCH threshold (0-1 normalized similarity)
TOFPOP_ANTI_ALIAS    = 0.35   # top-16 concentration threshold for anti-aliasing
PLAUSIBILITY_THRESH  = 0.64
SELF_MATCH_THRESH    = 0.32   # meters — restored: blocks "molasses effect" backward corrections
MATCH_THRESH = 0.001

# EMA trace parameters for temporal binding (Fix #3)
# Event cameras are sparse/async — EMA "synaptic afterglow" binds vision→pose
TRACE_ALPHA = 0.85   # slow decay: 95% carry-over, 5% new input


def build_pc_preferred_locs(map_size=MAP_SIZE, room_w=ROOM_W, room_h=ROOM_H):
    """Build (N_PLACE, 2) array of preferred (x,y) for each place cell.

    Each cell's preferred location is the center of its grid tile in world coords.
    For a 32×32 grid covering a 10×10m room:
      cell (gx, gy) → x = (gx + 0.5) * room_w/map_size
                       y = (gy + 0.5) * room_h/map_size
    Returns: (N_PLACE, 2) float32 in meters
    """
    gx = jnp.arange(map_size, dtype=jnp.float32)          # [0, 1, ..., 31]
    gy = jnp.arange(map_size, dtype=jnp.float32)
    gx_center = (gx + 0.5) * (room_w / map_size)            # [0.156, 0.469, ..., 9.844]
    gy_center = (gy + 0.5) * (room_h / map_size)

    xx, yy = jnp.meshgrid(gx_center, gy_center, indexing='xy')
    locs = jnp.stack([xx.ravel(), yy.ravel()], axis=1)     # (1024, 2)
    return locs


def build_ring_preferred_th(ring_n=RING_N):
    """Build (RING_N,) array of preferred heading angles in radians."""
    return jnp.arange(ring_n, dtype=jnp.float32) * (2 * jnp.pi / ring_n)


# ============================================================================
#  🧠  PlaceCellNetwork
# ============================================================================
class PlaceCellNetwork:
    """BATCH-ISOLATED place + ring cell associative memory.

    Each trajectory in the batch has its own independent associative memory
    (W_fused_to_place, W_fused_to_ring), preventing catastrophic interference.
    """

    SPARSITY_THRESH  = 0.010  # ≥0.5% of fused features active → distinctive
    MATCH_THRESH     = 0.0005  # ≥0.05% avg recalled → seeds memory at cold start

    def __init__(self, key, n_fused=N_FUSED):
        key_hebb, key_ring, key_vis, key_tof = random.split(key, 4)
        self.n_fused    = n_fused           # e.g. 2048
        self.n_place    = N_PLACE           # 1024
        self.ring_n     = RING_N            # 64
        self.map_size   = MAP_SIZE          # 32

        # ── Place cell preferred locations for decoding ───────────────────
        self.pc_preferred_locs = jnp.array(build_pc_preferred_locs())  # (1024, 2)
        self.ring_preferred_th  = build_ring_preferred_th()            # (64,)

        # ── NOTE: W_fused_to_place and W_fused_to_ring are NO LONGER
        #     initialized here. They are initialized in reset(B) with
        #     shape (B, n_fused, n_place) and (B, n_fused, ring_n).
        #     This is the KEY FIX for the batch contamination bug.
        self._is_initialized = False

    def reset(self, B):
        """Initialize per-batch associative memory matrices to ZERO."""
        n_fused = self.n_fused
        
        # 🟢 FIX: Initialize memory to Tabula Rasa (blank slate)
        # This prevents random noise from tripping the Anti-Aliasing gate
        self.W_fused_to_place = jnp.zeros((B, n_fused, N_PLACE), dtype=jnp.float32)
        self.W_fused_to_ring  = jnp.zeros((B, n_fused, RING_N), dtype=jnp.float32)

        # ── Batched temporal state ─────────────────────────────────────────
        self._I_correction_smooth_place = jnp.zeros((B, N_PLACE))  
        self._I_correction_smooth_ring  = jnp.zeros((B, RING_N))    

        # ── EMA temporal traces for Hebbian binding ──────────────────────
        self._trace_fused = jnp.zeros((B, n_fused))  
        self._trace_place = jnp.zeros((B, N_PLACE))   
        self._trace_ring  = jnp.zeros((B, RING_N))    
        self._confidence_ema = jnp.zeros((B,)) # Gate 6 temporal trace

        self._step = 0
        self._is_initialized = True

    def __call__(self, fused_features, pose_bump, ring_bump, learn=True, confidence=None):
        """Forward: recall + temporal smoothing + optional Hebbian update."""
        B = fused_features.shape[0]

        # ── Memory recall: each trajectory uses its OWN weight matrix ────
        # Einsum: (B, F) × (B, F, P) → (B, P)
        I_place = jnp.einsum('bf,bfp->bp', fused_features, self.W_fused_to_place)
        I_ring  = jnp.einsum('bf,bfr->br', fused_features, self.W_fused_to_ring)

        # ── Temporal smoothing (per-trajectory) ────────────────────────────
        I_place_smooth = (
            CORRECTION_SMOOTH * self._I_correction_smooth_place
            + (1 - CORRECTION_SMOOTH) * I_place
        )
        I_ring_smooth = (
            RING_CORRECTION_SMOOTH * self._I_correction_smooth_ring
            + (1 - RING_CORRECTION_SMOOTH) * I_ring
        )
        self._I_correction_smooth_place = I_place_smooth
        self._I_correction_smooth_ring  = I_ring_smooth

        # ── Hebbian learning: each trajectory learns its OWN associations ──
        if learn:
            # Pass confidence into Hebbian update for Novelty Gating
            delta_W_place, delta_W_ring = self._updateHebbian(
                fused_features, pose_bump, ring_bump, confidence=confidence
            )
            # Update EACH trajectory's weights independently
            self.W_fused_to_place = self.W_fused_to_place + delta_W_place
            self.W_fused_to_ring  = self.W_fused_to_ring  + delta_W_ring

        return I_place, I_ring, I_place_smooth, I_ring_smooth

    def forward_mapping(self, fused_features, pose_bump, ring_bump=None, learn=True, confidence=None):
        """Alias for __call__ — kept for API compat."""
        # 🟢 FIX: ensure confidence passes through the alias
        return self(fused_features, pose_bump, ring_bump, learn=learn, confidence=confidence)

    def _updateHebbian(self, fused_features, pose_bump, ring_bump, confidence=None):
        ETA_W = 0.0006
        W_MAX = 1.0  
        LAMBDA_W = 0.02  # 🟢 FIX: Bring back decay, but make it LOCAL!

        # ── NOVELTY GATING ──────────────────────────────────────────
        if confidence is not None:
            novelty = 1.0 - confidence
            eta_batch = ETA_W * novelty[:, None, None]
        else:
            eta_batch = ETA_W

        # ── Update EMA traces ───────────────────────────────────────────────
        self._trace_fused = TRACE_ALPHA * self._trace_fused + (1 - TRACE_ALPHA) * fused_features
        self._trace_place = TRACE_ALPHA * self._trace_place + (1 - TRACE_ALPHA) * pose_bump
        self._trace_ring  = TRACE_ALPHA * self._trace_ring  + (1 - TRACE_ALPHA) * ring_bump

        # ── Raw Hebbian outer products ─────────────────────────────────────
        raw_delta_place = eta_batch * jnp.einsum(
            'bf,bp->bfp', self._trace_fused, self._trace_place
        )
        raw_delta_ring = eta_batch * jnp.einsum(
            'bf,br->bfr', self._trace_fused, self._trace_ring
        )

        # ── Localized Decay (Oja's Principle) ──────────────────────────────
        # Only decay synapses for the place cells that are CURRENTLY ACTIVE.
        # We broadcast the active pose trace (B, P) to match the weights (B, F, P).
        decay_place = eta_batch * LAMBDA_W * self._trace_place[:, None, :] * self.W_fused_to_place
        decay_ring  = eta_batch * LAMBDA_W * self._trace_ring[:, None, :]  * self.W_fused_to_ring

        # ── Combine Potentiation (Soft Bounded) and Local Decay ────────────
        delta_W_place = raw_delta_place * (W_MAX - self.W_fused_to_place) - decay_place
        delta_W_ring  = raw_delta_ring  * (W_MAX - self.W_fused_to_ring)  - decay_ring

        return delta_W_place, delta_W_ring

    # ========================================================================
    #  🔑  GATING — SOTAv1 Dual-Key with Anti-Aliasing + Plausibility
    # ========================================================================
    def compute_confidence_with_gates(self, fused_features, pose_xy, heading):
        """SOTAv1 dual-key gating: SPARSITY + MATCH + ANTI-ALIAS + PLAUSIBILITY.

        Batch-isolated: each trajectory computes its own gates independently.
        fused_features: (B, N_FUSED)
        pose_xy:       (B, 2)   current CANN estimate [x, y] in world meters
        heading:        (B,)    current ring estimate in radians
        """
        B = fused_features.shape[0]

        # ── Memory recall (batched — each trajectory uses its own W) ───────
        I_place = jnp.einsum('bf,bfp->bp', fused_features, self.W_fused_to_place)
        I_ring  = jnp.einsum('bf,bfr->br', fused_features, self.W_fused_to_ring)

        recalled_place = I_place.sum(axis=1) / float(N_PLACE)  # (B,) — mean per trajectory
        recalled_ring  = I_ring.sum(axis=1)  / float(RING_N) # (B,)

        # ── Gate 1: SPARSITY — distinctive vision ⊗ ToF activation? ───────
        # Sufficiently active fused features → observation is distinctive
        vision_activity = fused_features.sum(axis=1) / float(self.n_fused)  # (B,)
        is_distinctive = vision_activity > self.SPARSITY_THRESH           # (B,)

        # ── Gate 2: MATCH — associative memory recalls something? ───────────
        # Cold-start hack: allow corrections even without learned associations
        has_learned_memory = recalled_place > 0.010  # (B,)
        is_match = recalled_place > self.MATCH_THRESH  # (B,)

        # ── Gate 3: ANTI-ALIASING — concentration of recalled distribution ──
        # Reject near-uniform recall patterns (would cause position aliasing).
        # Use top-K concentration: genuine bump has localized recall, not uniform.
        I_place_sort = jnp.sort(I_place, axis=1)[:, ::-1]   # (B, P) descending
        total_place  = I_place_sort.sum(axis=1, keepdims=True) + 1e-8  # (B, 1)
        top16_place  = I_place_sort[:, :16].sum(axis=1)        # (B,) top-16 recall
        conc_place   = top16_place / total_place.squeeze(-1)  # (B,) ∈ [0.016, 1]

        I_ring_sort  = jnp.sort(I_ring, axis=1)[:, ::-1]
        total_ring   = I_ring_sort.sum(axis=1, keepdims=True) + 1e-8
        top4_ring    = I_ring_sort[:, :4].sum(axis=1)
        conc_ring    = top4_ring / total_ring.squeeze(-1)

        is_place_anti_aliased = jnp.where(
            has_learned_memory,
            conc_place > 0.30, # Stricter spatial concentration
            True 
        ) # (B,)

        is_ring_anti_aliased = jnp.where(
            has_learned_memory,
            conc_ring > 0.30, # Stricter heading concentration
            True
        ) # (B,)
        is_anti_aliased = is_place_anti_aliased & is_ring_anti_aliased

        # ── Gate 4: PLAUSIBILITY — correction jump is physically plausible ──
        # Decode recalled place cell peak position (world meters)
        peak_idx_place = jnp.argmax(I_place, axis=1)  # (B,) scalar index

        # 🟢 FIX: Use pre-built locs instead of mismatched meshgrid reconstruction
        recalled_x = self.pc_preferred_locs[peak_idx_place, 0]  # (B,)
        recalled_y = self.pc_preferred_locs[peak_idx_place, 1]  # (B,)

        # Correction jump from current pose to recalled pose
        dx = recalled_x - pose_xy[:, 0]   # (B,)
        dy = recalled_y - pose_xy[:, 1]
        jump = jnp.sqrt(dx**2 + dy**2)    # (B,)
        is_plausible = jump < PLAUSIBILITY_THRESH  # (B,)

        # 🌟 NEW: Gate 4b: HEADING PLAUSIBILITY (The Missing Dual-Key Check!) ──
        # Decode recalled ring heading
        peak_idx_ring = jnp.argmax(I_ring, axis=1)  # (B,)
        recalled_heading = peak_idx_ring.astype(jnp.float32) * (2 * jnp.pi / RING_N)

        # Calculate shortest angular distance between current and recalled heading
        angle_diff = jnp.abs(recalled_heading - heading)
        angle_diff = jnp.minimum(angle_diff, 2 * jnp.pi - angle_diff)

        HEADING_THRESH = 0.52  # ~30 degrees (rejects matches facing the wrong way)
        is_heading_plausible = angle_diff < HEADING_THRESH

        # ── Gate 5: SELF-MATCH — reject immediate-past duplicate (Fix #2) ──
        # Without this, the "Molasses Effect" drags the robot backward into
        # its own immediate-past memory on every step (tiny self-corrections
        # accumulate into large backward drags).
        # Only allow corrections to places > SELF_MATCH_THRESH away from current.
        dist = jump  # reuse jump magnitude as dist
        is_not_self_matching = dist > SELF_MATCH_THRESH  # (B,)

        # ── Combine static gates (AND logic) ────────────────────────────────
        is_confident_raw = (
            is_distinctive
            & is_match
            & is_anti_aliased
            & is_plausible
            & is_heading_plausible 
            & is_not_self_matching 
        ) # (B,)

        # 🌟 NEW: Gate 6: TEMPORAL CONSISTENCY ─────────────────────────────
        # EMA requires raw gates to pass for several consecutive frames
        self._confidence_ema = 0.8 * self._confidence_ema + 0.2 * is_confident_raw.astype(jnp.float32)
        is_temporally_consistent = self._confidence_ema > 0.6 
        
        is_confident = is_confident_raw & is_temporally_consistent

        # ── Soft confidence: continuous [0, 1] blend value ──────────────────
        # Scale smoothly: 0.0 at zero recall, reaching 1.0 when recall hits 0.010
        raw_confidence = recalled_place / 0.010
        blend = jnp.clip(raw_confidence, 0.0, 1.0) # (B,)

        return jnp.where(is_confident, blend, 0.0) # (B,)

    # ========================================================================
    #  🔧  CORRECTION SIGNALS
    # ========================================================================
    def compute_spatial_correction(self, fused_features, pose_xy, heading):
        """Build ghost bump at recalled place cell location (batched).

        Returns (B, N_PLACE) correction current for CANN.
        BATCH-ISOLATED: each trajectory's recalled pattern is independent.
        """
        B = fused_features.shape[0]

        # Batched recall using batched weights
        I_place = jnp.einsum('bf,bfp->bp', fused_features, self.W_fused_to_place)  # (B, P)

        # Decode recalled position
        peak_idx = jnp.argmax(I_place, axis=1)  # (B,) scalar

        gx_1d = jnp.arange(MAP_SIZE, dtype=jnp.float32)
        gy_1d = jnp.arange(MAP_SIZE, dtype=jnp.float32)
        gx_center = (gx_1d + 0.5) * (ROOM_W / MAP_SIZE)
        gy_center = (gy_1d + 0.5) * (ROOM_H / MAP_SIZE)
        xx, yy = jnp.meshgrid(gx_center, gy_center, indexing='ij')
        pc_map_x = xx.ravel()
        pc_map_y = yy.ravel()

        # 🟢 FIX: Use pre-built locs instead of mismatched meshgrid reconstruction
        recalled_x = self.pc_preferred_locs[peak_idx, 0]  # (B,)
        recalled_y = self.pc_preferred_locs[peak_idx, 1]  # (B,)

        # World → CANN grid coords
        cell_width = ROOM_W / MAP_SIZE
        cell_height = ROOM_H / MAP_SIZE
        recalled_gx = (recalled_x / (ROOM_W / MAP_SIZE)).astype(jnp.int32)
        recalled_gy = (recalled_y / (ROOM_H / MAP_SIZE)).astype(jnp.int32)
        recalled_gx = jnp.clip(recalled_gx, 1, MAP_SIZE - 2)
        recalled_gy = jnp.clip(recalled_gy, 1, MAP_SIZE - 2)

        # Build ghost Gaussian bump at recalled location (Proper 3D Broadcasting)
        gx = jnp.arange(MAP_SIZE, dtype=jnp.float32)[None, None, :]  # (1, 1, 32)
        gy = jnp.arange(MAP_SIZE, dtype=jnp.float32)[None, :, None]  # (1, 32, 1)
        recalled_gx_f = recalled_gx.astype(jnp.float32)[:, None, None]  # (B, 1, 1)
        recalled_gy_f = recalled_gy.astype(jnp.float32)[:, None, None]  # (B, 1, 1)
        
        # Now this correctly broadcasts into a full (B, 32, 32) grid!
        d2 = (gx - recalled_gx_f)**2 + (gy - recalled_gy_f)**2

        # 🟢 FIX: Calculate standard deviation based on 1D cell width, not 2D cell area
        sigma_mpc = SIGMA_M / cell_width  # sigma in grid-cell units
        ghost = GAIN * jnp.exp(-d2 / (2 * sigma_mpc**2))          # (B, 32, 32)

        return CORRECTION_GAIN * ghost.reshape(B, N_PLACE)  # (B, 1024)

    def compute_ring_correction(self, fused_features, heading):
        """Build ring correction bump at recalled heading (batched).

        Returns (B, RING_N) correction current for ring attractor.
        """
        B = fused_features.shape[0]

        # Batched recall
        I_ring = jnp.einsum('bf,bfr->br', fused_features, self.W_fused_to_ring)  # (B, R)

        # Decode recalled heading
        peak_idx = jnp.argmax(I_ring, axis=1)  # (B,)
        recalled_th = peak_idx.astype(jnp.float32) * (2 * jnp.pi / RING_N)  # (B,)

        # Build ring Gaussian bump at recalled angle
        th_1d = jnp.arange(RING_N, dtype=jnp.float32) * (2 * jnp.pi / RING_N)  # (RING_N,)
        recalled_th_b = recalled_th[:, None]  # (B, 1)
        d2_ring = (th_1d[None, :] - recalled_th_b + jnp.pi) % (2 * jnp.pi) - jnp.pi  # (B, RING_N)
        d2_ring = d2_ring**2

        sigma_ring = 4.0 / RING_N * 2 * jnp.pi  # ~0.39 rad ≈ 22°
        ring_corr = jnp.exp(-d2_ring / (2 * sigma_ring**2))  # (B, RING_N)

        return RING_CORRECTION_GAIN * ring_corr

    def initialize_from_pose(self, pose_bump, ring_bump=None):
        """Initialize place cells from a pose bump (for startup).

        Seeds the memory with a pre-formed Gaussian bump at the
        current ground truth pose, ensuring non-zero recall even at t=0 
        (breaking the cold-start deadlock).
        """
        # Ensure states exist (reset(B) should have been called)
        if self._I_correction_smooth_place is None:
            B = pose_bump.shape[0]
            self._I_correction_smooth_place = jnp.zeros((B, self.n_place), dtype=jnp.float32)
            
        self._I_correction_smooth_place = jnp.maximum(
            self._I_correction_smooth_place, pose_bump[:, :self.n_place]
        )

        # Seed the ring memory with a bump at the current heading
        if ring_bump is not None:
            if self._I_correction_smooth_ring is None:
                B = ring_bump.shape[0]
                self._I_correction_smooth_ring = jnp.zeros((B, self.ring_n), dtype=jnp.float32)
                
            self._I_correction_smooth_ring = jnp.maximum(
                self._I_correction_smooth_ring, ring_bump
            )

    # ========================================================================
    #  📍  DECODING
    # ========================================================================
    def decode_position(self, r_place):
        """Decode (x, y) from place cell activity via peak argmax.

        Fix #1: Previously used weighted average, which dragged the decoded
        coordinate toward the geographic center due to noise across the 1024-
        neuron sheet. Now uses argmax to find the exact peak of the memory
        recall, then looks up that neuron's preferred location.

        r_place: (B, N_PLACE) — place cell activation (smoothed recall current)
        Returns: (B, 2) — decoded [x, y] in world meters
        """
        # Find peak activation index for each batch element
        peak_idx = jnp.argmax(r_place, axis=1)  # (B,) — scalar index into 1024
        # Look up preferred location of the most-active place cell
        x_dec = self.pc_preferred_locs[peak_idx, 0]  # (B,)
        y_dec = self.pc_preferred_locs[peak_idx, 1]  # (B,)
        return jnp.stack([x_dec, y_dec], axis=1)

    def decode_heading(self, r_ring):
        """Decode θ from ring cell activity via circular mean."""
        angles = self.ring_preferred_th  # (RING_N,)
        p = r_ring / (r_ring.sum(axis=1, keepdims=True) + 1e-8)
        sin_sum = jnp.sum(jnp.sin(angles) * p, axis=1)
        cos_sum = jnp.sum(jnp.cos(angles) * p, axis=1)
        return jnp.arctan2(sin_sum, cos_sum) % (2 * jnp.pi)

    def top_place_cells(self, r_place, k=8):
        """Return indices and activations of top-k most active place cells."""
        B = r_place.shape[0]
        top_idx = jnp.zeros((B, k), dtype=jnp.int32)
        top_val = jnp.zeros((B, k), dtype=jnp.float32)
        for b in range(B):
            sorted_idx = jnp.argsort(r_place[b])[::-1]
            top_idx = top_idx.at[b].set(sorted_idx[:k])
            top_val = top_val.at[b].set(r_place[b, sorted_idx[:k]])
        return top_idx, top_val

    def get_place_activity_flat(self):
        """Return smoothed recall current (for decoding/visualization)."""
        return self._I_correction_smooth_place   # (B, 1024)

    def get_ring_activity_flat(self):
        """Return smoothed ring recall current (for decoding/visualization)."""
        return self._I_correction_smooth_ring    # (B, 64)
