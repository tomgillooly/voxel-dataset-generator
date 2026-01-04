# Voxel Light Transport Learning: Experiment Plan and Architecture Specification

## Problem Statement

Learn a composable representation for subsurface scattering in homogeneous translucent voxel structures, such that:

- Light transport can be predicted without expensive Monte Carlo simulation
- The learned operator generalises across structure sizes
- The representation integrates into a ray tracing pipeline (query at bounding box → exitant radiance)

### Key Constraints

- Homogeneous material: all occupied voxels share identical optical properties
- Binary occupancy: voxels are present or absent
- Structures are axis-aligned regular grids
- Target resolution: up to 64³ voxels

### Core Hypothesis

Subsurface scattering exhibits sufficient locality and regularity that local transfer functions can be learned and composed to predict global transport behaviour.

---

## Data

### Source

Thingi10k dataset, voxelised at 64³ resolution, plus optimal transport interpolations between ~17 endpoint structures, yielding approximately 45k total structures including morphs.

### Preparation Pipeline

1. **Fill boundary shells**: Source structures are boundary-only; flood-fill interiors to produce solid volumes
2. **Validate watertightness**: Ensure no holes or ambiguous regions
3. **Store as dense binary arrays**: 64³ boolean tensors

```bash
# Pseudocode for filling
for structure in dataset:
    filled = flood_fill_interior(structure)
    validate_watertight(filled)
    save(filled, f"filled/{structure.id}.npy")
```

---

## Ground Truth Generation

### Approach

Use spherical harmonics (SH) to represent angular distributions of incident and exitant radiance. Accept fidelity loss from SH truncation to make ground truth generation tractable.

### SH Order Selection

| Order | Coefficients | Angular Resolution | Recommended For |
|-------|--------------|-------------------|-----------------|
| 2     | 9            | ~90° lobes        | Initial experiments |
| 3     | 16           | ~60° lobes        | Production if needed |
| 4     | 25           | ~45° lobes        | Only if lower orders demonstrably insufficient |

**Validation experiment**: Before committing to an SH order, render a few structures with Monte Carlo ground truth and compare against SH reconstruction at orders 2, 3, 4. Quantify error to establish ceiling.

### Ground Truth Pipeline

For each structure:

1. **Define boundary patches**: Discretise the bounding box surface into patches (e.g., 8×8 per face = 384 patches for a cube)
2. **For each incident condition** (patch × SH basis function):
   - Inject light with that angular distribution at that patch
   - Trace rays through volume using OptiX/Monte Carlo
   - Record exitant radiance at all boundary patches as SH coefficients
3. **Store as transfer matrix**: Shape `(n_patches × n_sh, n_patches × n_sh)`

```python
# Pseudocode
def generate_ground_truth(structure, n_patches, sh_order):
    n_sh = (sh_order + 1) ** 2
    dim = n_patches * n_sh
    transfer_matrix = np.zeros((dim, dim))
    
    for i, (patch_in, sh_in) in enumerate(all_incident_conditions):
        exitant = trace_rays(structure, patch_in, sh_in)  # Returns SH per patch
        transfer_matrix[i, :] = exitant.flatten()
    
    return transfer_matrix
```

### Computational Budget

For 64³ structures with 384 patches and order-2 SH (9 coefficients):
- Transfer matrix: 3456 × 3456 ≈ 12M elements
- Ray tracing calls per structure: 3456 incident conditions
- Parallelise across incident conditions; batch structures on cluster

**Recommendation**: Start with a subset of 100-500 structures to validate pipeline before scaling.

---

## Architecture Specification

### Tokenisation

Convert 2×2×2 voxel blocks into learned feature tokens.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Block size | 2×2×2 | 256 possible occupancy patterns |
| Token dimension | D (hyperparameter) | Start with D=64, test 32, 128, 256 |
| Token type | Continuous learned embedding | Not discrete lookup |

**Implementation options**:

(a) **Lookup table**: Learn 256 embeddings, index by occupancy pattern
```python
occupancy_to_token = nn.Embedding(256, D)
token = occupancy_to_token[block_pattern_id]
```

(b) **Small encoder**: Process 2×2×2 binary input through MLP
```python
encoder = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, D)
)
token = encoder(block.flatten())
```

Option (a) is simpler and sufficient given finite vocabulary. Start there.

### Graph Construction

After tokenisation, a 64³ volume becomes a 32³ graph of tokens.

- **Nodes**: Token features, shape `(32³, D)`
- **Edges**: 6-connectivity (face-adjacent blocks)
- **Boundary nodes**: Blocks touching the bounding box surface, marked for readout

### Message Passing

Learn a single operator applied iteratively:

```python
class MessagePassingLayer(nn.Module):
    def __init__(self, D, hidden=128):
        self.message_fn = nn.Sequential(
            nn.Linear(2 * D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, D)
        )
        self.update_fn = nn.Sequential(
            nn.Linear(2 * D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, D)
        )
    
    def forward(self, nodes, edge_index):
        # Aggregate messages from neighbors
        src, dst = edge_index
        messages = self.message_fn(torch.cat([nodes[src], nodes[dst]], dim=-1))
        aggregated = scatter_mean(messages, dst, dim=0)
        
        # Update node features
        updated = self.update_fn(torch.cat([nodes, aggregated], dim=-1))
        return updated
```

**Iterations**: Start with 8-16 iterations. The 32³ graph has diameter ~55 (corner to corner), so information needs many hops to propagate globally.

### Readout

Query boundary nodes for exitant radiance given incident illumination.

```python
class Readout(nn.Module):
    def __init__(self, D, n_sh=9):
        self.query_encoder = nn.Linear(n_sh, D)
        self.output_decoder = nn.Linear(D, n_sh)
    
    def forward(self, boundary_nodes, incident_sh):
        # incident_sh: (n_boundary_patches, n_sh)
        query = self.query_encoder(incident_sh)
        # Cross-attention or simple combination
        combined = boundary_nodes + query  # Simplest version
        exitant_sh = self.output_decoder(combined)
        return exitant_sh
```

### Hierarchical Variant

For the pooling experiments:

```python
class HierarchicalModel(nn.Module):
    def __init__(self, D, levels=3, iterations_per_level=4):
        self.tokenizer = nn.Embedding(256, D)
        self.message_passing = MessagePassingLayer(D)  # Shared across levels
        self.pool = nn.Linear(8 * D, D)  # 2×2×2 tokens → 1 token
        self.unpool = nn.Linear(D, 8 * D)
        
    def forward(self, voxels):
        # Tokenize
        tokens = self.tokenize(voxels)  # 32³ tokens
        
        # Downward pass: message pass then pool
        pyramid = [tokens]
        for level in range(self.levels):
            for _ in range(self.iterations_per_level):
                tokens = self.message_passing(tokens, edges)
            tokens = self.pool_2x2x2(tokens)
            pyramid.append(tokens)
        
        # Upward pass: unpool then message pass
        for level in reversed(range(self.levels)):
            tokens = self.unpool_2x2x2(tokens)
            tokens = tokens + pyramid[level]  # Skip connection
            for _ in range(self.iterations_per_level):
                tokens = self.message_passing(tokens, edges)
        
        return tokens
```

---

## Experiment Plan

### Phase 0: Validate Angular Representation (Per-Patch SH Truncation)

**Goal**: Confirm SH truncation doesn't introduce unacceptable *angular* error, independent of spatial discretization.

**Important note on error sources:**

The patch-based representation introduces two sources of discretization error:

1. **Angular (SH truncation)**: Continuous directional distribution → finite SH coefficients
2. **Spatial (patch discretization)**: Continuous boundary position → discrete patches

Spatial error is likely the dominant source of fidelity loss and will probably be large. We acknowledge this and defer addressing it to the coordinate-based approach (Appendix B). Phase 0 validates only the angular component to ensure we're not compounding two severe error sources.

**Method**:
1. Select 3-5 diverse structures
2. For a sampling of patch pairs (incident patch, exitant patch):
   - Run Monte Carlo: collect many (direction_in, direction_out, throughput) samples
   - For each exitant direction distribution, project onto SH basis at orders 2, 3, 4
   - Reconstruct the angular distribution from SH coefficients
   - Compare reconstructed distribution against raw MC samples

```python
def validate_sh_per_patch(mc_samples, sh_order):
    """
    mc_samples: exitant directions with associated throughput weights
    Returns error between empirical angular distribution and SH reconstruction
    """
    # Project onto SH basis
    sh_coeffs = project_to_sh(mc_samples.directions, mc_samples.weights, sh_order)
    
    # Reconstruct at sample directions
    reconstructed = evaluate_sh(sh_coeffs, mc_samples.directions)
    
    # Compare against empirical weights
    return relative_mse(reconstructed, mc_samples.weights)
```

3. Aggregate error statistics across patch pairs and structures

**Success criterion**: Per-patch angular reconstruction error < 10% relative to MC distribution, at order 2 or 3.

**Failure modes**:
- High error even at order 4: The angular distributions are too complex (high-frequency features). May indicate strong single-scattering component with sharp directional peaks. Consider higher SH order or alternative angular basis.
- Error varies wildly across patches: Some patch pairs have simple distributions (diffuse), others complex. May need adaptive SH order or accept non-uniform fidelity.

**Deliverable**: 
- Table of per-patch angular errors by SH order
- Decision on SH order for subsequent phases
- Documented angular ceiling (distinct from spatial ceiling)

**Note**: This does not validate the full pipeline. Rendered images will show spatial discretization artifacts regardless of SH accuracy. The coordinate-based approach (Appendix B) addresses this if the learning problem is otherwise solved.

---

### Phase 1: Single Object Overfit Test

**Goal**: Verify the architecture can fit *at all*. If it can't memorise one object, it can't generalise.

**Method**:
1. Select 1 medium-complexity structure
2. Train model (flat, D=64, 16 iterations) to predict transfer matrix
3. No regularisation, train until convergence or 10k epochs

**Success criteria**:
- Training loss approaches zero (< 1% of initial)
- Predicted transfer matrix matches ground truth visually and numerically

**Failure modes**:
- Loss plateaus high → architecture can't express the function
- Loss oscillates → optimisation issues (try lower LR, gradient clipping)

**Deliverable**: Learning curve, final error, go/no-go decision.

---

### Phase 2: Capacity Saturation

**Goal**: Determine if representation capacity is the bottleneck.

**Method**:
1. Using same single structure from Phase 1
2. Train with token dimensions D ∈ {32, 64, 128, 256}
3. Train with message passing iterations N ∈ {4, 8, 16, 32}
4. Record final training error for each configuration

**Analysis**:
- Plot error vs D (fixed N=16)
- Plot error vs N (fixed D=64)
- Identify knee points where increasing capacity stops helping

**Success criterion**: Find configuration where doubling capacity yields < 5% improvement.

**Deliverable**: Capacity curves, recommended D and N values.

---

### Phase 3: Multi-Object Generalisation

**Goal**: Test whether the model generalises beyond memorisation.

**Method**:
1. Split dataset: 80% train, 10% validation (interpolated structures), 10% test (held-out endpoints)
2. Train with configuration from Phase 2
3. Early stopping on validation loss

**Metrics**:
- Training loss
- Validation loss (structures on interpolation manifold)
- Test loss (novel endpoint structures)

**Analysis**:
- If train >> val ≈ test: model generalises well
- If train ≈ val >> test: overfitting to interpolation manifold
- If train >> val >> test: severe overfitting

**Deliverable**: Learning curves, generalisation gap analysis.

---

### Phase 4: Flat vs Hierarchical Comparison

**Goal**: Determine whether hierarchical pooling improves efficiency or accuracy.

**Configurations**:

| Name | Base Block | Graph Size | Message Passing |
|------|-----------|------------|-----------------|
| Flat-2 | 2×2×2 | 32³ | 16 iterations |
| Flat-4 | 4×4×4 | 16³ | 16 iterations |
| Flat-8 | 8×8×8 | 8³ | 16 iterations |
| Hier-2 | 2×2×2 | 32³→16³→8³→16³→32³ | 4 iter/level × 4 levels |

**Matching compute**: All configurations should have approximately equal FLOPs. Adjust iterations accordingly.

**Method**:
1. Train each configuration on same training set
2. Evaluate on same validation/test split
3. Record wall-clock time, memory usage, final error

**Analysis**:
- Error vs compute efficiency
- Error vs model complexity
- Qualitative: which structures does each configuration struggle with?

**Deliverable**: Comparison table, recommendation for best configuration.

---

### Phase 5: Error Analysis

**Goal**: Understand what causes prediction errors.

**Method**:
1. Take best model from Phase 4
2. Compute per-structure error on test set
3. Correlate error with structure properties:
   - Genus (topological complexity)
   - Surface area / volume ratio
   - Thin features (minimum thickness)
   - Convexity

**Analysis**:
- Scatter plots: error vs each property
- Identify failure cases: structures with highest error
- Visual inspection: where in the structure does prediction fail?

**Deliverable**: Error correlation analysis, identified failure modes.

---

## Error Metrics

### Primary: Transfer Matrix MSE

```python
def transfer_mse(predicted, ground_truth):
    return F.mse_loss(predicted, ground_truth)
```

### Secondary: Rendered Image Error

For interpretability, render under test illumination:

```python
def render_error(model, structure, test_illumination, ground_truth_image):
    predicted_transfer = model(structure)
    predicted_image = apply_transfer(predicted_transfer, test_illumination)
    return {
        'l2': F.mse_loss(predicted_image, ground_truth_image),
        'lpips': lpips_fn(predicted_image, ground_truth_image)
    }
```

### Diagnostic: Convergence Rate

For understanding whether eigenvalue structure matters:

```python
def measure_convergence(model, structure, max_iter=100):
    """Run message passing and record state at each iteration."""
    tokens = model.tokenize(structure)
    trajectory = [tokens.clone()]
    for i in range(max_iter):
        tokens = model.message_passing(tokens, edges)
        trajectory.append(tokens.clone())
    
    # Compute change between iterations
    deltas = [torch.norm(trajectory[i+1] - trajectory[i]) for i in range(max_iter)]
    return deltas
```

---

## Stopping Criteria

### Training Stopping

1. **Convergence**: Validation loss hasn't improved for 50 epochs
2. **Capacity**: Training loss < 1% of initial and not improving
3. **Time budget**: 24 hours per configuration (adjust based on resources)

### Hyperparameter Search Stopping

**"Good enough" criterion**: If a configuration achieves validation error within 2× of the SH ceiling (Phase 0), further tuning unlikely to help—the bottleneck is elsewhere.

**Diminishing returns**: If doubling a hyperparameter (D, N, data) yields < 10% improvement, stop increasing it.

### Experiment-Level Stopping

**Negative result thresholds**:
- Phase 1 fails (can't overfit single object): Fundamental architecture problem. Revisit representation.
- Phase 3 shows no generalisation: Model is memorising, not learning composable structure. Consider explicit inductive biases.
- Phase 4 shows all configurations perform similarly: The flat/hierarchical distinction doesn't matter for this problem.

---

## Implementation Checklist

### Prerequisites

- [ ] OptiX ray tracing pipeline operational
- [ ] SH projection/reconstruction code
- [ ] Voxel flood-fill utility
- [ ] PyTorch geometric or equivalent for graph operations

### Data Pipeline

- [ ] Fill all structures in dataset
- [ ] Implement boundary patch discretisation
- [ ] Implement ground truth transfer matrix generation
- [ ] Generate ground truth for pilot subset (100 structures)
- [ ] Validate SH truncation error (Phase 0)

### Model Implementation

- [ ] Tokeniser (embedding lookup)
- [ ] Message passing layer
- [ ] Graph construction from voxels
- [ ] Readout head
- [ ] Hierarchical pooling variant

### Training Infrastructure

- [ ] Data loader with batching
- [ ] Training loop with validation
- [ ] Checkpointing
- [ ] Logging (tensorboard/wandb)

### Evaluation

- [ ] Transfer matrix comparison
- [ ] Rendered image comparison
- [ ] Structure property extraction (genus, surface area, etc.)

---

## Appendix: Theoretical Considerations

### Why This Might Work

1. **Physics is local-ish**: Subsurface scattering has finite mean free path. Information decays exponentially with distance.

2. **Material is homogeneous**: Only geometry varies. The "rules" of transport are the same everywhere—only the boundary conditions differ.

3. **Interpolation manifold**: Training on smooth interpolations between structures might help the network learn continuous relationships rather than memorising discrete examples.

### Why This Might Fail

1. **Global coupling**: If light transport is fundamentally non-local (high-albedo materials where light bounces many times), local composition may not capture long-range correlations.

2. **Topological sensitivity**: If small geometric changes cause large transport changes (e.g., opening/closing a thin channel), smooth interpolation might not help.

3. **Representation bottleneck**: Finite token dimension may discard information needed for accurate composition. This is the information loss problem you've encountered before.

### Future Directions (If This Works)

- Extend to heterogeneous materials (weighted graph Laplacian)
- Learn in hyperbolic space to address information loss
- Explicit eigenbasis estimation for direct (non-iterative) solution
- Integration into real-time rendering pipeline
- Coordinate-based representation (see Appendix B)

---

## Appendix B: Coordinate-Based Representation (Future Work)

### Motivation

The patch-based approach discretises boundary position and direction into bins. This introduces resolution/accuracy tradeoffs and large transfer matrices. A coordinate-based approach would instead learn a continuous function:

```
f(structure, p_in, ω_in, p_out, ω_out) → radiance_transfer
```

Where `p` is position on the boundary and `ω` is direction. This sidesteps discretisation entirely and naturally handles arbitrary query resolution.

### Why Defer This

Coordinate-based adds complexity that conflates multiple hypotheses:

1. Can local transfer functions compose? (core question)
2. Can we represent transfer as a neural field? (representation question)
3. Can we condition fields on structure in a composable way? (architecture question)

The patch-based approach isolates question 1. If it works, questions 2-3 become engineering rather than research.

### Potential Architecture

**Per-block transfer field:**

Each 2×2×2 block gets a latent code `z_i`. The transfer function for that block is:

```python
class BlockTransferField(nn.Module):
    def __init__(self, latent_dim, hidden=256):
        self.net = nn.Sequential(
            # Input: latent + entry position (3) + entry direction (3) + exit position (3) + exit direction (3)
            nn.Linear(latent_dim + 12, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # Transfer coefficient
        )
    
    def forward(self, z, p_in, omega_in, p_out, omega_out):
        # Positions/directions in block-local coordinates
        x = torch.cat([z, p_in, omega_in, p_out, omega_out], dim=-1)
        return self.net(x)
```

**Composition challenge:**

For patch-based, composing two blocks is matrix multiplication. For coordinate-based, composition is an integral:

```
T_composed(p_in, ω_in, p_out, ω_out) = ∫∫ T_A(p_in, ω_in, p_mid, ω_mid) · T_B(p_mid, ω_mid, p_out, ω_out) dp_mid dω_mid
```

This integral over the shared interface is expensive. Options:

1. **Monte Carlo estimation**: Sample interface points/directions, average. Noisy but unbiased.

2. **Learned composition**: Train a network to predict the composed latent `z_AB` from `z_A` and `z_B` directly, bypassing explicit integration. This is back to learned composition, but now with a clear target (the integral) to supervise against.

3. **Basis function approach**: Represent the directional component in SH (keeping coordinate-based for position only). Now the directional integral is analytic (SH triple product), leaving only the spatial integral.

### Hybrid: Coordinate Position, SH Direction

A middle ground that might be practical:

```python
class HybridTransferField(nn.Module):
    def __init__(self, latent_dim, sh_order=2, hidden=256):
        self.n_sh = (sh_order + 1) ** 2
        self.net = nn.Sequential(
            # Input: latent + entry position (3) + exit position (3)
            nn.Linear(latent_dim + 6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_sh * self.n_sh)  # SH-to-SH transfer matrix
        )
    
    def forward(self, z, p_in, p_out):
        # Returns transfer matrix from incident SH to exitant SH
        # for this specific pair of boundary positions
        x = torch.cat([z, p_in, p_out], dim=-1)
        return self.net(x).reshape(-1, self.n_sh, self.n_sh)
```

Composition over directions is now matrix multiplication (SH domain). Composition over positions still requires integration, but only 2D (over the shared face) rather than 4D.

### Validation Strategy

If pursuing coordinate-based after patch-based succeeds:

1. **Train coordinate field to match patch ground truth**: Use patch-based transfer matrices as supervision. The coordinate field should reproduce patch values when queried at patch centers.

2. **Test interpolation**: Query at positions between patch centers. Compare to higher-resolution patch ground truth.

3. **End-to-end fine-tuning**: Once coordinate field matches patches, fine-tune on rendered image loss to recover any information lost in patch discretisation.

### When to Pursue This

Prerequisites:
- Patch-based Phase 3 (generalisation) succeeds
- Error analysis (Phase 5) suggests discretisation is a limiting factor
- Computational budget allows for more complex architecture

If patch-based achieves acceptable accuracy, coordinate-based may be unnecessary complexity. If patch resolution becomes a bottleneck, coordinate-based offers a principled upgrade path.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | Draft | Initial specification |