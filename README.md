# Universal Stripe Remover

A variational stripe removal algorithm for 2-D images.  
Removes stripes in **five directions simultaneously** using a Primal-Dual Hybrid Gradient Method (PDHGMp).

---

## Background

Stripe artifacts are a recurring problem across diverse imaging techniques — light-sheet fluorescence microscopy (LSFM), focused ion beam scanning electron microscopy (FIB-SEM), and satellite remote sensing. These narrow, elongated artifacts compromise image quality and interfere with quantitative analysis.

The challenge lies in **separating stripes from real structures**: aggressive filtering removes stripes but blurs edges, while conservative approaches leave artifacts behind. [Rottmayer et al. (2025)](#references) addressed this with a variational formulation that models stripes explicitly as sparse directional components, enabling their removal without degrading the underlying image.

This repository provides a **PyTorch re-implementation** of their method, extended with batch processing and tiled inference for practical use.

## How It Works

### Mathematical Formulation

The input image `F` is decomposed into a clean image `u` and five stripe components `sᵢ`:

```
argmin  μ₁‖∇u‖_{2,1}  +  Σᵢ ‖D_θᵢ sᵢ‖₁  +  μ₂ Σᵢ ‖sᵢ‖₁
s.t.   u + Σᵢ sᵢ = F,   u ∈ [0, 1]
```

| Term | Role |
|------|------|
| `μ₁‖∇u‖_{2,1}` | Isotropic total variation — keeps `u` smooth |
| `‖D_θᵢ sᵢ‖₁` | Directional difference along stripe angle — enforces each `sᵢ` to be stripe-shaped |
| `μ₂‖sᵢ‖₁` | L1 sparsity — prevents stripes from absorbing real structures |
| `u + Σsᵢ = F` | Data fidelity — the decomposition must reconstruct the input exactly |

### Five Stripe Directions

All directions are optimised **simultaneously** in a single solve — no manual mode selection required.

| Mode | Direction | Δrow | Δcol | Typical source |
|------|-----------|------|------|----------------|
| 0 | Vertical | 1 | 0 | Sensor column defects |
| 1 | 26.6° Left | 2 | +1 | Oblique scan artifacts |
| 2 | 45° Left | 1 | +1 | Diagonal striping |
| 3 | 26.6° Right | 2 | −1 | Oblique scan artifacts |
| 4 | 45° Right | 1 | −1 | Diagonal striping |

### Solver: PDHGMp

The optimisation is solved iteratively using the **Primal-Dual Hybrid Gradient Method with dual extrapolation** (PDHGMp). Each iteration consists of five steps:

1. **Primal descent** — Update `u` and `sᵢ` using adjoint operators of the dual variables
2. **Constraint projection** — Enforce `u + Σsᵢ = F` by distributing the residual equally
3. **Box projection** — Clamp `u ∈ [0, 1]` and redistribute the excess into stripes
4. **Dual update** — Moreau proximal step for smoothness (coupled shrinkage) and directional/sparsity duals (soft clipping)
5. **Convergence check** — Every 20 iterations, measure relative change in `u`; stop early if below tolerance

### Tiled Processing

For large images or limited GPU memory, `process_tiled()` provides a seamless workflow:

```
Input image (H × W)
  ↓  pad to be divisible by n
  ↓  add overlap margin (default 64 px)
  ↓  extract n × n tiles
  ↓  stack as batch (n², tile_h, tile_w)
  ↓  process entire batch at once
  ↓  weight each tile by 2-D cosine window
  ↓  accumulate into canvas, normalise by weight sum
  ↓  crop to original size
Output (H × W)
```

The **cosine blending window** ensures smooth transitions at tile boundaries: the centre of each tile has weight 1, fading to 0 at the edges over the overlap region. This eliminates visible seams without any post-processing.

## Features

- **Multi-direction** — Five stripe angles optimised jointly; no mode selection needed
- **Batch inference** — `(B, H, W)` input; all images processed in a single solver pass
- **Tiled processing** — *n × n* grid with overlap → batch → seamless reassembly
- **Cosine blending** — Artifact-free tile boundaries guaranteed by smooth windowing
- **GPU-accelerated** — CUDA by default, automatic CPU fallback
- **Memory-efficient** — Pre-allocated scratch buffer reused across all iterations
- **Early stopping** — Convergence-based termination saves unnecessary computation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- (Optional) CUDA-capable GPU

## Quick Start

```python
import torch
import numpy as np
from PIL import Image
from src.remover import UniversalStripeRemover

# Load image
img = np.array(Image.open("asset/sample.png").convert("L"), dtype=np.float32) / 255.0
F = torch.from_numpy(img)

# De-stripe (full image)
remover = UniversalStripeRemover(mu1=0.33, mu2=0.003)
clean = remover.process(F, iterations=500)
stripes = F - clean

# De-stripe (tiled, for large images)
clean = remover.process_tiled(F, n=3, overlap=64)
```

## API

### `UniversalStripeRemover(mu1, mu2, device)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu1` | `0.33` | Smoothness weight — higher = smoother `u` |
| `mu2` | `0.003` | Sparsity weight — higher = fewer stripes detected |
| `device` | auto | `'cuda'`, `'cpu'`, or `None` for auto-detect |

### `.process(image, iterations, tol, proj, verbose)`

Process a single image or pre-batched tensor directly.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | `(H, W)` or `(B, H, W)` tensor / numpy array |
| `iterations` | `500` | Max PDHGMp iterations |
| `tol` | `1e-5` | Early stopping threshold (checked every 20 it) |
| `proj` | `True` | Clamp output to [0, 1] |
| `verbose` | `True` | Print progress |

**Returns:** `torch.Tensor` on CPU, same shape as input.

### `.process_tiled(image, n, iterations, tol, overlap, proj, verbose)`

Split into *n × n* tiles → batch-process → reassemble with cosine blending.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | `(H, W)` single image |
| `n` | `1` | Tile grid size (`1` = no tiling) |
| `overlap` | `64` | Overlap margin in pixels |
| *others* | — | Same as `.process()` |

**Returns:** `torch.Tensor` `(H, W)` on CPU.

## Validation

All correctness guarantees are verified in `test.ipynb`:

### Mathematical Correctness

**Adjoint consistency test** — For every finite-difference operator `D` (5 directional + 2 gradient = 7 total), we verify:

```
⟨Df, g⟩ = ⟨f, D^T g⟩     (relative error < 1e-5)
```

This is the fundamental requirement for PDHGMp convergence. If the adjoint is wrong, the solver diverges or converges to a wrong solution. **All 7 operators pass.**

### Tiling Seamlessness

| Test | What it checks |
|------|----------------|
| **Boundary zoom** | 160×160 px crop around tile edges — no visible seams |
| **Line profile** | 1-D intensity plot across boundaries — no discontinuities |
| **Quantitative** | Relative diff between n=1 and n=2/3 results — near-zero |

### Result Quality

| Metric | Purpose |
|--------|---------|
| Relative difference vs n=1 | Confirms tiling doesn't degrade quality |
| Max pixel difference | Worst-case deviation |
| PSNR (input → result) | Overall destriping strength |

## Files

| File | Description |
|------|-------------|
| `src/remover.py` | Core algorithm — `UniversalStripeRemover` class (~350 lines) |
| `test.ipynb` | Test notebook — 7 sections covering all validations above |
| `asset/sample.png` | Example test image with stripe artifacts |

## References

1. N. Rottmayer, C. Redenbach, and F. O. Fahrbach, "A universal and effective variational method for destriping: application to light-sheet microscopy, FIB-SEM, and remote sensing images," *Optics Express* **33**(3), 5800–5809 (2025). [https://doi.org/10.1364/OE.542868](https://doi.org/10.1364/OE.542868)

2. Original implementation: [https://github.com/NiklasRottmayer/General-Stripe-Removal](https://github.com/NiklasRottmayer/General-Stripe-Removal)
