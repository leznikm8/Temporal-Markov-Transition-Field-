# Temporal-Markov-Transition-Field-
Temporal Markov Transition Field (TMTF), an extension that partitions the series into K contiguous temporal chunks, estimates a separate local transition matrix for each chunk, and assembles the image so that each row reflects the dynamics local to its chunk rather than the global average.



# Temporal Markov Transition Field (TMTF)

A Python implementation of the **Temporal Markov Transition Field**, introduced in:

> Leznik, M. (2026). *The Temporal Markov Transition Field: A Representation for Time-Varying Transition Dynamics in Time Series Analysis*. arXiv:2603.08803

The TMTF extends the classic Markov Transition Field (MTF) of Wang & Oates (2015) to capture **regime-switching behaviour** in time series. Instead of estimating a single global transition matrix, it partitions the series into *K* contiguous temporal chunks, estimates a separate local transition matrix for each, and assembles a *T Ã— T* image whose horizontal band structure directly encodes when and how the dynamics changed.

---

## The Core Idea

The global MTF has a fundamental limitation: it pools all transitions into one matrix, so rows from the same quantile state are pixel-for-pixel identical regardless of when they occur. A regime change leaves no trace.

The TMTF fixes this. Each row of the image is governed by the local matrix of the **chunk it belongs to**, not the global average:

$$M_{ij} = W^{(\text{chunk}(i))}_{s_i,\, s_j}$$

The result is a *T Ã— T* image with *K* visually distinct horizontal bands, each carrying the transition texture of its temporal segment â€” diagonal-heavy for persistent regimes, off-diagonal for mean-reverting ones, upper-triangular for trending ones.

| Global MTF | TMTF |
|:---:|:---:|
| Single matrix, uniform texture | *K* local matrices, horizontal band structure |
| Regime change is invisible | Regime change is encoded as a texture boundary |
| At most *Q* distinct row patterns | Up to *K Ã— Q* distinct row patterns |

---

## Visualisations

**Toy series with quantile state bands and chunk boundary**

![Series](tmtf_series.png)

**Local transition matrices â€” Chunk 1 (mean-reverting) vs Chunk 2 (persistent)**

![Local matrices](tmtf_local_matrices.png)

**TMTF image â€” horizontal band structure clearly visible**

![TMTF image](tmtf_image.png)

**Global MTF vs TMTF side by side**

![Comparison](tmtf_comparison.png)

---

## Installation

No special packages are needed beyond the standard scientific Python stack.

```bash
pip install numpy matplotlib
```

Clone the repository:

```bash
git clone https://github.com/<your-username>/tmtf.git
cd tmtf
```

---

## Quick Start

Run the script directly to reproduce the paper's worked example:

```bash
python tmtf.py
```

This will print the state sequence, both local transition matrices, the full 12Ã—12 TMTF image, and save four PNG plots.

**Expected output (truncated):**

```
============================================================
  Temporal Markov Transition Field (TMTF)
============================================================

Series  : [12.0, 85.0, 45.0, 18.0, 78.0, 42.0, 15.0, 22.0, 55.0, 48.0, 82.0, 91.0]
Length  : T = 12
States  : Q = 3
Chunks  : K = 2

  Chunk 1 (t=1â€“6, n=6): states [1, 3, 2, 1, 3, 2]
    W^(1) =
    [[0. 0. 1.]
     [1. 0. 0.]
     [0. 1. 0.]]

  Chunk 2 (t=7â€“12, n=6): states [1, 1, 2, 2, 3, 3]
    W^(2) =
    [[0.5 0.5 0. ]
     [0.  0.5 0.5]
     [0.  0.  1. ]]
```

---

## Usage as a Module

All core functions are importable independently.

### Build the TMTF

```python
import numpy as np
from tmtf import build_tmtf

x = np.array([12, 85, 45, 18, 78, 42, 15, 22, 55, 48, 82, 91])

result = build_tmtf(x, Q=3, K=2)

print(result['M'])          # 12Ã—12 TMTF image
print(result['W_local'])    # list of K local transition matrices
print(result['b'])          # state sequence
```

### Build the global MTF for comparison

```python
from tmtf import build_global_mtf

global_result = build_global_mtf(x, Q=3)

print(global_result['M'])   # 12Ã—12 global MTF image
print(global_result['W'])   # single global transition matrix
```

### Use the lower-level functions directly

```python
from tmtf import quantile_binning, local_transition_matrix

# Discretise into states
b, boundaries = quantile_binning(x, Q=3)
print(b)           # e.g. [1, 3, 2, 1, 3, 2, 1, 1, 2, 2, 3, 3]
print(boundaries)  # e.g. [12.0, 35.33, 62.67, 91.0]

# Estimate a transition matrix from a subsequence
W = local_transition_matrix(b[:6], Q=3)
print(W)
```

### Generate plots

```python
from tmtf import (build_tmtf, plot_series_with_states,
                  plot_transition_matrices, plot_mtf, compare_mtf_tmtf)

result = build_tmtf(x, Q=3, K=2)

# Raw series with state bands
fig1 = plot_series_with_states(
    x, result['b'], result['boundaries'], result['K'], result['chunk_of']
)

# Local transition matrices as heatmaps
fig2 = plot_transition_matrices(result)

# TÃ—T TMTF image
fig3 = plot_mtf(result['M'], result['b'], result['K'], result['chunk_of'])

# Global MTF vs TMTF side by side
fig4 = compare_mtf_tmtf(x, Q=3, K=2)

fig1.savefig('series.png', dpi=150, bbox_inches='tight')
```

---

## API Reference

### `quantile_binning(x, Q)`

Discretises a time series into `Q` quantile states (1-indexed).

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like, shape `(T,)` | Raw time series |
| `Q` | int | Number of quantile bins |

**Returns** `(b, boundaries)` â€” state sequence and quantile boundary values.

---

### `local_transition_matrix(b_chunk, Q)`

Estimates the empirical row-stochastic transition matrix from a state sequence. Only within-chunk consecutive pairs are counted. Rows with no observed departures fall back to a uniform distribution `1/Q`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `b_chunk` | array-like, shape `(n,)` | State sequence (values in `{1,â€¦,Q}`) |
| `Q` | int | Number of states |

**Returns** `W` â€” `(Q, Q)` transition matrix where `W[l-1, m-1] = P(next=m | current=l)`.

---

### `build_tmtf(x, Q=3, K=2)`

Builds the full TMTF representation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | array-like, shape `(T,)` | Raw time series |
| `Q` | int | Number of quantile bins (default `3`) |
| `K` | int | Number of temporal chunks (default `2`) |

**Returns** a `dict` with:

| Key | Shape | Description |
|-----|-------|-------------|
| `M` | `(T, T)` | TMTF image |
| `b` | `(T,)` | Global state sequence (1-indexed) |
| `boundaries` | `(Q+1,)` | Quantile boundary values |
| `W_local` | list of `K` arrays `(Q, Q)` | Local transition matrices |
| `chunk_of` | `(T,)` | Chunk index (0-based) for each time step |
| `chunk_sizes` | list of `K` ints | Number of observations per chunk |
| `Q`, `K`, `T` | int | Configuration parameters |

---

### `build_global_mtf(x, Q=3)`

Builds the standard global MTF using a single transition matrix over the full series.

**Returns** a `dict` with keys `M`, `b`, `boundaries`, `W`, `Q`, `T`.

---

## Choosing Q and K

The paper provides the following practical guidance:

**Choosing Q (number of states)**

Larger `Q` gives finer resolution but requires more data per state to estimate transition probabilities reliably. For series of length `T â‰¥ 200`, values `Q âˆˆ {6, 10, 14}` are recommended.

**Choosing K (number of chunks)**

Larger `K` captures finer temporal structure but increases estimation variance per local matrix. A minimum-transitions rule requires each chunk to contain at least `5QÂ²` transitions, giving:

$$K \leq \frac{T}{5Q^2 + 1}$$

| T | Q=6 | Q=10 | Q=14 |
|---|-----|------|------|
| 200 | K â‰¤ 1 | â€” | â€” |
| 400 | K â‰¤ 2 | â€” | â€” |
| 1000 | K â‰¤ 5 | K â‰¤ 2 | K â‰¤ 1 |

For `T âˆˆ [200, 1000]` with `Q âˆˆ {6, 10, 14}`, **K = 4 is recommended as a robust default**.

---

## Geometric Interpretation of Local Matrices

Each local matrix `W^(k)` encodes a geometric signature of the process dynamics in that chunk:

| Matrix pattern | Process signature | Image texture |
|----------------|------------------|---------------|
| Large diagonal entries | **Persistence** â€” series stays in current state | Dark stripe along diagonal |
| Small diagonal, spread off-diagonal | **Mean reversion** â€” series oscillates across states | Diffuse, near-uniform texture |
| Upper-triangular, near-zero lower triangle | **Upward trend** â€” series moves to higher states | Concentrated above diagonal |
| All entries â‰ˆ 1/Q | **Random walk** â€” current state carries no information | Flat, uniform texture |

---

## Key Properties

Three formal properties established in the paper:

**Amplitude agnosticism** â€” The TMTF is invariant to any strictly increasing transformation of the observations (rescaling, shifting, log transforms, etc.). Only the rank ordering matters.

**Graceful degradation** â€” If all local matrices are identical (`W^(1) = Â·Â·Â· = W^(K) = W`), the TMTF reduces exactly to the global MTF. No complexity is introduced for genuinely stationary series.

**Band structure** â€” The TMTF can produce up to `K Ã— Q` distinct row patterns, compared to at most `Q` for the global MTF. The additional `K`-fold discrimination is precisely the temporal regime information that the global MTF discards.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | â‰¥ 1.21 | Array operations |
| `matplotlib` | â‰¥ 3.4 | All plots |

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{leznik2026tmtf,
  title   = {The Temporal Markov Transition Field: A Representation for
             Time-Varying Transition Dynamics in Time Series Analysis},
  author  = {Leznik, Michael},
  journal = {arXiv preprint arXiv:2603.08803},
  year    = {2026}
}
```

---

## References

- Leznik, M. (2026). The Temporal Markov Transition Field. arXiv:2603.08803
- Wang, Z. and Oates, T. (2015). Imaging time-series to improve classification and imputation. *IJCAI*, pp. 3939â€“3945.
- Liu, Y. et al. (2024). Multi-scale Markov transition field for time series classification. *IEEE Journal of Biomedical and Health Informatics*, 28(2), 1078â€“1088.

