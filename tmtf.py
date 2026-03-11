'''
Created on 7 Mar 2026

@author: Dr. Michael Leznik
Temporal Markov Transition Field (TMTF)
========================================
Implementation following Leznik (2026) "The Temporal Markov Transition Field".
arXiv:2603.08803v1
 
The TMTF extends the global MTF by partitioning the series into K contiguous
temporal chunks, estimating a separate local transition matrix for each chunk,
and assembling the T×T image so that each row reflects the dynamics local to
its chunk rather than the global average.

Usage
-----
    python tmtf.py

or import as a module:
    from tmtf import quantile_binning, local_transition_matrix, build_tmtf, build_global_mtf
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

def quantile_binning(x: np.ndarray, Q: int) -> np.ndarray:
    """
    Discretise a time series into Q quantile states (1-indexed, following the paper).

    Each observation is assigned to a state k ∈ {1, …, Q} based on which
    quantile interval it falls in. The boundaries are chosen so that each
    interval contains approximately T/Q observations.

    Parameters
    ----------
    x : array-like, shape (T,)
        Raw time series.
    Q : int
        Number of quantile bins.

    Returns
    -------
    b : np.ndarray, shape (T,), dtype int
        State sequence with values in {1, …, Q}.
    boundaries : np.ndarray, shape (Q+1,)
        Quantile boundary values [q0, q1, …, qQ].
    """
    x = np.asarray(x, dtype=float)
    # Compute Q+1 evenly spaced quantile boundaries
    percentiles = np.linspace(0, 100, Q + 1)
    boundaries = np.percentile(x, percentiles)
    # Make the upper boundary slightly larger to include the maximum
    boundaries[-1] += 1e-10

    b = np.zeros(len(x), dtype=int)
    for k in range(Q):
        mask = (x >= boundaries[k]) & (x < boundaries[k + 1])
        b[mask] = k + 1  # 1-indexed states

    return b, boundaries


def local_transition_matrix(b_chunk: np.ndarray, Q: int) -> np.ndarray:
    """
    Estimate the empirical transition probability matrix from a chunk's state sequence.

    Only within-chunk consecutive pairs are counted; the final observation has
    no successor within the chunk and contributes no outgoing transition.

    Parameters
    ----------
    b_chunk : array-like, shape (n,)
        State sequence for one chunk, values in {1, …, Q}.
    Q : int
        Number of states.

    Returns
    -------
    W : np.ndarray, shape (Q, Q)
        Row-stochastic transition matrix. W[l-1, m-1] = P(next=m | current=l).
        Rows with no observed departures are set to uniform (1/Q) as a fallback.
    """
    b_chunk = np.asarray(b_chunk, dtype=int)
    W = np.zeros((Q, Q), dtype=float)

    for t in range(len(b_chunk) - 1):
        from_state = b_chunk[t] - 1      # 0-indexed for array
        to_state   = b_chunk[t + 1] - 1
        W[from_state, to_state] += 1

    # Normalise each row; unvisited rows → uniform distribution
    row_totals = W.sum(axis=1, keepdims=True)
    zero_rows  = (row_totals == 0).flatten()
    W = np.where(row_totals > 0, W / row_totals, 1.0 / Q)
    if zero_rows.any():
        print(f"  Warning: {zero_rows.sum()} state(s) had no outgoing transitions "
              f"in this chunk → set to uniform ({1/Q:.3f}).")

    return W


def build_tmtf(x: np.ndarray, Q: int = 3, K: int = 2) -> dict:
    """
    Build the Temporal Markov Transition Field (TMTF).

    Parameters
    ----------
    x : array-like, shape (T,)
        Raw time series.
    Q : int
        Number of quantile bins (default 3).
    K : int
        Number of temporal chunks (default 2).

    Returns
    -------
    result : dict with keys
        'M'           : np.ndarray (T, T)  — TMTF image
        'b'           : np.ndarray (T,)    — global state sequence (1-indexed)
        'boundaries'  : np.ndarray (Q+1,) — quantile boundaries
        'W_local'     : list of K np.ndarray (Q, Q) — local transition matrices
        'chunk_sizes' : list of K ints
        'Q'           : int
        'K'           : int
        'T'           : int
    """
    x = np.asarray(x, dtype=float)
    T = len(x)

    if T % K != 0:
        print(f"  Note: T={T} is not divisible by K={K}. "
              f"Chunk sizes will vary by ±1 observation.")

    # Step 1 – Quantile binning over the full series
    b, boundaries = quantile_binning(x, Q)

    # Step 2 – Partition into K chunks
    chunk_indices = np.array_split(np.arange(T), K)

    # Step 3 – Estimate one local transition matrix per chunk
    W_local = []
    chunk_sizes = []
    for k, idx in enumerate(chunk_indices):
        b_chunk = b[idx]
        W_k = local_transition_matrix(b_chunk, Q)
        W_local.append(W_k)
        chunk_sizes.append(len(idx))
        print(f"  Chunk {k+1} (t={idx[0]+1}–{idx[-1]+1}, n={len(idx)}): "
              f"states {b_chunk.tolist()}")
        print(f"    W^({k+1}) =\n{np.round(W_k, 4)}")

    # Step 4 – Assemble the T×T TMTF image
    # chunk_of[t] gives the chunk index (0-based) for time step t
    chunk_of = np.zeros(T, dtype=int)
    for k, idx in enumerate(chunk_indices):
        chunk_of[idx] = k

    M = np.zeros((T, T), dtype=float)
    for i in range(T):
        for j in range(T):
            k   = chunk_of[i]          # which chunk governs row i
            s_i = b[i] - 1            # 0-indexed state at time i
            s_j = b[j] - 1            # 0-indexed state at time j
            M[i, j] = W_local[k][s_i, s_j]

    return {
        'M'          : M,
        'b'          : b,
        'boundaries' : boundaries,
        'W_local'    : W_local,
        'chunk_sizes': chunk_sizes,
        'chunk_of'   : chunk_of,
        'Q'          : Q,
        'K'          : K,
        'T'          : T,
    }


def build_global_mtf(x: np.ndarray, Q: int = 3) -> dict:
    """
    Build the standard global Markov Transition Field for comparison.

    Parameters
    ----------
    x : array-like, shape (T,)
        Raw time series.
    Q : int
        Number of quantile bins.

    Returns
    -------
    result : dict with keys 'M', 'b', 'boundaries', 'W', 'Q', 'T'
    """
    x = np.asarray(x, dtype=float)
    T = len(x)

    b, boundaries = quantile_binning(x, Q)
    W = local_transition_matrix(b, Q)  # one matrix over full series

    M = np.zeros((T, T), dtype=float)
    for i in range(T):
        for j in range(T):
            M[i, j] = W[b[i] - 1, b[j] - 1]

    return {'M': M, 'b': b, 'boundaries': boundaries, 'W': W, 'Q': Q, 'T': T}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cmap():
    """White → light blue → dark blue colormap."""
    return LinearSegmentedColormap.from_list(
        'tmtf_blue',
        ['#FFFFFF', '#DEEAF1', '#BDD7EE', '#2E75B6', '#1F4E79']
    )


def plot_transition_matrices(result: dict, title_prefix: str = "Local") -> plt.Figure:
    """Plot the local (or global) transition matrices as heatmaps."""
    cmap  = _make_cmap()
    W_list = result.get('W_local', [result.get('W')])
    K     = len(W_list)
    Q     = result['Q']
    labels = [f"State {k+1}" for k in range(Q)]

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 3.8))
    if K == 1:
        axes = [axes]

    for k, (ax, W) in enumerate(zip(axes, W_list)):
        im = ax.imshow(W, vmin=0, vmax=1, cmap=cmap, aspect='auto')
        ax.set_xticks(range(Q)); ax.set_yticks(range(Q))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("To state", fontsize=10)
        ax.set_ylabel("From state", fontsize=10)
        chunk_label = f"Chunk {k+1}" if K > 1 else "Global"
        ax.set_title(f"{title_prefix} Matrix — {chunk_label}\n"
                     f"W^({k+1})" if K > 1 else f"Global Matrix W",
                     fontsize=11, color='#1F4E79', fontweight='bold')

        # Annotate cells with values
        for i in range(Q):
            for j in range(Q):
                val = W[i, j]
                color = 'white' if val > 0.6 else '#1F4E79'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Probability')

    fig.suptitle(f"Transition Matrices  (Q={Q})", fontsize=13,
                 fontweight='bold', color='#1F4E79', y=1.02)
    fig.tight_layout()
    return fig


def plot_mtf(M: np.ndarray, b: np.ndarray, K: int,
             chunk_of: np.ndarray, title: str = "TMTF") -> plt.Figure:
    """Plot the T×T MTF image with chunk band annotations."""
    T    = M.shape[0]
    cmap = _make_cmap()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(M, vmin=0, vmax=1, cmap=cmap, aspect='auto',
                   extent=[-0.5, T - 0.5, T - 0.5, -0.5])

    # Chunk boundary lines (horizontal — separate row bands)
    if K > 1:
        boundaries_chunks = []
        for k in range(K - 1):
            last_t = np.where(chunk_of == k)[0][-1]
            boundaries_chunks.append(last_t + 0.5)
        for b_line in boundaries_chunks:
            ax.axhline(b_line, color='#FF4444', linewidth=2,
                       linestyle='--', label='Chunk boundary')

    # Axis ticks — show time step and state
    tick_labels = [f"t{t+1}\ns{b[t]}" for t in range(T)]
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.set_yticklabels(tick_labels, fontsize=7.5)
    ax.set_xlabel("Column time step  j", fontsize=11)
    ax.set_ylabel("Row time step  i", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#1F4E79', pad=12)

    # Band labels on the right
    if K > 1:
        for k in range(K):
            rows_in_chunk = np.where(chunk_of == k)[0]
            mid = rows_in_chunk.mean()
            ax.text(T + 0.1, mid, f"Band {k+1}\nW^({k+1})",
                    va='center', ha='left', fontsize=9,
                    color='#1F4E79', fontweight='bold',
                    transform=ax.get_yaxis_transform())

    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04,
                 label='Transition probability')
    fig.tight_layout()
    return fig


def plot_series_with_states(x: np.ndarray, b: np.ndarray,
                            boundaries: np.ndarray, K: int,
                            chunk_of: np.ndarray) -> plt.Figure:
    """Plot the raw series with quantile bands and chunk boundaries."""
    T    = len(x)
    Q    = len(boundaries) - 1
    t    = np.arange(1, T + 1)

    # Colour per state
    state_colors = ['#2E75B6', '#ED7D31', '#70AD47',
                    '#7030A0', '#C55A11', '#833C00']
    band_colors  = ['#DEEAF1', '#FCE4D6', '#E2EFDA',
                    '#EAD1DC', '#D9E1F2', '#FFF2CC']

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Quantile bands
    for k in range(Q):
        ax.axhspan(boundaries[k], boundaries[k + 1],
                   alpha=0.25, color=band_colors[k % len(band_colors)], zorder=0)

    # Quantile boundary lines
    for bnd in boundaries[1:-1]:
        ax.axhline(bnd, color='#888888', linewidth=0.9, linestyle=':', alpha=0.7)
        ax.text(T + 0.15, bnd, f' {bnd:.1f}', va='center', fontsize=8, color='#555555')

    # Chunk boundaries (vertical)
    for k in range(K - 1):
        last_t = np.where(chunk_of == k)[0][-1]
        ax.axvline(last_t + 1.5, color='#CC0000', linewidth=1.8,
                   linestyle='--', alpha=0.8, zorder=2)

    # Chunk background shading
    chunk_bg = ['#EBF3FA', '#FEF0E6']
    for k in range(K):
        idx = np.where(chunk_of == k)[0]
        ax.axvspan(idx[0] + 0.5, idx[-1] + 1.5,
                   alpha=0.12, color=chunk_bg[k % 2], zorder=0)

    # Line and points
    ax.plot(t, x, color='#595959', linewidth=1.5, alpha=0.6, zorder=2)
    for i in range(T):
        sc = state_colors[(b[i] - 1) % len(state_colors)]
        ax.scatter(t[i], x[i], color=sc, s=90, zorder=3,
                   edgecolors='white', linewidths=0.9)

    # Legend
    patches = [mpatches.Patch(color=state_colors[k],
                               label=f'State {k+1}') for k in range(Q)]
    ax.legend(handles=patches, loc='upper left', fontsize=9, framealpha=0.9)

    ax.set_xlabel('Time step  t', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Time Series with Q={Q} States and K={K} Chunks',
                 fontsize=13, fontweight='bold', color='#1F4E79', pad=10)
    ax.set_xlim(0.5, T + 1)
    ax.set_xticks(t)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def compare_mtf_tmtf(x: np.ndarray, Q: int = 3, K: int = 2) -> plt.Figure:
    """Side-by-side comparison of Global MTF vs TMTF."""
    global_result = build_global_mtf(x, Q)
    tmtf_result   = build_tmtf(x, Q, K)

    cmap  = _make_cmap()
    T     = len(x)
    b     = tmtf_result['b']
    chunk_of = tmtf_result['chunk_of']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    tick_labels = [f"t{t+1}\ns{b[t]}" for t in range(T)]

    for ax, M, title in zip(
        axes,
        [global_result['M'], tmtf_result['M']],
        [f'Global MTF  (Q={Q})', f'TMTF  (Q={Q}, K={K})']
    ):
        im = ax.imshow(M, vmin=0, vmax=1, cmap=cmap, aspect='auto',
                       extent=[-0.5, T - 0.5, T - 0.5, -0.5])
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Column j", fontsize=10)
        ax.set_ylabel("Row i", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight='bold', color='#1F4E79', pad=10)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04,
                     label='Transition probability')

    # Draw chunk boundaries on TMTF panel only
    ax = axes[1]
    for k in range(K - 1):
        last_t = np.where(chunk_of == k)[0][-1]
        ax.axhline(last_t + 0.5, color='#FF4444', linewidth=2,
                   linestyle='--', label='Chunk boundary')
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Global MTF vs Temporal MTF (TMTF)', fontsize=15,
                 fontweight='bold', color='#1F4E79', y=1.01)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main — run with the paper's toy example
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Toy series from Leznik (2026) ────────────────────────────────────────
    x = np.array([12, 85, 45, 18, 78, 42, 15, 22, 55, 48, 82, 91], dtype=float)
    Q = 3   # number of quantile states
    K = 2   # number of temporal chunks

    print("=" * 60)
    print("  Temporal Markov Transition Field (TMTF)")
    print("=" * 60)
    print(f"\nSeries  : {x.tolist()}")
    print(f"Length  : T = {len(x)}")
    print(f"States  : Q = {Q}")
    print(f"Chunks  : K = {K}\n")

    # ── 1. Build TMTF ────────────────────────────────────────────────────────
    print("─" * 60)
    print("Step 1 — Quantile binning & local matrices")
    print("─" * 60)
    result = build_tmtf(x, Q=Q, K=K)

    print(f"\nGlobal state sequence b:")
    print(f"  {result['b'].tolist()}")
    print(f"\nQuantile boundaries: {np.round(result['boundaries'], 2).tolist()}")

    print("\n" + "─" * 60)
    print("Step 2 — TMTF image  M  (12×12)")
    print("─" * 60)
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(result['M'])

    # ── 2. Build global MTF for comparison ───────────────────────────────────
    print("\n" + "─" * 60)
    print("Global MTF image  M  (12×12)  [for comparison]")
    print("─" * 60)
    global_result = build_global_mtf(x, Q=Q)
    print(global_result['M'])
    print(f"\nGlobal transition matrix W:")
    print(global_result['W'])

    # ── 3. Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # Series with state bands
    fig1 = plot_series_with_states(
        x, result['b'], result['boundaries'], K, result['chunk_of']
    )
    fig1.savefig('tmtf_series.png', dpi=150, bbox_inches='tight')
    print("  Saved: tmtf_series.png")

    # Local transition matrices
    fig2 = plot_transition_matrices(result, title_prefix="Local")
    fig2.savefig('tmtf_local_matrices.png', dpi=150, bbox_inches='tight')
    print("  Saved: tmtf_local_matrices.png")

    # TMTF image
    fig3 = plot_mtf(
        result['M'], result['b'], K, result['chunk_of'],
        title=f'Temporal Markov Transition Field  (Q={Q}, K={K})'
    )
    fig3.savefig('tmtf_image.png', dpi=150, bbox_inches='tight')
    print("  Saved: tmtf_image.png")

    # Side-by-side comparison
    fig4 = compare_mtf_tmtf(x, Q=Q, K=K)
    fig4.savefig('tmtf_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: tmtf_comparison.png")

    plt.close('all')
    print("\nDone.")
