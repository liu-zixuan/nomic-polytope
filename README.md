# Nomic polytope and quantum bounds

Code and data accompanying *The simplest nomic inequalities and their violations* (by Julian Wechs, Nasra Daher Ahmed, Zixuan Liu, and Ravi Kunjwal). The scenario has three parties. Each party receives a binary setting `a_k`; its outcome `x_k` is fixed to zero for `a_k = 0` and is binary for `a_k = 1`. The scripts characterize the deterministically consistent (nomic, `DC`) polytope and estimate quantum process (`QP`) bounds using semidefinite programming.

## Coordinates and inequality convention

The 19 entries of `P` follow the order displayed in the paper's “Nomic polytope in the simplest tripartite scenario” section: `P(100|100), P(010|010), P(001|001), ... , P(111|111)`. A vertex file row is `(1, P)`. A facet or family row is a 20-component vector `u` representing `(1, P) · u >= 0`. All bound functions minimize this expression; a negative bound is a violation. The `QP` values are numerical SDP estimates, subject to solver tolerance.

## Requirements

- Python 3.10 or later. `generate_nomic_vertices.py` and `classify_nomic_facets.py` use only the standard library.
- `numpy`, `scipy`, and `cvxpy` with its SCS solver for quantum bounds (`LGYNP.py`, `qp_bound_sdp.py`, `compute_bounds.py`, and `pairwise_causal.py`). Install with `python3 -m pip install numpy scipy cvxpy scs` in your preferred virtual environment. The pairwise-causal correlation LP uses SciPy's HiGHS interface.
- [cddlib](https://github.com/cddlib/cddlib) with the `cddexec` command for regenerating the halfspace representation. The checked-in facet file makes cddlib optional for the other scripts.

Run commands from the repository root. These are standalone scripts; no package installation is needed.

## Reproduce the polytope

```sh
python3 generate_nomic_vertices.py
```

This enumerates all binary tripartite process functions with a unique fixed point for every choice of local operations and writes `nomic_vertices.ext` in cdd's V-representation format. It should contain 744 data rows with 19 probability coordinates. To regenerate the halfspace representation with cddlib 0.94m:

```sh
cddexec --rep < nomic_vertices.ext > nomic_facets.ine
```

The checked-in output contains 20,726 facets. To classify them and regenerate both `facet_families.txt` and the [LaTeX catalogue](nomic_facet_classes.tex), run:

```sh
python3 classify_nomic_facets.py
```

The catalogue lists each orbit size and representative. It interprets every facet as an exact upper bound on a seven-setting weighted winning game, with the referee's acceptance rule specified at the start of the note. It also checks which classes are single-target lazy guessing games: class 89 is the only one testing all seven settings, and five classes give partial games. Enumerating all 16 Boolean rules applied symmetrically to the neighbors' settings shows that the AND and OR optimal inequalities are not facets; XOR and XNOR give class 89. Three classes are event positivity constraints. The script checks that all 20,726 input facets lie in complete symmetry orbits and that its representatives match `facet_families.txt`. It uses only the standard library. To compile the note, run `pdflatex nomic_facet_classes.tex` twice (the second pass settles longtable widths).

## Reproduce bounds

```sh
python3 LGYNP.py
python3 compute_bounds.py
```

`LGYNP.py` prints the maximum LGYNP winning probability. The paper's seven-term form has quantum value about 4.7522; the game includes the always-winning `000` setting, so the probability is `(4.7522 + 1) / 8`, about 0.7190.

`compute_bounds.py` prints, for each of the 469 family representatives, the inequality vector followed by `[general, QP, PC, DC, causal]` lower bounds. It reads `causal_vertices.ext` (680 vertices) and `pc_vertices_superset.ext` (449,124 projected classical-process rows), as well as the nomic and family files. The `PC` file is a *superset* of the projected `PC` vertices; minimizing over it supplies the bound used in the paper. This full run is resource intensive: it solves an SDP for every family and scans the large classical-process file for every bound. Use `LGYNP.py` for a quick single-game SDP.

`processmatrixsdp.py` provides the dual tripartite process-matrix SDP shared by the bound scripts. The `QP` calculation fixes the instruments allowed by the paper's single-trigger reduction; it does not optimize instruments separately. The scripts use SCS with `eps=1e-6`, so small discrepancies from the paper's rounded values are expected.

On the `unitary_process_relaxation` branch, run `python3 pairwise_causal.py` for two LGYNP calculations. The script uses the canonical instruments from `LGYNP.py`: the lazy setting is the identity qubit channel with classical label 0, while the active setting measures and reprepares the computational-basis outcome with label 1. For every selected pair, the correlation constraints treat the third setting as a global-past input and retain the third outcome as a global-future output; this four-node correlation must be a random mixture of the two orders of the selected parties. The primal SDP imposes the same constraints on a full valid process matrix, with no block-diagonal restriction. `SingleTriggerTripartitePrimalSDP` in `processmatrixsdp.py` exposes the paper's 19-component probability vector, accepts additional linear constraints on that vector, and maximizes any supplied 19-component linear function. The correlation LP seven-term bound is 4.75; the constrained process value is recorded in `pairwise_causal.py` with its solver tolerance.

## Data-file note

The supplied `pc_vertices_superset.ext` has a cdd-style header declaring 65 columns, while its data rows have 20 columns `(1, P)`. `compute_bounds.py` reads the 20-component rows directly. The file is supplied as an input dataset; its generation procedure is not included in this repository.
