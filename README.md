# Nomic polytope and quantum bounds

Code and data accompanying *The simplest nomic inequalities and their violations*. The scenario has three parties. Each party receives a binary setting `a_k`; its outcome `x_k` is fixed to zero for `a_k = 0` and is binary for `a_k = 1`. The scripts characterize the deterministically consistent (nomic, `DC`) polytope and estimate quantum process (`QP`) bounds using semidefinite programming.

## Coordinates and inequality convention

The 19 entries of `P` follow the order displayed in the paper's “Nomic polytope in the simplest tripartite scenario” section: `P(100|100), P(010|010), P(001|001), ... , P(111|111)`. A vertex file row is `(1, P)`. A facet or family row is a 20-component vector `u` representing `(1, P) · u >= 0`. All bound functions minimize this expression; a negative bound is a violation. The `QP` values are numerical SDP estimates, subject to solver tolerance.

## Requirements

- Python 3.10 or later. `generate_nomic_vertices.py` and `process_facets.py` use only the standard library.
- `numpy` and `cvxpy` with its SCS solver for quantum bounds (`LGYNP.py`, `qp_bound_sdp.py`, and `compute_bounds.py`). Install with `python3 -m pip install numpy cvxpy scs` in your preferred virtual environment.
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

The checked-in output contains 20,726 facets. Then run:

```sh
python3 process_facets.py
```

This writes `facet_families.txt`, containing 469 representatives under party permutations and output flips. Its exact orbit search can take substantial time. The repository includes the generated `nomic_vertices.ext`, `nomic_facets.ine`, and `facet_families.txt` so that the SDP steps can use them directly.

## Reproduce bounds

```sh
python3 LGYNP.py
python3 compute_bounds.py
```

`LGYNP.py` prints the maximum LGYNP winning probability. The paper's seven-term form has quantum value about 4.7522; the game includes the always-winning `000` setting, so the probability is `(4.7522 + 1) / 8`, about 0.7190.

`compute_bounds.py` prints, for each of the 469 family representatives, the inequality vector followed by `[general, QP, PC, DC, causal]` lower bounds. It reads `causal_vertices.ext` (680 vertices) and `pc_vertices_superset.ext` (449,124 projected classical-process rows), as well as the nomic and family files. The `PC` file is a *superset* of the projected `PC` vertices; minimizing over it supplies the bound used in the paper. This full run is resource intensive: it solves an SDP for every family and scans the large classical-process file for every bound. Use `LGYNP.py` for a quick single-game SDP.

`processmatrixsdp.py` provides the dual tripartite process-matrix SDP shared by the bound scripts. The `QP` calculation fixes the instruments allowed by the paper's single-trigger reduction; it does not optimize instruments separately. The scripts use SCS with `eps=1e-6`, so small discrepancies from the paper's rounded values are expected.

## Data-file note

The supplied `pc_vertices_superset.ext` has a cdd-style header declaring 65 columns, while its data rows have 20 columns `(1, P)`. `compute_bounds.py` reads the 20-component rows directly. The file is supplied as an input dataset; its generation procedure is not included in this repository.
