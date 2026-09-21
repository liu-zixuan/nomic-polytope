"""
processmatrixsdp.py

Semidefinite programs (SDP) for calculating the optimal value achievable with process matrices.

The optimization problems are implemented using CVXPY and solved
numerically with the SCS solver.

``TripartiteSDP`` retains the original dual formulation in the span of
no-signaling channels. ``SingleTriggerTripartitePrimalSDP`` is the primal
formulation for the paper's binary single-trigger scenario. It exposes the
19-component probability vector directly and accepts linear constraints on it.
"""

from itertools import combinations

import numpy as np
import cvxpy as cp
from scipy import sparse


def partial_trace(A, dims, trace_list, discard=True):
    """
    Compute the partial trace of an operator over selected subsystems.

    Optionally, instead of discarding the traced-out systems, they can be
    replaced with the maximally mixed state.

    Args:
        A (cp.Expression): Operator acting on the tensor product Hilbert space.
        dims (list[int]): Dimensions of the local subsystems.
        trace_list (list[int]): Indices of subsystems to trace out.
        discard (bool):
            If True, return the reduced operator after tracing out systems.
            If False, return an operator on the full space where the traced
            systems are replaced by maximally mixed states.

    Returns:
        cp.Expression: The resulting operator after the partial trace
        (either reduced or reconstructed on the full space).
    """

    # Convert subsystem dimensions to a NumPy array
    dims = np.array(dims)

    # Reorder subsystems so traced systems appear first. cvxpy's partial
    # trace can then act on the combined first factor in one operation.
    new_order = trace_list + [i for i in range(dims.size) if i not in trace_list]

    # Dimension of the traced subsystem
    d_trace = np.prod(dims[trace_list])

    # Total Hilbert space dimension
    D = np.prod(dims)

    # -------------------------------------------------------------
    # Build permutation indices to reorder subsystems of the operator
    # -------------------------------------------------------------

    idx = np.arange(D)

    # Compute permutation vector corresponding to subsystem reordering
    p = np.einsum(idx.reshape(dims), range(dims.size), new_order).flatten()

    # Apply permutation to reorder rows and columns of A
    A = cp.partial_trace(A[p][:, p], [d_trace, D // d_trace])

    # -------------------------------------------------------------
    # Return either reduced operator or reconstructed full operator
    # -------------------------------------------------------------

    if discard:
        # Standard partial trace: traced subsystems are removed
        return A
    else:
        # The inverse index permutation restores the original party order
        # after reinserting maximally mixed states in the traced factors.
        p_inv = np.einsum(idx.reshape(dims[new_order]), new_order).flatten()

        # Reinsert traced systems as maximally mixed states
        return cp.kron(np.eye(d_trace) / d_trace, A)[p_inv][:, p_inv]


def process_matrix_constraints(W, dims, include_positivity=True):
    """Return the tripartite primal validity constraints for W.

    Eq. (90) of arXiv:2101.08796v3, with trivial global past and future,
    says that for each nonempty party subset K, the reduced process has no
    component traceless on every output in K. Together with positivity and
    Tr(W) = d_O1 d_O2 d_O3, these are exactly process-matrix validity.
    Subsystems are ordered I1,O1,I2,O2,I3,O3.
    """
    if len(dims) != 6:
        raise ValueError("Expected dimensions I1,O1,I2,O2,I3,O3")

    constraints = [cp.real(cp.trace(W)) == np.prod(dims[1::2])]
    if include_positivity:
        constraints.insert(0, W >> 0)
    for size in (1, 2, 3):
        for parties in combinations(range(3), size):
            kept = [index for party in parties for index in (2 * party, 2 * party + 1)]
            removed = [index for index in range(6) if index not in kept]
            reduced = partial_trace(W, dims, removed) if removed else W
            reduced_dims = [dims[index] for index in kept]

            # The maps (I - replace_Ok) commute for distinct parties.
            # Applying them successively avoids enumerating 2^|K| terms.
            traceless_on_outputs = reduced
            for output_index in range(1, 2 * size, 2):
                traceless_on_outputs -= partial_trace(
                    traceless_on_outputs, reduced_dims, [output_index], discard=False
                )
            constraints.append(traceless_on_outputs == 0)
    return constraints


class TripartiteSDP:
    """
    SDP for optimizing linear functionals over
    tripartite process matrices.

    The dual variable lies in the linear span of Choi operators corresponding
    to tripartite no-signaling channels.
    
    Given a performance operator Ω, the program computes the maximum
    value achievable by any valid process matrix.

    primal problem: Maximize Tr(SΩ) s.t. S is a process matrix
    dual problem: Minimize Tr(C)/D_in s.t. C >= Ω and C in Span{Choi(no-signaling channels)}
    """

    def __init__(self, dims):
        """Build the dual SDP for subsystem order I1,O1,I2,O2,I3,O3."""

        # -------------------------------------------------------------
        # Problem dimensions
        # -------------------------------------------------------------

        # Total Hilbert space dimension
        D = np.prod(dims)
        # Dimension of the joint input space (used for normalization)
        D_in = dims[0] * dims[2] * dims[4]

        # -------------------------------------------------------------
        # SDP variables and parameters
        # -------------------------------------------------------------

        # Performance operator for the requested linear functional.
        self.Omega = cp.Parameter((D, D), hermitian=True)

        # Optimization variable: Hermitian operator on the full space
        C = cp.Variable((D, D), hermitian=True)

        def ptrace(trace_list):
            """
            Convenience wrapper for the linear map

                S → Tr_P(S) ⊗ I_P / d_P

            where P denotes the subsystem(s) listed in trace_list.
            This form is convenient for expressing linear constraints
            characterizing no-signaling channels.
            """
            return partial_trace(C, dims, trace_list, discard=False)

        # -------------------------------------------------------------
        # SDP constraints
        # -------------------------------------------------------------

        cons = [
            # Enforce C ≥ Ω
            C >> self.Omega,
            
            # Replacing O_k by identity/d_k equals replacing I_k,O_k.
            # These are the three no-signaling channel-span conditions.
            ptrace([1]) == ptrace([0, 1]),
            ptrace([3]) == ptrace([2, 3]),
            ptrace([5]) == ptrace([4, 5]),
        ]

        # -------------------------------------------------------------
        # Objective function
        # -------------------------------------------------------------

        # Dual problem: minimize the normalized trace of C
        # (normalization corresponds to the input Hilbert-space dimension)
        self.problem = cp.Problem(
            cp.Minimize(cp.real(cp.trace(C) / D_in)),
            cons
        )

    def optimize(self, Omega_val, **solver_options):
        """
        Solve the SDP for a given performance operator.

        Args:
            Omega_val (np.ndarray):
                Numerical value assigned to the performance operator Ω.

            **solver_options: Overrides for SCS, such as eps or max_iters.

        Returns:
            float: Optimal value of the SDP.
        """

        # Assign the numerical value of the performance operator
        self.Omega.value = Omega_val

        # Retain the original dual solver defaults; callers may override them.
        options = {"eps": 1e-4, "max_iters": 20000,
                   "verbose": False, **solver_options}
        return self.problem.solve(solver=cp.SCS, **options)


SINGLE_TRIGGER_DIMS = (2, 4) * 3

# (outcome, setting) pairs in the probability-vector order of main.tex.
SINGLE_TRIGGER_EVENTS = (
    ("100", "100"), ("010", "010"), ("001", "001"),
    ("100", "110"), ("010", "110"), ("110", "110"),
    ("010", "011"), ("001", "011"), ("011", "011"),
    ("001", "101"), ("100", "101"), ("101", "101"),
    ("100", "111"), ("010", "111"), ("001", "111"),
    ("110", "111"), ("011", "111"), ("101", "111"),
    ("111", "111"),
)


def _basis_projector(bit):
    projector = np.zeros((2, 2))
    projector[bit, bit] = 1
    return projector


def single_trigger_instruments():
    """Return the canonical lazy map and the two active instrument maps."""
    identity_vector = np.eye(2).reshape(4)
    lazy = np.kron(
        np.outer(identity_vector, identity_vector), _basis_projector(0)
    )
    active = tuple(
        np.kron(np.kron(_basis_projector(x), _basis_projector(x)),
                _basis_projector(1))
        for x in (0, 1)
    )
    return lazy, active


def single_trigger_probability_operators():
    """Return the 19 canonical Choi operators corresponding to P in main.tex."""
    lazy, active = single_trigger_instruments()
    operators = []
    for outcome, setting in SINGLE_TRIGGER_EVENTS:
        local = [lazy if setting[k] == "0" else active[int(outcome[k])]
                 for k in range(3)]
        operators.append(sparse.kron(
            sparse.kron(sparse.csr_matrix(local[0]), local[1], format="csr"),
            local[2], format="csr",
        ))
    return tuple(operators)


def process_subspace_map(dims):
    """Return a sparse map whose kernel is the process-matrix linear space."""
    dims = tuple(dims)
    dimension = int(np.prod(dims))
    identity = sparse.eye(dimension * dimension, format="csr")
    replacements = [replacement_superoperator(dims, subsystem)
                    for subsystem in range(6)]
    maps = []
    for size in (1, 2, 3):
        for parties in combinations(range(3), size):
            process_map = identity
            for party in range(3):
                if party not in parties:
                    process_map = replacements[2 * party] @ process_map
                    process_map = replacements[2 * party + 1] @ process_map
            for party in parties:
                process_map = (process_map
                               - replacements[2 * party + 1] @ process_map)
            process_map.eliminate_zeros()
            nonzero_rows = np.flatnonzero(process_map.getnnz(axis=1))
            maps.append(process_map[nonzero_rows])
    return sparse.vstack(maps, format="csr")


class SingleTriggerTripartitePrimalSDP:
    """Full primal SDP for the paper's simplest tripartite scenario.

    The variable is one unrestricted 512-by-512 process matrix W; no block
    diagonal ansatz is used. ``probability_vector`` contains the 19 nontrivial
    probabilities in ``SINGLE_TRIGGER_EVENTS`` order. If supplied,
    ``additional_constraints(probability_vector)`` must return CVXPY linear
    constraints. ``optimize(c)`` maximizes the arbitrary linear functional
    ``c @ probability_vector``.
    """

    def __init__(self, additional_constraints=None):
        dimension = int(np.prod(SINGLE_TRIGGER_DIMS))
        self.process_matrix = cp.Variable(
            (dimension, dimension), hermitian=True, name="W"
        )
        self.probability_vector = cp.Variable(19, name="P")
        self.linear_function = cp.Parameter(19, name="c")
        vectorized_process = cp.vec(self.process_matrix, order="F")

        constraints = [
            self.process_matrix >> 0,
            cp.real(cp.trace(self.process_matrix)) == 4 ** 3,
            process_subspace_map(SINGLE_TRIGGER_DIMS) @ vectorized_process == 0,
        ]
        for index, operator in enumerate(single_trigger_probability_operators()):
            # Tr(W M) = vec(M^T)^T vec(W). Keeping M sparse avoids forming
            # dense 512-by-512 constants for the 19 Born-rule equations.
            coefficients = operator.reshape((1, -1), order="C")
            constraints.append(
                self.probability_vector[index]
                == cp.real(coefficients @ vectorized_process)
            )
        if additional_constraints is not None:
            constraints.extend(additional_constraints(self.probability_vector))

        self.problem = cp.Problem(
            cp.Maximize(self.linear_function @ self.probability_vector),
            constraints,
        )

    def optimize(self, coefficients, **solver_options):
        """Maximize a 19-component linear function of the probability vector."""
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (19,):
            raise ValueError("The linear function must contain 19 coefficients")
        self.linear_function.value = coefficients
        options = {"eps": 1e-6, "verbose": False, **solver_options}
        return self.problem.solve(solver=cp.SCS, **options)


def replacement_superoperator(dims, subsystem):
    """Sparse matrix for X -> I_s/d_s tensor Tr_s(X), in original order."""
    dims = tuple(map(int, dims))
    dimension = int(np.prod(dims))
    local_dimension = dims[subsystem]
    rest_dims = dims[:subsystem] + dims[subsystem + 1:]
    rest_dimension = dimension // local_dimension
    rest_coordinates = np.array(np.unravel_index(
        np.arange(rest_dimension), rest_dims
    )).T

    def full_indices(local_value):
        coordinates = np.insert(rest_coordinates, subsystem, local_value, axis=1)
        return np.ravel_multi_index(coordinates.T, dims)

    pair_rows = np.repeat(np.arange(rest_dimension), rest_dimension)
    pair_columns = np.tile(np.arange(rest_dimension), rest_dimension)
    rows, columns = [], []
    for output_value in range(local_dimension):
        output_indices = full_indices(output_value)
        output_rows = output_indices[pair_rows]
        output_columns = output_indices[pair_columns]
        output_vector_indices = output_rows + dimension * output_columns
        for input_value in range(local_dimension):
            input_indices = full_indices(input_value)
            input_rows = input_indices[pair_rows]
            input_columns = input_indices[pair_columns]
            rows.append(output_vector_indices)
            columns.append(input_rows + dimension * input_columns)
    rows = np.concatenate(rows)
    columns = np.concatenate(columns)
    data = np.full(rows.size, 1 / local_dimension)
    return sparse.csr_matrix(
        (data, (rows, columns)), shape=(dimension * dimension,) * 2
    )
