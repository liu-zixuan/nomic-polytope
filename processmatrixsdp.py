"""
processmatrixsdp.py

Semidefinite programs (SDP) for calculating the optimal value achievable with process matrices.

The optimization problems are implemented using CVXPY and solved
numerically with the SCS solver.
"""

import numpy as np
import cvxpy as cp


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

    # Reorder subsystems so traced systems appear first
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
        # Compute inverse permutation to restore subsystem ordering
        p_inv = np.einsum(idx.reshape(dims[new_order]), new_order).flatten()

        # Reinsert traced systems as maximally mixed states
        return cp.kron(np.eye(d_trace) / d_trace, A)[p_inv][:, p_inv]


class TripartiteSDP:
    """
    SDP for optimizing linear functionals over
    tripartite process matrices.

    The SDP variable represents an operator that lies in the linear span
    of Choi operators corresponding to tripartite no-signaling channels.
    
    Given a performance operator Ω, the program computes the maximum
    value achievable by any valid process matrix.

    primal problem: Maximize Tr(SΩ) s.t. S is a process matrix
    dual problem: Minimize Tr(C)/D_in s.t. C >= Ω and C in Span{Choi(no-signaling channels)}
    """

    def __init__(self, dims):

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

        # Optimization variable: Hermitian operator on the full space
        C = cp.Variable((D, D), hermitian=True)

        # Performance operator Ω specifying the linear functional
        self.Omega = cp.Parameter((D, D), hermitian=True)

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
            
            # Linear constraints ensuring that C lies in the span of
            # Choi operators of tripartite no-signaling channels
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

    def optimize(self, Omega_val):
        """
        Solve the SDP for a given performance operator.

        Args:
            Omega_val (np.ndarray):
                Numerical value assigned to the performance operator Ω.

        Returns:
            float: Optimal value of the SDP.
        """

        # Assign the numerical value of the performance operator
        self.Omega.value = Omega_val

        # Solve the SDP using the SCS solver
        return self.problem.solve(solver=cp.SCS, eps=1e-6)