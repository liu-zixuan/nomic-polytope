"""Compute the quantum-process lower bound of a paper-coordinate inequality.

For a 20-vector u representing (1, P) · u >= 0, build Choi operators
for the 19 nonconstant coordinates of P. A tripartite process-matrix SDP
maximizes the negative coefficient sum; u[0] minus that maximum is the
minimum inequality value. The instruments use the single-trigger
reduction described in main.tex.
"""

from processmatrixsdp import TripartiteSDP
import numpy as np
import functools


def std_basis_vec(d, i):
    '''
    Return the i-th standard basis vector in dimension d.
    '''
    v = np.zeros(d)
    v[i] = 1
    return v


def std_basis_mat(d, i, j=None):
    '''
    Return a standard matrix basis element in dimension d.

    If j is specified, returns |i⟩⟨j|.
    Otherwise returns the projector |i⟩⟨i|.
    '''
    M = np.zeros((d, d))

    if j is not None:
        M[i, j] = 1
        return M

    M[i, i] = 1
    return M


def outer_projector(v, w=None):
    '''
    Construct the outer product |v⟩⟨w|.

    If w is omitted, returns the projector |v⟩⟨v|.
    Works for vectors with tensor structure.
    '''
    if w is None:
        return np.einsum(v, range(v.ndim), v.conjugate(), range(v.ndim, v.ndim * 2))

    return np.einsum(v, range(v.ndim), w.conjugate(), range(v.ndim, v.ndim + w.ndim))


def kron_all(*operators):
    '''
    Tensor (Kronecker) product of multiple matrices.
    '''
    return functools.reduce(np.kron, operators)


# Identity operator on a qubit
identity_qubit = np.eye(2)


# ------------------------------------------------------------------
# Local instruments for the LGYNP game
# ------------------------------------------------------------------

# Instrument corresponding to input x = 0 ("lazy" operation):
# identity gate with a classical label
lazy_instrument = kron_all(
    outer_projector(identity_qubit.flatten()),
    std_basis_mat(2, 0)
)

# Instruments corresponding to input x = 1:
# projective measurement in the computational basis with a classical label
active_instrument = [
    kron_all(
        std_basis_mat(2, a),
        std_basis_mat(2, a),
        std_basis_mat(2, 1)
    )
    for a in [0, 1]
]

# Order matches the 19 probabilities P in main.tex and the generated
# nomic vertices. There is no component for setting 000: its outcome is
# certain, and that coordinate was removed from P.
performance_operator_components = [
    kron_all(active_instrument[1], lazy_instrument, lazy_instrument),
    kron_all(lazy_instrument, active_instrument[1], lazy_instrument),
    kron_all(lazy_instrument, lazy_instrument, active_instrument[1]),
    kron_all(active_instrument[1], active_instrument[0], lazy_instrument),
    kron_all(active_instrument[0], active_instrument[1], lazy_instrument),
    kron_all(active_instrument[1], active_instrument[1], lazy_instrument),
    kron_all(lazy_instrument, active_instrument[1], active_instrument[0]),
    kron_all(lazy_instrument, active_instrument[0], active_instrument[1]),
    kron_all(lazy_instrument, active_instrument[1], active_instrument[1]),
    kron_all(active_instrument[0], lazy_instrument, active_instrument[1]),
    kron_all(active_instrument[1], lazy_instrument, active_instrument[0]),
    kron_all(active_instrument[1], lazy_instrument, active_instrument[1]),
    kron_all(active_instrument[1], active_instrument[0], active_instrument[0]),
    kron_all(active_instrument[0], active_instrument[1], active_instrument[0]),
    kron_all(active_instrument[0], active_instrument[0], active_instrument[1]),
    kron_all(active_instrument[1], active_instrument[1], active_instrument[0]),
    kron_all(active_instrument[0], active_instrument[1], active_instrument[1]),
    kron_all(active_instrument[1], active_instrument[0], active_instrument[1]),
    kron_all(active_instrument[1], active_instrument[1], active_instrument[1])
]


def qp_bound(u):
    """Return the numerical QP lower bound for a 20-coefficient vector u."""

    if len(u) != 20:
        raise ValueError("An inequality must have 20 coefficients: (1, P) · u")

    coeffs = [-c for c in u[1:]]
    performance_operator = sum(c * M for c, M in zip(coeffs, performance_operator_components))

    # ------------------------------------------------------------------
    # Solve the SDP
    # ------------------------------------------------------------------

    # Instantiate the tripartite process-matrix SDP.
    # Each party has input dimension 2 and output dimension 4.
    sdp_instance = TripartiteSDP([2, 4] * 3)

    # Compute the optimal value of the linear functional defined by the performance operator.
    # Duality: min (u_0 + sum_j u_j P_j) = u_0 - max sum_j (-u_j) P_j.
    bound = u[0] - sdp_instance.optimize(performance_operator)

    return bound
