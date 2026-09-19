"""Compute the paper's quantum bound for Lazy Guess Your Neighbour's Parity.

For settings a_1,a_2,a_3, the winning outcomes are
x_k = a_k(a_j XOR a_l). We build the seven nonconstant winning-event
Choi operators, maximize their sum over tripartite process matrices,
then add the always-winning 000 setting and divide by eight.
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

# Instrument for setting a_k = 0 (the "lazy" operation):
# identity gate with a classical label
lazy_instrument = kron_all(
    outer_projector(identity_qubit.flatten()),
    std_basis_mat(2, 0)
)

# Instruments for setting a_k = 1:
# projective measurement in the computational basis with a classical label
active_instrument = [
    kron_all(
        std_basis_mat(2, a),
        std_basis_mat(2, a),
        std_basis_mat(2, 1)
    )
    for a in [0, 1]
]


# ------------------------------------------------------------------
# Performance operator for the LGYNP winning condition
# ------------------------------------------------------------------

# Success condition: x_k = a_k(a_j ⊕ a_l).
# The all-zero setting is certain to win and is added as 1/8 below.

performance_operator = \
    kron_all(active_instrument[0], active_instrument[0], active_instrument[0]) + \
    kron_all(active_instrument[1], active_instrument[1], lazy_instrument) + \
    kron_all(lazy_instrument, active_instrument[1], active_instrument[1]) + \
    kron_all(active_instrument[1], lazy_instrument, active_instrument[1]) + \
    kron_all(active_instrument[0], lazy_instrument, lazy_instrument) + \
    kron_all(lazy_instrument, active_instrument[0], lazy_instrument) + \
    kron_all(lazy_instrument, lazy_instrument, active_instrument[0])


# ------------------------------------------------------------------
# Solve the SDP
# ------------------------------------------------------------------

# Instantiate the tripartite process-matrix SDP.
# Each party has input dimension 2 and output dimension 4.
sdp_instance = TripartiteSDP([2, 4] * 3)

# Compute the optimal value of the linear functional defined by the performance operator.
# The seven modeled settings each have weight 1/8; 000 contributes 1/8.
p_succ = sdp_instance.optimize(performance_operator) / 8 + 1 / 8

# Print the optimal quantum success probability
print(p_succ)  # 4.752/8 + 1/8 ≈ 0.719
