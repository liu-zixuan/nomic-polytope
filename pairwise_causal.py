"""Optimize LGYNP subject to pairwise four-node causal constraints.

The local instruments are the canonical instruments used in ``LGYNP.py``.
Setting 0 applies the identity qubit channel and records the fixed classical
label 0. Setting 1 measures in the computational basis, prepares the measured
qubit state, and records the classical label 1. Thus every party has input
dimension 2 and output dimension 4 (prepared qubit times setting label).

For every selected pair, regard the third party's setting as a global-past
input and its outcome as a global-future output. The resulting four-node
correlation must be a random mixture of the two orders of the selected pair.
Each unnormalised order branch has a past-setting-dependent weight and does
not signal from the later selected party to the earlier one. The script first
optimizes over this correlation polytope and then over correlations generated
by a valid tripartite process matrix.
"""

from itertools import combinations, product

import cvxpy as cp

from processmatrixsdp import (
    SINGLE_TRIGGER_EVENTS,
    SingleTriggerTripartitePrimalSDP,
)


TRIPLES = tuple(product((0, 1), repeat=3))

# Numerical results with the canonical instruments:
# Pairwise four-node causal correlation LP (HiGHS, optimal):
#   seven-term bound = 4.750000000; winning probability = 0.718750000.
# Pairwise four-node causal process SDP (full W, SCS eps=1e-7, optimal):
#   seven-term bound = 4.677930557; winning probability = 0.709741320.


def bit_index(bits):
    """Index a binary triple in lexicographic order 000,001,...,111."""
    return 4 * bits[0] + 2 * bits[1] + bits[2]


def winning_outcomes(settings):
    """Return x_k = a_k(a_j XOR a_l), the LGYNP winning output."""
    return tuple(
        settings[k] * (settings[(k + 1) % 3] ^ settings[(k + 2) % 3])
        for k in range(3)
    )


def full_probability_table(probability_vector):
    """Reconstruct all P(x|a) from the 19 nontrivial coordinates."""
    coordinate = {event: probability_vector[index]
                  for index, event in enumerate(SINGLE_TRIGGER_EVENTS)}
    rows = []
    for outcomes in TRIPLES:
        entries = []
        outcome = "".join(map(str, outcomes))
        for settings in TRIPLES:
            setting = "".join(map(str, settings))
            if any(settings[k] == 0 and outcomes[k] == 1 for k in range(3)):
                entries.append(0)
            elif outcomes != (0, 0, 0):
                entries.append(coordinate[(outcome, setting)])
            else:
                nonzero = [value for (x, a), value in coordinate.items()
                           if a == setting]
                entries.append(1 - sum(nonzero))
        rows.append(cp.hstack(entries))
    return cp.vstack(rows)


def valid_lazy_correlation_constraints(p):
    """Return positivity, normalization, and fixed lazy-outcome constraints."""
    constraints = [p >= 0]
    for settings in TRIPLES:
        a_index = bit_index(settings)
        constraints.append(cp.sum(p[:, a_index]) == 1)
        for outcomes in TRIPLES:
            if any(settings[k] == 0 and outcomes[k] == 1 for k in range(3)):
                constraints.append(p[bit_index(outcomes), a_index] == 0)
    return constraints


def four_node_causal_constraints(p):
    """Require the four-node correlation for every selected pair to be causal.

    For pair (i,j), a_k is a global-past setting and x_k is retained as a
    global-future outcome. Branch q_ij represents i before j. At fixed a_k,
    its total weight is independent of a_i,a_j, and the marginal of x_i after
    summing over x_j,x_k cannot depend on the later setting a_j.
    """
    constraints = valid_lazy_correlation_constraints(p)
    for party_i, party_j in combinations(range(3), 2):
        third = 3 - party_i - party_j
        order_ij = cp.Variable((8, 8), nonneg=True)
        order_ji = cp.Variable((8, 8), nonneg=True)
        constraints.append(p == order_ij + order_ji)

        for first, second, branch in ((party_i, party_j, order_ij),
                                      (party_j, party_i, order_ji)):
            for third_setting in (0, 1):
                reference = [0, 0, 0]
                reference[third] = third_setting
                branch_weight = cp.sum(branch[:, bit_index(reference)])

                for first_setting, second_setting in product((0, 1), repeat=2):
                    settings = [0, 0, 0]
                    settings[first] = first_setting
                    settings[second] = second_setting
                    settings[third] = third_setting
                    constraints.append(
                        cp.sum(branch[:, bit_index(settings)]) == branch_weight
                    )

                for first_setting, first_outcome in product((0, 1), repeat=2):
                    marginals = []
                    for second_setting in (0, 1):
                        settings = [0, 0, 0]
                        settings[first] = first_setting
                        settings[second] = second_setting
                        settings[third] = third_setting
                        marginals.append(sum(
                            branch[bit_index(outcomes), bit_index(settings)]
                            for outcomes in TRIPLES
                            if outcomes[first] == first_outcome
                        ))
                    constraints.append(marginals[0] == marginals[1])
    return constraints


def correlation_score(p):
    """Return the sum of the seven nonconstant LGYNP win probabilities."""
    return sum(
        p[bit_index(winning_outcomes(a)), bit_index(a)]
        for a in TRIPLES if a != (0, 0, 0)
    )


def pairwise_causal_correlation_bound():
    """Optimize LGYNP over correlations passing every four-node causal test."""
    probabilities = cp.Variable((8, 8))
    problem = cp.Problem(
        cp.Maximize(correlation_score(probabilities)),
        four_node_causal_constraints(probabilities),
    )
    try:
        value = problem.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
    except cp.error.SolverError:
        value = problem.solve(solver=cp.CLARABEL)
    return value, problem.status


def pairwise_causal_process_bound():
    """Optimize LGYNP over processes passing every four-node causal test."""
    sdp = SingleTriggerTripartitePrimalSDP(
        additional_constraints=lambda probability_vector:
            four_node_causal_constraints(full_probability_table(probability_vector))
    )
    # S_7 = 4 + c.P after eliminating the omitted 000 outcome at each setting.
    coefficients = [
        -1, -1, -1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        -1, -1, -1, -1, -1, -1, -1,
    ]
    value = 4 + sdp.optimize(
        coefficients, eps=1e-7, max_iters=3000, verbose=False
    )
    return value, sdp.problem.status


def main():
    """Solve the correlation LP and probability-constrained process SDP."""
    causal_value, causal_status = pairwise_causal_correlation_bound()
    print(f"Four-node causal correlation seven-term bound: {causal_value:.9f}")
    print(f"Four-node causal correlation winning probability: {(causal_value + 1) / 8:.9f}")
    print(f"correlation LP status: {causal_status}")

    process_value, process_status = pairwise_causal_process_bound()
    print(f"Four-node causal process seven-term bound: {process_value:.9f}")
    print(f"Four-node causal process winning probability: {(process_value + 1) / 8:.9f}")
    print(f"process SDP status: {process_status}")


if __name__ == "__main__":
    main()
