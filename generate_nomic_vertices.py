"""Enumerate the 744 vertices of the tripartite nomic polytope.

Enumerate binary process functions omega with one fixed point under every
local operation. Each valid function gives the deterministic correlation
x_k = a_k i_k, expressed in the 19-coordinate order of main.tex.
Running this file writes the cdd V-representation nomic_vertices.ext.
"""

from itertools import product

TRIPLES = list(product([0, 1], repeat=3))

LOCAL_OPERATIONS = [
    {
        0: (v >> 0) & 1,
        1: (v >> 1) & 1,
    }
    for v in range(4)
]

def fixed_point_count(wA, wB, wC, hA, hB, hC):
    """Count solutions of o_k = h_k(omega_k(o)) for these local maps."""
    return sum(
        x == (hA[wA[x]], hB[wB[x]], hC[wC[x]])
        for x in TRIPLES
    )

def is_valid_process_function(wA, wB, wC):
    '''Determine if a tripartite function (wA: O_A x O_B x O_C -> I_A, wB: O_A x O_B x O_C -> I_B, wC: O_A x O_B x O_C -> I_C)
    with binary inputs I_A, I_B, I_C and outputs O_A, O_B, O_C, is a valid process function by composing it with all choices of local
    operations hA: I_A -> O_A, hB: I_B -> O_B, hC: I_C -> O_C and checking if there is exactly one fixed point'''
    for hA, hB, hC in product(LOCAL_OPERATIONS, repeat=3):
        if fixed_point_count(wA, wB, wC, hA, hB, hC) != 1:
            return False
    return True

def delta(a, b):
    """Kronecker delta for tuples of outcomes."""
    return 1 if a == b else 0

def generate_vertices():
    """Return the 19-coordinate vertices from all valid process functions."""
    valid_process_functions = []

    # A valid process cannot depend on its own outgoing bit; each component
    # is a Boolean function of the other two bits (16 choices per party).
    for wA_v in range(16):
        for wB_v in range(16):
            for wC_v in range(16):

                wA = {
                    (0,0,0): (wA_v >> 0) & 1,
                    (1,0,0): (wA_v >> 0) & 1,
                    (0,0,1): (wA_v >> 1) & 1,
                    (1,0,1): (wA_v >> 1) & 1,
                    (0,1,0): (wA_v >> 2) & 1,
                    (1,1,0): (wA_v >> 2) & 1,
                    (0,1,1): (wA_v >> 3) & 1,
                    (1,1,1): (wA_v >> 3) & 1
                }

                wB = {
                    (0,0,0): (wB_v >> 0) & 1,
                    (0,1,0): (wB_v >> 0) & 1,
                    (0,0,1): (wB_v >> 1) & 1,
                    (0,1,1): (wB_v >> 1) & 1,
                    (1,0,0): (wB_v >> 2) & 1,
                    (1,1,0): (wB_v >> 2) & 1,
                    (1,0,1): (wB_v >> 3) & 1,
                    (1,1,1): (wB_v >> 3) & 1
                }

                wC = {
                    (0,0,0): (wC_v >> 0) & 1,
                    (0,0,1): (wC_v >> 0) & 1,
                    (0,1,0): (wC_v >> 1) & 1,
                    (0,1,1): (wC_v >> 1) & 1,
                    (1,0,0): (wC_v >> 2) & 1,
                    (1,0,1): (wC_v >> 2) & 1,
                    (1,1,0): (wC_v >> 3) & 1,
                    (1,1,1): (wC_v >> 3) & 1
                }

                if is_valid_process_function(wA, wB, wC):
                    valid_process_functions.append([wA,wB,wC])

    vertex_list = []

    for [wA,wB,wC] in valid_process_functions:

        # With o = a and x_k = a_k i_k, the unique fixed point reduces to
        # i = omega(a). Lazy parties (a_k = 0) always return x_k = 0.
        x_of_a = {
            (0,0,0): (0,0,0),
            (0,0,1): (0,0,wC[(0,0,1)]),
            (0,1,0): (0,wB[(0,1,0)],0),
            (0,1,1): (0,wB[(0,1,1)],wC[(0,1,1)]),
            (1,0,0): (wA[(1,0,0)],0,0),
            (1,0,1): (wA[(1,0,1)],0,wC[(1,0,1)]),
            (1,1,0): (wA[(1,1,0)],wB[(1,1,0)],0),
            (1,1,1): (wA[(1,1,1)],wB[(1,1,1)],wC[(1,1,1)])
        }

        # Coordinates follow the 19-term vector P in main.tex exactly.
        vertex_list.append(
            [delta(x_of_a[(1,0,0)],(1,0,0)),delta(x_of_a[(0,1,0)],(0,1,0)),delta(x_of_a[(0,0,1)],(0,0,1)),
             delta(x_of_a[(1,1,0)],(1,0,0)),delta(x_of_a[(1,1,0)],(0,1,0)),delta(x_of_a[(1,1,0)],(1,1,0)),
             delta(x_of_a[(0,1,1)],(0,1,0)),delta(x_of_a[(0,1,1)],(0,0,1)),delta(x_of_a[(0,1,1)],(0,1,1)),
             delta(x_of_a[(1,0,1)],(0,0,1)),delta(x_of_a[(1,0,1)],(1,0,0)),delta(x_of_a[(1,0,1)],(1,0,1)),
             delta(x_of_a[(1,1,1)],(1,0,0)),delta(x_of_a[(1,1,1)],(0,1,0)),delta(x_of_a[(1,1,1)],(0,0,1)),
             delta(x_of_a[(1,1,1)],(1,1,0)),delta(x_of_a[(1,1,1)],(0,1,1)),delta(x_of_a[(1,1,1)],(1,0,1)),
             delta(x_of_a[(1,1,1)],(1,1,1))
             ]
        )

    return vertex_list


def write_cdd_vertices(vertex_list, path="nomic_vertices.ext"):
    """Write homogeneous vertices (1, P) in cdd V-representation format."""
    # Write homogeneous rows (1, P) in cdd V-representation format.
    with open(path, "w") as f:
        f.write("V-representation\n")
        f.write("begin\n")
        f.write(f"{len(vertex_list)} 20 integer\n")

        for v in vertex_list:
            row = " ".join(str(x) for x in v)
            f.write(f"1 {row}\n")

        f.write("end\n")


if __name__ == "__main__":
    write_cdd_vertices(generate_vertices())
