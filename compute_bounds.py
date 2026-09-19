"""Evaluate lower bounds for each nomic facet family in the paper.

Rows u in facet_families.txt represent (1, P) . u >= 0, with the 19
coordinates P ordered as in main.tex. Minimize over deterministic general,
causal and nomic vertices, the supplied classical-process (PC) superset,
and the tripartite quantum-process (QP) SDP. A negative value violates
that nomic inequality. Running all 469 QP SDPs is computationally costly.
"""

from fractions import Fraction
from qp_bound_sdp import qp_bound
from itertools import product

nomic_vertices = []  # Load nomic vertices in homogeneous (1, P) form.
with open("nomic_vertices.ext") as f:
    for line in f:
        parts = line.split()
        
        if len(parts) == 20:
            try:
                nomic_vertices.append([int(x) for x in parts])
            except ValueError:
                pass  # skip lines that contain non-numeric values


causal_vertices = []  # Load causal vertices in the same coordinates.
with open("causal_vertices.ext") as f:
    for line in f:
        parts = line.split()
        
        if len(parts) == 20:
            try:
                causal_vertices.append([int(x) for x in parts])
            except ValueError:
                pass  


# The source file declares 65 columns, but its projected data rows have
# 20 entries (1, P). Select those rows; its header is not a data row.
pc_vertices_superset = []
with open("pc_vertices_superset.ext") as f:
    for line in f:
        parts = line.split()
        
        if len(parts) == 20:
            try:
                pc_vertices_superset.append([float(Fraction(x)) for x in parts] )
            except ValueError:
                pass  


general_vertices = []
# For each setting a, choose one deterministic allowed outcome. There are
# 1 + 1 + 1 + 2 + 2 + 2 + 3 = 12 free bits across the seven nonzero
# settings, hence 2^12 vertices of the unrestricted probability polytope.
for c in product([0, 1], repeat=12):

    (x001_C, x010_B, x011_B, x011_C, x100_A, x101_A,
     x101_C, x110_A, x110_B, x111_A, x111_B, x111_C) = c

    # Each product is the indicator of one outcome x at its setting a.
    # The coordinate order is the 19-entry P vector from main.tex.
    general_vertices.append([
        1, x100_A, x010_B, x001_C,
        x110_A*(1-x110_B), (1-x110_A)*x110_B, x110_A*x110_B,
        x011_B*(1-x011_C), (1-x011_B)*x011_C, x011_B*x011_C,
        (1-x101_A)*x101_C, x101_A*(1-x101_C), x101_A*x101_C,
        x111_A*(1-x111_B)*(1-x111_C),
        (1-x111_A)*x111_B*(1-x111_C),
        (1-x111_A)*(1-x111_B)*x111_C,
        x111_A*x111_B*(1-x111_C),
        (1-x111_A)*x111_B*x111_C,
        x111_A*(1-x111_B)*x111_C,
        x111_A*x111_B*x111_C,
    ])

def nomic_bound(u):
    """Minimum of (1, P) . u over nomic (DC) vertices."""
    all_values_nomic = []
    for v in nomic_vertices:
        all_values_nomic.append(sum(x * y for x, y in zip(v, u)))
    return min(all_values_nomic)

def causal_bound(u):
    """Minimum of (1, P) . u over causal vertices."""
    all_values_causal = []
    for v in causal_vertices:
        all_values_causal.append(sum(x * y for x, y in zip(v, u)))
    return min(all_values_causal)

def pc_bound(u):
    """Minimum over the supplied PC vertex superset."""
    all_values_pc = []
    for v in pc_vertices_superset:
        all_values_pc.append(sum(x * y for x, y in zip(v, u)))
    return min(all_values_pc)

def gen_bound(u):
    """Minimum over all unrestricted deterministic correlations."""
    all_values_gen = []
    for v in general_vertices:
        all_values_gen.append(sum(x * y for x, y in zip(v, u)))
    return min(all_values_gen)

if __name__ == "__main__":
    # Read the 469 representative inequalities in the paper's order.
    families = []

    with open("facet_families.txt") as f:
        for line in f:
            parts = line.split()
            
            if len(parts) == 20:
                try:
                    numbers = [int(x) for x in parts]  
                    families.append(numbers)
                except ValueError:
                    pass  

    bounds = []

    for u in families:
        print(u)
        values = [gen_bound(u), qp_bound(u), pc_bound(u), nomic_bound(u), causal_bound(u)]
        print(values)
        bounds.append(values)

    print(bounds)
