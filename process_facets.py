"""Group cdd facets of the nomic polytope into symmetry families.

Read the 20-coefficient inequalities (1, P) . u >= 0 from
nomic_facets.ine. Clear half-integer coefficients, then identify facets
related by party permutations and binary-output relabelings. Write one
representative per orbit to facet_families.txt (469 rows for the paper data).
"""

from fractions import Fraction

#read in the list of facets from the CDD output file
facets_int = []

# cdd in double arithmetic may emit half-integer coefficients; scaling
# preserves the oriented inequality and makes orbit comparisons exact.
# Rescale such inequalities by a factor of 2 so that all coefficients are integers.
with open("nomic_facets.ine") as f:
    for line in f:
        parts = line.split()

        if len(parts) == 20:
            try:
                row = [Fraction(x) for x in parts]
            except ValueError:
                continue

            denominators = {x.denominator for x in row}

            if denominators <= {1}:
                facets_int.append([int(x) for x in row])
            elif denominators <= {1, 2}:
                facets_int.append([int(2*x) for x in row])
            else:
                raise ValueError(f"Unexpected non-half-integer row: {row}")

#group inequalities into families

def SwapAB(v): #transformation that the probability vector undergoes when relabeling A <-> B
    return [v[0],v[2],v[1],v[3],v[5],v[4],v[6],v[11],v[10],v[12],v[8],v[7],v[9],v[14],v[13],v[15],v[16],v[18],v[17],v[19]]

def SwapAC(v): #transformation that the probability vector undergoes when relabeling A <-> C
    return [v[0],v[3],v[2],v[1],v[8],v[7],v[9],v[5],v[4],v[6],v[11],v[10],v[12],v[15],v[14],v[13],v[17],v[16],v[18],v[19]]

def SwapBC(v): #transformation that the probability vector undergoes when relabeling B <-> C
    return [v[0],v[1],v[3],v[2],v[11],v[10],v[12],v[8],v[7],v[9],v[5],v[4],v[6],v[13],v[15],v[14],v[18],v[17],v[16],v[19]]

def RelabelA(v): #Relabeling A's output (0 <-> 1)
    return [v[0]+v[1]+v[4]+v[11]+v[13],-v[1],v[2],v[3],-v[4],-v[4]+v[6],-v[4]+v[5],v[7],v[8],v[9],-v[11]+v[12],-v[11],v[10]-v[11],-v[13],-v[13]+v[16],-v[13]+v[18],
            -v[13]+v[14],-v[13]+v[19],-v[13]+v[15],-v[13]+v[17]]

def RelabelB(v): #Relabeling B's output (0 <-> 1)
    return [v[0]+v[2]+v[5]+v[7]+v[14],v[1],-v[2],v[3],-v[5]+v[6],-v[5],v[4]-v[5],-v[7],-v[7]+v[9],-v[7]+v[8],v[10],v[11],v[12],-v[14]+v[16],-v[14],-v[14]+v[17],
            v[13]-v[14],-v[14]+v[15],-v[14]+v[19],-v[14]+v[18]]

def RelabelC(v): #Relabeling C's output (0 <-> 1)
    return [v[0]+v[3]+v[8]+v[10]+v[15],v[1],v[2],-v[3],v[4],v[5],v[6],-v[8]+v[9],-v[8],v[7]-v[8],-v[10],-v[10]+v[12],
            -v[10]+v[11],-v[15]+v[18],-v[15]+v[17],-v[15],-v[15]+v[19],+v[14]-v[15],+v[13]-v[15],-v[15]+v[16]]

def symmetry_orbit(u):
    """Return the coefficient vectors related by the paper's symmetries.

    Party permutations act on coordinates of P. Output flips act affinely
    on P, so each flip also changes the constant coefficient u[0].
    """
    # Six permutations of parties, followed by eight independent choices
    # of which active-measurement outcomes to flip.
    party_permutations = [u, SwapBC(u), SwapAB(u), SwapAC(SwapAB(u)),
                          SwapAC(u), SwapAB(SwapAC(u))]
    orbit = []
    for v in party_permutations:
        orbit.extend((v, RelabelA(v), RelabelB(v), RelabelC(v),
                      RelabelA(RelabelB(v)), RelabelA(RelabelC(v)),
                      RelabelB(RelabelC(v)), RelabelA(RelabelB(RelabelC(v)))))
    return orbit


def IsEquivalent(u1, u2):
    """Whether two facet vectors are related by party/output relabeling."""
    return list(u2) in symmetry_orbit(u1)


# A canonical orbit key avoids repeating up to 48 transformations for each
# comparison with an existing family. Keep the first cdd row as representative
# so the output order and coefficients match the supplied paper data.
families = []
seen_orbits = set()
for u in facets_int:
    key = min(tuple(v) for v in symmetry_orbit(u))
    if key not in seen_orbits:
        seen_orbits.add(key)
        families.append(u)

with open("facet_families.txt", "w") as f: #write families to a text file
    for fam in families:
        f.write(" ".join(str(x) for x in fam) + "\n")
