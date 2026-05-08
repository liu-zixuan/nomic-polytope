from fractions import Fraction

#read in the list of facets from the CDD output file
facets = []
facets_int = []
facets_half_integer = []

# Some facet inequalities contain half-integer coefficients.
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
                facets_half_integer.append(row)
                facets_int.append([int(2*x) for x in row])
            else:
                raise ValueError(f"Unexpected non-half-integer row: {row}")

print(facets_half_integer)
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

def IsEquivalent(v1,v2): #Generate all representations of v1 and check if v2 is among them
    party_relabelings_v1 = [v1,SwapBC(v1),SwapAB(v1),SwapAC(SwapAB(v1)),SwapAC(v1),SwapAB(SwapAC(v1))]
    equivalenceClass = []
    for v in party_relabelings_v1:
        equivalenceClass.append(v)
        equivalenceClass.append(RelabelA(v))
        equivalenceClass.append(RelabelB(v))
        equivalenceClass.append(RelabelC(v))
        equivalenceClass.append(RelabelA(RelabelB(v)))
        equivalenceClass.append(RelabelA(RelabelC(v)))
        equivalenceClass.append(RelabelB(RelabelC(v)))
        equivalenceClass.append(RelabelA(RelabelB(RelabelC(v))))
    if v2 in equivalenceClass:
        return True
    else:
        return False

families = []

for f in facets_int: #generate the list of families, containing one representant of each class
    if not any(IsEquivalent(f, fam) for fam in families):
        families.append(f)

with open("facet_families.txt", "w") as f: #write families to a text file
    for fam in families:
        f.write(" ".join(str(x) for x in fam) + "\n")

