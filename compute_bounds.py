from fractions import Fraction
from qp_bound_sdp import qp_bound
from generate_nomic_vertices import delta
from itertools import product

nomic_vertices = []  #load nomic vertices          
with open("nomic_vertices.ext") as f:
    for line in f:
        parts = line.split()
        
        if len(parts) == 20:
            try:
                nomic_vertices.append([int(x) for x in parts])
            except ValueError:
                pass  # skip lines that contain non-numeric values


causal_vertices = []  #load causal vertices         
with open("causal_vertices.ext") as f:
    for line in f:
        parts = line.split()
        
        if len(parts) == 20:
            try:
                causal_vertices.append([int(x) for x in parts])
            except ValueError:
                pass  


pc_vertices_superset = []    #load pc vertices         
with open("pc_vertices_superset.ext") as f:
    for line in f:
        parts = line.split()
        
        if len(parts) == 20:
            try:
                pc_vertices_superset.append([float(Fraction(x)) for x in parts] )
            except ValueError:
                pass  


general_vertices = [] #generate all 2^12 vertices of the probability polytope
df = list(product([0, 1], repeat=12))
for c in df:

    s1 = c[0]
    s2 = c[1]
    s3 = c[2]
    s4 = c[3]
    s5 = c[4]
    s6 = c[5]
    s7 = c[6]
    s8 = c[7]
    s9 = c[8]
    s10 = c[9]
    s11 = c[10]
    s12 = c[11]

    general_vertices.append([1,s5,s2,s1,s8*(1-s9),(1-s8)*s9,s8*s9,s3*(1-s4),(1-s3)*s4,s3*s4,
                             (1-s6)*s7,s6*(1-s7),s6*s7,s10*(1-s11)*(1-s12),(1-s10)*s11*(1-s12),
                             (1-s10)*(1-s11)*s12,s10*s11*(1-s12),(1-s10)*s11*s12,s10*(1-s11)*s12,s10*s11*s12])
                            
def nomic_bound(ineq): #nomic bound: minimize (1,v)*ineq over nomic vertices
    all_values_nomic = []
    for v in nomic_vertices:
        all_values_nomic.append(sum(x * y for x, y in zip(v, ineq)))
    return min(all_values_nomic)

def causal_bound(ineq): #causal bound: minimize (1,v)*ineq over causal vertices
    all_values_causal = []
    for v in causal_vertices:
        all_values_causal.append(sum(x * y for x, y in zip(v, ineq)))
    return min(all_values_causal)

def pc_bound(ineq): #pc bound: minimize (1,v)*ineq over superset of pc vertices
    all_values_pc = []
    for v in pc_vertices_superset:
        all_values_pc.append(sum(x * y for x, y in zip(v, ineq)))
    return min(all_values_pc)

def gen_bound(ineq): #nomic bound: minimize (1,v)*ineq over general vertices
    all_values_gen = []
    for v in general_vertices:
        all_values_gen.append(sum(x * y for x, y in zip(v, ineq)))
    return min(all_values_gen)

if __name__ == "__main__":
    #read in list of nomic families
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

    for c in families:
        print(c)
        values = [gen_bound(c),qp_bound(c),pc_bound(c),nomic_bound(c),causal_bound(c)]
        print(values)
        bounds.append(values)

    print(bounds)

