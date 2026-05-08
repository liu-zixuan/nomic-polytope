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
    return 1 if a == b else 0

valid_process_functions = []

'''
Generate the list of all tripartite process functions (wA: O_A x O_B x O_C -> I_A, wB: O_A x O_B x O_C -> I_B, wC: O_A x O_B x O_C -> I_C)
with binary inputs I_A, I_B, I_C and outputs O_A, O_B, O_C (Loop over all tripartite functions for which the input of each party does not depend on its own output,
which is a necessary condition for valid process functions; then check if it is indeed a valid process function)
'''    
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

    #deterministic function obtained by composing each process function with the fixed local operations
    df = {
        (0,0,0): (0,0,0),
        (0,0,1): (0,0,wC[(0,0,1)]),
        (0,1,0): (0,wB[(0,1,0)],0),
        (0,1,1): (0,wB[(0,1,1)],wC[(0,1,1)]),
        (1,0,0): (wA[(1,0,0)],0,0),
        (1,0,1): (wA[(1,0,1)],0,wC[(1,0,1)]),
        (1,1,0): (wA[(1,1,0)],wB[(1,1,0)],0),
        (1,1,1): (wA[(1,1,1)],wB[(1,1,1)],wC[(1,1,1)])
    }

    #corresponding probability vector, in the parametrisation chosen in the paper
    vertex_list.append(
        [delta(df[(1,0,0)],(1,0,0)),delta(df[(0,1,0)],(0,1,0)),delta(df[(0,0,1)],(0,0,1)),
         delta(df[(1,1,0)],(1,0,0)),delta(df[(1,1,0)],(0,1,0)),delta(df[(1,1,0)],(1,1,0)),
         delta(df[(0,1,1)],(0,1,0)),delta(df[(0,1,1)],(0,0,1)),delta(df[(0,1,1)],(0,1,1)),
         delta(df[(1,0,1)],(0,0,1)),delta(df[(1,0,1)],(1,0,0)),delta(df[(1,0,1)],(1,0,1)),
         delta(df[(1,1,1)],(1,0,0)),delta(df[(1,1,1)],(0,1,0)),delta(df[(1,1,1)],(0,0,1)),
         delta(df[(1,1,1)],(1,1,0)),delta(df[(1,1,1)],(0,1,1)),delta(df[(1,1,1)],(1,0,1)),
         delta(df[(1,1,1)],(1,1,1))
         ]
    )

#Write the vertices to a file "nomic_vertices.ext" in the appropriate input format for CDD.   
with open("nomic_vertices.ext", "w") as f:
    f.write("V-representation\n")
    f.write("begin\n")
    f.write(f"{len(vertex_list)} 20 integer\n")

    for v in vertex_list:
        row = " ".join(str(x) for x in v)
        f.write(f"1 {row}\n")

    f.write("end\n")