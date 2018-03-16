import exchangable_tensor
import cnfformula
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time

n = 2000000
variables = 20
clauses = int(4.258 * variables + 58.26 * variables **(-2/3))
k = 3
n_examples = 50000

for item in tqdm(xrange(n_examples)):
    a = cnfformula.RandomKCNF(k=k,n=variables,m=clauses)
    X = csr_matrix((clauses, variables), dtype="int")
    
    indices = []
    values = []

    for i, clause in enumerate(a.clauses()):
        for term in clause:
            v_i = int(term[1].split("_")[1]) - 1
            v_v = -(1 - 2*int(term[0]))
            X[i, v_i] = v_v
            indices.append([i,v_i])
            v = [1,0] if v_v == 1 else [0, 1]
            values.append(v)
    indices = np.array(indices)
    values = np.array(values)
    
    t = time.time()
    y = 1 - 2 * np.random.randint(0, 2, [variables, n])
    test = X.dot(y) == -3
    is_sat = test.sum(axis=0) == 0
    y_sat = y[:, is_sat]
    y_unsat = y[:, np.logical_not(is_sat)][:, 0:200]
    sat = "sat" if np.sum(test.sum(axis=0) == 0) > 0 else "unsat"
    np.savez_compressed("data/%s_%06d.npz" % (sat, item), 
                        indices=indices, values=values, 
                        y_sat=y_sat, y_unsat=y_unsat)


