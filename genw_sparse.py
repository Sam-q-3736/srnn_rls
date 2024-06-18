# returns a sparse NxN matrix with gaussian entries
def genw_sparse(N, m, std, p):
    weight = sp.stats.norm.rvs(m, std, (N, N))
    is_zero = np.random.choice([0, 1], (N, N), [1-p, p])    
    return np.multiply(weight, is_zero)