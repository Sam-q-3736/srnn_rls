def genw_sparse(N, m, std, p):
    # this function returns a NxN matrix 
    # each entry is non-zero with probability p
    # entries are gaussian with mean m and standard deviation std
    w = np.zeros((N, N)) # initialize NxN matrix of zeros
    for i in range(N):
        for j in range(N):
            weight = sp.stats.norm.rvs(m, std)
            is_zero = sp.stats.bernoulli.rvs(p)
            w[i][j] = weight*is_zero
    return w