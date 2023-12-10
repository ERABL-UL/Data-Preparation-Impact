from scipy import stats

def dens_kde(values):
    kde = stats.gaussian_kde(values)
    density = kde(values)
    return(density)
