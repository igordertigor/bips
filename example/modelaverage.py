import bips
import pymc
import numpy as np

__doc__ = """Illustrate Bayesian model averaging using bips"""

# Data
x = np.array ( [1,2,3,4,5,6], 'd' )
n = np.array([20]*6)
k = np.array ( [11,10,12,15,20,19], 'i' )

# Three models
models = []
for F in [bips.model.logistic,bips.model.gumbel_l,bips.model.gumbel_r]:
    pmf = bips.model.make_pmf ( x, k, n, gm=0.5, F=F )
    M = pymc.MCMC ( pmf )
    M.sample ( iter=10000, burn=1000, thin=10, burn_till_tuned=True )

    print "Model prediction at x=[1,2,3,4,5,6]:",bips.resample.evaluate_models ( x, M ).mean(0)

    models.append ( M )

model_posterior = bips.resample.model_posteriors ( *models )
print "Model posterior:",model_posterior[0]

print "Model average prediction at x=[1,2,3,4,5,6]:",bips.resample.evaluate_models ( x, *models ).mean(0)
