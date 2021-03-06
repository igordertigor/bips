import bips
import pymc
import numpy as np

x = np.array ( [1,2,3,4,5,6], 'd' )
n = np.array([20]*6)
k = np.array ( [11,10,12,15,20,19], 'i' )

pmf = bips.model.make_pmf ( x, k, n, gm=0.5, F=bips.model.logistic )

M = pymc.MCMC ( pmf )
M.sample ( iter=10000, burn=1000, thin=10, burn_till_tuned=True )

obs,sim = bips.check.posterior_predictive_trace ( M, bips.check.deviance )

print "Bayesian p-value",bips.check.bayesian_p ( obs, sim )
