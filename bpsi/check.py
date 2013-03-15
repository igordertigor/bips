#!/usr/bin/env python

__doc__ = """Model checking routines"""

import numpy as np

# Error functions
def deviance ( p, k, n ):
    """Deviance"""
    y = k.astype('d')/n
    return 2*(klogp(k,y/p)+klogp(n-k,(1-y)/(1-p))).sum()

def rpd ( p, k, n ):
    """Correlation between model prediction and deviance residuals"""
    y = k.astype('d')/n
    dr = np.sign(y-p)*np.sqrt(2*(klogp(k,y/p)+klogp(n-k,(1-y)/(1-p))))
    return np.corrcoef ( dr, p )[0,1]

def rkd ( p, k, n ):
    """Correlation between block order and deviance residuals"""
    y = k.astype('d')/n
    dr = np.sign(y-p)*np.sqrt(2*(klogp(k,y/p)+klogp(n-k,(1-y)/(1-p))))
    return np.corrcoef ( dr, k )[0,1]

# Bayesian p value
def bayesian_p ( observed, simulated ):
    """Determine Bayesian p value from posterior predictive simulations

    :Parameters:
        *observed*
            observed values of the parameter of interest
        *simulated*
            simulated values of the parameter of interest
    """
    return np.mean ( simulated>observed )

def posterior_predictive_trace ( model, T ):
    """Get traces of a statistic based on posterior predictive sampling

    :Parameters:
        *model*
            the sampled model
        *T*
            the statistic to be evaluated
    """
    db = model.db
    n_samples = db.trace('deviance').length()
    observed  = np.empty ( n_samples, 'd' )
    simulated = np.empty ( n_samples, 'd' )

    k = model.k.value

    for i in xrange ( n_samples ):
        k_sim = db.trace('k_sim')[i]
        p     = db.trace('p')[i]
        n     = model.ntrials
        observed[i]  = T(p,k,n)
        simulated[i] = T(p,k_sim,n)
    return observed,simulated


# Helper functions
def klogp ( k, p ):
    return np.where ( k==0, 0, k*np.log(p) )

def klog1p ( k, p ):
    return np.where ( k==0, 0, k*np.log1p(-p) )

