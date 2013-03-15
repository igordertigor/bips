#!/usr/bin/env python

__doc__ = """resampling methods for model averaging an similar things"""

import numpy as np

def posterior_trace ( model1, model2=None ):
    """Evaluate the posterior probability of model2 for all samples of model1"""
    if model2 is None:
        model2 = model1
    db1 = model1.db
    db2 = model2.db
    n_samples1 = db1.trace ( 'deviance' ).length()
    n_samples2 = db2.trace ( 'deviance' ).length()
    assert n_samples1==n_samples2
    assert len(model1.stochastics)==len(model2.stochastics)

    logp = np.empty ( n_samples1, 'd' )
    parnames = []
    for parname in ['alpha','beta','gamma','lambda']:
        if model1.get_node(parname) in model1.stochastics:
            parnames.append ( parname )

    for i in xrange ( n_samples1 ):
        for parname in parnames:
            try:
                value = db1.trace ( parname )[i]
                model2.get_node ( parname ).value = value
            except KeyError:
                print "No trace available for %s." % (parname,)
        logp[i] = model2.logp

    return logp

def model_posteriors ( *models ):
    """Determine posterior distribution in model space

    We think of a gibbs sampler in the space of parametersXmodels. For this
    sampler, we use an analytical expression for the marginal posterior of
    models.
    """
    n = len(models)
    T = np.zeros ( (n,n), 'd' )

    # Set up the transition matrix
    for k,M1 in enumerate ( models ):
        p = []
        for M2 in models:
            p.append ( np.exp ( posterior_trace ( M1, M2 ) ) )
        p = np.array(p)
        p /= p.sum(0).reshape((1,-1))
        T[k] = p.sum(1)
        T[k] /= T[k].sum()

    # Now solve the stochastic matrix equation pi*T=pi
    A = (T-np.eye(n)).T
    A[0,:] = 1.
    p = np.zeros ( n, 'd' )
    p[0] = 1.

    return np.linalg.solve ( A, p ),T


