#!/usr/bin/env python

__doc__ = """resampling methods for model averaging an similar things"""

import numpy as np
import pymc
import model

def posterior_trace ( model1, model2=None ):
    """Evaluate the posterior probability of model2 for all samples of model1"""
    if model2 is None:
        model2 = model1
    db1 = model1.db
    n_samples1 = db1.trace ( 'deviance' ).length()
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

def evaluate_models ( x, *models, **kwargs ):
    """Evaluate the model posterior

    :Parameters:
        *x*
            x values at which to evaluate the posterior
        *models*
            for one model, we simply evaluate its posterior, for more than a
            single model, we evaluate the posterior p(theta,M|data)

    :Optional Keyword Arguments:
        *nsamples*
            number of samples
         *p_models*
            posterior probabilities of the models (will be determined using
            model_posteriors() by default)
   """
    p        = kwargs.setdefault ( 'p_models', model_posteriors ( *models )[0] )
    nsamples = kwargs.setdefault ( 'nsamples', 500 )

    y = np.zeros ( (nsamples,len(x)), 'd' )

    for i in xrange ( nsamples ):
        k = np.where(np.random.multinomial ( 1, p, size=1 ))[0]
        M = models[k]
        j = np.random.randint ( M.db.trace('deviance').length() )
        y[i,:] = evaluate_single_sample ( x, M, j )
    return y
def evaluate_single_sample ( x, model, j ):
    th = get_prm ( model, j )
    return th['gm'] + (1-th['gm']-th['lm'])*th['F'](x,th['al'],th['bt'])

def get_thres ( p, model, j ):
    """Determine the p-threshold for a certain sample"""
    th = get_prm ( model, j )
    iF = eval('model.i'+th['F'].func_name)
    q = p-th['gm']
    q /= 1-th['gm']-th['lm']
    if q >= 1:
        return np.inf
    else:
        return iF(q)

def get_prm ( model, j ):
    """Get parameters of a certain sample"""
    th = {'F': model.F}
    for prm in ['al','bt','lm','gm']:
        if isinstance ( getattr(model,prm), pymc.Stochastic ):
            th[prm] = model.db.trace ( prm )[j]
        else:
            th[prm] = getattr(model,prm)
    return th
