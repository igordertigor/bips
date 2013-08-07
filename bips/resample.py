#!/usr/bin/env python

__doc__ = """resampling methods for model averaging an similar things"""

import numpy as np
import pymc
import model
from  scipy.stats import norm
sq2 = np.sqrt(2.)

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

def evaluate_thres ( th_p, *models, **kwargs ):
    """Evaluate posterior threshold distribution

    :Parameters:
        *th_p*
            performance to be associated with the threshold
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

    th = np.zeros ( (nsamples,), 'd' )

    for i in xrange ( nsamples ):
        k = np.where(np.random.multinomial ( 1, p, size=1 ))[0]
        M = models[k]
        j = np.random.randint ( M.db.trace('deviance').length() )
        th[i] = get_thres ( th_p, M, j )

    return th

def sample_weber_fraction ( base, increment, p=0.75, **kwargs ):
    """Determine the weber fraction from base to increment

    Weber fractions are defined as (th_increment-th_base)/th_base.

    :Parameters:
        *base*
            a model or a sequence of models at the base condition
        *increment*
            a model or a sequence of models at the incremented
            condition
        *p*
            the level at which to determine the threshold

    :Optional Keyword Arguments:
        *nsamples*
            number of samples used (default: 500)
        *p_base,p_increment*
            posterior probabilities of the models (will be determined using
            model_posteriors() by default)
    """
    p_base   = kwargs.setdefault ( 'p_base', model_posteriors ( *base )[0] )
    p_incr   = kwargs.setdefault ( 'p_increment', model_posteriors ( *increment )[0] )
    nsamples = kwargs.setdefault ( 'nsamples', 500 )

    y = np.zeros ( nsamples, 'd' )
    for i in xrange ( nsamples ):
        k_base = np.where(np.random.multinomial ( 1, p_base, size=1 ) )[0]
        k_incr = np.where(np.random.multinomial ( 1, p_incr, size=1 ) )[0]
        M_base = base[k_base]
        M_incr = increment[k_base]
        j_base = np.random.randint ( M_base.db.trace('deviance').length() )
        j_incr = np.random.randint ( M_incr.db.trace('deviance').length() )
        th_base = get_thres ( p, M_base, j_base )
        th_incr = get_thres ( p, M_incr, j_incr )
        y[i] = (th_incr-th_base)/th_base
    return y

def sample_efficiency ( x, ideal_performance, models, **kwargs ):
    """Determine efficiency

    :Parameters:
        *x*
            stimulus values at which to determine efficiency
        *ideal_performance*
            performance of an ideal observer at stimulus levels given by x
        *models*
            fitted psychometric function models or samples from the models
            evaluated at x
    """
    dprime_ideal = dprime ( ideal_performance )
    nsamples = kwargs.setdefault ( 'nsamples', 500 )

    if hasattr(models,'shape'):
        assert models.shape[1]==len(x)
        nsamples = models.shape[0]
        dprime_human = dprime ( models )
    else:
        p_models = kwargs.setdefault ( 'p_models', model_posteriors (*models)[0] )
        dprime_human = dprime (
                evaluate_models(*models,p_models=p_models,nsamples=nsamples) )

    efficiency = dprime_human/dprime_ideal
    efficiency **= 2

    return efficiency

def dprime ( p ):
    """Convert 2afc performance p to d'"""
    return sq2*norm.ppf ( np.clip(p,1e-8,1-1e-8) )

def evaluate_single_sample ( x, model, j ):
    th = get_prm ( model, j )
    return th['gm'] + (1-th['gm']-th['lm'])*th['F'](x,th['al'],th['bt'])

def get_thres ( p, M, j ):
    """Determine the p-threshold for a certain sample"""
    th = get_prm ( M, j )
    iF = eval('model.i'+th['F'].func_name)
    q = p-th['gm']
    q /= 1-th['gm']-th['lm']
    if q >= 1:
        return np.inf
    else:
        return iF(q,th['al'],th['bt'])

def get_prm ( model, j ):
    """Get parameters of a certain sample"""
    th = {'F': model.F}
    for prm in ['al','bt','lm','gm']:
        if isinstance ( getattr(model,prm), pymc.Stochastic ):
            th[prm] = model.db.trace ( prm )[j]
        else:
            th[prm] = getattr(model,prm)
    return th
