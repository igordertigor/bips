#!/usr/bin/env python

__doc__ = """Utilities to fit psychometric functions using pymc"""

import pymc
import numpy as np

# Constants
z_logistic = np.log(9.)
z1_gumbel = np.log(-np.log(.1))-np.log(-np.log(.9))
z2_gumbel = np.log(-np.log(.5))

# Sigmoids
def logistic ( x, al, bt ):
    """Logistic function sigmoid"""
    return 1./(1.+np.exp(-z_logistic*(x-al)/bt))

def gumbel_l ( x, al, bt ):
    """left skewed gumbel sigmoid"""
    return 1-np.exp(-np.exp(z1_gumbel*(x-al)/bt+z2_gumbel))

def gumbel_r ( x, al, bt ):
    """right skewed gumbel sigmoid"""
    return np.exp(-np.exp(-z1_gumbel*(x-al)/bt+z2_gumbel))

# Generating the model
def make_pmf ( stimulus_intensities, response_counts, ntrials, **kwargs ):
    """Create a psychometric function model with for a given data set

    The model has the form:

    .. math::
        \Psi(x) = \gamma + (1-\gamma-\lambda) F (x, \alpha, \beta)

    :Parameters:
        *stimulus_intensities*
            an array with the stimulus intensities for each block
        *response_counts*
            an array with the fitted response counts for each block
        *ntrials*
            an array with the number of trials delivered in each block

    :Optional parameters:
        *al,bt,lm,gm*
            specify prior distribution of model parameters or fix parameters to
            certain values. If these are omitted, 'reasonable' defaults are
            assumed and all parameters are fitted.
        *F*
            specify the sigmoidal function that relates stimulus and response
            probabilities. Default: logistic
    """
    d = (np.triu(stimulus_intensities.reshape((1,-1))
        -stimulus_intensities.reshape((-1,1))).ravel())
    y = response_counts.astype('d')/ntrials
    i = np.arange ( len(response_counts), dtype='d' )

    al = kwargs.setdefault ( 'al', pymc.Normal ( 'alpha', stimulus_intensities.mean(), .0001 ) )
    bt = kwargs.setdefault ( 'bt', pymc.Normal ( 'beta',  .5*(d.min()+d.max()), .0001 ) )
    gm = kwargs.setdefault ( 'gm', pymc.Beta (   'gamma', 2, 20 ) )
    lm = kwargs.setdefault ( 'lm', pymc.Beta (   'lambda',2, 20 ) )
    F  = kwargs.setdefault ( 'F',  logistic )
    posterior_args = {'al':al,'bt':bt,'gm':gm,'lm':lm}

    @pymc.deterministic
    def p ( stim=stimulus_intensities, a=al, b=bt, l=lm, g=gm ):
        return g+(1-g-l)*F(stim,a,b)

    k = pymc.Binomial ( 'k', n=ntrials, p=p, value=response_counts, observed=True )

    # Posterior predictive simulations
    k_sim = pymc.Binomial ( 'k_sim', n=ntrials, p=p )

    return locals()

