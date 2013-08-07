#!/usr/bin/env python

__doc__ = """In [FHW2011], a method is introduced to correct confidence
intervals for bias that might result from nonstationarity in the data. This
module implements the estimation of the respective correction factor nu. See
the original reference for details.

.. [FHW2011] Fründ, Haenel, Wichmann (2011) J Vis
"""

import numpy as np
from scipy.special import polygamma,gamma
import resample

__all__ = ['estimate_nu']

def l ( nu, psi, k, n ):
    """log likelihood for psi

    :Parameters:
        *nu* :
            scalar value nu, for which the likelihood should be evaluated
        *psi* :
            m array that gives the psychometric function evaluated at stimulus
            levels x_i, i=1,...,m
        *k* :
            m array that gives the numbers of correct responses (in Yes/No: Yes responses)
            at stimulus levels x_i, i=1,...,m
        *n* :
            m array that gives the numbers of presented trials at stimulus
            levels x_i, i=1,...,m

    :Example:

    >>> psi = [ 0.52370051,  0.58115041,  0.70565915,  0.83343107,  0.89467234,  0.91364765,  0.91867512]
    >>> k   = [28, 29, 35, 41, 46, 46, 45]
    >>> n   = [50]*7
    >>> l ( 1, psi, k, n )
    13.752858759933943
    """ 
    psi = np.array(psi,'d')
    k = np.array(k,'d')
    n = np.array(n,'d')
    p = k/n
    return (np.log(gamma(nu*n+2))).sum()-(np.log(gamma(psi*nu*n+1))).sum()-(np.log(gamma((1-psi)*nu*n+1))).sum() \
            + (psi*nu*n*np.log(p)).sum() + ((1-psi)*nu*n*np.log(1-p)).sum()

def dl ( nu, psi, k, n ):
    """first derivative of the likelihood function with respect to nu

    :Parameters:
        *nu* :
            scalar value nu, for which the likelihood should be evaluated
        *psi* :
            m array that gives the psychometric function evaluated at stimulus
            levels x_i, i=1,...,m
        *k* :
            m array that gives the numbers of correct responses (in Yes/No: Yes responses)
            at stimulus levels x_i, i=1,...,m
        *n* :
            m array that gives the numbers of presented trials at stimulus
            levels x_i, i=1,...,m
    """ 
    p = k/n
    return (n*polygamma( 0, nu*n+2 )).sum() \
            - (psi*n*polygamma(0,nu*psi*n+1)).sum() \
            - ((1-psi)*n*polygamma(0,nu*(1-psi)*n+1)).sum() \
            + (psi*n*np.log(p)).sum() \
            + ((1-psi)*n*np.log(1-p)).sum()

def ddl ( nu, psi, k, n ):
    """second derivative of the likelihood function with respect to nu

    :Parameters:
        *nu* :
            scalar value nu, for which the likelihood should be evaluated
        *psi* :
            m array that gives the psychometric function evaluated at stimulus
            levels x_i, i=1,...,m
        *k* :
            m array that gives the numbers of correct responses (in Yes/No: Yes responses)
            at stimulus levels x_i, i=1,...,m
        *n* :
            m array that gives the numbers of presented trials at stimulus
            levels x_i, i=1,...,m
    """
    return (n**2*polygamma( 1, nu*n+2)).sum() \
            - (psi**2*n**2*polygamma(1,nu*psi*n+1)).sum() \
            - ((1-psi)**2*n**2*polygamma(1,nu*(1-psi)*n+1)).sum()

def estimate_nu ( model ):
    """Perform a couple of newton iterations to estimate nu

    :Parameters:
        *model* :
            the model object for which we want to estimate nu

    :Return:
        nu,nu_i,l_i
        *nu* :
            estimated nu parameter 
        *nu_i* :
            sequence of nu values during optimization
        *l_i* : 
            sequence of likelihoods associated with the nu values
    """
    psi = resample.evaluate_models ( model.stimulus_intensities, model )
    k = model.response_counts.astype('d')
    n = model.ntrials.astype('d')
    k = N.where ( k==n, k-.01, k )
    k = N.where ( k==0, .01, k )

    nu = 1.
    nu_i = [nu]
    l_i = [l(nu,psi,k,n)]

    for i in xrange(10):
        nu -= dl(nu,psi,k,n)/ddl(nu,psi,k,n)
        if nu>1:
            nu=1
        elif nu<0:
            nu=0
        nu_i.append ( nu )
        l_i.append ( l(nu,psi,k,n) )

    return nu, nu_i, l_i

def correction_for_inference ( th, nu=1 ):
    """Correct the values in th for nonstationarity

    :Parameters:
        *th*
            an array of values to be corrected each parameter has one column,
            each sample one row

    :Optional keyword arguments:
        *nu*
            estimated nonstationarity factor
    """
    m = th.mean(0).reshape((1,-1))
    th_ = m + (th-m)/np.sqrt(nu)
    return th_
