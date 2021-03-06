bips
====

Bayesian Inference for PSychometric functions

Yes, there is psignifit, and yes psignifit offers lots of nice and
established features. At the same time, psignifit tries to address a number
of issues that make it really tricky to maintain. First of all: The need to
support (i) matlab and (ii) all platforms. Taken together, these two things
resulted (amoung other things) in psignifit development being essentially
stalled.

I have lately been playing around with a number of features in `pymc <https://github.com/pymc-devs/pymc>`_. The result is **bips** for **B** ayesian **I** nference for **Ps** ychometric functions. It provides *much* reduced pure python framework, that does not attempt to have extraordinarily clever defaults, that does not attempt to show any figures to help you decide if your model is appropriate or not. In other words, bips is for me, not for impact.

Getting started
---------------

To get you started using bips, we will now walk you through the most basic
analysis of a small data set. bips draws heavily on the functionality of
`pymc<http://pymc-devs.github.com/pymc/index.html>`_. We will address the most
important bits of pymc here in passing, but for a deeper understanding of the
internal workings of sampling from the psychometric function, you are referred
to the pymc-documentation. In addition, this tutorial assumes that you are
superficially familiar with the ideas of Bayesian inference. Bips is not meant
to be a tool that provides smart default settings or makes the analysis
foolproof. You have to know what you are doing. That said, let us take a look
at a dataset::

    >>> stimuli = [ 2,  3,3.5,  4,4.5,  5,  6,  8]
    >>> correct = [10, 11, 15, 19, 20, 17, 20, 20]
    >>> ntrials = [20, 20, 20, 20, 20, 20, 20, 20]

This is a small dataset from a 2-alternatives forced choice experiment. We
would like to fit the probability of a correct response. Let us first import
bips and build a model::

    >>> import bips
    >>> pmf = bips.make_pmf ( stimuli, correct, ntrials, gm=0.5 )

This code, constructs all the internal things needed to set up a pymc model.
Not that we explicitely set gamma to 0.5. This way, we fix gamma to 0.5 rather
than leave it a free parameter. We can fix any parameter of the psychometric
function in this way. In addition, we can change the prior distribution of the
parameters at this point. See the documentation for `make_pmf`_ for more
information.

Our model is now in a state that it can be handed over to pymc::

    >>> import pymc
    >>> mcmc = pymc.MCMC(pmf)

This is now a regular pymc model and we can do all the things we can do with
other pymc models (e.g. sampling, ...). After we sampled from the model, we can
use the functions in `bips.check`_ to test the goodness of fit. In addition, we
may want to use resampling as in `bips.resample`_ to derive posterior intervals
for thresholds, weber fractions, observer efficiency, ...

Model checking using `bips.check`_
----------------------------------

Bips implements the model checking methods published in [1]_. This paper
describes two model checking strategies: (i) Using posterior predictive
sampling to compare the observed data to data expected if the model was correct
and (ii) determining influential observations using KL-Divergence.

Before we can start, we should first sample from our model::

    >>> mcmc.sample ( iter=10000, thin=10, burn_till_tuned=True )

The model automatically creates posterior predictive samples and we can use
them to do the deviance test from [1]_::

    >>> observed_deviance,simulated_deviance = bips.check.posterior_predictive_trace (
    ...    mcmc, bips.check.deviance )

You can see that the last argument here is a function. The bips.check module
includes other functions that can be applied to the data in order to compare
observed and simulated data. We can now generate a plot of the observed and
simulated deviances, or look at the Bayesian p-value::

    >>> print "Bayesian p-value:", bips.check.bayesian_p ( observed_deviance, simulated_deviance )
    ... # doctest: +ELLIPSIS
    Bayesian p-value: ...

In [1]_ we discuss another way to check a model: An observed block of trials is
considered "influential", if including or excluding it changes the posterior
distribution (in terms of KL-divergence). This is determined in bips by calling::

    >>> infl = bips.check.influential ( mcmc )

You can now plot the variable infl against to inspect the influence of
individual blocks on the estimated posterior.

References
----------

.. [1] Fründ, Haenel, Wichmann (2011): Inference for psychometric functions in
    the presence of nonstationary behavior, J Vis, 11(6): 16, doi: 10.1167/11.6.16.
