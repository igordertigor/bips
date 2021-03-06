#!/usr/bin/env python

__doc__ = """
bips: Bayesian Inference for PSychometric functions
===================================================

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
at a dataset:

>>> stimuli = [ 2,  3,3.5,  4,4.5,  5,  6,  8]
>>> correct = [10, 11, 15, 19, 20, 17, 20, 20]
>>> ntrials = [20, 20, 20, 20, 20, 20, 20, 20]

This is a small dataset from a 2-alternatives forced choice experiment. We
would like to fit the probability of a correct response. Let us first import
bips and build a model:

>>> import bips
>>> pmf = bips.make_pmf ( stimuli, correct, ntrials, gm=0.5 )

This code, constructs all the internal things needed to set up a pymc model.
Not that we explicitely set gamma to 0.5. This way, we fix gamma to 0.5 rather
than leave it a free parameter. We can fix any parameter of the psychometric
function in this way. In addition, we can change the prior distribution of the
parameters at this point. See the documentation for `make_pmf`_ for more
information.

Our model is now in a state that it can be handed over to pymc:

>>> import pymc
>>> mcmc = pymc.MCMC(pmf)

This is now a regular pymc model and we can do all the things we can do with
other pymc models (e.g. sampling, ...). After we sampled from the model, we can
use the functions in `bips.check`_ to test the goodness of fit. In addition, we
may want to use resampling as in `bips.resample`_ to derive posterior intervals
for thresholds, weber fractions, observer efficiency, ...
"""

import model
import check
import resample
import nonstationary

from model import make_pmf
