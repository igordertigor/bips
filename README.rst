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
