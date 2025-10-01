# Inference example

A very simplistic inference example, demonstrating the self-consistency
of the likelihood implementation.

In `bilby_inference.py` we perform a zero-noise injection of a BBH signal.
We recover the signal varying only the sky position, and with very tight priors.
This makes it so inference only takes a couple minutes.

The likelihood in this example uses relative binning with 2000 geometrically spaced
grid points.

Then, in `resampling_efficiency.py` we check that this was sufficient. 
We reweight the samples with a new likelihood using 10000 geometrically spaced 
points, and check that the sample efficiency is near-unity.
It is: I get 0.9999999477341882.

