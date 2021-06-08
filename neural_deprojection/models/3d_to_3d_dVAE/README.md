step 1:

discrete VAE for 3d -> 3d
latent space = 32x32 tokens

this reduces weights needed in model:

no CNN bc we have graph net
we want: nodes -> nodes (the tokens)

then nodes (tokens) -> nodes (gaussian parameters)
this way the output graph is much smaller.

encoder return distribution from which we have to sample.

like eq 1 without y.

step 2:

learn the prior (eq. 1 most right term, mistake in eq)

