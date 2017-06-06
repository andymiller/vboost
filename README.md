# vboost

code for [Variational Boosting: Iteratively Refining Posterior Approximations](https://arxiv.org/abs/1611.06585)

### Abstract

> We propose a black-box variational inference method to approximate
> intractable distributions with an increasingly rich approximating class.
> Our method, termed variational boosting, iteratively refines an existing
> variational approximation by solving a sequence of optimization problems,
> allowing the practitioner to trade computation time for accuracy.
> We show how to expand the variational approximating class by incorporating
> additional covariance structure and by introducing new components to form a
> mixture. We apply variational boosting to synthetic and real statistical
> models, and show that resulting posterior inferences compare favorably to
> existing posterior approximation algorithms in both accuracy and efficiency.

Authors:
[Andrew Miller](http://andymiller.github.io/),
[Nick Foti](http://nfoti.github.io/), and
[Ryan Adams](http://people.seas.harvard.edu/~rpa/).

### Requires

* [`autograd`](https://github.com/HIPS/autograd) + its requirements (`numpy`, etc).  Our code is compatible with [this `autograd` commit](https://github.com/HIPS/autograd/tree/42a57226442417785efe3bd5ba543b958680b765).
* [`pyprind `](https://github.com/rasbt/pyprind)
