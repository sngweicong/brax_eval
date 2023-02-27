# brax_eval

Currently only support evaluating f(x). Maybe can do f'(x) later.

basic.py shows how to call brax environment evaluations.

train_turbo.py shows how to integrate with TuRBO easily,
where all environments are used to evaluate a single x.

train_turbo_batch.py shows how to do batch-TuRBO,
where the environments are split up into batch_size portions and used to evaluate the batch of x separately.
This may be ideal because brax easily supports 2**15 parallel environments with reasonably low cost,
and it may be beneficial to want to evaluate batches of 2**3 where each x will have 2**12 evaluations 
and still have a low associated standard error.
Caveat is that one needs to edit the TuRBO files from where one pip-installed from (files included in this repo).


Tested under:
$ pip freeze | grep 'jax\|brax\|gym\|numpy\|flax\|turbo'
-e git+https://github.com/google/brax.git@f0de0e56126b7cd7cdf941f0fc02f3480a02716a#egg=brax
flax==0.6.4
gym==0.23.0
gym-notices==0.0.8
jax==0.3.25
jaxlib==0.3.25+cuda11.cudnn82
jaxopt==0.6
numpy==1.23.5
-e git+https://github.com/uber-research/TuRBO.git@de0db39f481d9505bb3610b7b7aa0ebf7702e4a5#egg=turbo
