'''
You need to provide:
1. env: Choice of brax environment, for example: "swimmer". See Line 112 of brax_caller.py for full list of envs.
2. arch1/arch2: Number of units per hidden layer in 2-hidden-layer-MLP.
3. nenv: Number of parallel brax environments
4. the x that is to be evaluated, in the form of a numpy array with dimensions implied by the MLP architecture, in the swimmer case 222.
5. batch size, if you want to evaluate a batch of x's.

$ python simple.py
Expected output:
Environment is "swimmer"
Architecture is [8, 10, 10, 2], resultant x dimension: 222
f_X is 1.0470927953720093,
single f_X across 32768 envs computed in 4.879093885421753 seconds.
f_Xs are [ 0.92772925  1.120687    1.1900027   1.0475277  -0.01021799  1.3422477
  1.1406054   1.6161344 ],
8 f_Xs across 32768 envs computed in 5.645813941955566 seconds.
'''

from brax_caller import BraxCaller
import numpy as np
import time

'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--env",type=str, required=True)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--arch1", type=int, default=10) #1st hidden layer dim
parser.add_argument("--arch2", type=int, default=10) #2nd hidden layer dim
parser.add_argument("--nenv", type=int, default=32768) #number of parallel envs
parser.add_argument("--batch-size", type=int, default=8) #turbo batch size
args = parser.parse_args()
'''

env = "swimmer"
seed = 11
arch1 = 10
arch2 = 10
nenv = 32768
batch_size = 8

brx_env = BraxCaller(env, arch1, arch2, nenv, batch_size, seed)
D = arch1*(brx_env.env.observation_space.shape[1]+1) + (arch1+1) * arch2 +  (arch2+1) * brx_env.env.action_space.shape[1]

ctime = time.time()
f_X = brx_env.fix1ksteps_single_numpy_eval(np.random.rand(D))
print(f'f_X is {f_X},') 
print(f'single f_X across {nenv} envs computed in {time.time()-ctime} seconds.')

ctime = time.time()
f_Xs = brx_env.fix1ksteps_batched_numpy_eval(np.random.rand(batch_size, D))
print(f'f_Xs are {f_Xs},') 
print(f'{batch_size} f_Xs across {nenv} envs computed in {time.time()-ctime} seconds.') 