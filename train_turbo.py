'''
Illustrates how easy it is to integrate with TuRBO directly.
'''


from turbo import Turbo1
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
seed = np.random.randint(1e6)
arch1 = 10
arch2 = 10
nenv = 2048
batch_size = 8

brx_env = BraxCaller(env, arch1, arch2, nenv, batch_size, seed)
D = arch1*(brx_env.env.observation_space.shape[1]+1) + (arch1+1) * arch2 +  (arch2+1) * brx_env.env.action_space.shape[1]

turbo1 = Turbo1(
    f=-brx_env.single_numpy_eval,  # Handle to objective function, NEGATIVE BECAUSE WE TYPICALLY WANT TO MAXIMIZE RETURNS IN RL
    lb=-np.ones(D),  # Numpy array specifying lower bounds
    ub=np.ones(D),  # Numpy array specifying upper bounds
    n_init = 512,  # Number of initial bounds from an Latin hypercube design
    max_evals = 1536,  # Maximum number of evaluations
    batch_size = batch_size,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
    seed=seed
)

turbo1.optimize()

ind_best = np.argmin(turbo1.fX)
x_best = turbo1.X[ind_best, :]
with open('best_'+env+'_'+str(arch1)+'_'+str(arch2)+'.npy', 'wb') as f:
    np.save(f, x_best)

