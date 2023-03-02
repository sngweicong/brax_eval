'''
Note: fix1ksteps_ functions are strictly for profiling time-cost, 
since we want the environments to rollout for a fixed 1000 steps instead of terminating when all environments are done==True.
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.5'

import argparse
import jax
import jax.numpy as jnp
import gym
import functools
import numpy as np
from brax import envs
import flax
import flax.linen as nn
from typing import Sequence
import time

class Actor(nn.Module):
    architecture: Sequence[int]
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.architecture[1])(x)
        x = nn.relu(x)
        x = nn.Dense(self.architecture[2])(x)
        x = nn.relu(x)
        x = nn.Dense(self.architecture[-1])(x)
        x = nn.tanh(x)
        #x = x * self.action_scale + self.action_bias
        return x

class BraxCaller():
    def __init__(self, env_name, arch1, arch2, nenv, batch_size, max_or_min, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, self.actor_key = jax.random.split(self.key, 2)

        self.env_name = env_name 
        self.nenv = nenv
        self.batch_size = batch_size
        self.max_or_min = max_or_min
        self.make_env()
        self.architecture = [self.env.observation_space.shape[1], arch1, arch2, self.env.action_space.shape[1]] 
        self.D = arch1*(self.env.observation_space.shape[1]+1) + (arch1+1) * arch2 +  (arch2+1) * self.env.action_space.shape[1]

        self.actor = Actor(self.architecture)
        self.actor.apply = jax.jit(self.actor.apply)
        print(f'Environment is "{self.env_name}"')
        print(f'Architecture is {self.architecture}, resultant x dimension: {self.D}')

    def dict_builder(self, flat_array):
        '''
        Given a 1d numpy array with all the parameters of the network, e.g., proposed by some acquisition function of BO,
        and also the 2-hidden-layer MLP architecture, in the form of a list,
        build a flax.core.frozen_dict.FrozenDict for jax forward-pass.

        end_indices is for bias of 1st layer, weights of 1st layer, bias of 2nd layer etc. ordering.
        '''
        end_indices = [self.architecture[1], self.architecture[1]*(self.architecture[0]+1), self.architecture[1]*(self.architecture[0]+1)+self.architecture[2], 
        self.architecture[1]*(self.architecture[0]+1)+self.architecture[2]*(self.architecture[1]+1),self.architecture[1]*(self.architecture[0]+1)+self.architecture[2]*(self.architecture[1]+1)+self.architecture[3],
        self.architecture[1]*(self.architecture[0]+1)+self.architecture[2]*(self.architecture[1]+1)+self.architecture[3]*(self.architecture[2]+1)]
        #print(end_indices)

        to_return = dict()
        to_return['params'] = dict()
        to_return['params']['Dense_0'] = dict()
        to_return['params']['Dense_1'] = dict()
        to_return['params']['Dense_2'] = dict()
        to_return['params']['Dense_0']['bias'] = jnp.array(flat_array[0:end_indices[0]])
        to_return['params']['Dense_0']['kernel'] = jnp.array(flat_array[end_indices[0]:end_indices[1]]).reshape(self.architecture[0],self.architecture[1])
        to_return['params']['Dense_1']['bias'] = jnp.array(flat_array[end_indices[1]:end_indices[2]])
        to_return['params']['Dense_1']['kernel'] = jnp.array(flat_array[end_indices[2]:end_indices[3]]).reshape(self.architecture[1],self.architecture[2])
        to_return['params']['Dense_2']['bias'] = jnp.array(flat_array[end_indices[3]:end_indices[4]])
        to_return['params']['Dense_2']['kernel'] = jnp.array(flat_array[end_indices[4]:end_indices[5]]).reshape(self.architecture[2],self.architecture[3])

        froz_dict = flax.core.frozen_dict.freeze(to_return)

        return froz_dict


    def dict_unbuilder(self, froz_dict):
        '''
        The reverse of dict_builder().
        '''
        end_indices = [self.architecture[1], self.architecture[1]*(self.architecture[0]+1), self.architecture[1]*(self.architecture[0]+1)+self.architecture[2], 
        self.architecture[1]*(self.architecture[0]+1)+self.architecture[2]*(self.architecture[1]+1),self.architecture[1]*(self.architecture[0]+1)+self.architecture[2]*(self.architecture[1]+1)+self.architecture[3],
        self.architecture[1]*(self.architecture[0]+1)+self.architecture[2]*(self.architecture[1]+1)+self.architecture[3]*(self.architecture[2]+1)]
        #print(end_indices)
        D = self.architecture[1]*(self.architecture[0]+1) + (self.architecture[1]+1) * self.architecture[2] +  (self.architecture[2]+1) * self.architecture[3]
        flat_array = jnp.zeros(D)
        flat_array = flat_array.at[0:end_indices[0]].set(froz_dict['params']['Dense_0']['bias'])
        flat_array = flat_array.at[end_indices[0]:end_indices[1]].set(froz_dict['params']['Dense_0']['kernel'].reshape(self.architecture[0]*self.architecture[1]))
        flat_array = flat_array.at[end_indices[1]:end_indices[2]].set(froz_dict['params']['Dense_1']['bias'])
        flat_array = flat_array.at[end_indices[2]:end_indices[3]].set(froz_dict['params']['Dense_1']['kernel'].reshape(self.architecture[1]*self.architecture[2]))
        flat_array = flat_array.at[end_indices[3]:end_indices[4]].set(froz_dict['params']['Dense_2']['bias'])
        flat_array = flat_array.at[end_indices[4]:end_indices[5]].set(froz_dict['params']['Dense_2']['kernel'].reshape(self.architecture[2]*self.architecture[3]))

        flat_array = np.array(flat_array)

        return flat_array
        
    def make_env(self):
        '''
        env_name must be from:
            _envs = {
            'acrobot': acrobot.Acrobot,
            'ant': functools.partial(ant.Ant, use_contact_forces=True),
            'fast': fast.Fast,
            'fetch': fetch.Fetch,
            'grasp': grasp.Grasp,
            'halfcheetah': half_cheetah.Halfcheetah,
            'hopper': hopper.Hopper,
            'humanoid': humanoid.Humanoid,
            'humanoidstandup': humanoid_standup.HumanoidStandup,
            'inverted_pendulum': inverted_pendulum.InvertedPendulum,
            'inverted_double_pendulum': inverted_double_pendulum.InvertedDoublePendulum,
            'pusher': pusher.Pusher,
            'reacher': reacher.Reacher,
            'reacherangle': reacherangle.ReacherAngle,
            'swimmer': swimmer.Swimmer,
            'ur5e': ur5e.Ur5e,
            'walker2d': walker2d.Walker2d,
        }
        '''
        env_id = "brax-"+self.env_name+"-v0"
        entry_point = functools.partial(envs.create_gym_env, self.env_name)
        if env_id not in gym.envs.registry.env_specs:
            gym.register(env_id, entry_point=entry_point)

        # create a gym environment that contains n parallel environments
        self.env = gym.make(env_id, batch_size=self.nenv)

        # wrap it to interoperate with torch data structures
        #gym_env = to_torch.JaxToTorchWrapper(gym_env, device='cuda')

        # jit compile
        self.env.reset()
        for _ in range(2):
            action = jax.random.uniform(jax.random.PRNGKey(758493),shape = self.env.action_space.shape)
            self.env.step(action)
        
        #obs = self.env.reset()
        #params=self.actor.init(self.actor_key, obs)
        #print(jax.tree_util.tree_map(lambda x: x.shape, params))# Checking output shapes

    def evaluate_single_x_on_all_envs(self, fdict):
        cumulative_return = jnp.zeros(self.nenv)
        cumu_done = jnp.full(self.nenv,False)
        obs = self.env.reset()
        for iter in range(999999):
            actions = self.actor.apply(fdict, obs)
            next_obs, rewards, dones, infos = self.env.step(actions)
            obs = next_obs
            #print(dones.shape) #(8192,)
            cumu_done = jnp.logical_or(cumu_done,dones)
            cumulative_return += rewards * jnp.logical_not(cumu_done)
            if all(cumu_done):
                break
        mu_return = jnp.mean(cumulative_return)
        return mu_return

    def evaluate_multiple_x_on_all_envs(self, batched_fdict):
        cumulative_return = jnp.zeros(self.nenv)
        cumu_done = jnp.full(self.nenv,False)
        obs = self.env.reset() 
        for iter in range(999999):
            all_actions=jnp.array([]).reshape(0,self.env.action_space.shape[1])
            #all_actions = jnp.zeros((nenv, 2)) #SLOWER
            for i in range(self.batch_size):
                actions = self.actor.apply(batched_fdict[i],obs[i*self.nenv//self.batch_size:(i+1)*self.nenv//self.batch_size,:])
                all_actions = jnp.concatenate((all_actions,actions),axis=0)
                #all_actions = all_actions.at[i*nenv//batch_size:(i+1)*nenv//batch_size,:].set(actions) #SLOWER
            next_obs, rewards, dones, infos = self.env.step(all_actions)
            obs = next_obs
            #print(dones.shape) #(8192,)
            cumu_done = jnp.logical_or(cumu_done,dones)
            cumulative_return += rewards * jnp.logical_not(cumu_done)
            if all(cumu_done):
                break
        mu_returns = jnp.mean(cumulative_return.reshape(self.batch_size, self.nenv//self.batch_size), axis=1)
        return mu_returns
    
    def fix1ksteps_evaluate_single_x_on_all_envs(self, fdict):
        cumulative_return = jnp.zeros(self.nenv)
        cumu_done = jnp.full(self.nenv,False)
        obs = self.env.reset()
        for iter in range(1000):
            actions = self.actor.apply(fdict, obs)
            next_obs, rewards, dones, infos = self.env.step(actions)
            obs = next_obs
            #print(dones.shape) #(8192,)
            cumu_done = jnp.logical_or(cumu_done,dones)
            cumulative_return += rewards * jnp.logical_not(cumu_done)
        mu_return = jnp.mean(cumulative_return)
        return mu_return
    
    def fix1ksteps_evaluate_multiple_x_on_all_envs(self, batched_fdict):
        cumulative_return = jnp.zeros(self.nenv)
        cumu_done = jnp.full(self.nenv,False)
        obs = self.env.reset() 
        for iter in range(1000):
            all_actions=jnp.array([]).reshape(0,self.env.action_space.shape[1])
            #all_actions = jnp.zeros((nenv, 2)) #SLOWER
            for i in range(self.batch_size):
                actions = self.actor.apply(batched_fdict[i],obs[i*self.nenv//self.batch_size:(i+1)*self.nenv//self.batch_size,:])
                all_actions = jnp.concatenate((all_actions,actions),axis=0)
                #all_actions = all_actions.at[i*nenv//batch_size:(i+1)*nenv//batch_size,:].set(actions) #SLOWER
            next_obs, rewards, dones, infos = self.env.step(all_actions)
            obs = next_obs
            #print(dones.shape) #(8192,)
            cumu_done = jnp.logical_or(cumu_done,dones)
            cumulative_return += rewards * jnp.logical_not(cumu_done)
        mu_returns = jnp.mean(cumulative_return.reshape(self.batch_size, self.nenv//self.batch_size), axis=1)
        return mu_returns

    def single_numpy_eval(self, numpy_x):
        fdict = self.dict_builder(numpy_x)
        #converted_params = dict_unbuilder(fdict, architecture)
        #print("DELTA CHECK", og_params - converted_params) #check for loopclosure
        f_X = self.evaluate_single_x_on_all_envs(fdict)
        f_X = -f_X if self.max_or_min == "min" else f_X  #NEGATIVE BECAUSE WE TYPICALLY WANT TO MAXIMIZE RETURNS IN RL BUT SOME DOMAINS ARE FOCUSED ON MINIMIZING
        return f_X

    def batched_numpy_eval(self, batched_numpy_x):
        batched_fdict = [self.dict_builder(x) for x in batched_numpy_x]
        f_Xs = self.evaluate_multiple_x_on_all_envs(batched_fdict)
        f_Xs = -f_Xs if self.max_or_min == "min" else f_Xs  #NEGATIVE BECAUSE WE TYPICALLY WANT TO MAXIMIZE RETURNS IN RL BUT SOME DOMAINS ARE FOCUSED ON MINIMIZING
        return f_Xs
    
    def fix1ksteps_single_numpy_eval(self, numpy_x):
        fdict = self.dict_builder(numpy_x)
        #converted_params = dict_unbuilder(fdict, architecture)
        #print("DELTA CHECK", og_params - converted_params) #check for loopclosure
        f_X = self.fix1ksteps_evaluate_single_x_on_all_envs(fdict)
        return f_X

    def fix1ksteps_batched_numpy_eval(self, batched_numpy_x):
        batched_fdict = [self.dict_builder(x) for x in batched_numpy_x]
        f_Xs = self.fix1ksteps_evaluate_multiple_x_on_all_envs(batched_fdict)
        return f_Xs

