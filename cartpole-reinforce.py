#!/usr/bin/env python3

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *

import sys

env = gym.make('BipedalWalker-v2')

RNG_SEED=1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

alpha = 0.01
TINY = 1e-8
gamma = 0.99

# try:
#     output_units = env.action_space.shape[0]
# except AttributeError:
#     output_units = env.action_space.n

#input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 4 #the state vector #CHANGE
x = tf.placeholder("float",[None,NUM_INPUT_FEATURES], name='x') #state
#y - prob of actions from A_0 to A_T-1
#probablity of actions (num_actions = output_units)
y = tf.placeholder("float",[None,1], name='y')


#The policy function should have two outputs – the probability of “left” 
#and the probability of “right.’ Implement the policy function as a softmax 
#layer (i.e., a linear layer that is then passed through softmax.) 
#Note that a softmax layer is simply a fully-connected layer with a 
#softmax actication.

params = tf.get_variable("policy_parameters",[NUM_INPUT_FEATURES,2])
linear_layer = tf.matmul(x,params)
all_vars = tf.global_variables()
pi = tf.nn.softmax(linear_layer)
# tf.reduce_sum(tf.mul(probabilities, actions),reduction_indices=[1])
#pi_sample = tf.tanh(pi.sample(), name='pi_sample')
#log_pi = pi.log_prob(y, name='log_pi') #log prob


# log of prob of actions A_0 and A_1 for every state for episode
#if you compute the log-probabilities log_pi tensor of dimension [None, 2]
log_pi = tf.log(pi) #2 prob values for every state in vector of time steps 


#We make a one-hot vector of 0/1 actions stored in y
#with a one at the action we want to increase the probability of.
# (T x 2)*(2 * T) ????
# 1*2 vector, each column is sum of log prob for each action over all t's???

act_pi = tf.reduce_sum(tf.mul(log_pi, tf.one_hot(y, 2, axis=1)), reduction_indices=[1])
#original: act_pi = tf.matmul(tf.expand_dims(log_pi, 1), tf.one_hot(y, 2, axis=1))

#Returns contains all the G_t's from t=1 to t=T     
Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)

vect = act_pi * Returns #elementwise multiplication
loss = -tf.reduce_sum(vect)
train_op = optimizer.minimize(loss)

#train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=25
MAX_STEPS = 50#env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

track_returns = []
for ep in range(200): #number of simulations of policy 
    # reset the environment
    obs = env.reset() #root state

    # generate an episode following the policy with current theta
    # generating all the states and actions and rewards
    G = 0 # total discounted reward
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        ep_states.append(obs) #record the state
        env.render()
        # pick a single random action for time step t, based on state obs 
        action_probs = sess.run(pi, feed_dict={x:[obs]})
        action = 0 if random.uniform(0,1) < action_probs[0][0] else 1
        
        ep_actions.append(action)
        
        # Rewards are always 1.0, so its difficult
        # to distinguish between bad actions and good actions.
        
        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward * I) #R_t - reward after action a
        # G is the total discounted reward, starting from time t=0
        #R1 + gamma*R2 + gamma^2*R3 + ... + gamma^(MAX_STEPS - 1)*RN
        G += reward * I 
        I *= gamma

        t += 1
        #t >= MAX_STEPS:
        if done: #run an episode until the pole drops
            break

    #if not args.load_model:
    # np.cumsum: for each index i of ep_rewards, compute sum of entry at i,
    # as well as all entries before it (like a series or a cumulative sum)
    # G_t definition: total discounted reward starting from time t
    # G_t = total - culmulative up to time t
    #returns: - set/array of all G_t's
    
    returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
    index = ep % MEMORY

    returns = np.expand_dims(advantages, axis=1)
    # ep_states contains all the state S_0 to S_T-1
    # ep_actions contains all the actions from A_0 to A_T-1
    # returns contains all the G_t's from t=1 to t=T       
    _ = sess.run([train_op],
                feed_dict={x:np.array(ep_states),
                            y:np.array(ep_actions),
                            Returns:returns })

    track_returns.append(G) 
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)
    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    print("Mean return over the last {} episodes is {}".format(MEMORY,
                                                               mean_return))


    # with tf.variable_scope("mus", reuse=True):
    #     print("incoming weights for the mu's from the first hidden unit:", sess.run(tf.get_variable("weights"))[0,:]) #print theta - weights for mus


sess.close()
