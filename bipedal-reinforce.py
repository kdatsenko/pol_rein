#!/usr/bin/env python3

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *

import sys



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--load-model', metavar='NPZ',
                    help='NPZ file containing model weights/biases')
args = parser.parse_args()


#The policy function pi is implemented as a single-hidden-layer neural network.

#Explain precisely why the code corresponds to the pseudocode on p. 271 of 
#Sutton & Barto . Specifically, in your report, explain how all the terms 
#( Gt , pi , and the update to theta(i.e. parameters)) are computed, quoting the 
#relevant lines of Python.


env = gym.make('BipedalWalker-v2')

RNG_SEED=1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

hidden_size = 64
alpha = 0.01
TINY = 1e-8
gamma = 0.98

weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

if args.load_model:
    model = np.load(args.load_model)
    hw_init = tf.constant_initializer(model['hidden/weights'])
    hb_init = tf.constant_initializer(model['hidden/biases'])
    mw_init = tf.constant_initializer(model['mus/weights'])
    mb_init = tf.constant_initializer(model['mus/biases'])
    sw_init = tf.constant_initializer(model['sigmas/weights'])
    sb_init = tf.constant_initializer(model['sigmas/biases'])
else:
    hw_init = weights_init
    hb_init = relu_init
    mw_init = weights_init
    mb_init = relu_init
    sw_init = weights_init
    sb_init = relu_init

try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 24 #the state vector
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
#probablity of actions (num_actions = output_units)
y = tf.placeholder(tf.float32, shape=(None, output_units), name='y')

#The policy function pi is implemented as a single-hidden-layer neural network.

hidden = fully_connected(
    inputs=x, #inputs x
    num_outputs=hidden_size,
    activation_fn=tf.nn.relu,
    weights_initializer=hw_init,
    weights_regularizer=None,
    biases_initializer=hb_init,
    scope='hidden')
    
    
#Gaussian distribution policy for the actions

# mus = phi(s, a)^T times theta
mus = fully_connected(
    inputs=hidden, #hidden layer
    num_outputs=output_units,
    activation_fn=tf.tanh,
    weights_initializer=mw_init,
    weights_regularizer=None,
    biases_initializer=mb_init,
    scope='mus')

# the sigmas of the gaussian distribution,
#also computed based on the hidden unit values
sigmas = tf.clip_by_value(fully_connected(
    inputs=hidden, #units of hidden layer 
    num_outputs=output_units,
    activation_fn=tf.nn.softplus,
    weights_initializer=sw_init,
    weights_regularizer=None,
    biases_initializer=sb_init,
    scope='sigmas'),
    TINY, 5)

all_vars = tf.global_variables()

#policy function
# probs computed for continuous values of actions, for some state,
# depending on state input x (input to network)
pi = tf.contrib.distributions.Normal(mus, sigmas, name='pi')
pi_sample = tf.tanh(pi.sample(), name='pi_sample')
# log of prob of actions from A_0 to A_T-1
log_pi = pi.log_prob(y, name='log_pi') #log prob

#Returns contains all the G_t's from t=1 to t=T     
Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)

#maximize (Returns * log_pi) - JavV expected value for every time t
# dot prod of all Gts and all log prob for all At (Gt dot At)
#(Returns * log_pi) - represents rewards from policy with current theta
# Rather than doing an update for every time step of an episode,
# does an update for entire episode at once (by computing cumulative reward for
# all time steps)
# tensorflow computes gradient and runs gradient descent...
train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=25
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

track_returns = []
for ep in range(16384): #number of simulations of policy 
    # reset the environment
    obs = env.reset() #root state

    # generate an episode following the policy with current theta
    # generating all the states and actions and rewards
    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        ep_states.append(obs)
        env.render()
        # pick a single random action for time step t, based on state obs - input x
        # pi_sample is a randomly generated action based on probablities in pi
        action = sess.run([pi_sample], feed_dict={x:[obs]})[0][0]
        ep_actions.append(action)
        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward * I) #R_t - reward after action a
        # G is the total discounted reward, starting from time t=0
        #R1 + gamma*R2 + gamma^2*R3 + ... + gamma^(MAX_STEPS - 1)*RN
        G += reward * I 
        I *= gamma #rewards farther away in time count less (greedy approach)
        #gamma between 0 and 1

        t += 1
        if t >= MAX_STEPS:
            break
            # done generating the episode

    if not args.load_model:
        # np.cumsum: for each index i of ep_rewards, compute sum of entry at i,
        # as well as all entries before it (like a series or a cumulative sum)
        # G_t definition: total discounted reward starting from time t
        # G_t = total - culmulative up to time t
        #returns: - set/array of all G_t's
        returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
        index = ep % MEMORY
        
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


    with tf.variable_scope("mus", reuse=True):
        print("incoming weights for the mu's from the first hidden unit:", sess.run(tf.get_variable("weights"))[0,:]) #print theta - weights for mus


sess.close()
