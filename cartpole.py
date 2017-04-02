import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *

import os
import sys
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

RNG_SEED=1
np.random.seed(0)
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

alpha = 1e-6
gamma = 0.99

try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

NUM_INPUT_FEATURES = env.observation_space.shape[0]

#Declared these variables in the order as in Bipedal reinforce
params = tf.get_variable("thetas", shape=[NUM_INPUT_FEATURES, output_units])
bias = tf.get_variable("bias", shape=[output_units])

#the state vector #CHANGE
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
#y - At (actions selected from A_0 to A_T-1 via bernoulli)
#probablity of actions (num_actions = output_units)
y = tf.placeholder(tf.int32, shape=(None), name='y')

#The policy function should have two outputs - the probability of "left"
#and the probability of "right." Implement the policy function as a softmax 
#layer (i.e., a linear layer that is then passed through softmax.) 
#Note that a softmax layer is simply a fully-connected layer with a 
#softmax actication.

linear_layer = tf.matmul(x, params)+bias
pi = tf.nn.softmax(linear_layer)

# log of prob of actions A_0 and A_1 for every state for episode
#if you compute the log-probabilities log_pi tensor of dimension [None, 2]
log_pi = tf.log(pi)

#We make a one-hot vector of 0/1 actions stored in y
#with a one at the action we want to increase the probability of.

#1. log_pi - should be (T x 2), 2 action probabilities for every time step
#2. tf.one_hot(y, 2, axis=1) - every row corresponds to one time step T, 
#columns are left & right probabilities - (T x 2)

#tf.multiply - Returns x * y elementwise, will set the prob of action != At to 0
#tf.reduce_sum - squeezes vector values across two rows into flat vector,
#act_pi is probability of action At for every time step t

act_pi = tf.reduce_sum(tf.multiply(log_pi, tf.one_hot(y, 2, axis=1)), reduction_indices=[1])

#Returns contains all the G_t's from t=1 to t=T  
Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * act_pi)

vect = act_pi * Returns #elementwise multiplication
#for gradient update for entire episode we take the sum
#Multiply by -1 because we want to maximize JavV reinforcement reward by following policy theta

loss = -tf.reduce_sum(vect) 
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



MEMORY=25
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

track_returns = []
total_time_steps = 0
avg_time_steps_per_episode = 0

for ep in range(10000): #number of simulations of policy
    #print avg_time_steps_per_episode
    if avg_time_steps_per_episode >= 50 and ep > 500: #wait until avg is at least 50
        break
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
    while not done: #run an episode until the pole drops
        ep_states.append(obs)
        #cartpole._render()
        
        # pick a single random action for time step t, based on state obs 
        action_probs = sess.run(pi, feed_dict={x:[obs]})
        #Get action sample using Bernoulli, probs already provided thanks to policy
        if np.random.uniform(0,1) < action_probs[0][0]:
            action = 0#tf.constant(0, dtype=tf.int32)
        else:
            action = 1#tf.constant(1, dtype=tf.int32)

        ep_actions.append(action)

        # Rewards are always 1.0, so its difficult
        # to distinguish between bad actions and good actions.

        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward * I)
        # G is the total discounted reward, starting from time t=0
        #R1 + gamma*R2 + gamma^2*R3 + ... + gamma^(MAX_STEPS - 1)*RN
        G += reward * I
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break # done generating the episode

    #if not args.load_model:
    # np.cumsum: for each index i of ep_rewards, compute sum of entry at i,
    # as well as all entries before it (like a series or a cumulative sum)
    # G_t definition: total discounted reward starting from time t
    # G_t = total - culmulative up to time t
    #returns: - set/array of all G_t's

    returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
    index = ep % MEMORY
    
    # = np.expand_dims(y, axis=1)
    # ep_states contains all the state S_0 to S_T-1
    # ep_actions contains all the actions from A_0 to A_T-1
    # returns contains all the G_t's from t=1 to t=T    
    _ = sess.run([train_op],
                feed_dict={x:np.array(ep_states),
                            y:np.array(ep_actions),
                            Returns:returns })
    
    
    #track_returns.append(G)
    #track_returns = track_returns[-MEMORY:]
    #mean_return = np.mean(track_returns)
    total_time_steps += t
    avg_time_steps_per_episode = (float(total_time_steps) / float(ep + 1))
    if ep % 100 == 0 or avg_time_steps_per_episode >= 50:
        #print("Episode {} finished after {} steps with return {}".format(ep, t, G))
        #print("Mean return over the last {} episodes is {}".format(MEMORY, mean_return))
        print("Cost: {}".format(sess.run(loss, feed_dict={x:np.array(ep_states),y:np.array(ep_actions), Returns:returns })))

        print("At Episode {} average number of time steps per episode = {}".format(ep, avg_time_steps_per_episode))
        # how the weights of the policy function changed ... (theta)

        with tf.variable_scope("", reuse=True):
            print("Weights of the policy function: \n {}".format(sess.run(tf.get_variable("thetas")))) 
            # ll1 = sess.run(pi, feed_dict={x:np.array(ep_states),y:np.array(ep_actions), Returns:returns })
            # print("pi: \n {}".format(ll1)) 
            # ll2 = sess.run(log_pi, feed_dict={x:np.array(ep_states),y:np.array(ep_actions), Returns:returns })
            # print("log_pi: \n {}".format(ll2)) 
            
            # print("returns: \n {}".format(np.array(ep_actions))) 
            # ll3 = sess.run(act_pi, feed_dict={x:np.array(ep_states),y:np.array(ep_actions), Returns:returns })
            # print("act_pi: \n {}".format(ll3)) 

sess.close()