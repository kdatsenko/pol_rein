"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *
import sys

logger = logging.getLogger(__name__)

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

#=================PART 2==============================================
#env = gym.make('Cartpole-v0')
env = CartPoleEnv()
#???


RNG_SEED=1
np.random.seed(0)
tf.set_random_seed(RNG_SEED)
env._seed(RNG_SEED)


alpha = 0.0001
TINY = 1e-8
gamma = 0.99

# xavier initialization is another way to init weight randomly, but better
weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

w_init = weights_init
b_init = relu_init

# try:
#     output_units = env.action_space.shape[0]
# except AttributeError:
#     output_units = env.action_space.n

#input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 4 #the state vector #CHANGE
x = tf.placeholder("float",[None,NUM_INPUT_FEATURES], name='x') #state
#y - At (actions selected from A_0 to A_T-1 via bernoulli)
#probablity of actions (num_actions = output_units)
y = tf.placeholder("float",[None,1], name='y')


#The policy function should have two outputs - the probability of "left"
#and the probability of "right." Implement the policy function as a softmax 
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

#1. log_pi - should be (T x 2), 2 action probabilities for every time step
#2. tf.one_hot(y, 2, axis=1) - every row corresponds to one time step T, 
#columns are left & right probabilities - (T x 2)

#tf.multiply - Returns x * y elementwise, will set the prob of action != At to 0
#tf.reduce_sum - squeezes vector values across two rows into flat vector,
#act_pi is probability of action At for every time step t

act_pi = tf.reduce_sum(tf.multiply(log_pi, tf.one_hot(y, 2, axis=1)), reduction_indices=[1])
#original: act_pi = tf.matmul(tf.expand_dims(log_pi, 1), tf.one_hot(y, 2, axis=1))

#Returns contains all the G_t's from t=1 to t=T     
Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)

vect = act_pi * Returns #elementwise multiplication
#for gradient update for entire episode we take the sum
#Multiply by -1 because we want to maximize JavV reinforcement reward by following policy theta
loss = -tf.reduce_sum(vect) 
train_op = optimizer.minimize(loss)

#train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=25
MAX_STEPS = 50#env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

track_returns = []
total_time_steps = 0

for ep in range(200): #number of simulations of policy 
    # reset the environment
    obs = env._reset() #root state

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
        #env._render()

        # pick a single random action for time step t, based on state obs 
        action_probs = sess.run(pi, feed_dict={x:[obs]})
        #Get action sample using Bernoulli, probs already provided thanks to policy
        if random.uniform(0,1) < action_probs[0][0]:
            action = 0
        else:
            action = 1
        
        ep_actions.append(action)
        
        # Rewards are always 1.0, so its difficult
        # to distinguish between bad actions and good actions.
        
        obs, reward, done, info = env._step(action)
        ep_rewards.append(reward * I) #R_t - reward after action a
        # G is the total discounted reward, starting from time t=0
        #R1 + gamma*R2 + gamma^2*R3 + ... + gamma^(MAX_STEPS - 1)*RN
        G += reward * I 
        I *= gamma

        t += 1
        #t >= MAX_STEPS:
        if done or t >= MAX_STEPS: #run an episode until the pole drops
            break

    #if not args.load_model:
    # np.cumsum: for each index i of ep_rewards, compute sum of entry at i,
    # as well as all entries before it (like a series or a cumulative sum)
    # G_t definition: total discounted reward starting from time t
    # G_t = total - culmulative up to time t
    #returns: - set/array of all G_t's
    
    returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
    index = ep % MEMORY

    returns = np.expand_dims(returns, axis=1)
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
    total_time_steps += t
    avg_time_steps_per_episode = (float(total_time_steps) / float(ep))
    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    #print("Mean return over the last {} episodes is {}".format(MEMORY, mean_return))
    print("Cost: {}".format(sess.run(loss, feed_dict={x:np.array(ep_states),y:np.array(ep_actions), Returns:returns })))

    print("At Episode {} average number of time steps per episode = {}".format(ep, avg_time_steps_per_episode))
    # how the weights of the policy function changed ... (theta)
    print("Weights of the policy function: \n {}".format(sess.run(tf.get_variable("policy_parameters")))) 

sess.close()
