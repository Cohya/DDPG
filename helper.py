import argparse
import gym 
from ReplayMemory import ReplayMemory
import numpy as np 
import tensorflow as tf 
# parser = argparse.ArgumentParser()
# # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
# parser.add_argument('--env', type=str, default='Pendulum-v0')
# parser.add_argument('--hidden_layer_sizes', type=int, default=300)
# parser.add_argument('--num_layers', type=int, default=1)
# parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--num_train_episodes', type=int, default=200)
# parser.add_argument('--save_folder', type=str, default='ddpg_monitor')

# args = parser.parse_args()


# env = gym.make('Pendulum-v0', g=9.81)

# obs = env.reset()

# rm = ReplayMemory(capacity = 10000, number_of_channels = env.observation_space.shape[0] ,
#                  agent_history_length = 1, batch_size = 32)


# done = False 
# count = 0
# while not done:
#     a = env.action_space.sample()
    
#     next_obs, r, done, info = env.step(a)
    
#     rm.add_experience(a[0], np.expand_dims(next_obs, axis = 1), r, done)
    
#     state = next_obs
#     count += 1
    
    
# states, actions, rewards, next_states, dones = rm.get_minibatch()

# s = states[0:5]
# n,h,w,c = s.shape
# s_e = tf.reshape(s, shape = (n, h*w*c))



x = tf.Variable(3.)
y =tf.Variable(2.0)

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch([x,y])
    z = x * y 
    
gradients = tape.gradient(z, [x,y])

    