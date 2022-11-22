import numpy as np 
from DDPG import DDPG
from ReplayMemory import ReplayMemory
import gym 
import argparse
import tensorflow as tf 
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, default='HalfCheetah-v2')
parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--hidden_layer_sizes_mu', default = [300])
parser.add_argument('--hidden_layer_sizes_Q', default = [300])
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_train_episodes', type=int, default=200)
parser.add_argument('--save_folder', type=str, default='ddpg_monitor')

args = parser.parse_args()


def test_agent(agent, env):
    accumulat_r = 0
    state = env.reset()
    
    done = False
    
    while not done:
        state = np.reshape(state, newshape = (1,3)).astype(np.float32)
        a = agent.get_action(state, noise_scale = 0)[0] 
        next_obs, r, done, _ = env.step(a)
        
        accumulat_r += r 
        state = next_obs
    return accumulat_r



env = gym.make('Pendulum-v0', g=9.81)


#int(1e6)
rm = ReplayMemory(capacity = int(1e6) , number_of_channels = env.observation_space.shape[0],
                 agent_history_length = 1, batch_size = 100)
ddpg = DDPG(action_dims= 1, observation_dims= 3, args = args, replay_memory= rm,
           action_clip=[-2,2])

num_train_episodes = 100
start_steps = 10000 #10000
num_steps = 0

q_loss_vec = []
mu_loss_vec = []
accumulat_r_vec = []
for episode in range(num_train_episodes):
    obs = env.reset()
    state = obs
    done = False 
    episode_length = 0 
    while not done:
        
        if num_steps > start_steps:
          state = np.reshape(state, newshape = (1,3)).astype(np.float32)
          a = ddpg.get_action(state, noise_scale = 0.1)[0]
        else:
          a = env.action_space.sample()
    
        next_obs, r, done, info = env.step(a)
        next_obs = np.expand_dims(next_obs, axis = 1).astype(np.float32)
        ddpg.replay_memory.add_experience(a[0], next_obs, r, done)
        num_steps += 1
        state = next_obs
        episode_length += 1
        
        if episode_length >=201:
            break 
        
    
    for i in range(episode_length):
        ## train the net 
        
        q_loss, mu_loss = ddpg.train()
        q_loss_vec.append(q_loss.numpy())
        mu_loss_vec.append(mu_loss.numpy())
        if i %  300 == 0 :
            accumulat_r = test_agent(ddpg, env)
            accumulat_r_vec.append(accumulat_r )
    print("Episode:", episode, "q_loss:",q_loss.numpy(), "mu_loss:", mu_loss.numpy(), "R:",accumulat_r )
        
plt.plot(q_loss_vec)

plt.figure(2)
plt.plot(mu_loss_vec)

plt.figure(3)
plt.plot(accumulat_r_vec)
    
        
    
    
    



