import numpy as np 
from DDPG import DDPG
from Nets import ActorNet, CriticNet
from ReplayMemory import ReplayMemory
import gym 
from gym import wrappers
import argparse
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib
from utils import smooting, test_agent, record_agent, main_statistic
import os 

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_train_episodes', type=int, default=200)
    parser.add_argument('--decay', type = float, default = 0.995)
    
    args = parser.parse_args()
    
    
    env = gym.make(args.env)
    
    ## DEFINE NETS ACTOR - CRITIC 
    mu_Net  = ActorNet(observation_dims = 3, action_dims = 1)
    mu_Net_target  = ActorNet(observation_dims = 3, action_dims = 1)
    
    q_Net =  CriticNet(observation_dims= 3, action_dims=1)
    q_Net_target =  CriticNet(observation_dims= 3, action_dims=1)
    
    batch_size = 64
    rm = ReplayMemory(capacity = int(50000) , number_of_channels = env.observation_space.shape[0],
                     agent_history_length = 1, batch_size = batch_size) # 32
    
    
    ddpg = DDPG(mu_Net = mu_Net, mu_Net_targ= mu_Net_target, 
                q_Net=q_Net, q_Net_targ= q_Net_target,  replay_memory= rm,
               action_clip=[-2,2], 
               gamma = args.gamma,
               decay = args.decay)
    
    num_train_episodes = 100 
    start_steps = 200 # After that number of steps start the trainig (till then sample random action from action space )
    num_steps = 0
    
    q_loss_vec = []
    mu_loss_vec = []
    accumulat_r_vec = []
    for episode in range(num_train_episodes):
        obs = env.reset()
        state = obs
        done = False 
        episode_length = 0 
        episode_rewards = 0
        
        noise = 'OU'
        if noise == 'OU':
            ddpg.ou_noise.reset()
            
        
        while not done:
            
            if num_steps > start_steps:
              state = np.reshape(state, newshape = (1,3)).astype(np.float32)
              a = ddpg.get_action(state, noise_scale = noise)[0] # Ornstein–Uhlenbeck noise ('OU')
            else:
              a = env.action_space.sample()
        
            next_obs, r, done, info = env.step(a)
            next_obs = np.expand_dims(next_obs, axis = 1).astype(np.float32)
            ddpg.replay_memory.add_experience(a[0], next_obs, r, done)
            num_steps += 1
            state = next_obs
            episode_length += 1
            episode_rewards += r
    
            if num_steps > batch_size:
                q_loss, mu_loss = ddpg.train()
                if episode > 0 :
                    q_loss_vec.append(q_loss.numpy())
                    mu_loss_vec.append(mu_loss.numpy())
        if episode %  20 == 0 :
            accumulat_r = test_agent(ddpg, env)
            print('#####################################')
            print("Test agent result: %.2f" % accumulat_r)
            print('#####################################')
            
        accumulat_r_vec.append(episode_rewards)
        print("Episode:", episode,
              "q_loss:",np.mean(q_loss_vec[-20:]),
              "mu_loss:",np.mean(mu_loss_vec[-20:]),
              "R:",np.mean(accumulat_r_vec[-20:]) )
                
        
    plt.rcParams["font.family"] = "Times New Roman"
    font = {'family' : 'Times New Roman',
            'size'   : 12}
    matplotlib.rc('font', **font)
    
    plt.plot(smooting(vec = q_loss_vec, size = 200))
    plt.xlabel('Episodes')  
    plt.ylabel('Critic Loss')
    
    plt.figure(2)
    plt.plot(smooting(mu_loss_vec, size = 200))
    plt.xlabel('Episodes')  
    plt.ylabel('Actor Loss')
    
    plt.figure(3)
    plt.plot(accumulat_r_vec)
    plt.xlabel('Episodes')   
    plt.ylabel("Accumulated rewards") 
    
    
    ddpg.save_weghts()
           
    record_agent(env = env, model = ddpg)
        
    main_statistic(model = ddpg,
                   env = env,
                   num_of_games = 300,
                   statistics = True,
                   video= False)

if __name__ == "__main__":
    main()

