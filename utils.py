
import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from gym import wrappers
import matplotlib.cm as cm

def smooting(vec, size):
    
    n = len(vec) // size
    
    smooted_vec = []
    
    for i in range(n):
        y = vec[ i * size : (i+1) * size]
        smooted_vec.append(np.mean(y))
        
    return smooted_vec

def test_agent(agent, env):
    accumulat_r = 0
    state = env.reset()
    
    done = False
    
    while not done:
        state = np.reshape(state, newshape = (1,3)).astype(np.float32)
        a = agent.get_action(state, noise_scale = False)[0] 
        next_obs, r, done, _ = env.step(a)
        
        accumulat_r += r 
        state = next_obs
    return accumulat_r


def record_agent(env, model, videoNum= 0):
    if not os.path.isdir('videos'):
        os.makedirs('videos')
        
    dire = './videos/' + 'vid_' + str(videoNum)
    env = wrappers.Monitor(env, dire, force = True)

    obs = env.reset()
    state = obs 
    done = False
    
    episode_reward = 0
    print("The agent is playing, please be patient...")
    
    while not done:
        state = np.reshape(state, newshape = (1,3)).astype(np.float32)
        a = model.get_action(state, noise_scale = False)[0]
        next_obs, r, done, info = env.step(a)
        next_obs = np.expand_dims(next_obs, axis = 1).astype(np.float32)
        
        state = next_obs
        episode_reward += r
        # time.sleep(0.02)
        
    print("record video game in folder video %s / " % 'vid_' + str(videoNum), "episode reward: ", episode_reward)
    return episode_reward


def main_statistic(model,env, num_of_games = 300, statistics = True, video= False):
    
    model.load_weights(mu_Net_weights_file= 'weights/weights_mu_Net.pk',
               q_Net_weights_file = 'weights/Weights_q_Net.pk')
    
    if statistics :
        rewards = []
        for i in range(num_of_games):
            r = test_agent(model, env)
            rewards.append(r)
            
            print("Episode %i: %.f" % (i, r))
            
        with open('rewards_stat.pickle', 'wb') as file:
            pickle.dump(rewards, file)
        
        colors = cm.rainbow(np.linspace(0, 1, 20))
        plt.style.use('dark_background')    
        
        fig, ax = plt.subplots()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['xtick.labelsize']= 12
        plt.rcParams['ytick.labelsize']= 12
        N, bins, patches = ax.hist(rewards, bins = 20)

        for i in range(20):
            patches[i].set_facecolor(colors[i])
    
        plt.xlabel('Accumulated Rewards', fontsize= 14)
        plt.ylabel('#Episodes', fontsize=14)
        
    if video:
        r = 0
        count = 0
        while r <= 428:
            r = record_agent(env, model, videoNum= 'epsilon_0.01_'+"best_game")
            
            count += 1
            
            print("reward:", r, "Episode:", count)
