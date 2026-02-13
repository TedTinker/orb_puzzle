#%%
# Pretty sure im not using transitions correctly

import os 
folder = r"/home/ted/Desktop/orb_puzzle"
os.chdir(folder) 

import pickle

import torch

from utils import folder
from plotting import plot_positions, plot_images
from environment import Environment
from episode import episode, step, push
from agent import agent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
%matplotlib inline



agent.load_state_dict(
    file = "saved_agent", 
    keys = ["world_model", "observation_models"])



#agent.world_model.summary()
env = Environment()



max_steps_per_episode = 100
complete_epoch_dict = {}
batch_size = 64



while True:        
    episode_dict = episode(agent, env, sleep_time = .0001, human_action = True)
    push(agent, episode_dict["step_dict_list"], episode_dict["terminal_obs"], human_action = True) 
            
    with open(f'saved_buffers/saved_buffer.pickle', 'wb') as handle:
        pickle.dump(agent.buffer, handle)

