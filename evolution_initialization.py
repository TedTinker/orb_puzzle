#%%
import os
import re

import torch

from general_FEP_RL.buffer import RecurrentReplayBuffer

from utils import plot_images, add_to_epoch_dict, plot_complete_epoch_dict, plot_positions, folder
from environment import Environment
from agent import agent
from episode import episode, push

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
%matplotlib inline



start_epochs = 0
pattern = re.compile(r"saved_agent_(\d+)\.pth")

numbers = []

for filename in os.listdir(folder + "/saved_agents_evo"):
    match = pattern.fullmatch(filename)
    if match:
        numbers.append(int(match.group(1)))

if numbers:
    start_epochs = max(numbers)
    print(f"\n\n\n\nStarting at {start_epochs}\n\n\n\n")
    agent.load_state_dict(
        file = f"saved_agents_evo/saved_agent_{start_epochs}", 
        keys = None)



#agent.world_model.summary() # Using this makes all models use CUDA if available!
env = Environment()



agent.buffer = RecurrentReplayBuffer(
    agent.world_model.observation_model_dict, 
    agent.world_model.action_model_dict, 
    capacity = 256, 
    max_steps = 1)






                    
max_epochs = 10000
episodes_per_epoch = 16
max_steps_per_episode = 1
batch_size = 128



complete_epoch_dict = {}



for e in range(start_epochs, max_epochs): 
    print(f"\nEpoch {e}")
        
    for ep in range(episodes_per_epoch):        
        episode_dict = episode(agent, env)
        for step_dict in episode_dict["step_dict_list"]:
            step_dict["reward"] = 0
        push(agent, episode_dict["step_dict_list"],  episode_dict["terminal_obs"])
        
    epoch_dict = agent.epoch(batch_size = batch_size)
    add_to_epoch_dict(complete_epoch_dict, epoch_dict)
    
    if(e % 10 == 0):
        plot_positions(episode_dict["positions_for_plot_list"])
                
        real_images = [obs[1, :, :, :-1] for obs in epoch_dict['obs']['see_image']][:25]
        pred_images = [obs[:, :, :, :-1].squeeze(0) for obs in epoch_dict['pred_obs_q']['see_image']][:25]

        plot_images(real_images, title = "REAL")
        
        plot_images(pred_images, title = "PRED")
        
        plot_complete_epoch_dict(
            complete_epoch_dict, 
            folder = folder, 
            epoch = e)
        
        agent.save_state_dict(file = f"saved_agents_evo/saved_agent_{e}")



# %%
