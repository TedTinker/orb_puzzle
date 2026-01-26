#%%
import os 
folder = r"/home/ted/Desktop/orb_puzzle"
os.chdir(folder) 

import torch

from utils import plot_images, add_to_epoch_dict, plot_complete_epoch_dict, plot_positions, folder, plot_results
from environment import Environment
from agent import agent
from episode import episode, push



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
%matplotlib inline



agent.load_state_dict(
    file = "saved_agent", 
    keys = ["world_model", "observation_models"])


#agent.world_model.summary()
env = Environment()


                    
epochs = 5000
episodes_per_epoch = 16
batch_size = 64



results = []
complete_epoch_dict = {}



for e in range(epochs): 
    print(f"\nEpoch {e}")
        
    for ep in range(episodes_per_epoch):        
        episode_dict = episode(agent, env, sleep_time = 0) #.0001)
        push(agent, episode_dict["step_dict_list"], episode_dict["terminal_obs"])
        result = episode_dict["step_dict_list"][-1]["reward"]
        result = "None" if result == -1 else "Orb" if result == 3 else "Good Orb" 
        results.append(result)
            
        
        
    epoch_dict = agent.epoch(batch_size = batch_size)
    add_to_epoch_dict(complete_epoch_dict, epoch_dict)
    
    if(e % 10 == 0):
        plot_positions(episode_dict["positions_for_plot_list"])
        
        real_images = [obs[:, :, :-1] for obs in epoch_dict['obs']['see_image'][0]][:25]
        pred_images = [obs[:, :, :-1] for obs in epoch_dict['pred_obs_q']['see_image'][0]][:24]
        pred_images = [real_images[0] * 0] + pred_images
        
        # Can we multiply by 0 the pred_images after end of episode?

        plot_images(real_images, title = "REAL")
        
        plot_images(pred_images, title = "PRED")
        
        plot_results(results, episodes_per_epoch)
        
        plot_complete_epoch_dict(
            complete_epoch_dict, 
            folder = folder, 
            epoch = e)
        
        agent.save_state_dict(file = f"saved_agents_play/saved_agent_{e}")



# %%
