#%%
import os 
folder = r"/home/ted/Desktop/orb_puzzle"
os.chdir(folder) 

import torch

from general_FEP_RL.plot_training_log import plot_training_log

from utils import plot_images, plot_positions, folder, plot_results
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
        
    for ep in range(episodes_per_epoch):        
        episode_dict = episode(agent, env, sleep_time = 0) #.0001)
        push(agent, episode_dict["step_dict_list"], episode_dict["terminal_obs"])
        result = episode_dict["step_dict_list"][-1]["orb"]
        results.append(result)
            
    epoch_dict, epoch_dict_actor = agent.epoch(batch_size = batch_size)
    
    if(e % 10 == 0):

        plot_positions(episode_dict["positions_for_plot_list"])
                
        real_images = [obs[:, :, :-1] for obs in agent.training_log['obs']['see_image'][-1][0]][:25]
        pred_images_p = [obs[:, :, :-1] for obs in agent.training_log['pred_obs_p']['see_image'][-1][0]][:24]
        pred_images_p = [real_images[0] * 0] + pred_images_p
        pred_images_q = [obs[:, :, :-1] for obs in agent.training_log['pred_obs_q']['see_image'][-1][0]][:24]
        pred_images_q = [real_images[0] * 0] + pred_images_q
        
        plot_images(real_images, title = "REAL")
        #plot_images(pred_images_p, title = "PRED PRIOR")
        plot_images(pred_images_q, title = "PRED POSTERIOR")
        
        plot_results(results, episodes_per_epoch)
                
        plot_training_log(
            agent, 
            folder = folder, 
            epoch = e)
        
        print(f"\nEpoch {e}")
        
        #agent.save_state_dict(file = f"saved_agents_play/saved_agent_{e}")
        
# %%
