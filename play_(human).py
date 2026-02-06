#%%
# Pretty sure im not using transitions correctly

import os 
folder = r"/home/ted/Desktop/orb_puzzle"
os.chdir(folder) 

import torch

from utils import plot_positions, folder, plot_images
from environment import Environment
from episode import episode, step, push
from agent import agent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
%matplotlib inline



#agent.load_state_dict(
#    file = "saved_agent", 
#    keys = None)



#agent.world_model.summary()
env = Environment()



max_steps_per_episode = 100
complete_epoch_dict = {}
batch_size = 64



while True:   
    step_num = 0
    while True:
        
        step_dict, positions_for_plot = step(agent, env, sleep_time = .01, human_action = True, step_num = step_num)
        step_num += 1
        
        plot_images(
            [
                step_dict["obs"]["see_image"][0][0][:,:,:-1], 
                step_dict["pred_obs_p"]["see_image"][0][0][:,:,:-1], 
                step_dict["pred_obs_q"]["see_image"][0][0][:,:,:-1]], 
            "OBS, PRED P, PRED Q", 
            show=True, name="", folder="")

        if(step_dict["done"]):
            break
    
