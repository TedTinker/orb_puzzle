#%%
import os 
import pickle
import matplotlib.pyplot as plt

import torch

from general_FEP_RL.plot_training_log import plot_training_log

from utils import folder, args
from plotting import plot_images, plot_positions, plot_results 
from environment import Environment
from agent import agent
from episode import episode, push



def train_agent(q = None, agent_num = 0):
    agent.load_state_dict(
        file = "saved_agent", 
        keys = ["world_model", "observation_models"])
    with open(f'saved_buffer.pickle', 'rb') as f:
        agent.buffer = pickle.load(f)



    #agent.world_model.summary()
    env = Environment()



    results = []
    complete_epoch_dict = {}



    for e in range(args.epochs): 
        
        percent_done = str(e / args.epochs)
        if q is not None:
            q.put((agent_num, percent_done))
            
        for ep in range(args.episodes_per_epoch):        
            episode_dict = episode(agent, env, sleep_time = 0) #.0001)
            push(agent, episode_dict["step_dict_list"], episode_dict["terminal_obs"])
            result = episode_dict["step_dict_list"][-1]["orb"]
            results.append(result)
                
        epoch_dict, epoch_dict_actor = agent.epoch(batch_size = args.batch_size)
        
        if(e % 10 == 0):

            plot_positions(episode_dict["positions_for_plot_list"], show=False, name=f"positions_agent_{agent_num}_epoch_{e}", folder="positions")
                    
            real_images = [obs[:, :, :-1] for obs in agent.training_log['obs']['see_image'][-1][0]][:25]
            pred_images_p = [obs[:, :, :-1] for obs in agent.training_log['pred_obs_p']['see_image'][-1][0]][:24]
            pred_images_p = [real_images[0] * 0] + pred_images_p
            pred_images_q = [obs[:, :, :-1] for obs in agent.training_log['pred_obs_q']['see_image'][-1][0]][:24]
            pred_images_q = [real_images[0] * 0] + pred_images_q
            
            plot_images(real_images, title = "REAL", show=True) #, name=f"images_{str(agent_num).zfill(3)}_real_agent", folder="images")
            plot_images(pred_images_p, title = "PRED PRIOR")
            plot_images(pred_images_q, title = "PRED POSTERIOR", show=True) #, name=f"images_agent_{str(agent_num).zfill(3)}_predicted", folder="images")
            
            plot_results(results, args.episodes_per_epoch, show = True, name=f"results_agent_{agent_num}", folder="results")
             
            #os.makedirs(f'saved_{args.comp}/thesis_pics/training_log', exist_ok=True)
        
            fig = plot_training_log(agent)
            #fig.savefig(f"saved_{args.comp}/thesis_pics/training_log/{args.arg_name}_training_log_agent_{str(agent_num).zfill(3)}.png")
            plt.show()
            plt.close(fig)
                        
            print("Number of epochs in training log:", len(agent.training_log["epoch_num"]))
            print("Epoch numbers:", agent.training_log["epoch_num"])
                        
            #with open(f'{folder}/agent_training_log_{str(agent_num).zfill(3)}.pickle', 'wb') as handle:
            #    pickle.dump(agent.training_log, handle)
                
                
                
if __name__ == "__main__":
    train_agent(q = None, agent_num = 0)

        
        
# %%