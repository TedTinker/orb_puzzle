#%%
# Pretty sure im not using transitions correctly

import os 
folder = r"/home/ted/Desktop/orb_puzzle"
os.chdir(folder) 

import tkinter as tk

import torch

from utils import plot_positions, folder
from environment import Environment
from agent import agent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



agent.load_state_dict(
    file = "saved_agent", 
    keys = None



#agent.world_model.summary()
env = Environment()



def human_wheel_override(default_left, default_right):
    """
    Opens a small window with two sliders for wheel speeds.
    Returns (left_speed, right_speed).
    """
    result = {"left": default_left, "right": default_right}

    root = tk.Tk()
    root.title("Wheel Speed Override")

    tk.Label(root, text="Left Wheel").pack()
    left_slider = tk.Scale(
        root,
        from_=-1.0,
        to=1.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        length=300
    )
    left_slider.set(default_left)
    left_slider.pack()

    tk.Label(root, text="Right Wheel").pack()
    right_slider = tk.Scale(
        root,
        from_=-1.0,
        to=1.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        length=300
    )
    right_slider.set(default_right)
    right_slider.pack()

    def submit():
        result["left"] = left_slider.get()
        result["right"] = right_slider.get()
        root.destroy()

    tk.Button(root, text="Submit", command=submit).pack(pady=10)

    root.mainloop()
    return result["left"], result["right"]

                    

max_steps_per_episode = 100
complete_epoch_dict = {}



while True:   
        step_dict_list = []
        positions_for_plot_list = []
        
        agent.begin()
        env.begin()
        step = 0
        done = False 
        
        while True:
            step += 1
            image = torch.tensor(env.photo_for_agent()).unsqueeze(0).unsqueeze(0).to(dtype = torch.float)
            obs_dict = {"see_image" : image}
            
            step_dict = agent.step_in_episode(obs_dict)
            wheel_speeds = step_dict["action"]["make_wheel_speeds"].squeeze(0).squeeze(0).tolist()
            
            
            
            default_left, default_right = wheel_speeds[0], wheel_speeds[1]
            
            # Human override via sliders
            left_speed, right_speed = human_wheel_override(default_left, default_right)
            
            positions_for_plot = env.step(left_speed, right_speed, sleep_time = .1)
            
            
            
            positions_for_plot_list.append(positions_for_plot)
            reward = env.reward()
            
            done = step == max_steps_per_episode or reward > 0
            if done and reward == 0:
                reward = -1
            
            step_dict["reward"] = reward
            step_dict["done"] = done 
            step_dict_list.append(step_dict)
            
            if(done):
                print(f"{step} steps, {reward}")
                break
    
        plot_positions(positions_for_plot_list)
    
