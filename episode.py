import tkinter as tk

import torch

from general_FEP_RL.utils import device



def human_wheel_override(wheel_speeds):
    """
    Opens a small window with two sliders for wheel speeds.
    Returns (left_speed, right_speed).
    """
    default_left = wheel_speeds[0] 
    default_right = wheel_speeds[1]
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



def episode(agent, env, sleep_time = 0, human_action = False):
    step_num = 0
    step_dict_list = []
    positions_for_plot_list = []
    
    while True:
        step_dict, positions_for_plot = step(agent, env, sleep_time, human_action, step_num)
        step_num += 1
        step_dict_list.append(step_dict)
        positions_for_plot_list.append(positions_for_plot)
        if(step_dict["done"]):
            #print(f"DONE: {step_num}")
            break
            
    image = torch.tensor(env.photo_for_agent()).unsqueeze(0).unsqueeze(0).to(dtype = torch.float)
    terminal_obs = {"see_image" : image}
    
    return(
        {"step_dict_list" : step_dict_list,
        "terminal_obs" : terminal_obs,
        "positions_for_plot_list" : positions_for_plot_list})
    
    
    
def step(agent, env, sleep_time, human_action = False, step_num = 0):
    if step_num == 0:
        agent.begin()
        env.begin()
    image = torch.tensor(env.photo_for_agent()).unsqueeze(0).unsqueeze(0).to(dtype = torch.float, device = device)
    obs_dict = {"see_image" : image}
    
    step_dict = agent.step_in_episode(obs_dict)
    wheel_speeds = step_dict["action"]["make_wheel_speeds"].squeeze(0).squeeze(0).tolist()

    if human_action:
        wheel_speeds = human_wheel_override(wheel_speeds)
    positions_for_plot = env.step(wheel_speeds[0], wheel_speeds[1], sleep_time = sleep_time)
    reward, orb = env.reward()
    
    done = (step_num == agent.buffer.max_episode_len - 1) or reward > 0
    
    step_dict["reward"] = reward
    step_dict["orb"] = orb
    step_dict["done"] = done 
    
    return step_dict, positions_for_plot
         
    
    
def push(agent, step_dict_list, terminal_obs):
    for i in range(len(step_dict_list)):
        if(step_dict_list[i]["done"]):
            next_obs = terminal_obs
        else:
            next_obs = step_dict_list[i+1]["obs"]
        agent.buffer.push(
            observation_dict = step_dict_list[i]["obs"], 
            action_dict = step_dict_list[i]["action"], 
            reward = step_dict_list[i]["reward"], 
            next_observation_dict = next_obs,
            done = step_dict_list[i]["done"],
            best_action_dict = None)