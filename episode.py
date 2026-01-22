import torch



def episode(agent, env, sleep_time = 0):
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
        positions_for_plot = env.step(wheel_speeds[0], wheel_speeds[1], sleep_time = sleep_time)
        positions_for_plot_list.append(positions_for_plot)
        reward = env.reward()
        
        done = step == agent.buffer.max_episode_len or reward > 0
        if done and reward == 0:
            reward = -1
        
        step_dict["reward"] = reward
        step_dict["done"] = done 
        step_dict_list.append(step_dict)
        
        if(done):
            print(f"{step} steps, {reward}")
            break
            
    image = torch.tensor(env.photo_for_agent()).unsqueeze(0).unsqueeze(0).to(dtype = torch.float)
    terminal_obs = {"see_image" : image}
    
    return(
        {"step_dict_list" : step_dict_list,
        "terminal_obs" : terminal_obs,
        "positions_for_plot_list" : positions_for_plot_list}
         )
    
    
    
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