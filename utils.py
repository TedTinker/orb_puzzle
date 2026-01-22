# I think entropy isn't applied correctly. 



import os 
os.chdir(r"C:\Users\Ted\onedrive\Desktop\orb_puzzle") 

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def relative_to(this, min, max):
    """Convert a value in [-1, 1] to the range [min, max]."""
    this = min + ((this + 1) / 2) * (max - min)
    this = [min, max, this]
    this.sort()
    return this[1]



def opposite_relative_to(this, min, max):
    """Convert a value in [min, max] to [-1, 1]."""
    return ((this - min) / (max - min)) * 2 - 1



def plot_images(images, title, show=True, name="", folder=""):
    n_images = len(images)
    columns = math.ceil(math.sqrt(n_images))
    rows = math.ceil(n_images / columns)
    fig = plt.figure(figsize=(columns + 1, rows + 1))
    fig.suptitle(title)
    for i in range(1, rows * columns + 1):
        ax = fig.add_subplot(rows, columns, i)
        if i <= n_images:
            ax.imshow(images[i - 1], vmin=0, vmax=1)
        ax.axis("off")
        rect = patches.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    if name:
        os.makedirs(f"{folder}", exist_ok=True)
        plt.savefig(f"{folder}/{name}.png")
    if show:
        plt.show()
    plt.close()
    
    
    
def add_to_epoch_dict(complete_epoch_dict, epoch_dict):
    for key, value in epoch_dict.items():
                
        if(type(value) == float):
            if(not key in complete_epoch_dict):
                complete_epoch_dict[key] = [] 
            complete_epoch_dict[key].append(value)

        if(type(value) == list):
            if(not key in complete_epoch_dict):
                complete_epoch_dict[key] = []
                for i, v in enumerate(value):
                    complete_epoch_dict[key].append([])
            for i, v in enumerate(value):
                complete_epoch_dict[key][i].append(v) 
            
        if(type(value) == dict):
            if(not key in complete_epoch_dict):
                complete_epoch_dict[key] = {}
            for k, v in value.items():
                if(not k in complete_epoch_dict[key]):
                    complete_epoch_dict[key][k] = []
                complete_epoch_dict[key][k].append(v)
                
                
                
def plot_complete_epoch_dict(complete_epoch_dict, folder="", epoch = 0):
        
    plt.figure(figsize=(6, 6))
    for key, value in complete_epoch_dict["accuracy_losses"].items():
        plt.plot(value, label=f"accuracy loss {key}")
    for key, value in complete_epoch_dict["complexity_losses"].items():
        plt.plot(value, label=f"complexity loss {key}")
    plt.title(f"Losses for Accuracy and Complexity over epochs")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #if folder != "":
    #    os.makedirs(f"{folder}/accuracy", exist_ok=True)
    #    plt.savefig(f"{folder}/accuracy/{epoch}.png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.plot(complete_epoch_dict["reward"], label="reward")
    for key, value in complete_epoch_dict["curiosities"].items():
        plt.plot(value, label=f"curiosity {key}")
    plt.plot(complete_epoch_dict["total_reward"], label="total")
    plt.title(f"Rewards over epochs")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #if folder != "":
    #    os.makedirs(f"{folder}/reward", exist_ok=True)
    #    plt.savefig(f"{folder}/reward/{epoch}.png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.plot(complete_epoch_dict["actor_loss"], label="actor loss")
    for key, alpha_entropy in complete_epoch_dict["alpha_entropies"].items():
        plt.plot(alpha_entropy, label=f"alpha entropy {key}")
    for key, alpha_normal_entropy in complete_epoch_dict["alpha_normal_entropies"].items():
        plt.plot(alpha_normal_entropy, label=f"alpha normal entropy {key}")
    for key, total_entropy in complete_epoch_dict["total_entropies"].items():
        plt.plot(total_entropy, label=f"total entropy {key}")
    plt.title(f"Actor loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #if folder != "":
    #    os.makedirs(f"{folder}/actor", exist_ok=True)
    #    plt.savefig(f"{folder}/actor/{epoch}.png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(6, 6))
    for i, critic_loss in enumerate(complete_epoch_dict["critic_losses"]):
        plt.plot(critic_loss, label=f"critic {i} loss")
    plt.title(f"Critic loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #if folder != "":
    #    os.makedirs(f"{folder}/critic", exist_ok=True)
    #    plt.savefig(f"{folder}/critic/{epoch}.png")
    plt.show()
    plt.close()
    
    
    
def plot_positions(positions_for_plot_list):
    robot_path = [step["robot"]["pos"] for step in positions_for_plot_list]
    robot_x = [pos[0] for pos in robot_path]
    robot_y = [pos[1] for pos in robot_path]
    if positions_for_plot_list:
        walls_pos = positions_for_plot_list[0]["walls"]
        walls_x = [pos[0] for pos in walls_pos]
        walls_y = [pos[1] for pos in walls_pos]
    else:
        walls_x, walls_y = [], []

    plt.figure(figsize=(6, 6))
    plt.scatter(walls_x, walls_y, c='black', marker='s', s=100, label='Walls')
    plt.plot(robot_x, robot_y, c='blue', linewidth=2, label='Robot Trajectory')
    if robot_x:
        plt.scatter(robot_x[0], robot_y[0], c='green', marker='o', s=150, zorder=5, label='Start')
        plt.scatter(robot_x[-1], robot_y[-1], c='red', marker='x', s=150, zorder=5, label='End')
    plt.title("Robot Trajectory and Wall Positions")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis('equal') # Ensures X and Y scales are the same to prevent distortion
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    #plt.savefig('trajectory_plot.png')
    plt.show()
    
    
    
