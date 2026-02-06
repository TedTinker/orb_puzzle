#%%

# I think entropy isn't applied correctly. 



import os 
folder = r"/home/ted/Desktop/orb_puzzle"
os.chdir(folder) 

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

import torch
torch.set_default_device("cpu")



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
    
    
    
def plot_positions(positions_for_plot_list):
    robot_path = [step["robot"]["pos"] for step in positions_for_plot_list]
    robot_x = [pos[0] for pos in robot_path]
    robot_y = [pos[1] for pos in robot_path]
    
    orbs = positions_for_plot_list[0]["orbs"]
    orbs_x = [orb.pos[0] for orb in orbs]
    orbs_y = [orb.pos[1] for orb in orbs]
    orb_colors = [orb.color for orb in orbs]
    
    walls = positions_for_plot_list[0]["walls"]
    walls_x = [wall.pos[0] for wall in walls]
    walls_y = [wall.pos[0] for wall in walls]

    plt.figure(figsize=(6, 6))
    plt.scatter(orbs_x, orbs_y, c=orb_colors, marker='s', s=100, label='Orbs')
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
    
    

def plot_results(results, episodes_per_epoch):
    df = pd.DataFrame({'Result': results})
    df['Epoch'] = df.index // episodes_per_epoch
    df_counts = df.groupby(['Epoch', 'Result']).size().unstack(fill_value=0)
    for col in ["None", "Orb", "Good Orb"]:
        if col not in df_counts.columns:
            df_counts[col] = 0
    df_counts = df_counts[["None", "Orb", "Good Orb"]]
    plt.figure(figsize=(10, 6))
    plt.plot(df_counts.index, df_counts["None"], label="None", marker='o', markersize=3)
    plt.plot(df_counts.index, df_counts["Orb"], label="Orb", marker='o', markersize=3)
    plt.plot(df_counts.index, df_counts["Good Orb"], label="Good Orb", marker='o', markersize=3)

    plt.title("Frequency of Results Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Count per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()