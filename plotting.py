import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import math 

from utils import args



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
    if name and not args.local:
        os.makedirs(f'saved_{args.comp}/thesis_pics/{folder}', exist_ok=True)
        plt.savefig(f'saved_{args.comp}/thesis_pics/{folder}/{args.arg_name}_{name}.png')
    if show and args.local:
        plt.show()
    plt.close()
    
    
    
def plot_positions(positions_for_plot_list, show=True, name="", folder=""):
    robot_path = [step["robot"]["pos"] for step in positions_for_plot_list]
    robot_x = [pos[0] for pos in robot_path]
    robot_y = [pos[1] for pos in robot_path]
    
    orbs = positions_for_plot_list[0]["orbs"]
    orbs_x = [orb.pos[0] for orb in orbs]
    orbs_y = [orb.pos[1] for orb in orbs]
    orb_colors = [orb.color for orb in orbs]
    
    final_orb = positions_for_plot_list[-1]["final_orb"]
    
    walls = positions_for_plot_list[0]["walls"]
    walls_x = [wall.pos[0] for wall in walls]
    walls_y = [wall.pos[0] for wall in walls]

    plt.figure(figsize=(6, 6))
    plt.scatter(orbs_x, orbs_y, c=orb_colors, marker='s', s=100, label='Orbs')
    plt.scatter(walls_x, walls_y, c='black', marker='s', s=100, label='Walls')
    plt.plot(robot_x, robot_y, c='blue', linewidth=2, label='Robot Trajectory')
    if robot_x:
        plt.scatter(robot_x[0], robot_y[0], c='green', marker='o', s=150, zorder=5, label='Start')
        if final_orb == "Orb":
            plt.scatter(robot_x[-1], robot_y[-1], c='yellow', marker='x', s=150, zorder=5, label='End')
        elif final_orb == "Good Orb":
            plt.scatter(robot_x[-1], robot_y[-1], c='green', marker='x', s=150, zorder=5, label='End')
        else:
            plt.scatter(robot_x[-1], robot_y[-1], c='red', marker='x', s=150, zorder=5, label='End')
    plt.title("Robot Trajectory and Wall Positions")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis('equal') # Ensures X and Y scales are the same to prevent distortion
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    if name and not args.local:
        os.makedirs(f'saved_{args.comp}/thesis_pics/{folder}', exist_ok=True)
        plt.savefig(f'saved_{args.comp}/thesis_pics/{folder}/{args.arg_name}_{name}.png')
    if(show) and args.local:
        plt.show()
    plt.close()
    


def plot_results(results, episodes_per_epoch, rolling_window=10, show=True, name="", folder=""):
    df = pd.DataFrame({'Result': results})
    df['Epoch'] = df.index // episodes_per_epoch

    # Count results per epoch
    df_counts = df.groupby(['Epoch', 'Result']).size().unstack(fill_value=0)

    # Ensure all expected columns exist
    for col in ["None", "Orb", "Good Orb"]:
        if col not in df_counts.columns:
            df_counts[col] = 0

    df_counts = df_counts[["None", "Orb", "Good Orb"]]

    # Convert counts to proportions
    df_props = df_counts.div(df_counts.sum(axis=1), axis=0).fillna(0)

    # Rolling average smoothing
    df_props_smooth = df_props.rolling(window=rolling_window, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df_props_smooth.index, df_props_smooth["None"], label="None", marker='o', markersize=3)
    plt.plot(df_props_smooth.index, df_props_smooth["Orb"], label="Orb", marker='o', markersize=3)
    plt.plot(df_props_smooth.index, df_props_smooth["Good Orb"], label="Good Orb", marker='o', markersize=3)

    plt.title(f"Rolling Proportion of Results Over Epochs (window={rolling_window})")
    plt.xlabel("Epoch")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if name and not args.local:
        os.makedirs(f'saved_{args.comp}/thesis_pics/{folder}', exist_ok=True)
        plt.savefig(f'saved_{args.comp}/thesis_pics/{folder}/{args.arg_name}_{name}.png', dpi=300)

    if(show) and args.local:
        plt.show()

    plt.close()