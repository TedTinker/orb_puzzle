#%%

# I think entropy isn't applied correctly. 



import os 
folder = r"/home/ted/Desktop/orb_puzzle"
os.chdir(folder) 

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import argparse, ast

import torch
torch.set_default_device("cpu")
device = "cpu"



# ---------------------------------------
# LIST OF ARGUMENTS
# ---------------------------------------



# Type for booleons in arguments.
def literal(arg_string): 
    return(ast.literal_eval(arg_string))


# Arguments to parse. 
parser = argparse.ArgumentParser()

    # Meta 
parser.add_argument('--arg_title',                      type=str,           default = 'default',
                    help='Title of argument-set containing all non-default arguments.') 
parser.add_argument('--arg_name',                       type=str,           default = 'default',
                    help='Title of argument-set for human-understanding.') 
parser.add_argument('--agents',                         type=int,           default = 36,
                    help='How many agents are trained in this job?')
parser.add_argument('--previous_agents',                type=int,           default = 0,
                    help='How many agents with this argument-set are trained in previous jobs?')
parser.add_argument('--init_seed',                      type=int,         default = 777,
                    help='Random seed.')
parser.add_argument('--comp',                           type=str,           default = 'deigo',
                    help='Cluster name (deigo or saion).')
parser.add_argument('--device',                         type=str,           default = device,
                    help='Which device to use for Torch.')
parser.add_argument('--cpu',                            type=int,           default = 0,
                    help='Which cpu for affinity.')
parser.add_argument('--local',                          type=bool,          default = False,
                    help='Is this running on a local machine for testing?')
parser.add_argument('--show_duration',                  type=bool,          default = False,
                    help='Should durations be printed?')
parser.add_argument('--load_agents',                    type=literal,       default = False,
                    help='Are we loading agents?')      



# Make arguments.
try:
    default_args = parser.parse_args([])
    try:    
        args = parser.parse_args()
    except: 
        args, _ = parser.parse_known_args()
except:
    import sys 
    sys.argv=[''] ; del sys           
    default_args = parser.parse_args([])
    try:    
        args = parser.parse_args()
    except: 
        args, _ = parser.parse_known_args()



# Based on arguments, adjust other arguments.
def update_args(arg_set):
    return(arg_set)

for arg_set in [default_args, args]:
    default_args = update_args(default_args) 
    args = update_args(args)
    
    

# ---------------------------------------
# MAKE A TITLE FOR ARGUMENTS, COMPARED TO DEFAULT ARGUMENTS
# ---------------------------------------

    
        
# Don't include these parameters in title.
args_not_in_title = ['arg_title', 'id', 'agents', 'previous_agents', 'init_seed']

# Make a title for the arguments. 
def get_args_title(default_args, args):
    if(args.arg_title[:3] == '___'): 
        return(args.arg_title)
    name = '' 
    first = True
    arg_list = list(vars(default_args).keys())
    arg_list.insert(0, arg_list.pop(arg_list.index('arg_name')))
    for arg in arg_list:
        if(arg in args_not_in_title): 
            pass 
        else: 
            default = getattr(default_args, arg)
            try:
                this_time = getattr(args, arg)
            except:
                this_time = 'NONE'
            if(this_time == default): 
                pass
            elif(arg == 'arg_name'):
                name += '{} ('.format(this_time)
            else: 
                if first: 
                    first = False
                else: 
                    name += ', '
                name += '{}: {}'.format(arg, this_time)
    if(name == ''): 
        name = 'default' 
    else:           
        name += ')'
    if(name.endswith(' ()')): 
        name = name[:-3]
    parts = name.split(',')
    name = '' 
    line = ''
    for i, part in enumerate(parts):
        if(len(line) > 50 and len(part) > 2): 
            name += line + '\n' ; line = ''
        line += part
        if(i+1 != len(parts)): 
            line += ','
    name += line
    return(name)

args.arg_title = get_args_title(default_args, args)

# Generate folders for saving agents and plots.
save_file = f'saved_{args.comp}'
os.makedirs(f'{save_file}', exist_ok=True)
os.makedirs(f'{save_file}/thesis_pics', exist_ok=True)
os.makedirs(f'{save_file}/thesis_pics/final', exist_ok=True)
folder = f'{save_file}/{args.arg_name}'

if(args.arg_title[:3] != '___' and not args.arg_name in ['default', 'finishing_dictionaries', 'plotting', 'plotting_predictions', 'plotting_positions']):
    os.makedirs(f'{folder}', exist_ok=True)
    os.makedirs(f'{folder}/agents', exist_ok=True)
    with open(f'{folder}/agents/args.pickle', 'wb') as handle:
        pickle.dump(args, handle)

# Print information about arguments.
if(args == default_args): 
    print('Using default arguments.')
else:
    for arg in vars(default_args):
        default = getattr(default_args, arg)
        try:
            this_time = getattr(args, arg)
        except:
            this_time = 'NONE'
        if(this_time != default):
            print('{}:\n\tDefault:\t{}\n\tThis time:\t{}'.format(arg, default, this_time))
        elif(arg == 'device'):
            print('{}:\n\tDefault:\t{}\n\tThis time:\t{}'.format(arg, default, this_time))



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