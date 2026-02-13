#%%

import os 
if os.getcwd().split('/')[-1] != 'orb_puzzle': 
    os.chdir('orb_puzzle')
print(f'\n\nWorking in: {os.getcwd()}\n\n')

import argparse, ast
import pickle
import datetime
import builtins
import matplotlib

import torch
device = torch.device('cpu')



def print(*args, **kwargs):
    """Override built-in print to auto-flush."""
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

start_time = datetime.datetime.now()

def duration(start_time=start_time):
    """Return elapsed time since given start time (default: script start)."""
    delta = datetime.datetime.now() - start_time
    return delta

def print_duration(start_time, end_time, text=None, end_text=''):
    """Print the duration between two times with optional prefix text."""
    delta = end_time - start_time
    if text:
        print(f'{text}: {delta}{end_text}')
    else:
        print(f'{delta}{end_text}')

def estimate_total_duration(proportion_completed, start_time=start_time):
    """Estimate total time given progress percentage and elapsed time."""
    if proportion_completed == 0:
        return '?:??:??'
    so_far = datetime.datetime.now() - start_time
    estimated_total = so_far / proportion_completed
    estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    return estimated_total



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
parser.add_argument('--init_seed',                      type=int,           default = 777,
                    help='Random seed.')
parser.add_argument('--comp',                           type=str,           default = 'deigo',
                    help='Cluster name (deigo or saion).')
parser.add_argument('--device',                         type=str,           default = device,
                    help='Which device to use for Torch.')
parser.add_argument('--cpu',                            type=int,           default = 0,
                    help='Which cpu for affinity.')
parser.add_argument('--local',                          type=bool,          default = True,
                    help='Is this running on a local machine for testing?')  

    # Agent Arguments
parser.add_argument('--accuracy_scalar',                type=float,         default = 10,
                    help='')  
parser.add_argument('--beta_obs',                       type=float,         default = .001,
                    help='') 
parser.add_argument('--target_entropy',                 type=float,         default = -1,
                    help='')  
parser.add_argument('--alpha_normal',                   type=float,         default = 0,
                    help='')  
parser.add_argument('--eta',                            type=float,         default = 0,
                    help='')  

    # Other
parser.add_argument('--epochs',                         type=int,           default = 1000,
                    help='')  
parser.add_argument('--episodes_per_epoch',             type=int,           default = 4,
                    help='')  
parser.add_argument('--batch_size',                     type=float,         default = 64,
                    help='')  

    # Misc
parser.add_argument('--GUI',                            type=str,           default = True,
                    help='Show pybullet with GUI?') 

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
    
    
    
# I believe this should plot correctly.
#%matplotlib inline
if not args.local:
    matplotlib.use('Agg')
    
    

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