#%% 

import os 
folder = r"/home/ted/Desktop/orb_puzzle"
os.chdir(folder) 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, model_start, model_end



# Encode Wheel Speeds (ews).
class Encode_Wheel_Speeds(nn.Module):
    def __init__(
            self, 
            arg_dict = {
                "encode_size" : 16,
                "zp_zq_sizes" : [16, 16]}, 
            verbose = False):
        super(Encode_Wheel_Speeds, self).__init__()
        
        self.arg_dict = arg_dict
                
        self.example_input = torch.zeros(1, 1, 2)
        if(verbose):
            print("\nEWS Start:", self.example_input.shape)
        


        self.a = nn.Sequential(
            nn.Linear(
                in_features = 2,
                out_features = self.arg_dict["encode_size"]),
            nn.PReLU(),
            nn.Linear(
                in_features = self.arg_dict["encode_size"],
                out_features = self.arg_dict["encode_size"]),
            nn.PReLU())
        
        
        
        example = self.a(self.example_input)
        if(verbose):
            print("EWS End:")
            print("\toutput:", example.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, wheel_speeds):
        a = self.a(wheel_speeds)
        return(a)
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    ews = Encode_Wheel_Speeds(verbose = True)
    print("\n\n")
    print(ews)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(ews, ews.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
