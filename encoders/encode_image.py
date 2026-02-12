#%% 
import os 
#folder = r"/home/ted/Desktop/orb_puzzle"
#os.chdir(folder) 

import torch
torch.set_default_device("cpu")
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, model_start, model_end

from utils_torch import Interpolate



# Encode Image (ei).
class Encode_Image(nn.Module):
    def __init__(
            self, 
            arg_dict = {
                "encode_size" : 128,
                "zp_zq_sizes" : [128, 128]}, 
            verbose = False):
        super(Encode_Image, self).__init__()
        
        self.arg_dict = arg_dict
                
        self.example_input = torch.zeros((1, 1, 16, 16, 4), device = "cpu")
        if(verbose):
            print("\nEI Start:", self.example_input.shape)

        episodes, steps, [example] = model_start([(self.example_input, "cnn")])
        if(verbose): 
            print("\tReshaped:", example.shape)
        
        self.a = nn.Sequential(
            nn.Conv2d(
                in_channels = self.example_input.shape[-1], 
                out_channels = 16, 
                kernel_size = 3, 
                padding=1, 
                padding_mode='reflect'),
            nn.LeakyReLU(),
            
            nn.PixelUnshuffle(
                downscale_factor = 2),
            
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 16, 
                kernel_size = 3, 
                padding=1, 
                padding_mode='reflect'),
            nn.LeakyReLU(),
            
            nn.PixelUnshuffle(
                downscale_factor = 2),
            
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                padding=1, 
                padding_mode='reflect'),
            nn.LeakyReLU(),)
        
        example = self.a(example)
        if(verbose): 
            print("\ta:", example.shape)
        example = example.reshape(example.shape[0], 64 * example.shape[2] * example.shape[3])
        if(verbose): 
            print("\tReshaped:", example.shape)
                
        self.b = nn.Sequential(
            nn.Linear(
                in_features = example.shape[-1],
                out_features = self.arg_dict["encode_size"]))
                
        example = self.b(example)
        if(verbose): 
            print("\toutput:", example.shape)
        
        [example] = model_end(episodes, steps, [(example, "lin")])
        self.example_output = example
        if(verbose):
            print("EI End:")
            print("\toutput:", example.shape, "\n")
                    
        self.apply(init_weights)
        
        
        
    def forward(self, image):
        image = (image * 2) - 1
        episodes, steps, [image] = model_start([(image, "cnn")])
        a = self.a(image)
        a = a.reshape(image.shape[0], 64 * a.shape[2] * a.shape[3])
        output = self.b(a)
        [output] = model_end(episodes, steps, [(output, "lin")])
        return(output)
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    ei = Encode_Image(verbose = True)
    print("\n\n")
    print(ei)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(ei, ei.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
    example_dict = {
        "encoder" : ei,
        "target_entropy" : 1,
        "accuracy_scalar" : 1,                               
        "complexity_scalar" : 1,                                 
        "eta" : 1                                   
        }
    
