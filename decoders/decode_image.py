#%%

import os 
#folder = r"/home/ted/Desktop/orb_puzzle"
#os.chdir(folder) 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, model_start, model_end, mu_std

from utils_torch import rgb_to_circular_hsv, Interpolate



# Decode Image (di).
class Decode_Image(nn.Module):
    def __init__(
            self, 
            hidden_state_size, 
            encoded_action_size = 0, 
            entropy = False, 
            arg_dict = {}, 
            verbose = False):
        super(Decode_Image, self).__init__()
                        
        self.example_input = torch.zeros(32, 16, hidden_state_size + encoded_action_size)
        if(verbose): 
            print("\nDI Start:", self.example_input.shape)

        episodes, steps, [example] = model_start([(self.example_input, "lin")])
        if(verbose): 
            print("\tReshaped:", example.shape)
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = hidden_state_size,
                out_features = 16 * 4 * 4),
            nn.LeakyReLU())
        
        example = self.a(example)
        if(verbose): 
            print("\ta:", example.shape)
        example = example.reshape(example.shape[0], 16, 4, 4)
        if(verbose): 
            print("\tReshaped:", example.shape)
                
        self.b = nn.Sequential(
            nn.Conv2d(
                in_channels = 16, 
                out_channels = 256, 
                kernel_size = 3, 
                padding=1, 
                padding_mode='reflect'),
            nn.LeakyReLU(),
            
            nn.PixelShuffle(
                upscale_factor = 2),
            
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 256, 
                kernel_size = 3, 
                padding=1, 
                padding_mode='reflect'),
            nn.LeakyReLU(),
            
            nn.PixelShuffle(
                upscale_factor = 2),
            
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                padding=1, 
                padding_mode='reflect'),
            nn.LeakyReLU())
        
        example = self.b(example)
        if(verbose): 
            print("\tb:", example.shape)

        mu = nn.Sequential(
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 4,
                kernel_size = 1))
        
        self.mu_std = mu_std(mu, entropy = entropy)
        
        example_output, example_log_prob = self.mu_std(example)
        if(verbose): 
            print("\toutput:", example_output.shape)
            print("\tlog_prob:", example_log_prob.shape)
            
        example_log_prob = example_log_prob.mean(dim=(1, 2))
        
        [example_output, example_log_prob] = model_end(episodes, steps, [(example_output, "cnn"), (example_log_prob, "lin")])
        example_output = example_output.reshape(episodes, steps, 16, 16, 4)
        self.example_output = example_output
        if(verbose): 
            print("DI End:")
            print("\toutput:", example_output.shape)
            print("\tlog_prob:", example_log_prob.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, hidden_state):
        episodes, steps, [hidden_state] = model_start([(hidden_state, "lin")])
        a = self.a(hidden_state)
        a = a.reshape(episodes * steps, 16, 4, 4)
        b = self.b(a)
        output, log_prob = self.mu_std(b)
        output = F.tanh(output)
        output = (output + 1) / 2
        log_prob = log_prob.mean(dim = (1, 2))
        [output, log_prob] = model_end(episodes, steps, [(output, "cnn"), (log_prob, "lin")])
        return(output, log_prob)
    
    
    
    @staticmethod
    def loss_func(predicted_values, target_values):
        #loss_value = F.mse_loss(predicted_values, target_values, reduction = "none")
        #return loss_value
        
        episodes, steps, H, W, C = predicted_values.shape
        # total_loss = F.mse_loss(predicted_values, target_values, reduction='none')
        
        predicted_rgbd = predicted_values.reshape((episodes * steps, H, W, C)).permute(0, 3, 1, 2)
        predicted_rgb = predicted_rgbd[:, :-1]
        predicted_d = predicted_rgbd[:, -1]
        predicted_hsv = rgb_to_circular_hsv(predicted_rgb)
        
        target_rgbd = target_values.reshape((episodes * steps, H, W, C)).permute(0, 3, 1, 2)
        target_rgb = target_rgbd[:, :-1]
        target_d = target_rgbd[:, -1]
        target_hsv = rgb_to_circular_hsv(target_rgb)
        saturation = target_hsv[:, 2:3, :, :] 
        
        h_loss = F.mse_loss(predicted_hsv[:, :2, :, :], target_hsv[:, :2, :, :], reduction='none') * saturation
        sv_loss = F.mse_loss(predicted_hsv[:, 2:, :, :], target_hsv[:, 2:, :, :], reduction='none')
        #d_loss = F.mse_loss(predicted_d, target_d, reduction='none').unsqueeze(1)
        total_loss = torch.cat([h_loss, sv_loss], dim = 1)
        total_loss = total_loss.reshape(episodes, steps, total_loss.shape[1], H, W).permute(0, 1, 3, 4, 2)
    
        return(total_loss)
    
    
# Let's check it out!
if(__name__ == "__main__"):
    di = Decode_Image(hidden_state_size = 128, verbose = True)
    print("\n\n")
    print(di)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(di, di.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    

    di = Decode_Image(hidden_state_size = 128, entropy = True, verbose = True)
    print("\n\n")
    print(di)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(di, di.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
    example_dict = {
        "decoder" : di,
        "target_entropy" : 1,
        "accuracy_scalar" : 1,                               
        "complexity_scalar" : 1,                                 
        "eta" : 1                                   
        }
    
