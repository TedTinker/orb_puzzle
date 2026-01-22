import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, model_start, model_end, mu_std



# Decode Wheel Speeds (dws).
class Decode_Wheel_Speeds(nn.Module):
    def __init__(
            self, 
            hidden_state_size, 
            encoded_action_size = 0, 
            entropy = False, 
            arg_dict = {}, 
            verbose = False):
        super(Decode_Wheel_Speeds, self).__init__()
                        
        self.example_input = torch.zeros(32, 16, hidden_state_size + encoded_action_size)
        if(verbose): 
            print("\nDWS Start:", self.example_input.shape)

        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = hidden_state_size,
                out_features = 256),
            nn.PReLU())
        
        example = self.a(self.example_input)
        if(verbose):
            print("\toutput:", example.shape, "\n")

        mu = nn.Sequential(
            nn.Linear(
                in_features = 256,
                out_features = 2))
        
        self.mu_std = mu_std(mu, entropy = entropy)
        
        self.example_output, example_log_prob = self.mu_std(example)
        if(verbose): 
            print("DWS End:")
            print("\toutput:", self.example_output.shape)
            print("\tlog_prob:", example_log_prob.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, hidden_state):
        a = self.a(hidden_state)
        output, log_prob = self.mu_std(a)
        output = F.sigmoid(output)
        return(output, log_prob)
    
    
    
    @staticmethod
    def loss_func(true_values, predicted_values):
        loss_value = F.binary_cross_entropy(predicted_values, true_values, reduction = "none")
        return loss_value
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    dws = Decode_Wheel_Speeds(hidden_state_size = 128, verbose = True)
    print("\n\n")
    print(dws)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dws, dws.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    

    dws = Decode_Wheel_Speeds(hidden_state_size = 128, entropy = True, verbose = True)
    print("\n\n")
    print(dws)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dws, dws.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    

    