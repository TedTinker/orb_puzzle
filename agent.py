from general_FEP_RL.agent import Agent
from encoders.encode_image import Encode_Image
from decoders.decode_image import Decode_Image
from encoders.encode_wheel_speeds import Encode_Wheel_Speeds
from decoders.decode_wheel_speeds import Decode_Wheel_Speeds


observation_dict = {
    "see_image" : {
        "encoder" : Encode_Image,
        "encoder_arg_dict" : {
            "encode_size" : 128,
            "zp_zq_sizes" : [128]},
        "decoder" : Decode_Image,
        "decoder_arg_dict" : {},
        "accuracy_scalar" : 10,                               
        "beta_obs" : .001,                      
        "eta_before_clamp" : .01,
        "eta" : 0,
        }
    }

action_dict = {
    "make_wheel_speeds" : {
        "encoder" : Encode_Wheel_Speeds,
        "encoder_arg_dict" : {
            "encode_size" : 64,
            "zp_zq_sizes" : [64]},
        "decoder" : Decode_Wheel_Speeds,
        "decoder_arg_dict" : {},
        "target_entropy" : -1,
        "alpha_normal" : 0,
        "delta" : 0
        }
    }



agent = Agent(
    observation_dict = observation_dict,       
    action_dict = action_dict,  
    hidden_state_sizes = [256],
    time_scales = [1], 
    beta_hidden = [],
    eta_before_clamp = [],
    eta = [],
    number_of_critics = 2, 
    tau = .1,
    lr_world_model = .001,
    lr_critic = .001,
    lr_actor = .001,
    lr_alpha = .1,
    weight_decay = .00001,
    gamma = .99,
    d = 2,
    capacity = 128, 
    max_steps = 25,
    max_epochs_in_log = 64)



if __name__ == '__main__':
    agent.world_model.summary()