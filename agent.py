from general_FEP_RL.agent import Agent
from encoders.encode_image import Encode_Image
from decoders.decode_image import Decode_Image
from encoders.encode_wheel_speeds import Encode_Wheel_Speeds
from decoders.decode_wheel_speeds import Decode_Wheel_Speeds


observation_dict = {
    "see_image" : {
        "encoder" : Encode_Image,
        "encoder_arg_dict" : {
            "encode_size" : 1024,
            "zp_zq_sizes" : [1024]},
        "decoder" : Decode_Image,
        "decoder_arg_dict" : {},
        "accuracy_scalar" : 1000,                               
        "beta_obs" : .001,                      
        "eta_before_clamp" : 1,
        "eta" : 0,
        }
    }

action_dict = {
    "make_wheel_speeds" : {
        "encoder" : Encode_Wheel_Speeds,
        "encoder_arg_dict" : {
            "encode_size" : 256,
            "zp_zq_sizes" : [256]},
        "decoder" : Decode_Wheel_Speeds,
        "decoder_arg_dict" : {},
        "target_entropy" : -2,
        "alpha_normal" : 25,
        "delta" : 0
        }
    }



agent = Agent(
    observation_dict = observation_dict,       
    action_dict = action_dict,  
    hidden_state_sizes = [1024],
    time_scales = [1], 
    beta_hidden = [],
    eta_before_clamp = [],
    eta = [],
    number_of_critics = 1, 
    tau = .99,
    lr = .001,
    weight_decay = .00001,
    gamma = .99,
    capacity = 128, 
    max_steps = 25)



if __name__ == '__main__':
    agent.world_model.summary()