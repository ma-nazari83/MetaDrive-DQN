# from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive import MetaDriveEnv
# from metadrive.component.vehicle.vehicle_type import DefaultVehicle
# from metadrive.component.vehicle.base_vehicle import BaseVehicle
import torch.nn as nn
# from proj import DQNNetwork
import torch
import time
from model import DQNNetwork
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "race.pth"

# class DQN(nn.Module):
#     def __init__(self,state_size,hidden_size,num_actions):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_size,hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size,hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size,num_actions)
#         )

    # def forward(self,x):
    #     return self.net(x)


def discrete_to_continous(action):
    if action[0] == "right":
        direction = -1
    elif action[0]=="left":
        direction = 1
    else:
        direction = 0
    if action[1]=="accelerate":
        acceleration = 1
    else:
        acceleration = 0.3
    return (direction ,acceleration)


# def flattening_obs(obs):
#     pass
    # return torch(obs).flatten()
index_to_actions = {1:("right" , "accelerate") , 2:("right" , "brake") , 3:("left" , "accelerate") , 4:("left" ,"brake") , 5:("straight" , "accelerate") , 6:("straight" , "brake")}


def infer(state_action , model):
    state_action = torch.Tensor(state_action)
    discrete_action = model(state_action).argmax()
            # print("discrete action" ,discrete_action)
    ind = index_to_actions[int(discrete_action)+1]
    
            # print("discrete action" , discrete_action)
    contiues_action = discrete_to_continous(ind)
    return contiues_action
model = DQNNetwork(259,6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
env = MetaDriveEnv(config={"use_render": True})



# model2 = DQN(259 , 128 , 6)
# model2.load_state_dict(torch.load("target_network.pth" , map_location=torch.device('cpu')))
# print(info)
# print("conf" , env.config.keys())
# d = BaseVehicle
# print(env.config["vehicle_config"])
# vehicle = env.vehicle
# print(v.config["max_steering"])
# max_steering_angle = vehicle.config["max_steering"]
# max_engine_force = vehicle.con2max_brake_force"]


# print(max_steering_angle)

# steering_degree = action[0]*max_steering_angle
time.sleep(2)
obs, info = env.reset()

for i in range(1000):
    # action = (0,-1)
    action = infer(obs , model)
    obs, reward, terminated, truncated, info = env.step(action)
   
    # # break
    # print("step reward" , info["step_reward"])
    # print("episode reward" ,info["episode_reward"])
    # print(info)
    # break
    # print(reward)
    # random_action = env.action_space.sample()
    # print("len obs" , len(obs))
    # print('obs' , obs.shape)
    # break
    # print("obs" , obs)
    # print(random_action[0]*max_steering_angle)
    # print("action" , env.action_space.sample())
    # print(flattening_obs(obs))
    # break
    if terminated or truncated:
        env.reset()
env.close()