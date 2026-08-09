import torch 
# import pandas as pd
import numpy as np
import torch.nn as nn
import random
from torch.utils.data import Dataset 
from metadrive import MetaDriveEnv
import math
import argparse
from collections import deque
from model import DQNNetwork
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "race.pth"

parser = argparse.ArgumentParser()
parser.add_argument("--train_steps", type=int, default=200000)
args = parser.parse_args()
train_steps = args.train_steps
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(device)
def is_whole_vehicle_in_lane(vehicle, lane=None):
    # Use the currently assigned lane by default
    if lane is None:
        lane = vehicle.lane

    if lane is None:
        return False

    # True only when every corner of the vehicle body is inside the lane polygon
    if all(lane.point_on_lane(corner) for corner in vehicle.bounding_box):
        return 0.2
    else:
        return -20

def new_reward_fn(info , action , prev_action , vehicle ):
    reward= 0
    speed = info["velocity"]
    if speed <=30:
        reward += (0.2*speed)/30
    elif speed > 30:
        reward -= ((speed-30)*2/30)
    delta = np.linalg.norm((action[0] - prev_action[0] , action[1] - prev_action[1]))
    if delta <1 and delta >0:
        reward += 0.05*(1-delta)
    elif delta<=0 :
        reward+=0.05
    reward += is_whole_vehicle_in_lane(vehicle)
    if  info["crash"] or info["out_of_road"] or speed>45:
        reward -=50
    if info["arrive_dest"]:
        reward+=100
    #TODO add smoothing reward
    reward+=info["step_reward"]
   
    return reward

def returning_probs(buffer):
    t = torch.arange(1, len(buffer) + 1, dtype=torch.float64)
    probs = torch.softmax(t, dim=0).numpy()

    # final safety normalization for np.random.choice
    probs = probs / probs.sum()

    return probs
# print(returning_probs(list(range(10))))
    
    # probs = list(torch.softma)
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
        acceleration = 0
    return (direction ,acceleration)
def continous_to_discrete(actions):
    keys = []

    for action in list(actions):
        action = action.squeeze()
        action[0] , action[1] = float(action[0]) , float(action[1])
        if action[0]==-1:
            direction = "right"
        elif action[0]==1:
            direction = "left"
        else:
            direction = "straight"
        if action[1]== 1:
            accelerate = "accelerate"
        else:
            accelerate = "brake"
        for key  , value in index_to_actions.items():
            if value[0]==direction and value[1]== accelerate:
                keys.append(key-1)
    # print(len(keys))
    return (keys)
     
# print(continous_to_discrete([(0, 1), (1, 0)]))

# def flattening_obs(obs):
#     pass
    # return torch(obs).flatten()
index_to_actions = {1:("right" , "accelerate") , 2:("right" , "brake") , 3:("left" , "accelerate") , 4:("left" ,"brake") , 5:("straight" , "accelerate") , 6:("straight" , "brake")}

class CustomDataset(Dataset):
    def __init__(self , data):
        self.data = data
        # self.states = np.array(list(map(lambda x : x[0] , self.data)))
        # self.actions = np.array(list(map(lambda x : x[1] , self.data)))
        # self.rewards = np.array(list(map(lambda  x : x[2] , self.data)))
        # self.terminated = np.array(list(map(lambda x : x[-2] , self.data)))
        # self.next_states = np.array(list(map(lambda x : x[3] , self.data)))
        # self.truncated = np.array(list(map(lambda x : x[-1] , self.data)))
    def __getitem__(self , idx):
        state = np.array(self.data[idx][0] , dtype = np.float32)
        action = np.array(self.data[idx][1] , dtype = np.float32)
        reward = np.array(self.data[idx][2] , dtype = np.float32)
        terminated = np.array(self.data[idx][-2] , dtype = np.float32)
        next_state = np.array(self.data[idx][3] , dtype = np.float32)
        truncated = np.array(self.data[idx][-1] , dtype = np.float32)

        # print(self.states[idx])
        # print(self.rewards[idx])
        return torch.from_numpy(state) , torch.from_numpy(reward) , torch.from_numpy(terminated) , torch.from_numpy(next_state) , torch.from_numpy(action) , torch.from_numpy(truncated)
    def __len__(self):
        return len(self.data)


# class DQNNetwork(nn.Module):
#     def __init__(self ,input_size , ouput_size):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size , 128)
#         self.relu1 = nn.ReLU()
#         # self.batchn1 = nn.BatchNorm1d(128)
#         # self.dropout = nn.Dropout(0.2)
#         self.fc2  = nn.Linear(128 , 128)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(128 , 6)


#     def forward(self , x):
#         x = self.fc1(x)
#         # print("shape" , x.dim())
#         # if x.dim()>1:
#         #     x = x.reshape(-1 , 16)
                
#         #     x = self.batchn1(x)
#         x= self.relu1(x)
#         # x = self.dropout(x)

#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.fc3(x)
#         return x
        
class ReplayBuffer():
    def __init__(self ,batch_size):
        # pass
        self.trajectories = deque(maxlen=10000)
        self.batch_size = batch_size


    def sample_batch(self):
        prob = returning_probs(self.trajectories)
        indices = np.random.choice(len(self.trajectories), size=self.batch_size, p=prob)   
        samples = [self.trajectories[i] for i in indices]
        # (previous_obs ,action ,new_reward , obs , terminated , truncated )
        # Batch lists directly
        states = np.array([s[0] for s in samples], dtype=np.float32)
        actions = np.array([s[1] for s in samples], dtype=np.float32)
        rewards = np.array([s[2] for s in samples], dtype=np.float32)
        next_states = np.array([s[3] for s in samples], dtype=np.float32)
        terminated = np.array([s[4] for s in samples], dtype=np.float32)
        truncated = np.array([s[5] for s in samples], dtype=np.float32)
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        terminated = torch.from_numpy(terminated)
        truncated = torch.from_numpy(truncated)
        
        return states, rewards, terminated, next_states, actions, truncated
    # def sample_batch(self ):
    #     prob = returning_probs(self.trajectories)
    #     # print(*prob)
    #     # exit()
    #     # print(len(self.trajectories))
    #     indices = np.random.choice(
    #         len(self.trajectories),
    #         size=self.batch_size,
    #         p=prob
    #         )   

    #     samples = [self.trajectories[i] for i in indices]
    #     # print(samples)
    #     # samples = random.sample(self.trajectories , k = 1*self.batch_size)
    #     custom_data = CustomDataset(samples)
    #     # for states , rewards in custom_data:
    #         # print(states , rewards)
    #     data_loader =DataLoader(custom_data , batch_size=  self.batch_size , shuffle = True)
    #     # print("length of dataloader" , len(data_loader))
    #     return data_loader
        # pass
    def add_new_transitions(self , state1 , action , reward , state2 , end_or_not , truncated):
        self.trajectories.append((state1, action ,reward , state2 , end_or_not , truncated))
    def __len__(self):
        return len(self.trajectories)
    def emptying_buffer(self):
        self.trajectories.pop(0)

class DQNAgent():
    def __init__(self , Q_network , target_network , ReplayBuffer , optimizer , train_steps):
        self.Q_network = Q_network
        self.target_network = target_network
        self.ReplayBuffer = ReplayBuffer
        self.optimizer = optimizer
        self.epsilon = 1
        self.gamma = 0.99
        self.k = 1
        decay_steps = int(train_steps * 0.60)
        self.epsilon_decay_value = (1.0 - 0.01) / decay_steps
    def choosing_action(self , state_action):
        number =  random.random()
        # print(self.epsilon)
        if number> self.epsilon:
            state_action = torch.Tensor(state_action).to(device)

            self.Q_network.eval()
            with torch.no_grad():
                discrete_action = self.Q_network(state_action).argmax()
            # print("discrete action" ,discrete_action)
            ind = index_to_actions[int(discrete_action)+1]
            
            # print("discrete action" , discrete_action)
            contiues_action = discrete_to_continous(ind)
            # print(contiues_action)
        else:
            contiues_action = (random.sample([1,-1,0] , k=1)[0] , random.sample([1 ,0.3] , k=1)[0])
        return contiues_action
        
        
    def train(self , epochs , x):
        criterion = nn.MSELoss()
        # if len(self.ReplayBuffer)>32:
        data = [self.ReplayBuffer.sample_batch()]
        all_losses = np.zeros(epochs)
        for i in range(epochs):
            train_loss = 0
            for batch in  data:
                states, rewards, terminated, next_state, actions, truncated = batch
                states = states.to(device)
                next_state = next_state.to(device)
                rewards = rewards.to(device)
                terminated = terminated.to(device)
                truncated = truncated.to(device)
                # actions = actions.to(device)
                action_indexes = continous_to_discrete(actions)
                action_indexes = torch.tensor(action_indexes, dtype=torch.long).to(device)
                # print(action_indexes)
                # print(states)
                # print(next_state)
                self.optimizer.zero_grad()
                # x, y = x.to(device) , y.to(device)
                # x  =torch.cat(states , actions)
                self.Q_network.train()
                result = self.Q_network(states)
                
                # print(result)
                # print(action_indexes)
                # result = [result[i][action_indexes[i]-1] for i in range(len(action_indexes))]
                # result = torch.Tensor(result)
                # print(result.shape)
                # result = result[: , action_indexes]
                result = result.gather(1, action_indexes.unsqueeze(1)).squeeze(1)  # 
                # print(result.shape)
                # exit()
                # for a in all_actions:
                # all_qs = []
                # with 
                self.target_network.eval()

                with torch.no_grad():
                    qs = self.target_network(next_state)
                    qs = qs.reshape(self.ReplayBuffer.batch_size,6).max(dim=1).values
                    qs = qs.reshape(self.ReplayBuffer.batch_size , 1)
                    # print("q shape" , qs.shape)
                    # print("qs shape" , qs.reshape(16,6).max(dim=1).values.shape)
                    # print("qs for a batch" ,qs.shape)
                    # print(qs.max(dim=1).values)
                    # print(self.gamma*qs.shape)
                    # print((1-terminated).shape)

                    # done = terminated.int() | truncated.int()
                    # done = (terminated.bool() | truncated.bool()).float()
                    done = terminated.float()
                    # print("fs shape" , fs.shape)
                    # print(done.shape)
                    qs = qs.view(-1)
                    ss =  (1-done)*self.gamma*qs
                    # print(rewards)
                    # print(fs.shape)
                    # print(ss.shape)
                    # ss = ss.reshape(self.ReplayBuffer.batch_size,1)
                    # print(ss.shape)
                    # print(rewards.shape)
                    ys = rewards+(ss)
                    # print(ys.shape)
                    # exit()
                    # ys = fs*(ss.reshape(-1  ,1))
                    # print(ys.shape)
                    # print()
                    # ys = 1 + (1-terminated)*
                    # exit()
                    #TODO fix here
                    # y = rewards + 
                # target = self.target_network(x)
                # print(ys.shape)
                ys = ys.reshape(self.ReplayBuffer.batch_size)
                # print(result.shape , ys.shape)
                # exit()

                # exit()
                # print(result)
                # print(ys)
                # exit()
                loss = criterion(result , ys)
                # print(loss)
                train_loss+= loss.item()
                # print("result" , result)
                # print("ys" , ys)
                # exit()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Q_network.parameters(), max_norm=1.0) # Add this line!
                self.optimizer.step()
            # for states  , rewards , terminated in data:
            #     with torch.inference_mode:
            #        result = self.Q_network(states) 
            # print(f"epoch {i} train loss : {train_loss}")
            all_losses[i] =loss.item()
        # print(f"time step{x} ,epoch {i}", all_losses.mean())
        # return self.Q_network
    
    def updating_target_network(self, tau=0.005):
        for target_param, q_param in zip(self.target_network.parameters(), self.Q_network.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)
    # def updating_target_network(self):
    #     self.target_network.load_state_dict(self.Q_network.state_dict())
        # return 
    def decrease_epsilon(self):
        self.epsilon  = self.epsilon*0.999993
        if self.epsilon< 0.01:
            self.epsilon=0.01
        # if self.epsilon<0.3:
            # self.epsilon=0.
        # return

# training_env = MetaDriveEnv(dict(
#     num_scenarios=3,
#     traffic_density  = 0.3,
#     start_seed=1000,
#     random_lane_width=True,
#     random_agent_model=False,
#     random_lane_num=True
# ))
# env_config = {
#     "use_render": False,

#     # Dense reward: make forward route progress valuable
#     "driving_reward": 1.0,

#     # Keep this small initially. Too much speed reward makes the car rush and crash.
#     "speed_reward": 0.02,

#     # Multiply progress reward by a factor based on distance from lane center.
#     # This encourages lane keeping.
#     "use_lateral_reward": True,

#     # Terminal rewards
#     "success_reward": 20.0,
#     "out_of_road_penalty": 10.0,
#     "crash_vehicle_penalty": 10.0,
#     "crash_object_penalty": 10.0,
#     "crash_sidewalk_penalty": 10.0,

#     # Keep episode lengths bounded while training
#     "horizon": 1000,
# }
training_env = MetaDriveEnv(config={"use_render": False , "vehicle_config":{"show_lidar":False , "show_side_detector":False , "show_lane_line_detector":False}})
# print("trainig env length" ,len(training_env))
q_net = DQNNetwork(259 , 6)
target_net = DQNNetwork(259 , 6)
target_net.load_state_dict(q_net.state_dict())
replay_buffer = ReplayBuffer(64)

optimizer = torch.optim.Adam(q_net.parameters() , lr = 0.0001)
Agent = DQNAgent(q_net.to(device) , target_net.to(device)  , replay_buffer  ,optimizer , train_steps)

    
trained = False

obs , info = training_env.reset()
previous_obs = obs
previous_action = (0, 0)
for i in range(train_steps):
    
    action  = Agent.choosing_action(obs)
    # print(action)
    # execute 10 step
    obs, reward, terminated, truncated, info  =training_env.step(action)
    
    if previous_action!= None:
        new_reward = new_reward_fn(info , action, previous_action, training_env.agent)

        Agent.ReplayBuffer.add_new_transitions(previous_obs ,action ,new_reward , obs , terminated , truncated )
    # print(trained)
    # print(terminated)
    if len(Agent.ReplayBuffer)> Agent.ReplayBuffer.batch_size* Agent.k and i > 5000 :
        Agent.train(1,i)
        Agent.decrease_epsilon()
        trained = True
    if trained and i%(train_steps//200)==0 :
        Agent.updating_target_network()
        print("agent updated" , i)
        # Agent.ReplayBuffer.emptying_buffer()
    # if trained and len(Agent.ReplayBuffer)> 10000:
    #     Agent.ReplayBuffer.emptying_buffer()
        # print("buffer len" , len(Agent.ReplayBuffer) , i)
        
    # if trained:
        
    # print(Agent.epsilon)
    if terminated or truncated:
        obs, info = training_env.reset()
        previous_action = None
    else:
    # print("epsilon",Agent.epsilon)
        # print("new epsilon" , Agent.epsilon)
    # if i%50==0:
    #     print(f"time step {i}")
        previous_obs  = obs
        previous_action = (0, 0)

training_env.close()
weights = Agent.Q_network.state_dict()
torch.save(weights , MODEL_PATH)

    # # evaluation
    # print("Evaluate checkpoint for training epoch {}...\n".format(training_epoch))
    # test_env.reset()
    # for _ in range(10):
    #     # execute 10 evaluation step
    #     test_env.step(test_env.action_space.sample())
    # test_env.close()


