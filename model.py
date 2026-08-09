import torch.nn as nn
# from proj import DQNNetwork
import torch

class DQNNetwork(nn.Module):
    def __init__(self ,input_size , ouput_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size , 128)
        self.relu1 = nn.ReLU()
        # self.batchn1 = nn.BatchNorm1d(128)
        # self.dropout = nn.Dropout(0.2)
        self.fc2  = nn.Linear(128 , 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128 , 6)


    def forward(self , x):
        x = self.fc1(x)
        # print("shape" , x.dim())
        # if x.dim()>1:
        #     x = x.reshape(-1 , 16)
                
        #     x = self.batchn1(x)
        x= self.relu1(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


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