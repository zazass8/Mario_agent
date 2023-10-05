import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict
import os
import datetime
import random


def agent_net(input_dim, output_dim):
    c, h, w = input_dim
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(in_channels = c, out_channels=32 , kernel_size=8, stride = 4)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(in_channels = 32, out_channels=64 , kernel_size=4, stride = 2)),
        ('relu2', nn.ReLU()),
        ('conv3', nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride = 1)),
        ('relu3', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('dence1', nn.Linear(3136, 512)),
        ('relu4', nn.ReLU()),
        ('out1', nn.Linear(512, output_dim))
    ]))
    return model

class dueling_agent_net(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers separated into A/V branches
    """

    def __init__(self, input_dim, output_dim):
        super(dueling_agent_net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self.get_conv_out(input_dim)
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU())

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU())

        self.value = nn.Sequential(nn.Linear(512, 1))

        self.advantage = nn.Sequential(nn.Linear(512, output_dim))

        self.optimizer   = torch.optim.Adam(self.parameters(), lr=0.00025)
        self.loss=nn.SmoothL1Loss()

    def get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """ Forward bass including collapsing of trunks"""

        conv_out = self.conv(x).view(x.size()[0], -1)

        value = self.fc_value(conv_out)
        advantage = self.fc_advantage(conv_out)

        value = self.value(value)
        advantage = self.advantage(advantage)

        avg_advantage = torch.mean(advantage, dim=1, keepdim=True)
        Q = value + advantage - avg_advantage  

        return Q


class Deep_Agent(object):
    def __init__(self, state_dim, action_dim, algo):
        
        self.algo        = algo
        
        self.counter     = 0
        self.explo       = 1.0
        self.explo_decay = 0.9993
        self.explo_min   = 0.01
        self.gamma       = 0.9
        
        self.Table       = deque(maxlen = 30000)
        
        if self.algo=="dqn" or self.algo=="ddqn":
          self.actval_q    = agent_net(state_dim, action_dim)
          self.target_q    = agent_net(state_dim, action_dim)
          self.optimizer   = torch.optim.Adam(self.actval_q.parameters(), lr=0.00025)

        else:
          self.actval_q    = dueling_agent_net(state_dim, action_dim)
          self.target_q    = dueling_agent_net(state_dim, action_dim)
          for i in self.target_q.parameters():
            i.requires_grad = False

        
        self.batch_size  = 64
        
        self.save_dir = os.path.join(
            "checkpoints",
            f"{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    #Get Q values from online network        
    def get_actval_q_values(self, state):
        state_float = torch.FloatTensor(np.array(state))
        state_float = state_float / 255.
        if self.algo=="dqn" or self.algo=="ddqn":
          return self.actval_q(state_float)

        else:
          return self.actval_q.forward(state_float)
    
    #Get Q values from target network
    def get_target_q_values(self, state):
        state_float = torch.FloatTensor(np.array(state))
        state_float = state_float / 255.
        if self.algo=="dqn" or self.algo=="ddqn":
          return self.target_q(state_float)
          
        else:
          return self.target_q.forward(state_float)
          print("hello")
    
    
    #Returns best action at each state
    def training_act(self, state, action_dim):
        if np.random.rand() < self.explo:
            a = np.random.randint(action_dim)
        else:
            state = np.expand_dims(state, 0)
            action_values = self.get_actval_q_values(state)
            a = torch.argmax(action_values, axis=1).item()
        self.counter += 1
                    
        return a
      
    #Return best action after training with zero exploration rate
    def trained_act(self, state, action_dim):
        state = np.expand_dims(state, 0)
        action_values = self.get_actval_q_values(state)
        a = torch.argmax(action_values, axis=1).item()
        self.counter += 1
        return a

    #Store experiences in the replay buffer
    def get_xp(self, xp):
        self.Table.append(xp)


    #Get sample for training online neural network
    def get_sample(self):
        batch = random.sample(self.Table, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))
        return state, next_state, action, reward, done
    
    #Update the weights of the online neural network after backpropagation
    def updt(self, state, action, next_state, reward):
        curr_state_q = self.get_actval_q_values(state)
        pred_actval_q = curr_state_q[np.arange(0, self.batch_size), action]

        next_target_q = self.get_target_q_values(next_state)
        if(self.algo == "ddqn" or self.algo=="duelddqn"):
            next_actval_q = self.get_actval_q_values(next_state)
            a = torch.argmax(next_actval_q, axis=1)
        else:
            a = torch.argmax(next_target_q, axis=1)
        
        pred_target_q = torch.FloatTensor(reward) + (1. - done) * next_target_q[np.arange(0, self.batch_size), a] * self.gamma

        if self.algo == "duelddqn":
          loss = self.actval_q.loss(input = pred_actval_q, target = pred_target_q)
          self.actval_q.optimizer.zero_grad()
          loss.backward()
          self.actval_q.optimizer.step()

        else:
          loss=nn.functional.smooth_l1_loss(input=pred_actval_q,target=pred_target_q)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

  
        
    #Sync the target network with the online network every 5000 steps
    def sync_target_q(self):
        self.target_q.load_state_dict(self.actval_q.state_dict())
        
    def learn(self):
        state, next_state, action, reward, done = self.get_sample()
        self.updt(state, action, next_state, reward)
