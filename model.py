import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


fc = [512, 256, 128, 64]

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc[0])
        self.fc2 = nn.Linear(fc[0], fc[1])
        self.fc3 = nn.Linear(fc[1], fc[2])
        self.fc4 = nn.Linear(fc[2], fc[3])
        self.fc5 = nn.Linear(fc[-1], action_size)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.bn1 = nn.BatchNorm1d(fc[0])
        self.bn2 = nn.BatchNorm1d(fc[1])
        self.bn3 = nn.BatchNorm1d(fc[2])
        self.bn4 = nn.BatchNorm1d(fc[3])
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        #x = self.bn0(state)
        #x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.fc2(x)))
        #x = F.relu(self.bn3(self.fc3(x)))
        #x = F.relu(self.bn4(self.fc4(x)))
        #x = self.bn0(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.fc5(x)
        return torch.clamp(x,-1,1)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fc[0])
        self.fc2 = nn.Linear(fc[0]+action_size, fc[1])
        self.fc3 = nn.Linear(fc[1], fc[2])
        self.fc4 = nn.Linear(fc[2], fc[3])
        self.fc5 = nn.Linear(fc[3], 1)
        self.reset_parameters()

        self.bn0 = nn.BatchNorm1d(state_size)
        self.bn1 = nn.BatchNorm1d(fc[0])
        self.bn2 = nn.BatchNorm1d(fc[1])
        self.bn3 = nn.BatchNorm1d(fc[2])
        self.bn4 = nn.BatchNorm1d(fc[3])


    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        #xs = self.bn0(state)
        #xs = F.relu(self.bn1(self.fcs1(xs)))
        #x = torch.cat((xs, action), dim=1)
        #x = F.relu(self.bn2(self.fc2(x)))
        #x = F.relu(self.bn3(self.fc3(x)))
        #x = F.relu(self.bn4(self.fc4(x)))
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
