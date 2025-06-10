import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
        """
        #Q1
        self.linear1_S_1 = nn.Linear(num_inputs, 2*hidden_dim)
        self.linear2_S_1 = nn.Linear(2*hidden_dim, hidden_dim)
        
        self.linear1_A_1 = nn.Linear(num_actions, hidden_dim)
        
        #self.linear1_cat_1 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.linear2_cat_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear3_cat_1 = nn.Linear(hidden_dim, 1)
        
        #Q2
        
        self.linear1_S_2 = nn.Linear(num_inputs, 2*hidden_dim)
        self.linear2_S_2 = nn.Linear(2*hidden_dim, hidden_dim)
        
        self.linear1_A_2 = nn.Linear(num_actions, hidden_dim)
        
        #self.linear1_cat_2 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.linear2_cat_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear3_cat_2 = nn.Linear(hidden_dim, 1)


        self.apply(weights_init_)
        """

    def forward(self, state, action):
        
        
        xu = torch.cat([state, action], len(action.shape)-1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x1, x2
        
        """
        #Q1
        xs_1 = F.relu(self.linear1_S_1(state))
        xs_1 = F.relu(self.linear2_S_1(xs_1))
        
        xa_1 = F.relu(self.linear1_A_1(action))
        
        xc_1 = torch.cat([xs_1, xa_1], len(action.shape)-1)
        #xc_1 = F.relu(self.linear1_cat_1(xc_1))
        xc_1 = F.relu(self.linear2_cat_1(xc_1))
        xc_1 = self.linear3_cat_1(xc_1)
        
        #Q2
        
        xs_2 = F.relu(self.linear1_S_2(state))
        xs_2 = F.relu(self.linear2_S_2(xs_2))
        
        xa_2 = F.relu(self.linear1_A_2(action))
        
        xc_2 = torch.cat([xs_2, xa_2], len(action.shape)-1)
        #xc_2 = F.relu(self.linear1_cat_2(xc_2))
        xc_2 = F.relu(self.linear2_cat_2(xc_2))
        xc_2 = self.linear3_cat_2(xc_2)
        
        return xc_1, xc_2

        
"""
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        
        super(GaussianPolicy, self).__init__()
        '''
        ## Added 4 coefficient for correlation
        self.linear1 = nn.Linear(num_inputs, 4*hidden_dim)
        self.linear2 = nn.Linear(4*hidden_dim, 2*hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim) ### ADDED
        '''
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        #self.linear3 = nn.Linear(hidden_dim, hidden_dim) ### ADDED
        
        
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        #print("X after ")
        x = F.relu(self.linear2(x))
        #x = F.relu(self.linear3(x)) ### ADDED
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        #print("Mean:",mean)
        #print("log_std:",log_std)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        #print("x_t:",x_t)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        num_actions = action.shape[-1]
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        #print("log_prob size:", log_prob.size())
        log_prob = log_prob.reshape((-1,1,num_actions))
        #print("log_prob size before sum:", log_prob.size())
        log_prob = log_prob.sum(2, keepdim=True)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)