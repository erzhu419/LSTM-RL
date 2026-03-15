import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math
import os
from bus_feature_utils import create_embedding_layer, build_bus_categorical_info
from normalization import Normalization, RunningMeanStd

# ----------------- SAC Components -----------------

class SAC_PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(SAC_PolicyNetwork, self).__init__()

        self.embedding_layer = embedding_layer
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]

        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)

        x = F.relu(self.linear1(state_with_embeddings))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = (self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state, deterministic, device):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range/2 * torch.tanh(mean + std * z) + self.action_range/2

        action = self.action_range/2 * torch.tanh(mean).detach().cpu().numpy()[0] + self.action_range/2 if deterministic else action.detach().cpu().numpy()[0]
        return action

class SAC_SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, init_w=3e-3):
        super(SAC_SoftQNetwork, self).__init__()

        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]

        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        x = torch.cat([state_with_embeddings, action], 1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

# ----------------- Ensemble Components -----------------

class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weight + self.bias

class VectorizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_critics, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        self.num_critics = num_critics

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        q_values = self.critic(state_action).squeeze(-1)
        return q_values

class Ensemble_SoftQNetwork(VectorizedCritic):
    def __init__(self, state_dim, action_dim, hidden_dim, embedding_layer, ensemble_size=10):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_critics=ensemble_size,
            embedding_layer=embedding_layer
        )

    def forward(self, state, action):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        return super().forward(state_with_embeddings, action)


# ----------------- Model Loader -----------------

class ModelLoader:
    def __init__(self, env, hidden_dim, action_range, device):
        self.env = env
        self.hidden_dim = hidden_dim
        self.action_range = action_range
        self.device = device
        
        # Build embedding info using bus_feature_utils
        self.cat_cols, self.cat_code_dict = build_bus_categorical_info(env)
        
        self.num_cat_features = len(self.cat_cols)
        self.num_cont_features = env.state_dim - self.num_cat_features
        self.action_dim = env.action_space.shape[0]

    def _init_default_norm(self):
        # Default initialization values from sac_ensemble_original_logging.py
        initial_mean = [360., 360., 90.]
        initial_std = [165., 133., 45.]
        
        # Pad to full num_cont_features (e.g. 24)
        padded_mean = np.zeros(self.num_cont_features, dtype=np.float32)
        padded_std = np.ones(self.num_cont_features, dtype=np.float32)
        
        # Fill in the known stats
        for i in range(min(len(initial_mean), self.num_cont_features)):
            padded_mean[i] = initial_mean[i]
            padded_std[i] = initial_std[i]
            
        running_ms = RunningMeanStd(shape=(self.num_cont_features,), init_mean=padded_mean, init_std=padded_std)
        return Normalization(num_categorical=self.num_cat_features, num_numerical=self.num_cont_features, running_ms=running_ms)

    def load_sac(self, model_path):
        # Hardcode cardinalities to match ensemble_10 architecture (19 embedding dims + 24 numerical = 43)
        cat_code_dict = {
            'bus_id': {i: i for i in range(25)},
            'station_id': {i: i for i in range(23)},
            'time_period': {i: i for i in range(15)},
            'direction': {0: 0, 1: 1}
        }
        embedding_layer = create_embedding_layer('full', cat_code_dict, self.cat_cols)
        
        state_dim = embedding_layer.output_dim + self.num_cont_features
        
        policy_net = SAC_PolicyNetwork(state_dim, self.action_dim, self.hidden_dim, embedding_layer, self.action_range).to(self.device)
        soft_q_net1 = SAC_SoftQNetwork(state_dim, self.action_dim, self.hidden_dim, embedding_layer).to(self.device)
        soft_q_net2 = SAC_SoftQNetwork(state_dim, self.action_dim, self.hidden_dim, embedding_layer).to(self.device)
        
        state_norm = self._init_default_norm()
        
        try:
            policy_net.load_state_dict(torch.load(model_path + '_policy', map_location=self.device, weights_only=True))
            soft_q_net1.load_state_dict(torch.load(model_path + '_q1', map_location=self.device, weights_only=True))
            soft_q_net2.load_state_dict(torch.load(model_path + '_q2', map_location=self.device, weights_only=True))
            
            if os.path.exists(model_path + '_norm'):
                state_norm = torch.load(model_path + '_norm', map_location=self.device)
        except Exception:
             pass

        policy_net.eval()
        soft_q_net1.eval()
        soft_q_net2.eval()
        
        return {
            "type": "sac",
            "policy": policy_net,
            "q1": soft_q_net1,
            "q2": soft_q_net2,
            "state_norm": state_norm
        }

    def load_ensemble(self, model_path, ensemble_size=10):
        # Same hardcoding for Ensemble
        cat_code_dict = {
            'bus_id': {i: i for i in range(25)},
            'station_id': {i: i for i in range(23)},
            'time_period': {i: i for i in range(15)},
            'direction': {0: 0, 1: 1}
        }
        embedding_layer = create_embedding_layer('full', cat_code_dict, self.cat_cols)
        
        state_dim = embedding_layer.output_dim + self.num_cont_features
        
        policy_net = SAC_PolicyNetwork(state_dim, self.action_dim, self.hidden_dim, embedding_layer, self.action_range).to(self.device)
        soft_q_net = Ensemble_SoftQNetwork(state_dim, self.action_dim, self.hidden_dim, embedding_layer, ensemble_size).to(self.device)
        
        state_norm = self._init_default_norm()

        try:
            policy_net.load_state_dict(torch.load(model_path + '_policy', map_location=self.device, weights_only=True))
            soft_q_net.load_state_dict(torch.load(model_path + '_q', map_location=self.device, weights_only=True))
            
            if os.path.exists(model_path + '_norm'):
                state_norm = torch.load(model_path + '_norm', map_location=self.device)
        except Exception:
            pass
            
        policy_net.eval()
        soft_q_net.eval()
        
        return {
            "type": "ensemble",
            "policy": policy_net,
            "q_net": soft_q_net,
            "ensemble_size": ensemble_size,
            "state_norm": state_norm
        }
