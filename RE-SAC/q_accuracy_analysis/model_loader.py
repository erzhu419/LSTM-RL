import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math
import os
from bus_feature_utils import create_embedding_layer, build_bus_categorical_info
from normalization import Normalization, RunningMeanStd

# ----------------- Helper for Dynamic Slicing -----------------

def get_state_with_embeddings(state, embedding_layer, linear_layer, action_dim=0):
    if len(embedding_layer.cat_cols) == 4:
        cat_indices = [0, 1, 2, 3]
    else:
        cat_indices = [1, 2, 3]
        
    num_numerical_expected = linear_layer.in_features - embedding_layer.output_dim - action_dim
    cat_tensor = state[:, cat_indices]
    all_num_indices = [i for i in range(state.shape[1]) if i not in cat_indices]
    num_indices = all_num_indices[:num_numerical_expected]
    num_tensor = state[:, num_indices]
    
    if num_tensor.shape[1] < num_numerical_expected:
        padding_size = num_numerical_expected - num_tensor.shape[1]
        padding = torch.zeros(num_tensor.shape[0], padding_size).to(state.device)
        num_tensor = torch.cat([num_tensor, padding], dim=1)
    
    embedding = embedding_layer(cat_tensor.long())
    state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
    
    return state_with_embeddings

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
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.action_range = action_range

    def forward(self, state):
        state_with_embeddings = get_state_with_embeddings(state, self.embedding_layer, self.linear1, action_dim=0)
        x = F.relu(self.linear1(state_with_embeddings))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
        return mean, log_std

class SAC_SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer):
        super(SAC_SoftQNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        state_with_embeddings = get_state_with_embeddings(state, self.embedding_layer, self.linear1, action_dim=action.shape[1])
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
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        for i in range(ensemble_size): nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x): return x @ self.weight + self.bias

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
        state_action = torch.cat([state, action], dim=-1).unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        return self.critic(state_action).squeeze(-1)

class Ensemble_SoftQNetwork(VectorizedCritic):
    def forward(self, state, action):
        state_with_embeddings = get_state_with_embeddings(state, self.embedding_layer, self.critic[0], action_dim=action.shape[1])
        return super().forward(state_with_embeddings, action)

# ----------------- DSAC Components -----------------

class DSAC_QuantileNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, num_quantiles=10):
        super(DSAC_QuantileNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_quantiles)

    def forward(self, state, action):
        state_with_embeddings = get_state_with_embeddings(state, self.embedding_layer, self.fc1, action_dim=action.shape[1])
        x = torch.cat([state_with_embeddings, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# ----------------- BAC Components -----------------

class BAC_PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, action_range=1., log_std_min=-20, log_std_max=2):
        super(BAC_PolicyNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.action_range = action_range

    def forward(self, state):
        state_with_embeddings = get_state_with_embeddings(state, self.embedding_layer, self.linear1, action_dim=0)
        x = F.relu(self.linear1(state_with_embeddings))
        x = F.relu(self.linear2(x))
        x = self.LayerNorm(x)
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
        return mean, log_std

class BAC_QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer):
        super(BAC_QNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm1 = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm2 = nn.LayerNorm(hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        state_with_embeddings = get_state_with_embeddings(state, self.embedding_layer, self.linear1, action_dim=action.shape[1])
        xu = torch.cat([state_with_embeddings, action], 1)
        x1 = self.linear3(self.LayerNorm1(F.relu(self.linear2(F.relu(self.linear1(xu))))))
        x2 = self.linear6(self.LayerNorm2(F.relu(self.linear5(F.relu(self.linear4(xu))))))
        return x1, x2

# ----------------- Model Loader -----------------

class ModelLoader:
    def __init__(self, env, hidden_dim, action_range, device):
        self.env, self.hidden_dim, self.action_range, self.device = env, hidden_dim, action_range, device
        self.cat_cols, self.cat_code_dict = build_bus_categorical_info(env)
        self.num_cat_features = len(self.cat_cols)
        self.num_cont_features = env.state_dim - self.num_cat_features
        self.action_dim = env.action_space.shape[0]

    def _init_default_norm(self):
        running_ms = RunningMeanStd(shape=(self.num_cont_features,), init_mean=[360., 360., 90.], init_std=[165., 133., 45.])
        return Normalization(num_categorical=self.num_cat_features, num_numerical=self.num_cont_features, running_ms=running_ms)

    def load_sac(self, path):
        dict_cd = {'bus_id': {i:i for i in range(25)}, 'station_id': {i:i for i in range(23)}, 'direction': {0:0, 1:1}}
        emb = create_embedding_layer('full', dict_cd, ['station_id', 'bus_id', 'direction'])
        pol = SAC_PolicyNetwork(18+25, self.action_dim, self.hidden_dim, emb, self.action_range).to(self.device)
        q1 = SAC_SoftQNetwork(18+25, self.action_dim, self.hidden_dim, emb).to(self.device)
        q2 = SAC_SoftQNetwork(18+25, self.action_dim, self.hidden_dim, emb).to(self.device)
        try:
            pol.load_state_dict(torch.load(path+'_policy', map_location=self.device, weights_only=True))
            q1.load_state_dict(torch.load(path+'_q1', map_location=self.device, weights_only=True))
            q2.load_state_dict(torch.load(path+'_q2', map_location=self.device, weights_only=True))
        except Exception as e: print(f"FAILED SAC: {e}")
        return {"type":"sac","policy":pol,"q1":q1,"q2":q2,"state_norm":self._init_default_norm()}

    def load_ensemble(self, path, ens_size=10):
        dict_cd = {'bus_id': {i:i for i in range(25)}, 'station_id': {i:i for i in range(23)}, 'direction': {0:0, 1:1}}
        emb = create_embedding_layer('full', dict_cd, ['station_id', 'bus_id', 'direction'])
        pol = SAC_PolicyNetwork(18+25, self.action_dim, self.hidden_dim, emb, self.action_range).to(self.device)
        q = Ensemble_SoftQNetwork(18+25, self.action_dim, self.hidden_dim, emb, ens_size).to(self.device)
        try:
            pol.load_state_dict(torch.load(path+'_policy', map_location=self.device, weights_only=True))
            q.load_state_dict(torch.load(path+'_q', map_location=self.device, weights_only=True))
        except Exception as e: print(f"FAILED ENS: {e}")
        return {"type":"ensemble","policy":pol,"q_net":q,"ensemble_size":ens_size,"state_norm":self._init_default_norm()}

    def load_dsac(self, path, n_q=10):
        dict_cd = {'bus_id': {i:i for i in range(25)}, 'station_id': {i:i for i in range(22)}, 'direction': {0:0, 1:1}}
        emb = create_embedding_layer('full', dict_cd, ['station_id', 'bus_id', 'direction'], layer_norm=True)
        pol = SAC_PolicyNetwork(18+25, self.action_dim, self.hidden_dim, emb, self.action_range).to(self.device)
        z1 = DSAC_QuantileNetwork(18+25, self.action_dim, self.hidden_dim, emb, n_q).to(self.device)
        z2 = DSAC_QuantileNetwork(18+25, self.action_dim, self.hidden_dim, emb, n_q).to(self.device)
        try:
            pol.load_state_dict(torch.load(path+'_policy', map_location=self.device, weights_only=True))
            z1.load_state_dict(torch.load(path+'_z1', map_location=self.device, weights_only=True))
            z2.load_state_dict(torch.load(path+'_z2', map_location=self.device, weights_only=True))
        except Exception as e: print(f"FAILED DSAC: {e}")
        return {"type":"dsac","policy":pol,"z1":z1,"z2":z2,"num_quantiles":n_q, "state_norm":self._init_default_norm()}

    def load_bac(self, path):
        dict_cd = {'bus_id': {i:i for i in range(25)}, 'station_id': {i:i for i in range(22)}, 'time_period': {i:i for i in range(15)}, 'direction': {0:0, 1:1}}
        # BAC specific embedding dims: 7, 11, 12, 1 (total 31)
        emb = create_embedding_layer('full', dict_cd, ['time_period', 'station_id', 'bus_id', 'direction'], layer_norm=True)
        # Manually override embedding dimensions if create_embedding_layer doesn't match
        emb.embeddings.time_period = nn.Embedding(15, 7)
        emb.embeddings.station_id = nn.Embedding(22, 11)
        emb.embeddings.bus_id = nn.Embedding(25, 12)
        emb.embeddings.direction = nn.Embedding(2, 1)
        emb.layer_norm = nn.LayerNorm(31)
        emb.output_dim = 31

        pol = BAC_PolicyNetwork(31+24, self.action_dim, self.hidden_dim, emb, self.action_range).to(self.device)
        q = BAC_QNetwork(31+24, self.action_dim, self.hidden_dim, emb).to(self.device)
        try:
            pol.load_state_dict(torch.load(path+'_policy', map_location=self.device, weights_only=True))
            q.load_state_dict(torch.load(path+'_q', map_location=self.device, weights_only=True))
        except Exception as e: print(f"FAILED BAC: {e}")
        return {"type":"bac","policy":pol,"q_net":q,"state_norm":self._init_default_norm()}
