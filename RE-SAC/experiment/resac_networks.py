"""
RE-SAC Networks: Vectorized Ensemble Q-Networks for MuJoCo benchmarks.
Adapted from sac_ensemble_SUMO_linear_penalty.py, stripped of bus-env specifics.
"""
import math
import torch
import torch.nn as nn


class VectorizedLinear(nn.Module):
    """Batched linear layer for ensemble computation.

    Processes all ensemble members in parallel via a single batched matmul.
    Weight shape: (ensemble_size, in_features, out_features)
    """

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
        # x: (ensemble_size, batch_size, in_features)
        return x @ self.weight + self.bias


class VectorizedQNetwork(nn.Module):
    """Vectorized ensemble of Q-networks.

    All ensemble members share the same architecture but have independent
    parameters, computed in parallel via VectorizedLinear layers.

    Args:
        obs_dim: observation dimension
        action_dim: action dimension
        hidden_dim: hidden layer size
        ensemble_size: number of ensemble members
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, ensemble_size):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.ensemble_size = ensemble_size

        self.net = nn.Sequential(
            VectorizedLinear(input_dim, hidden_dim, ensemble_size),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, ensemble_size),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, ensemble_size),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, ensemble_size),
        )

    def forward(self, obs, action):
        """
        Args:
            obs: (batch_size, obs_dim)
            action: (batch_size, action_dim)

        Returns:
            q_values: (ensemble_size, batch_size)
        """
        sa = torch.cat([obs, action], dim=-1)
        # Expand for all ensemble members: (ensemble_size, batch_size, input_dim)
        # Use expand() (zero-copy view) instead of repeat_interleave (allocates copy)
        sa = sa.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        q = self.net(sa).squeeze(-1)  # (ensemble_size, batch_size)
        return q
