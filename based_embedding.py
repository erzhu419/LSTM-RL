import copy
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    """Original ensemble embedding implementation (no layer norm / dropout)."""

    def __init__(self, cat_code_dict: Dict[str, Dict[int, int]], cat_cols: Iterable[str], embedding_dims=None):
        super().__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = list(cat_cols)

        modules = {}
        total_dim = 0
        for col in self.cat_cols:
            cardinality = len(cat_code_dict[col])
            if embedding_dims and col in embedding_dims:
                dim = embedding_dims[col]
            else:
                dim = min(50, max(1, cardinality // 2))
            modules[col] = nn.Embedding(cardinality, dim)
            total_dim += dim

        self.embeddings = nn.ModuleDict(modules)
        self.output_dim = total_dim

    def forward(self, cat_tensor: torch.Tensor) -> torch.Tensor:
        embedding_tensor_group = []
        for idx, col in enumerate(self.cat_cols):
            layer = self.embeddings[col]
            out = layer(cat_tensor[:, idx])
            embedding_tensor_group.append(out)

        embed_tensor = torch.cat(embedding_tensor_group, dim=1)
        return embed_tensor

    def clone(self):
        return copy.deepcopy(self)


class OneHotEmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict: Dict[str, Dict[int, int]], cat_cols: Iterable[str]):
        super().__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = list(cat_cols)
        self.cardinalities = {}
        for col in self.cat_cols:
            codes = list(cat_code_dict[col].values())
            cardinality = max(codes) + 1 if codes else 0
            self.cardinalities[col] = cardinality
        self.output_dim = sum(self.cardinalities.values())

    def forward(self, cat_tensor):
        if cat_tensor.dim() == 1:
            cat_tensor = cat_tensor.unsqueeze(0)

        encoded_groups = []
        for idx, col in enumerate(self.cat_cols):
            cardinality = self.cardinalities[col]
            if cardinality == 0:
                continue
            indices = cat_tensor[:, idx].long().clamp(0, cardinality - 1)
            one_hot = F.one_hot(indices, num_classes=cardinality).float()
            encoded_groups.append(one_hot)

        if encoded_groups:
            return torch.cat(encoded_groups, dim=1)
        return torch.zeros(cat_tensor.size(0), 0, device=cat_tensor.device)

    def clone(self):
        return copy.deepcopy(self)


class NullEmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict: Dict[str, Dict[int, int]], cat_cols: Iterable[str]):
        super().__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = list(cat_cols)
        self.output_dim = 0

    def forward(self, cat_tensor):
        if cat_tensor.dim() == 1:
            batch_size = 1
        else:
            batch_size = cat_tensor.size(0)
        return cat_tensor.new_zeros((batch_size, 0))

    def clone(self):
        return copy.deepcopy(self)


def build_bus_categorical_info(env):
    cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'bus_id': {i: i for i in range(env.max_agent_num)},
        'station_id': {i: i for i in range(round(len(env.stations) / 2))},
        'time_period': {i: i for i in range(env.timetables[-1].launch_time // 3600 + 2)},
        'direction': {0: 0, 1: 1},
    }
    return cat_cols, cat_code_dict
