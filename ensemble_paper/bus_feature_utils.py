import copy
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict: Dict[str, Dict[int, int]], cat_cols: Iterable[str], embedding_dims=None,
                 layer_norm=False, dropout=0.0):
        super().__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = list(cat_cols)

        self.embedding_dims = {}
        self.cardinalities = {}
        modules = {}
        for col in self.cat_cols:
            codes = list(cat_code_dict[col].values())
            if len(codes) == 0:
                raise ValueError(f"Categorical column '{col}' has no encoding values defined.")
            cardinality = max(codes) + 1
            self.cardinalities[col] = cardinality
            dim = embedding_dims[col] if embedding_dims and col in embedding_dims else self._suggest_dim(cardinality)
            self.embedding_dims[col] = dim
            modules[col] = nn.Embedding(cardinality, dim)

        self.embeddings = nn.ModuleDict(modules)
        self.output_dim = sum(self.embedding_dims.values())
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm and self.output_dim > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def _suggest_dim(cardinality: int) -> int:
        if cardinality <= 1:
            return 1
        return min(32, max(2, int(round(cardinality ** 0.5)) + 1))

    @classmethod
    def compute_output_dim(cls, cat_code_dict, cat_cols, embedding_dims=None) -> int:
        total = 0
        for col in cat_cols:
            codes = list(cat_code_dict[col].values())
            if len(codes) == 0:
                continue
            cardinality = max(codes) + 1
            if embedding_dims and col in embedding_dims:
                total += embedding_dims[col]
            else:
                total += cls._suggest_dim(cardinality)
        return total

    def forward(self, cat_tensor):
        if cat_tensor.dim() == 1:
            cat_tensor = cat_tensor.unsqueeze(0)

        embedding_tensor_group = []
        for idx, col in enumerate(self.cat_cols):
            indices = cat_tensor[:, idx].long()
            max_index = self.cardinalities[col] - 1
            indices = torch.clamp(indices, 0, max_index)
            embedding_tensor_group.append(self.embeddings[col](indices))

        if embedding_tensor_group:
            embed_tensor = torch.cat(embedding_tensor_group, dim=1)
            if self.layer_norm is not None:
                embed_tensor = self.layer_norm(embed_tensor)
            if self.dropout is not None:
                embed_tensor = self.dropout(embed_tensor)
        else:
            embed_tensor = torch.empty(cat_tensor.size(0), 0, device=cat_tensor.device)

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


def create_embedding_layer(mode: str, cat_code_dict: Dict[str, Dict[int, int]], cat_cols: Iterable[str], **kwargs):
    mode = mode.lower()
    if mode == 'full':
        return EmbeddingLayer(cat_code_dict, cat_cols, **kwargs)
    if mode == 'one_hot':
        return OneHotEmbeddingLayer(cat_code_dict, cat_cols)
    if mode == 'none':
        return NullEmbeddingLayer(cat_code_dict, cat_cols)
    raise ValueError(f"Unsupported embedding mode: {mode}")


def build_bus_categorical_info(env):
    cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'bus_id': {i: i for i in range(env.max_agent_num)},
        'station_id': {i: i for i in range(round(len(env.stations) / 2))},
        'time_period': {i: i for i in range(env.timetables[-1].launch_time // 3600 + 2)},
        'direction': {0: 0, 1: 1},
    }
    return cat_cols, cat_code_dict
