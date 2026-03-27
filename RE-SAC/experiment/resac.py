"""
RE-SAC: Regularized Ensemble Soft Actor-Critic — MuJoCo benchmark entry point.

Usage:
    python resac.py --config configs/resac/hopper.yaml --gpu 0 --seed 0

This script mirrors the structure of dsac.py from the DSAC codebase,
plugging in the RE-SAC trainer with vectorized ensemble Q-networks.
"""
import argparse
import os
import sys

import torch

# Add DSAC rlkit to path
DSAC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../dsac'))
if DSAC_ROOT not in sys.path:
    sys.path.insert(0, DSAC_ROOT)
# Add experiment dir for local imports
EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
if EXPERIMENT_ROOT not in sys.path:
    sys.path.insert(0, EXPERIMENT_ROOT)

# Patch gym.spaces.Box.seed() for gym 0.10.x compatibility
import gym_patch  # noqa: F401

import yaml
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
from rlkit.envs import make_env
from rlkit.envs.vecenv import SubprocVectorEnv, VectorEnv
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.samplers.data_collector import (VecMdpPathCollector, VecMdpStepCollector)
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.torch_rl_algorithm import TorchVecOnlineRLAlgorithm

from copy import deepcopy
from resac_networks import VectorizedQNetwork
from resac_trainer import RESACTrainer

torch.set_num_threads(4)
torch.set_num_interop_threads(4)


def experiment(variant):
    dummy_env = make_env(variant['env'])
    obs_dim = dummy_env.observation_space.low.size
    action_dim = dummy_env.action_space.low.size

    expl_env = VectorEnv([lambda: make_env(variant['env']) for _ in range(variant['expl_env_num'])])
    expl_env.seed(variant["seed"])
    if hasattr(expl_env.action_space, 'seed'):
        expl_env.action_space.seed(variant["seed"])
    eval_env = SubprocVectorEnv([lambda: make_env(variant['env']) for _ in range(variant['eval_env_num'])])
    eval_env.seed(variant["seed"])

    M = variant['layer_size']
    ensemble_size = variant.get('ensemble_size', 10)

    # Ensemble Q-network
    qf = VectorizedQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=M,
        ensemble_size=ensemble_size,
    )
    target_qf = deepcopy(qf)

    # Policy (standard TanhGaussian, same as SAC/DSAC)
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = VecMdpStepCollector(
        expl_env,
        policy,
    )
    replay_buffer = TorchReplayBuffer(
        variant['replay_buffer_size'],
        dummy_env,
    )

    trainer_kwargs = variant.get('trainer_kwargs', {})
    trainer = RESACTrainer(
        env=dummy_env,
        policy=policy,
        qf=qf,
        target_qf=target_qf,
        ensemble_size=ensemble_size,
        **trainer_kwargs,
    )
    algorithm = TorchVecOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs'],
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RE-SAC: Regularized Ensemble Soft Actor-Critic')
    parser.add_argument('--config', type=str, default="configs/resac/hopper.yaml")
    parser.add_argument('--gpu', type=int, default=0, help="using cpu with -1")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    variant["seed"] = args.seed

    log_prefix = "_".join(["resac", variant["env"][:-3].lower(), str(variant["version"])])
    setup_logger(log_prefix, variant=variant, seed=args.seed)

    if args.gpu >= 0:
        ptu.set_gpu_mode(True, args.gpu)
    set_seed(args.seed)
    experiment(variant)
