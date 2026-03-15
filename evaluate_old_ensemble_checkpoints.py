#!/usr/bin/env python3
"""
Batch evaluation utility for historical SAC ensemble checkpoints.

Given the legacy checkpoint layout under model/sac_v2_bus_ensemble,
this script finds every policy checkpoint that contains at least
`--min-episode` training iterations (e.g. episode index >= 499),
replays the policy in the bus environment, and records reward curves.

Outputs:
- CSV with evaluation statistics per checkpoint.
- Line plot (episode index vs. reward / avg. agent reward).

The evaluation logic mirrors the policy rollout used during training,
using deterministic actions by default (can enable stochastic sampling
with --stochastic).

Usage example:
    python evaluate_old_ensemble_checkpoints.py \\
        --model-root model/sac_v2_bus_ensemble \\
        --min-episode 499 \\
        --num-eval-episodes 5 \\
        --output-dir analysis/ensemble_eval_may_run
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from based_embedding import EmbeddingLayer as LegacyEmbeddingLayer, build_bus_categorical_info
from env_original.sim import env_bus


DEVICE = torch.device("cpu")
CHECKPOINT_PATTERN = re.compile(r"(weight_reg_[^ ]+)\s+(\d+)$")


def format_float(value: float) -> str:
    """Replace '.' with 'p' for file-safe float formatting."""
    if float(value).is_integer():
        return f"{int(value)}"
    return str(value).replace(".", "p")


@dataclass(frozen=True)
class RunMetadata:
    replay_buffer_size: int
    critic_actor_ratio: int
    maximum_alpha: float
    weight_reg: float

    def to_label(self) -> str:
        return (
            f"replay{self.replay_buffer_size}"
            f"_car{self.critic_actor_ratio}"
            f"_amax{format_float(self.maximum_alpha)}"
            f"_wreg{format_float(self.weight_reg)}"
        )


class PolicyNetwork(nn.Module):
    """Replica of the SAC ensemble policy network used for evaluation."""

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_size: int,
        embedding_layer: nn.Module,
        num_categorical: int,
        action_range: float,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_categorical = num_categorical
        self.action_range = action_range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        self.to(DEVICE)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cat_tensor = state[:, : self.num_categorical]
        num_tensor = state[:, self.num_categorical :]

        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)

        x = torch.relu(self.linear1(state_with_embeddings))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    @torch.no_grad()
    def get_action(self, state: np.ndarray, deterministic: bool) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mean, log_std = self.forward(state_tensor)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
        else:
            normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
            z = normal.sample()
            action = torch.tanh(mean + std * z)

        # Map from (-1, 1) to (0, action_range)
        action = self.action_range / 2 * action + self.action_range / 2
        return action.squeeze(0).cpu().numpy()


def extract_metadata(path: Path) -> Optional[RunMetadata]:
    """Extract replay buffer size, critic_actor_ratio, maximum alpha, and weight_reg."""
    try:
        replay_buffer = int(next(p for p in path.parts if p.startswith("replay_buffer_size_")).split("_")[-1])
        critic_ratio = int(next(p for p in path.parts if p.startswith("critic_actor_ratio_")).split("_")[-1])
        max_alpha = float(next(p for p in path.parts if p.startswith("maximum_alpha_")).split("_")[-1])
    except StopIteration:
        return None
    return RunMetadata(
        replay_buffer_size=replay_buffer,
        critic_actor_ratio=critic_ratio,
        maximum_alpha=max_alpha,
        weight_reg=float("nan"),  # placeholder until we parse from filename
    )


def collect_checkpoints(
    root: Path, min_episode: int
) -> Dict[RunMetadata, List[Tuple[int, Path]]]:
    """Group checkpoint files by metadata; only keep runs with max episode >= min_episode."""
    grouped: Dict[RunMetadata, List[Tuple[int, Path]]] = defaultdict(list)

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        match = CHECKPOINT_PATTERN.match(file_path.name)
        if not match:
            continue

        meta = extract_metadata(file_path.parent)
        if meta is None:
            continue

        weight_reg_str, episode_str = match.groups()
        weight_reg_value = float(weight_reg_str.split("_")[-1])
        episode_idx = int(episode_str)

        enriched_meta = RunMetadata(
            replay_buffer_size=meta.replay_buffer_size,
            critic_actor_ratio=meta.critic_actor_ratio,
            maximum_alpha=meta.maximum_alpha,
            weight_reg=weight_reg_value,
        )
        grouped[enriched_meta].append((episode_idx, file_path))

    # Filter runs that do not reach min_episode
    filtered: Dict[RunMetadata, List[Tuple[int, Path]]] = {}
    for meta, checkpoints in grouped.items():
        max_episode = max(ep for ep, _ in checkpoints)
        if max_episode >= min_episode:
            filtered[meta] = sorted(checkpoints, key=lambda x: x[0])
    return filtered


def build_policy_for_env(env, state_dict: Dict[str, torch.Tensor]) -> PolicyNetwork:
    """Construct a policy network replicating the original ensemble architecture."""
    cat_cols, cat_code_dict = build_bus_categorical_info(env)
    embedding = LegacyEmbeddingLayer(cat_code_dict, cat_cols)
    num_categorical = len(cat_cols)
    num_continuous = env.state_dim - num_categorical
    state_dim = embedding.output_dim + num_continuous
    action_dim = env.action_space.shape[0]
    hidden_dim = state_dict["linear1.weight"].shape[0]

    policy = PolicyNetwork(
        num_inputs=state_dim,
        num_actions=action_dim,
        hidden_size=hidden_dim,
        embedding_layer=embedding,
        num_categorical=num_categorical,
        action_range=float(env.action_space.high[0]),
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def run_episode(env, policy: PolicyNetwork, deterministic: bool) -> Tuple[float, float]:
    """Roll out a single episode and return total reward and mean per-agent reward."""
    state_dict, reward_dict, _ = env.initialize_state(render=False)
    action_dict = {key: None for key in range(env.max_agent_num)}
    agent_rewards = {key: 0.0 for key in range(env.max_agent_num)}

    episode_reward = 0.0

    while True:
        for key in state_dict:
            if len(state_dict[key]) == 1:
                if action_dict[key] is None:
                    raw_state = np.asarray(state_dict[key][0], dtype=np.float32)
                    action_dict[key] = policy.get_action(raw_state, deterministic=deterministic)
            elif len(state_dict[key]) == 2:
                if state_dict[key][0][1] != state_dict[key][1][1]:
                    reward_val = reward_dict[key]
                    episode_reward += reward_val
                    agent_rewards[key] += reward_val

                state_dict[key] = state_dict[key][1:]
                raw_state = np.asarray(state_dict[key][0], dtype=np.float32)
                action_dict[key] = policy.get_action(raw_state, deterministic=deterministic)

        state_dict, reward_dict, done = env.step(action_dict, render=False)
        if done:
            break

    mean_agent_reward = float(np.mean(list(agent_rewards.values()))) if agent_rewards else 0.0
    return episode_reward, mean_agent_reward


def evaluate_checkpoint(
    checkpoint_path: Path,
    env_path: str,
    route_sigma: float,
    num_eval_episodes: int,
    deterministic: bool,
    seed: int,
) -> Dict[str, float]:
    """Evaluate a single checkpoint by averaging returns over several episodes."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = env_bus(env_path, debug=False, route_sigma=route_sigma)
    env.reset()

    policy_state = torch.load(str(checkpoint_path), map_location=DEVICE)
    policy = build_policy_for_env(env, policy_state)

    rewards = []
    agent_rewards = []

    for _ in range(num_eval_episodes):
        env.reset()
        with torch.no_grad():
            episode_reward, mean_agent_reward = run_episode(env, policy, deterministic)
        rewards.append(episode_reward)
        agent_rewards.append(mean_agent_reward)

    if hasattr(env, "close"):
        env.close()

    rewards = np.asarray(rewards, dtype=np.float32)
    agent_rewards = np.asarray(agent_rewards, dtype=np.float32)
    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=0)),
        "mean_agent_reward": float(agent_rewards.mean()),
        "std_agent_reward": float(agent_rewards.std(ddof=0)),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_metrics(
    episodes: np.ndarray,
    mean_reward: np.ndarray,
    std_reward: np.ndarray,
    mean_agent_reward: np.ndarray,
    std_agent_reward: np.ndarray,
    figure_path: Path,
) -> None:
    """Create a two-panel plot for reward and average per-agent reward."""
    ensure_dir(figure_path.parent)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(episodes, mean_reward, label="Avg Episode Reward", color="#1f77b4")
    axes[0].fill_between(
        episodes,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color="#1f77b4",
        alpha=0.2,
    )
    axes[0].set_ylabel("Episode Reward")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].plot(episodes, mean_agent_reward, label="Avg Agent Reward", color="#ff7f0e")
    axes[1].fill_between(
        episodes,
        mean_agent_reward - std_agent_reward,
        mean_agent_reward + std_agent_reward,
        color="#ff7f0e",
        alpha=0.2,
    )
    axes[1].set_xlabel("Checkpoint (Episode Index)")
    axes[1].set_ylabel("Mean Agent Reward")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def save_csv(records: List[Dict[str, float]], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = ["episode", "mean_reward", "std_reward", "mean_agent_reward", "std_agent_reward"]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate legacy SAC ensemble checkpoints.")
    parser.add_argument("--model-root", type=Path, default=Path("model/sac_v2_bus_ensemble"))
    parser.add_argument("--min-episode", type=int, default=499, help="Minimum checkpoint episode index to require.")
    parser.add_argument("--num-eval-episodes", type=int, default=5)
    parser.add_argument("--env-path", type=str, default="env_original")
    parser.add_argument("--route-sigma", type=float, default=1.5)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic sampling when drawing actions (default: deterministic).",
    )
    parser.add_argument("--checkpoint-step", type=int, default=1, help="Evaluate every Nth checkpoint (default 1).")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/ensemble_eval"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.model_root.exists():
        raise FileNotFoundError(f"Model root {args.model_root} does not exist.")

    grouped = collect_checkpoints(args.model_root, args.min_episode)
    if not grouped:
        print("No checkpoints found that meet the minimum episode requirement.")
        return

    ensure_dir(args.output_dir)

    for meta, checkpoints in grouped.items():
        run_label = meta.to_label()
        print(f"Evaluating run: {run_label} (total checkpoints: {len(checkpoints)})")

        records: List[Dict[str, float]] = []
        for idx, (episode, ckpt_path) in enumerate(checkpoints):
            if idx % max(1, args.checkpoint_step) != 0:
                continue
            metrics = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                env_path=args.env_path,
                route_sigma=args.route_sigma,
                num_eval_episodes=args.num_eval_episodes,
                deterministic=not args.stochastic,
                seed=args.seed,
            )
            record = {
                "episode": episode,
                **metrics,
            }
            records.append(record)
            print(
                f"  Episode {episode:4d} -> "
                f"Reward {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}, "
                f"AgentReward {metrics['mean_agent_reward']:.2f} ± {metrics['std_agent_reward']:.2f}"
            )

        if not records:
            print(f"  (No checkpoints selected after applying checkpoint_step={args.checkpoint_step})")
            continue

        records.sort(key=lambda x: x["episode"])
        episodes = np.array([r["episode"] for r in records], dtype=np.int32)
        mean_reward = np.array([r["mean_reward"] for r in records], dtype=np.float32)
        std_reward = np.array([r["std_reward"] for r in records], dtype=np.float32)
        mean_agent_reward = np.array([r["mean_agent_reward"] for r in records], dtype=np.float32)
        std_agent_reward = np.array([r["std_agent_reward"] for r in records], dtype=np.float32)

        run_dir = args.output_dir / run_label
        save_csv(records, run_dir / "evaluation.csv")
        plot_metrics(
            episodes=episodes,
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_agent_reward=mean_agent_reward,
            std_agent_reward=std_agent_reward,
            figure_path=run_dir / "reward_curves.png",
        )


if __name__ == "__main__":
    main()
