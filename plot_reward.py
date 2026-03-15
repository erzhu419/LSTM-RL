import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_series(directory: Path, reward_file: str = "rewards.npy", std_file: str = "eval_reward_stds.npy") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    reward_path = directory / reward_file
    std_path = directory / std_file

    if not reward_path.exists():
        print(f"[WARN] Reward file not found: {reward_path}")
        return None

    rewards = np.load(reward_path)
    stds = np.load(std_path) if std_path.exists() else None
    if stds is not None and len(stds) != len(rewards):
        min_len = min(len(rewards), len(stds))
        print(f"[WARN] Length mismatch in {directory}, clipping to {min_len}")
        rewards = rewards[:min_len]
        stds = stds[:min_len]
    return rewards, stds


def smooth_series(values: np.ndarray, window: int = 10, alpha: float = 0.3) -> Dict[str, pd.Series]:
    df = pd.DataFrame(values, columns=["values"])
    return {
        "rolling": df["values"].rolling(window=window, min_periods=1).mean(),
        "ewm": df["values"].ewm(alpha=alpha).mean(),
    }


def smooth_std(values: Optional[np.ndarray], window: int = 10, alpha: float = 0.3) -> Dict[str, pd.Series]:
    if values is None:
        return {"rolling": None, "ewm": None}
    df = pd.DataFrame(values, columns=["values"])
    return {
        "rolling": df["values"].rolling(window=window, min_periods=1).mean(),
        "ewm": df["values"].ewm(alpha=alpha).mean(),
    }


def default_directory(base: Path, script_name: str, sigma: float, embedding: str, run_name: str) -> Path:
    sigma_token = f"sigma{str(sigma).replace('.', 'p')}"
    experiment_id = f"{script_name}_{sigma_token}_embed-{embedding}_{run_name}"
    return base / "logs" / experiment_id


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot reward curves for SAC, DDPG, TD3, and MADDPG baselines.")
    parser.add_argument("--base_dir", type=str, default="comparison_runs", help="Base directory containing algorithm outputs")
    parser.add_argument("--sigma", type=float, default=1.5, help="Sigma used for plotting runs")
    parser.add_argument("--embedding_mode", type=str, default="full", choices=["full", "one_hot", "none"],
                        help="Embedding mode used for SAC/DDPG/TD3 plots")
    parser.add_argument("--run_name", type=str, default="train-{sigma}", help="Run name template for SAC/DDPG/TD3 directories (use {sigma} placeholder)")
    parser.add_argument("--window", type=int, default=10, help="Rolling window size")
    parser.add_argument("--alpha", type=float, default=0.3, help="EWMA smoothing factor")
    parser.add_argument("--show_ewm", action="store_true", help="Overlay EWMA curves")
    parser.add_argument("--maddpg_dir", type=str, default="model/MADDPG", help="Directory containing MADDPG metrics")
    parser.add_argument("--sac_dir", type=str, default=None, help="Override directory for SAC rewards/stds")
    parser.add_argument("--ddpg_dir", type=str, default=None, help="Override directory for DDPG rewards/stds")
    parser.add_argument("--td3_dir", type=str, default=None, help="Override directory for TD3 rewards/stds")
    return parser


def plot_rewards(args):
    base_dir = Path(args.base_dir)
    sigma = args.sigma
    sigma_label = str(sigma).replace(".", "p")
    run_name = args.run_name.format(sigma=sigma_label)

    sac_dir = Path(args.sac_dir) if args.sac_dir else default_directory(base_dir, "sac_v2_bus", sigma, args.embedding_mode, run_name)
    ddpg_dir = Path(args.ddpg_dir) if args.ddpg_dir else default_directory(base_dir, "ddpg_bus", sigma, args.embedding_mode, run_name)
    td3_dir = Path(args.td3_dir) if args.td3_dir else default_directory(base_dir, "td3_bus", sigma, args.embedding_mode, run_name)
    maddpg_dir = Path(args.maddpg_dir)

    datasets = {
        "SAC": load_series(sac_dir),
        "DDPG": load_series(ddpg_dir),
        "TD3": load_series(td3_dir),
        "MADDPG NPS": load_series(maddpg_dir, reward_file="rewards_individual.npy", std_file="eval_reward_stds_individual.npy"),
        "MADDPG PS": load_series(maddpg_dir, reward_file="rewards_ps.npy", std_file="eval_reward_stds_ps.npy"),
    }

    plt.figure(figsize=(14, 7))
    plt.axhline(y=-980000, color='gray', linestyle='--', label='No control average')

    colors = {
        "SAC": "tab:blue",
        "DDPG": "tab:red",
        "TD3": "tab:purple",
        "MADDPG NPS": "tab:orange",
        "MADDPG PS": "tab:green",
    }

    for label, series in datasets.items():
        if series is None:
            continue
        rewards, stds = series
        reward_smooth = smooth_series(rewards, window=args.window, alpha=args.alpha)
        std_smooth = smooth_std(stds, window=args.window, alpha=args.alpha)

        x_axis = np.arange(len(rewards))
        plt.plot(x_axis, reward_smooth["rolling"], label=f"{label} Rolling", color=colors.get(label))

        if std_smooth["rolling"] is not None:
            lower = reward_smooth["rolling"] - std_smooth["rolling"]
            upper = reward_smooth["rolling"] + std_smooth["rolling"]
            plt.fill_between(x_axis, lower, upper, color=colors.get(label), alpha=0.2)

        if args.show_ewm:
            plt.plot(x_axis, reward_smooth["ewm"], linestyle='--', color=colors.get(label), label=f"{label} EWM")
            if std_smooth["ewm"] is not None:
                lower = reward_smooth["ewm"] - std_smooth["ewm"]
                upper = reward_smooth["ewm"] + std_smooth["ewm"]
                plt.fill_between(x_axis, lower, upper, color=colors.get(label), alpha=0.1)

    plt.title(f"Reward Comparison (sigma={sigma}, embedding={args.embedding_mode})", fontsize=16)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Rewards", fontsize=16)
    plt.legend()
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig('reproduced_five_algo_v2.png')
    print("Plot saved to reproduced_five_algo_v2.png")
    plt.show()


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    plot_rewards(args)


if __name__ == "__main__":
    main()
