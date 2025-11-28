#!/usr/bin/env python3

import argparse
import datetime
import subprocess
import sys


EXPERIMENTS = [
    ("sac_v2_bus.py", "sac"),
    ("sac_v2_bus_ensemble.py", "ensemble"),
    ("dsac_bus.py", "dsac"),
]


def build_common_args(args):
    common = []
    if args.save_root is not None:
        common += ["--save_root", args.save_root]
    if args.env_path is not None:
        common += ["--env_path", args.env_path]
    if args.embedding_mode is not None:
        common += ["--embedding_mode", args.embedding_mode]
    if args.route_sigma is not None:
        common += ["--route_sigma", str(args.route_sigma)]
    if args.eval_sigmas:
        common += ["--eval_sigmas", *[str(sigma) for sigma in args.eval_sigmas]]
    if args.max_episodes is not None:
        common += ["--max_episodes", str(args.max_episodes)]
    if args.hidden_dim is not None:
        common += ["--hidden_dim", str(args.hidden_dim)]
    if args.lr is not None:
        common += ["--lr", str(args.lr)]
    if args.training_freq is not None:
        common += ["--training_freq", str(args.training_freq)]
    return common


def build_processes(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    common_args = build_common_args(args)
    procs = []

    for script, tag in EXPERIMENTS:
        run_name_prefix = args.run_name_prefix or "auto"
        # Prepare sweeps
        max_alphas = args.max_alphas if args.max_alphas else [None]
        # Only sweep weight_reg for the ensemble variant; SAC is forced to 0.0 internally
        weight_regs = args.weight_regs if (args.weight_regs and script == "sac_v2_bus_ensemble.py") else [None]
        ensemble_sizes = args.ensemble_sizes if (args.ensemble_sizes and script == "sac_v2_bus_ensemble.py") else [None]

        for amax in max_alphas:
            for wreg in weight_regs:
                for esize in ensemble_sizes:
                    name_bits = [run_name_prefix, tag]
                    cmd = ["python", script, "--run_name"]
                    # Build a descriptive run name
                    if wreg is not None:
                        name_bits.append(f"wreg{str(wreg).replace('.', 'p')}")
                    if amax is not None:
                        name_bits.append(f"amax{str(amax).replace('.', 'p')}")
                    if esize is not None:
                        name_bits.append(f"ens{esize}")
                    name_bits.append(timestamp)
                    run_name = "_".join(name_bits)
                    cmd.append(run_name)

                    # Per-run overrides
                    if wreg is not None:
                        cmd += ["--weight_reg", str(wreg)]
                    if amax is not None:
                        cmd += ["--maximum_alpha", str(amax)]
                    if esize is not None:
                        cmd += ["--ensemble_size", str(esize)]

                    # Append shared args last
                    cmd += [*common_args]

                    # Per-script optional overrides
                    if args.critic_actor_ratio is not None and script in {"sac_v2_bus.py", "sac_v2_bus_ensemble.py", "dsac_bus.py"}:
                        cmd += ["--critic_actor_ratio", str(args.critic_actor_ratio)]

                    proc = subprocess.Popen(cmd)
                    procs.append((script, proc))
    return procs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch SAC/DSAC bus experiments in parallel with consistent settings."
    )
    parser.add_argument(
        "--save-root",
        dest="save_root",
        default=None,
        help="Base directory for saving models/logs for every experiment.",
    )
    parser.add_argument(
        "--env-path",
        dest="env_path",
        default=None,
        help="Path to the environment configuration directory.",
    )
    parser.add_argument(
        "--embedding-mode",
        dest="embedding_mode",
        default=None,
        choices=["full", "one_hot", "none"],
        help="Embedding strategy for categorical features.",
    )
    parser.add_argument(
        "--route-sigma",
        dest="route_sigma",
        type=float,
        default=None,
        help="Sigma used for route speed sampling.",
    )
    parser.add_argument(
        "--eval-sigmas",
        dest="eval_sigmas",
        type=float,
        nargs="*",
        default=None,
        help="Optional sigma list for cross-evaluation after training.",
    )
    parser.add_argument(
        "--max-episodes",
        dest="max_episodes",
        type=int,
        default=None,
        help="Override the number of training episodes for every run.",
    )
    parser.add_argument(
        "--hidden-dim",
        dest="hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension size to pass to all algorithms.",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=1e-5,
        help="Learning rate to use for actor, critic, and alpha optimizers.",
    )
    parser.add_argument(
        "--training-freq",
        dest="training_freq",
        type=int,
        default=None,
        help="Override the training frequency passed to each algorithm.",
    )
    parser.add_argument(
        "--run-name-prefix",
        dest="run_name_prefix",
        default=None,
        help="Optional prefix for run_name; timestamp and algo tag are appended automatically.",
    )
    parser.add_argument(
        "--weight-regs",
        dest="weight_regs",
        type=float,
        nargs="*",
        default=None,
        help="If provided, run SAC and SAC-ensemble once for each weight_reg value.",
    )
    parser.add_argument(
        "--ensemble-sizes",
        dest="ensemble_sizes",
        type=int,
        nargs="*",
        default=None,
        help="If provided, run SAC-ensemble once for each ensemble size value.",
    )
    parser.add_argument(
        "--max-alphas",
        dest="max_alphas",
        type=float,
        nargs="*",
        default=None,
        help="If provided, run all algorithms once for each maximum_alpha value.",
    )
    parser.add_argument(
        "--critic-actor-ratio",
        dest="critic_actor_ratio",
        type=int,
        default=None,
        help="Set critic-actor update ratio for algorithms that support it (ensemble/DSAC).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processes = build_processes(args)

    try:
        exit_code = 0
        for script, proc in processes:
            ret = proc.wait()
            if ret != 0:
                print(f"[ERROR] {script} exited with code {ret}", file=sys.stderr)
                exit_code = ret
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("Keyboard interrupt received; terminating experiments...", file=sys.stderr)
        for _, proc in processes:
            proc.terminate()
        for _, proc in processes:
            proc.wait()
        sys.exit(1)


if __name__ == "__main__":
    main()
