import argparse
import concurrent.futures
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


ALGORITHMS = [
    {"name": "SAC", "script": "sac_v2_bus.py"},
    {"name": "DDPG", "script": "ddpg_bus.py"},
    {"name": "TD3", "script": "td3_bus.py"},
]

EMBEDDING_MODES = ["full", "one_hot", "none"]
TRAIN_SIGMAS = [1.0, 1.5, 2.0]
EVAL_SIGMAS = [1.0, 1.5, 2.0]


def sigma_label(value: float) -> str:
    return f"{value}".replace(".", "p")


def build_experiment_id(script_name: str, train_sigma: float, embedding_mode: str, run_name: str) -> str:
    sigma_token = f"sigma{sigma_label(train_sigma)}"
    embedding_token = f"embed-{embedding_mode}"
    components = [script_name, sigma_token, embedding_token]
    if run_name:
        components.append(run_name)
    return "_".join(components)


def run_training_task(task: Dict, log_dir: Path, env_path: str, max_episodes: int, plot_freq: int,
                      training_freq: int, save_root: str, eval_sigmas: List[float], disable_gpu: bool) -> Dict:
    script_path = Path(__file__).resolve().parent / task["script"]
    run_name = task["run_name"]

    cmd = [
        sys.executable, str(script_path),
        "--train",
        "--max_episodes", str(max_episodes),
        "--plot_freq", str(plot_freq),
        "--training_freq", str(training_freq),
        "--route_sigma", str(task["train_sigma"]),
        "--embedding_mode", task["embedding_mode"],
        "--env_path", env_path,
        "--save_root", save_root,
        "--run_name", run_name,
        "--eval_sigmas", *[str(sig) for sig in eval_sigmas],
    ]

    stdout_path = log_dir / f"{task['id']}.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    if disable_gpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""

    with open(stdout_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=str(Path(__file__).resolve().parent),
        )

    return {
        "task": task,
        "returncode": result.returncode,
        "log_file": str(stdout_path),
    }


def collect_results(save_root: str) -> List[Dict]:
    records = []
    save_root_path = Path(save_root).resolve()
    for algo in ALGORITHMS:
        script_name = Path(algo["script"]).stem
        for embedding_mode in EMBEDDING_MODES:
            for train_sigma in TRAIN_SIGMAS:
                run_name = f"train-{sigma_label(train_sigma)}"
                experiment_id = build_experiment_id(script_name, train_sigma, embedding_mode, run_name)
                summary_path = save_root_path / "logs" / experiment_id / "cross_sigma_eval.json"
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        data = json.load(f)
                    records.extend(data)
    return records


def persist_summary(records: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = output_dir / "cross_sigma_all_results.csv"
    best_csv = output_dir / "cross_sigma_best_results.csv"

    if not records:
        detailed_csv.write_text("algorithm,embedding_mode,train_sigma,eval_sigma,mean_reward,reward_std\n")
        best_csv.write_text("algorithm,embedding_mode,eval_sigma,best_train_sigma,best_mean_reward,reward_std\n")
        return

    headers = ["algorithm", "embedding_mode", "train_sigma", "eval_sigma", "mean_reward", "reward_std"]
    with open(detailed_csv, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in records:
            f.write(",".join([
                str(row.get("algorithm", "")),
                str(row.get("embedding_mode", "")),
                str(row.get("train_sigma", "")),
                str(row.get("eval_sigma", "")),
                f"{row.get('mean_reward', '')}",
                f"{row.get('reward_std', '')}",
            ]) + "\n")

    best_map = {}
    for row in records:
        key = (row.get("algorithm"), row.get("embedding_mode"), row.get("eval_sigma"))
        current_best = best_map.get(key)
        if current_best is None or row.get("mean_reward", float("-inf")) > current_best["mean_reward"]:
            best_map[key] = {
                "best_train_sigma": row.get("train_sigma"),
                "mean_reward": row.get("mean_reward"),
                "reward_std": row.get("reward_std"),
            }

    with open(best_csv, "w") as f:
        f.write("algorithm,embedding_mode,eval_sigma,best_train_sigma,best_mean_reward,reward_std\n")
        for (algorithm, embedding_mode, eval_sigma), stats in sorted(best_map.items()):
            f.write(",".join([
                str(algorithm),
                str(embedding_mode),
                str(eval_sigma),
                str(stats["best_train_sigma"]),
                f"{stats['mean_reward']}",
                f"{stats['reward_std']}",
            ]) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run cross-sigma experiments across algorithms and embedding modes.")
    parser.add_argument("--max_workers", type=int, default=6, help="Maximum concurrent training processes")
    parser.add_argument("--max_episodes", type=int, default=50, help="Training episodes per run")
    parser.add_argument("--plot_freq", type=int, default=10, help="Plot/log frequency for individual runs")
    parser.add_argument("--training_freq", type=int, default=10, help="Update frequency inside algorithms")
    parser.add_argument("--save_root", type=str, default="comparison_runs", help="Base directory for algorithm outputs")
    parser.add_argument("--env_path", type=str, default="env", help="Path to environment configuration directory")
    parser.add_argument("--log_dir", type=str, default="comparison_runs/stdout", help="Directory for subprocess logs")
    parser.add_argument("--disable_gpu", action="store_true", help="Force subprocesses to run on CPU (recommended for parallel runs)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip runs when cross-evaluation results already exist")
    args = parser.parse_args()

    tasks = []
    for algo, embedding_mode, train_sigma in itertools.product(ALGORITHMS, EMBEDDING_MODES, TRAIN_SIGMAS):
        script_name = Path(algo["script"]).stem
        run_name = f"train-{sigma_label(train_sigma)}"
        experiment_id = build_experiment_id(script_name, train_sigma, embedding_mode, run_name)
        summary_path = Path(args.save_root) / "logs" / experiment_id / "cross_sigma_eval.json"

        if args.skip_existing and summary_path.exists():
            continue

        task_id = f"{script_name}_sigma{sigma_label(train_sigma)}_{embedding_mode}"
        tasks.append({
            "id": task_id,
            "script": algo["script"],
            "algorithm": algo["name"],
            "embedding_mode": embedding_mode,
            "train_sigma": train_sigma,
            "run_name": run_name,
        })

    if not tasks:
        print("No tasks scheduled. Consider adjusting parameters or disabling --skip_existing.")
    else:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    run_training_task,
                    task,
                    log_dir,
                    args.env_path,
                    args.max_episodes,
                    args.plot_freq,
                    args.training_freq,
                    args.save_root,
                    EVAL_SIGMAS,
                    args.disable_gpu,
                )
                for task in tasks
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                status = "SUCCESS" if result["returncode"] == 0 else "FAILURE"
                print(f"[{status}] {result['task']['id']} (log: {result['log_file']})")

    all_records = collect_results(args.save_root)
    summary_dir = Path(args.save_root) / "summaries"
    persist_summary(all_records, summary_dir)
    print(f"Wrote summaries to {summary_dir}")


if __name__ == "__main__":
    main()
