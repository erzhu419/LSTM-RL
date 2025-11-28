"""
Evaluate all May 2024 Ensemble model configurations
Load the final checkpoints from different parameter settings and compare
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Configuration for evaluation
configs_to_eval = [
    # buffer_size, critic_ratio, max_alpha, weight_reg, expected_episodes
    ("1000000", "2", "0.3", "0.03", 511),  # Most complete training
    ("1000000", "2", "0.3", "0.01", 11),
    ("100000", "2", "0.3", "0.01", 426),
    ("100000", "3", "0.3", "0.01", 235),
    ("100000", "4", "0.3", "0.01", 425),
    ("100000", "10", "2", "0.01", 18),  # High exploration config
]

BASE_DIR = "/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus_ensemble"

print("="*80)
print("May 2024 Ensemble Models - Final Checkpoint Evaluation")
print("="*80)
print("\nScanning available models...")

results = []

for buffer, ratio, alpha, wreg, expected_ep in configs_to_eval:
    model_dir = os.path.join(
        BASE_DIR,
        f"replay_buffer_size_{buffer}",
        f"critic_actor_ratio_{ratio}",
        f"maximum_alpha_{alpha}"
    )

    config_name = f"buf{buffer}_r{ratio}_a{alpha}_w{wreg}"

    # Find the latest checkpoint
    latest_ep = None
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        episodes = []
        for f in files:
            if f.startswith(f"weight_reg_{wreg} ") and f != f"weight_reg_{wreg}":
                try:
                    ep = int(f.split()[-1])
                    episodes.append(ep)
                except:
                    pass

        if episodes:
            latest_ep = max(episodes)
            model_path = os.path.join(model_dir, f"weight_reg_{wreg} {latest_ep}")

            # Load model to check if it's valid
            try:
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

                results.append({
                    'config': config_name,
                    'buffer': buffer,
                    'ratio': ratio,
                    'alpha': alpha,
                    'wreg': wreg,
                    'episodes': latest_ep + 1,  # 0-indexed, so +1 for total count
                    'expected': expected_ep + 1,
                    'path': model_path,
                    'has_embedding': 'embedding_layer.embeddings.bus_id.weight' in state_dict,
                    'hidden_dim': state_dict['linear1.bias'].shape[0],
                    'input_dim': state_dict['linear1.weight'].shape[1],
                })

                status = "✓" if latest_ep == expected_ep else "⚠"
                print(f"{status} {config_name:40} Episodes: {latest_ep+1:4}/{expected_ep+1:4}")

            except Exception as e:
                print(f"✗ {config_name:40} Error: {e}")
        else:
            print(f"✗ {config_name:40} No checkpoints found")
    else:
        print(f"✗ {config_name:40} Directory not found")

# Summary table
print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)
print(f"{'Config':<40} {'Episodes':<12} {'Hidden':<10} {'Input':<10} {'Embed':<8}")
print("-"*80)

for r in results:
    embed_str = "Yes" if r['has_embedding'] else "No"
    print(f"{r['config']:<40} {r['episodes']:>4}/{r['expected']:<6} "
          f"{r['hidden_dim']:<10} {r['input_dim']:<10} {embed_str:<8}")

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

if results:
    # Most complete training
    most_complete = max(results, key=lambda x: x['episodes'])
    print(f"\nMost complete training:")
    print(f"  Config: {most_complete['config']}")
    print(f"  Episodes: {most_complete['episodes']}")
    print(f"  Path: {most_complete['path']}")

    # Find high exploration config
    high_explore = [r for r in results if r['alpha'] == '2']
    if high_explore:
        print(f"\nHigh exploration config (alpha=2.0):")
        print(f"  Config: {high_explore[0]['config']}")
        print(f"  Episodes: {high_explore[0]['episodes']}")
        print(f"  Path: {high_explore[0]['path']}")

    # Architecture summary
    print(f"\nArchitecture:")
    print(f"  All models use embedding layer: {all(r['has_embedding'] for r in results)}")
    print(f"  Hidden dimension: {results[0]['hidden_dim']}")
    print(f"  Input dimension (after embedding): {results[0]['input_dim']}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\nBased on the analysis document (ensemble_good_performance_analysis.md),")
    print("the best performing config was likely:")
    print("  - critic_actor_ratio = 10 (充分训练critic)")
    print("  - maximum_alpha = 2.0 (高探索)")
    print("  - weight_reg = 0.03")
    print("\nHowever, this config only trained 19 episodes.")
    print("\nThe most complete training (512 episodes) was:")
    print("  - buffer = 1M, ratio = 2, alpha = 0.3, wreg = 0.03")
    print("\nTo see actual performance, you would need to:")
    print("1. Run these models on the environment (requires matching architecture)")
    print("2. Or retrain with the fixed code using recommended parameters")

else:
    print("No valid models found!")

print("="*80)
