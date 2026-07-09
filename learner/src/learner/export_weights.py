import argparse
import json
import os
from stable_baselines3 import PPO

def export_model(model_path: str, output_path: str):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    policy = model.policy

    # PPO MlpPolicy architecture:
    # policy.mlp_extractor.policy_net[0] -> Linear(8, 64)
    # policy.mlp_extractor.policy_net[2] -> Linear(64, 64)
    # policy.action_net -> Linear(64, 3)

    weights = {
        "l1_w": policy.mlp_extractor.policy_net[0].weight.detach().cpu().numpy().tolist(),
        "l1_b": policy.mlp_extractor.policy_net[0].bias.detach().cpu().numpy().tolist(),
        "l2_w": policy.mlp_extractor.policy_net[2].weight.detach().cpu().numpy().tolist(),
        "l2_b": policy.mlp_extractor.policy_net[2].bias.detach().cpu().numpy().tolist(),
        "out_w": policy.action_net.weight.detach().cpu().numpy().tolist(),
        "out_b": policy.action_net.bias.detach().cpu().numpy().tolist(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(weights, f)
        
    print(f"Exported weights successfully to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PPO weights to JSON.")
    parser.add_argument("--model", type=str, default="models/snake_model", help="Path to SB3 zip (without .zip)")
    parser.add_argument("--out", type=str, default="models/weights.json", help="Path to output JSON")
    args = parser.parse_args()
    
    export_model(args.model, args.out)
