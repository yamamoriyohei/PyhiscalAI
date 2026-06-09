import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate CESN+ in Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import time
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


class CESNPlus:
    def __init__(self, input_dim=25, action_dim=6, reservoir_size=1000,
                 spectral_radius=0.9, ridge_param=1e-4):
        self.res_size = reservoir_size
        self.ridge_param = ridge_param
        self.W_in = torch.rand((reservoir_size, input_dim), device='cuda') * 2 - 1
        W_res = torch.rand((reservoir_size, reservoir_size), device='cuda') * 2 - 1
        max_eig = torch.max(torch.abs(torch.linalg.eigvals(W_res)))
        self.W_res = (W_res / max_eig) * spectral_radius
        self.W_out = None
        self.cov_matrix = None

    def get_states(self, X):
        return torch.tanh(X @ self.W_in.T)

    def train(self, X, Y):
        R = self.get_states(X)
        R_T_R = R.T @ R
        reg = self.ridge_param * torch.eye(self.res_size, device='cuda')
        self.cov_matrix = torch.linalg.inv(R_T_R + reg)
        self.W_out = Y.T @ R @ self.cov_matrix

    def predict(self, X):
        R = self.get_states(X)
        action_pred = R @ self.W_out.T
        variance = torch.sum(R * (R @ self.cov_matrix), dim=-1)
        return action_pred, variance


def main():
    print("[INFO] Loading expert demonstrations...")
    demos = np.load("/workspace/cesn_plus/data/demos.npy", allow_pickle=True)
    obs_list, action_list = [], []
    for demo in demos:
        for step in demo:
            obs_list.append(step["obs"].squeeze())
            action_list.append(step["action"].squeeze())

    X_obs = torch.tensor(np.array(obs_list), dtype=torch.float32).cuda()
    Y_act = torch.tensor(np.array(action_list), dtype=torch.float32).cuda()
    input_dim = X_obs.shape[1]
    action_dim = Y_act.shape[1]

    print(f"\n[INFO] Training CESN+ Model (arXiv:2412.00541)... input_dim={input_dim}, action_dim={action_dim}")
    model = CESNPlus(input_dim=input_dim, action_dim=action_dim, reservoir_size=1000)

    start_time = time.time()
    model.train(X_obs, Y_act)
    end_time = time.time()

    print("======================================")
    print(f"[RESULT] CESN+ Training finished in {end_time - start_time:.4f} seconds!")
    print("======================================\n")

    print("[INFO] Initializing Isaac Lab Environment...")
    env_cfg = parse_env_cfg("Isaac-Reach-Franka-IK-Rel", device="cuda:0", num_envs=1)
    env = gym.make("Isaac-Reach-Franka-IK-Rel", cfg=env_cfg)
    obs, _ = env.reset()

    _, base_var = model.predict(X_obs[:10])
    threshold = base_var.mean().item() * 10.0
    print(f"[INFO] Starting Evaluation Loop. Baseline Variance: {base_var.mean().item():.6f}")

    for sim_step in range(1000):
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        if not isinstance(policy_obs, torch.Tensor):
            policy_obs = torch.tensor(policy_obs, dtype=torch.float32)
        policy_obs = policy_obs.cuda()

        obs_input = policy_obs[:, :input_dim]
        action, variance = model.predict(obs_input)

        current_var = variance.mean().item()
        if current_var > threshold:
            print(f"  [Step {sim_step}] High Variance Detected! (Var: {current_var:.4f}) OOD")
        elif sim_step % 100 == 0:
            print(f"  [Step {sim_step}] Generating Trajectory... (Var: {current_var:.6f}) Normal")

        obs, reward, terminated, truncated, info = env.step(action)

    print("[INFO] Evaluation finished successfully!")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
