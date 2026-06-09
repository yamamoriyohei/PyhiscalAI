import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train & evaluate CESN+ on Reach.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--reservoir_size", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import time
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


class CESNPlus:
    def __init__(self, input_dim, action_dim, reservoir_size=1000,
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
        reg = self.ridge_param * torch.eye(self.res_size, device='cuda')
        self.cov_matrix = torch.linalg.inv(R.T @ R + reg)
        self.W_out = Y.T @ R @ self.cov_matrix

    def predict(self, X):
        R = self.get_states(X)
        action_pred = R @ self.W_out.T
        variance = torch.sum(R * (R @ self.cov_matrix), dim=-1)
        return action_pred, variance


def main():
    # --- 1. PPOデモでCESN+を学習 ---
    print("[INFO] Loading PPO expert demos...")
    data = np.load("/workspace/cesn_plus/data/demos_ppo.npz")
    X_obs = torch.tensor(data["obs"], dtype=torch.float32).cuda()
    Y_act = torch.tensor(data["action"], dtype=torch.float32).cuda()
    input_dim = X_obs.shape[1]
    action_dim = Y_act.shape[1]
    print(f"[INFO] obs_dim={input_dim}, action_dim={action_dim}, samples={X_obs.shape[0]}")

    model = CESNPlus(input_dim, action_dim, reservoir_size=args_cli.reservoir_size)
    start = time.time()
    model.train(X_obs, Y_act)
    train_time = time.time() - start
    print("======================================")
    print(f"[RESULT] CESN+ Training time: {train_time:.4f} seconds")
    print("======================================")

    # --- 2. 環境でCESN+を動かして精度評価 ---
    task = "Isaac-Reach-Franka-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    ee_idx = base_env.scene["robot"].body_names.index("panda_hand")
    errors = []
    print("[INFO] Evaluating CESN+...")
    for step in range(args_cli.num_steps):
        action, variance = model.predict(obs)
        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]

        robot = base_env.scene["robot"]
        cmd = base_env.command_manager.get_command("ee_pose")
        ee_pos = robot.data.body_pos_w[:, ee_idx, :] - base_env.scene.env_origins
        dist = torch.norm(ee_pos - cmd[:, :3], dim=1)
        errors.append(dist.mean().item())

    final_error = torch.tensor(errors[-50:]).mean().item()
    success = (torch.tensor(errors[-50:]) < 0.05).float().mean().item()
    print("======================================")
    print(f"[RESULT] CESN+ Accuracy Evaluation")
    print(f"  Training time:        {train_time:.4f} sec")
    print(f"  Mean position error:  {final_error*100:.2f} cm")
    print(f"  Success rate (<5cm):  {success*100:.1f} %")
    print("======================================")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
