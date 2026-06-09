import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect expert demos (per-env trajectories).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner


def main():
    task = "Isaac-Reach-Franka-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")

    env = gym.make(task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)

    obs, _ = env.reset()

    obs_seq, act_seq = [], []
    print(f"[INFO] Collecting {args_cli.num_steps} steps x {args_cli.num_envs} envs...")
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            outputs = runner.agent.act(states=obs, observations=obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
        obs_seq.append(obs.cpu().numpy())       # [num_envs, obs_dim]
        act_seq.append(actions.cpu().numpy())   # [num_envs, act_dim]
        obs, reward, terminated, truncated, info = env.step(actions)

    # [steps, num_envs, dim] -> [num_envs, steps, dim] に転置して軌道単位で保存
    obs_arr = np.stack(obs_seq, axis=0).transpose(1, 0, 2)   # [num_envs, steps, obs_dim]
    act_arr = np.stack(act_seq, axis=0).transpose(1, 0, 2)   # [num_envs, steps, act_dim]
    print(f"[INFO] obs trajectories shape: {obs_arr.shape}")  # (num_envs, steps, obs_dim)
    print(f"[INFO] act trajectories shape: {act_arr.shape}")

    save_path = "/workspace/cesn_plus/data/demos_ppo_traj.npz"
    np.savez(save_path, obs=obs_arr, action=act_arr)
    print(f"[INFO] Saved per-trajectory demos to {save_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
