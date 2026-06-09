import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Noisy Expert Demos")
parser.add_argument("--num_demos", type=int, default=64)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

def main():
    env_name = "Isaac-Reach-Franka-v0"
    print(f"[INFO] Loading environment: {env_name}")
    
    env_cfg = parse_env_cfg(env_name, device="cuda:0", num_envs=args_cli.num_envs)
    
    # 可視化マーカーを無効化（エラー回避）
    try: env_cfg.commands.ee_pose.debug_vis = False
    except: pass
    for _attr in ["robot", "table", "ground", "light"]:
        try: getattr(env_cfg.scene, _attr).debug_vis = False
        except: pass

    env = gym.make(env_name, cfg=env_cfg)

    all_obs = []
    all_actions = []
    print(f"[INFO] Collecting {args_cli.num_demos} NOISY Expert demos...")

    robot = env.unwrapped.scene["robot"]
    body_names = robot.data.body_names
    ee_name = next((name for name in body_names if "hand" in name or "link7" in name), body_names[-1])
    ee_idx = robot.find_bodies(ee_name)[0][0]

    for i in range(args_cli.num_demos):
        obs_dict, _ = env.reset()
        cur_obs = obs_dict["policy"]
        
        obs_traj = []
        act_traj = []
        
        for step in range(300):
            ee_pos = robot.data.body_pos_w[:, ee_idx]
            goal_pos = env.unwrapped.command_manager.get_command("ee_pose")[:, :3]
            
            pos_error = goal_pos - ee_pos
            
            # 【重要】最適行動に意図的にノイズ（ブレ）を加える
            # 標準偏差0.15のノイズを足すことで、フラフラと遠回りする軌道を作る
            noise = torch.randn_like(pos_error) * 0.15 
            
            action = torch.zeros((args_cli.num_envs, 7), device="cuda:0")
            action[:, :3] = (pos_error * 5.0) + noise
            action = torch.clamp(action, -1.0, 1.0)
            
            obs_traj.append(cur_obs.cpu().numpy()[0])
            act_traj.append(action.cpu().numpy()[0])
            
            obs_dict, _, _, _, _ = env.step(action)
            cur_obs = obs_dict["policy"]
            
        all_obs.append(obs_traj)
        all_actions.append(act_traj)
        
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{args_cli.num_demos}] noisy demos collected")
            
    save_path = "/workspace/cesn_plus/data/demos_noisy_traj.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.savez(save_path, obs=np.array(all_obs), action=np.array(all_actions))
    print(f"[INFO] Saved NOISY demos to {save_path}")
    print(f"       obs shape: {np.array(all_obs).shape}, act shape: {np.array(all_actions).shape}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
