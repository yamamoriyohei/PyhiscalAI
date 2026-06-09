import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Heuristic Demos for Lift")
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
from isaaclab_tasks.utils import parse_env_cfg

def main():
    env_name = "Isaac-Lift-Cube-Franka-IK-Rel-v0"
    env_cfg = parse_env_cfg(env_name, device="cuda:0", num_envs=args_cli.num_envs)
    
    try: env_cfg.commands.ee_pose.debug_vis = False
    except: pass
    for _attr in ["robot", "object", "table", "ground", "light"]:
        try: getattr(env_cfg.scene, _attr).debug_vis = False
        except: pass

    env = gym.make(env_name, cfg=env_cfg)
    
    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["object"]
    body_names = robot.data.body_names
    ee_name = next((name for name in body_names if "hand" in name or "link7" in name), body_names[-1])
    ee_idx = robot.find_bodies(ee_name)[0][0]

    act_dim = env.action_space.shape[1] if len(env.action_space.shape) > 1 else env.action_space.shape[0]

    all_obs, all_actions = [], []
    success_count = 0
    attempts = 0
    
    print(f"[INFO] Collecting {args_cli.num_demos} Heuristic Demos for Lift Task...")
    
    while success_count < args_cli.num_demos:
        obs_dict, _ = env.reset()
        cur_obs = obs_dict["policy"]
        obs_traj, act_traj = [], []
        
        state = 0 # 0:接近, 1:下降, 2:把持, 3:持ち上げ
        wait_ticks = 0
        
        for step in range(300):
            ee_pos = robot.data.body_pos_w[:, ee_idx]
            cube_pos = cube.data.root_pos_w
            
            action = torch.zeros((1, act_dim), device="cuda:0")
            gripper_idx = act_dim - 1
            
            # -1.0 = 開く, 1.0 = 閉じる
            action[:, gripper_idx] = -1.0 
            
            if state == 0:
                target = cube_pos.clone()
                target[:, 2] += 0.10
                pos_err = target - ee_pos
                action[:, :3] = pos_err * 6.0
                if torch.norm(pos_err) < 0.05: state = 1
                    
            elif state == 1:
                target = cube_pos.clone()
                target[:, 2] -= 0.01 # キューブにめり込ませるくらい下げる
                pos_err = target - ee_pos
                action[:, :3] = pos_err * 4.0
                if torch.norm(pos_err) < 0.03: state = 2
                    
            elif state == 2:
                pos_err = cube_pos - ee_pos
                action[:, :3] = pos_err * 4.0
                action[:, gripper_idx] = 1.0 # 閉じる
                wait_ticks += 1
                if wait_ticks > 15: state = 3
                    
            elif state == 3:
                pos_err = cube_pos - ee_pos
                action[:, :2] = pos_err[:, :2] * 4.0 # XYはズレないようにキープ
                action[:, 2] = 1.0 # Zは全力で上へ
                action[:, gripper_idx] = 1.0 # 閉じたまま
                
            # 多峰性テストのための微小ノイズ
            action[:, :3] += torch.randn_like(action[:, :3]) * 0.05
            action = torch.clamp(action, -1.0, 1.0)
            
            obs_traj.append(cur_obs.cpu().numpy()[0])
            act_traj.append(action.cpu().numpy()[0])
            
            obs_dict, _, _, _, _ = env.step(action)
            cur_obs = obs_dict["policy"]
            
        attempts += 1
        final_cube_z = cube.data.root_pos_w[:, 2] - env.unwrapped.scene.env_origins[:, 2]
        
        if final_cube_z.item() > 0.05:
            all_obs.append(obs_traj)
            all_actions.append(act_traj)
            success_count += 1
            print(f"  [SUCCESS] {success_count}/{args_cli.num_demos} (Total attempts: {attempts})")
        else:
            print(f"  [FAILED] Attempt {attempts}, max_z: {final_cube_z.item():.3f} - Retrying...")
            
    save_path = "/workspace/cesn_plus/data/demos_ppo_lift_traj.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, obs=np.array(all_obs), action=np.array(all_actions))
    print(f"[INFO] Saved {len(all_obs)} successful demos to {save_path}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
