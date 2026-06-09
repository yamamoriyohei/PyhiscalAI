import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=50)
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
    env_name = "Isaac-Reach-Franka-IK-Rel-v0"
    print(f"[INFO] Loading environment: {env_name}")
    
    env_cfg = parse_env_cfg(env_name, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(env_name, cfg=env_cfg)

    demos = []
    print(f"[INFO] Collecting {args_cli.num_demos} Expert demos...")

    # --- エキスパート（お手本）AIの準備 ---
    # シミュレータの内部データからロボットの情報を取得
    robot = env.unwrapped.scene["robot"]
    # エンドエフェクタ（手先）のパーツ名を自動検索
    body_names = robot.data.body_names
    ee_name = next((name for name in body_names if "hand" in name or "link7" in name), body_names[-1])
    ee_idx = robot.find_bodies(ee_name)[0][0]
    print(f"[INFO] Using body '{ee_name}' as end-effector.")
    # ------------------------------------

    for i in range(args_cli.num_demos):
        obs, _ = env.reset()
        trajectory = []
        done = False
        step = 0
        
        while not done and step < 100:
            # === エキスパートポリシー（お手本の動き） ===
            # 1. 現在の手先の座標を取得
            ee_pos = robot.data.body_pos_w[:, ee_idx]
            
            # 2. 目標の座標を取得
            goal_pose = env.unwrapped.command_manager.get_command("ee_pose")
            goal_pos = goal_pose[:, :3]
            
            # 3. 誤差を計算し、目標方向への移動コマンド（アクション）を生成
            pos_error = goal_pos - ee_pos
            
            action = torch.zeros((args_cli.num_envs, 6), device="cuda:0")
            action[:, :3] = pos_error * 5.0  # P制御のゲイン（強さ）
            
            # 安全のため、アクションの値を -1.0 〜 1.0 に制限
            action = torch.clamp(action, -1.0, 1.0)
            # ==========================================
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated.any() or truncated.any()
            
            obs_data = obs["policy"].cpu().numpy() if "policy" in obs else obs.cpu().numpy()
            
            trajectory.append({
                "obs": obs_data,
                "action": action.cpu().numpy(),
                "reward": reward.cpu().numpy(),
            })
            step += 1
            
        demos.append(trajectory)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args_cli.num_demos}] expert demos collected")
            
    save_path = "/workspace/cesn_plus/data/demos.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 既存のランダムデータを上書きして保存
    np.save(save_path, demos, allow_pickle=True)
    print(f"[INFO] Saved {len(demos)} Expert demos to {save_path}")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
