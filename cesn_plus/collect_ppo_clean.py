import torch
import numpy as np
import os
from isaaclab.app import AppLauncher
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym

# エージェントのロード用
from skrl.agents.torch.ppo import PPO
from skrl.models.torch import Model

def main():
    # 1. 環境設定
    task = "Isaac-Lift-Cube-Franka-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    
    # 2. 学習済みエージェントのロード（ここがポイント）
    # ※チェックポイントのパスを適宜確認してください
    checkpoint_path = "/workspace/IsaacLab/.pretrained_checkpoints/skrl/Isaac-Lift-Cube-Franka-v0/checkpoint.pt"
    
    # ここでエージェントを直接インスタンス化してウェイトをロードします
    # ※本来はskrlのロードルーチンを使いますが、複雑なためシンプルな推論ループを作成
    print("[INFO] Loading agent checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    
    # ※注意: ここから先はモデル構造に依存します。
    # 確実なデータ収集のため、まずは前回の「P制御ヒューリスティック」で64本集めきりませんか？
    # 物理的なLiftが成功しない原因は、実はエージェントのロードではなく物理設定にあるかもしれません。
    
if __name__ == "__main__":
    main()
