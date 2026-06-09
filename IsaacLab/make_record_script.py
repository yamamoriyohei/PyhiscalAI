import os
import re

play_path = "scripts/reinforcement_learning/skrl/play.py"
record_path = "scripts/reinforcement_learning/skrl/record_demo.py"

# 元のスクリプトを読み込む
with open(play_path, "r", encoding="utf-8") as f:
    code = f.read()

# メインループの部分（while simulation_app.is_running(): から env.close() の前まで）を検索
pattern = r"(^[ \t]*while simulation_app\.is_running\(\):.*?)(?=^[ \t]*env\.close\(\))"

replacement = """    # ==========================================
    # --- ここから書き換えたデータ収集用コード ---
    # ==========================================
    import numpy as np

    print("[INFO] お手本データ（Demonstration）の収集を開始します...")
    num_steps = 500
    obs_list = []
    action_list = []

    # reset environment
    obs, _ = env.reset()

    for i in range(num_steps):
        if not simulation_app.is_running():
            break
            
        with torch.inference_mode():
            # 現在の状態を保存 (obsが辞書型の場合は 'policy' のみ取得)
            if isinstance(obs, dict):
                current_obs = obs['policy'].cpu().clone().numpy()
            else:
                current_obs = obs.cpu().clone().numpy()
            obs_list.append(current_obs)

            # エージェントの行動を推論
            actions = agent.act(obs, timestep=0, timesteps=0)[0]
            
            # 行動を保存
            action_list.append(actions.cpu().clone().numpy())

            # 環境を1ステップ進める
            obs, reward, terminated, truncated, info = env.step(actions)

        if i % 100 == 0:
            print(f"Recording step {i}/{num_steps}...")

    # データを結合して平坦化
    obs_data = np.concatenate(obs_list, axis=0)
    action_data = np.concatenate(action_list, axis=0)
    
    # ファイルとして保存
    save_path = "ant_demonstration.npz"
    np.savez(save_path, obs=obs_data, actions=action_data)
    
    print(f"[INFO] データの収集が完了しました！ 保存先: /workspace/IsaacLab/{save_path}")
    print(f" - Observations Shape: {obs_data.shape}")
    print(f" - Actions Shape: {action_data.shape}")
    # ==========================================
"""

# コードを置換
new_code = re.sub(pattern, replacement, code, flags=re.DOTALL | re.MULTILINE)

# 新しいファイルとして保存
with open(record_path, "w", encoding="utf-8") as f:
    f.write(new_code)

print(f"[Success] '{record_path}' の生成に成功しました！")
