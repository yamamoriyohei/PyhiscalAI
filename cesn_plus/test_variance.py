import numpy as np
from train_cesn import CESNPlus

def main():
    # 1. データの読み込みと学習（0.1秒で終わるので毎回やります）
    data_path = "/workspace/cesn_plus/data/demos.npy"
    demos = np.load(data_path, allow_pickle=True)
    
    episodes_states, episodes_contexts, episodes_actions = [], [], []
    for demo in demos:
        states, contexts, actions = [], [], []
        for step_data in demo:
            obs = step_data["obs"].squeeze()
            states.append(obs[0:18])
            contexts.append(obs[18:25])
            actions.append(step_data["action"].squeeze())
        episodes_states.append(np.array(states))
        episodes_contexts.append(np.array(contexts))
        episodes_actions.append(np.array(actions))
        
    model = CESNPlus(state_dim=18, context_dim=7, reservoir_size=500)
    
    # ログ出力を抑えつつ学習
    print("[INFO] Training model for variance test...")
    model.train(episodes_states, episodes_contexts, episodes_actions)
    
    # ========================================================
    # 2. パニックテスト（未知のコンテキストの入力）
    # ========================================================
    print("\n" + "="*50)
    print(" 🚀 STARTING OUT-OF-DISTRIBUTION (OOD) TEST")
    print("="*50)
    
    # テスト用の初期状態（学習データの最初のステップを借用）
    test_state = episodes_states[0][0]
    x_prev = np.zeros(model.reservoir_size)
    
    # --- テストA: 知っている状況（学習データにある普通の目標座標） ---
    known_context = episodes_contexts[0][0]
    _, var_known, _ = model.predict_with_confidence(test_state, known_context, x_prev)
    
    print("\n[Test A] Known Context (Normal Target):")
    print(f"  Target Context: {known_context[:3]}...")
    print(f"  Confidence Variance (X, Y, Z):")
    print(f"  {var_known[:3]}")
    print("  -> AIの反応：「知っている状況です。自信を持って予測できます！」")

    # --- テストB: 未知の状況（絶対にありえない遠くの目標座標） ---
    # 目標のX, Y, Z座標を、学習データの範囲外（例：100m先）に設定
    unknown_context = np.array([100.0, -100.0, 50.0, 0.0, 0.0, 0.0, 1.0])
    _, var_unknown, _ = model.predict_with_confidence(test_state, unknown_context, x_prev)
    
    print("\n[Test B] Unknown Context (Extreme Target 100m away):")
    print(f"  Target Context: {unknown_context[:3]}...")
    print(f"  Confidence Variance (X, Y, Z):")
    print(f"  {var_unknown[:3]}")
    
    # 分散が何倍に跳ね上がったかを計算
    ratio = var_unknown[0] / var_known[0]
    print(f"\n[RESULT] Variance increased by {ratio:.1f} TIMES!")
    if ratio > 10:
        print("  -> AIの反応：「こんな座標見たことない！自信がない！危険です！」")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
