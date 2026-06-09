import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# CNMPの核となるニューラルネットワーク構造（多層パーセプトロン）
class CNMPLite(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNMPLite, self).__init__()
        # 論文に合わせ、隠れ層を持つネットワークを定義
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def main():
    # 1. データの読み込み（CESN+と全く同じ）
    data_path = "/workspace/cesn_plus/data/demos.npy"
    print(f"[INFO] Loading data from {data_path}")
    demos = np.load(data_path, allow_pickle=True)
    
    all_inputs = []
    all_targets = []
    
    for demo in demos:
        for step_data in demo:
            obs = step_data["obs"].squeeze()
            action = step_data["action"].squeeze()
            
            # 状態(18次元)とコンテキスト(7次元)を結合して25次元の入力にする
            u_t = np.concatenate([obs[0:18], obs[18:25]])
            all_inputs.append(u_t)
            all_targets.append(action)
            
    # PyTorchのテンソルに変換し、GPU(CUDA)へ転送
    X = torch.tensor(np.array(all_inputs), dtype=torch.float32).cuda()
    Y = torch.tensor(np.array(all_targets), dtype=torch.float32).cuda()
    
    print(f"[INFO] Dataset size: {X.shape[0]} steps")
    
    # 2. モデル、損失関数、最適化手法の定義
    model = CNMPLite(input_dim=25, output_dim=6).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 1000  # ディープラーニングなので何度も学習を回す必要がある
    
    print(f"[INFO] Training CNMP model for {epochs} epochs on GPU...")
    start_time = time.time()
    
    # 3. 学習ループ（バックプロパゲーション）
    for epoch in range(epochs):
        optimizer.zero_grad()    # 勾配のリセット
        predictions = model(X)   # 順伝播（予測）
        loss = criterion(predictions, Y) # 誤差の計算
        loss.backward()          # 逆伝播（重みの更新量の計算）
        optimizer.step()         # 重みの更新
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
    end_time = time.time()
    
    train_time = end_time - start_time
    print(f"\n======================================")
    print(f"[RESULT] CNMP Training finished in {train_time:.4f} seconds!")
    print(f"======================================\n")
    
    # 4. 精度の確認（RMSE）
    model.eval()
    with torch.no_grad():
        final_preds = model(X)
        rmse = torch.sqrt(torch.mean((Y - final_preds)**2)).item()
        print(f"[INFO] CNMP Training RMSE (Error): {rmse:.6f}")

if __name__ == "__main__":
    main()
