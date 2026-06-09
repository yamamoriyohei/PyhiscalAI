import numpy as np
from sklearn.linear_model import Ridge
import time

class CESNPlus:
    def __init__(self, state_dim, context_dim, reservoir_size=500, spectral_radius=0.9, leak_rate=0.1):
        """
        CESN+ (Context-based Echo State Network) の初期化
        """
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.input_dim = state_dim + context_dim
        
        np.random.seed(42)
        self.W_in = np.random.uniform(-1, 1, (self.reservoir_size, self.input_dim))
        
        W = np.random.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))
        eigenvalues = np.linalg.eigvals(W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        self.W_res = (W / max_eigenvalue) * spectral_radius
        
        self.alpha = 1e-4
        self.readout = Ridge(alpha=self.alpha)

    def train(self, episodes_states, episodes_contexts, episodes_actions):
        print("[INFO] Generating reservoir states...")
        all_reservoir_states = []
        all_targets = []
        
        for ep_idx in range(len(episodes_states)):
            states = episodes_states[ep_idx]
            contexts = episodes_contexts[ep_idx]
            actions = episodes_actions[ep_idx]
            
            x = np.zeros(self.reservoir_size)
            for t in range(len(states)):
                u_t = np.concatenate([states[t], contexts[t]])
                x_update = np.tanh(np.dot(self.W_in, u_t) + np.dot(self.W_res, x))
                x = (1 - self.leak_rate) * x + self.leak_rate * x_update
                all_reservoir_states.append(x)
                all_targets.append(actions[t])
                
        X_res = np.array(all_reservoir_states)
        Y_tgt = np.array(all_targets)
        
        print(f"[INFO] Training readout layer on {X_res.shape[0]} steps...")
        start_time = time.time()
        
        # 出力層の学習
        self.readout.fit(X_res, Y_tgt)
        end_time = time.time()
        
        # 1. 精度（RMSE）の計算
        predictions = self.readout.predict(X_res)
        rmse = np.sqrt(np.mean((Y_tgt - predictions)**2))
        
        # 2. 予測信頼度（PI）のための残差分散（ベースノイズ）の計算
        residuals = Y_tgt - predictions
        self.sigma2 = np.var(residuals, axis=0)
        
        # 3. 未知度判定のための共分散行列の逆行列を計算
        I = np.eye(self.reservoir_size)
        self.cov_matrix = np.linalg.inv(X_res.T @ X_res + self.alpha * I)
        
        # 結果の出力
        train_time = end_time - start_time
        print(f"\n======================================")
        print(f"[RESULT] CESN+ Training & Covariance calc finished in {train_time:.4f} seconds!")
        print(f"======================================\n")
        
        print(f"[INFO] Training RMSE (Error): {rmse:.6f}")
        print(f"[INFO] Base Prediction Interval Variance (per action dim): \n{self.sigma2}\n")

    def predict_with_confidence(self, state, context, x_prev):
        """
        推論（テスト実行）時にアクションと同時に現在の『不確実性（分散）』を動的に返す関数
        """
        u_t = np.concatenate([state, context])
        x_update = np.tanh(np.dot(self.W_in, u_t) + np.dot(self.W_res, x_prev))
        x_new = (1 - self.leak_rate) * x_prev + self.leak_rate * x_update
        
        action = self.readout.predict(x_new.reshape(1, -1))[0]
        
        # 入力データの未知度（学習データからどれくらい離れているか）を計算
        distance_factor = x_new.T @ self.cov_matrix @ x_new
        dynamic_variance = self.sigma2 * (1 + distance_factor)
        
        return action, dynamic_variance, x_new

def main():
    data_path = "/workspace/cesn_plus/data/demos.npy"
    print(f"[INFO] Loading data from {data_path}")
    demos = np.load(data_path, allow_pickle=True)
    
    episodes_states = []
    episodes_contexts = []
    episodes_actions = []
    
    for demo in demos:
        states, contexts, actions = [], [], []
        for step_data in demo:
            obs = step_data["obs"].squeeze() 
            action = step_data["action"].squeeze()
            
            states.append(obs[0:18])
            contexts.append(obs[18:25])
            actions.append(action)
            
        episodes_states.append(np.array(states))
        episodes_contexts.append(np.array(contexts))
        episodes_actions.append(np.array(actions))
        
    print(f"[INFO] Loaded {len(episodes_states)} episodes.")
    
    model = CESNPlus(state_dim=18, context_dim=7, reservoir_size=500)
    model.train(episodes_states, episodes_contexts, episodes_actions)

if __name__ == "__main__":
    main()
