import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Faithful CESN+ trained on per-env trajectories.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--reservoir_size", type=int, default=1000)
parser.add_argument("--spectral_radius", type=float, default=0.95)
parser.add_argument("--leaking_rate", type=float, default=0.3)
parser.add_argument("--input_scaling", type=float, default=1.0)
parser.add_argument("--ridge_param", type=float, default=1e-4)
parser.add_argument("--connectivity", type=float, default=0.1)
parser.add_argument("--washout", type=int, default=20)
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

DEVICE = "cuda"


class CESNPlus:
    def __init__(self, input_dim, action_dim, reservoir_size=1000,
                 spectral_radius=0.95, leaking_rate=0.3, input_scaling=1.0,
                 ridge_param=1e-4, connectivity=0.1):
        self.Nx = reservoir_size
        self.Nu = input_dim
        self.Ny = action_dim
        self.alpha = leaking_rate
        self.ridge = ridge_param
        self.W_in = (torch.rand((self.Nx, 1 + self.Nu), device=DEVICE) * 2 - 1) * input_scaling
        W = torch.rand((self.Nx, self.Nx), device=DEVICE) * 2 - 1
        mask = (torch.rand((self.Nx, self.Nx), device=DEVICE) < connectivity).float()
        W = W * mask
        max_eig = torch.max(torch.abs(torch.linalg.eigvals(W)))
        self.W = W / max_eig * spectral_radius
        self.u_mean = None
        self.u_std = None
        self.W_out = None
        self.XtX_inv = None
        self.s = None

    def _normalize(self, u):
        return (u - self.u_mean) / (self.u_std + 1e-6)

    def _run_traj(self, U_traj):
        """1本の軌道 U_traj:[T, Nu] を走らせ拡張状態 [T, 1+Nx] を返す。状態は軌道頭でリセット。"""
        T = U_traj.shape[0]
        x = torch.zeros(self.Nx, device=DEVICE)
        states = torch.empty((T, self.Nx), device=DEVICE)
        ones = torch.ones(1, device=DEVICE)
        Un = self._normalize(U_traj)
        for t in range(T):
            pre = self.W_in @ torch.cat([ones, Un[t]]) + self.W @ x
            x_tilde = torch.tanh(pre)
            x = (1 - self.alpha) * x + self.alpha * x_tilde
            states[t] = x
        bias = torch.ones((T, 1), device=DEVICE)
        return torch.cat([bias, states], dim=1)

    def _run_batch_step(self, U_batch):
        ones = torch.ones((U_batch.shape[0], 1), device=DEVICE)
        u = self._normalize(U_batch)
        pre = (torch.cat([ones, u], dim=1) @ self.W_in.T) + (self.x_batch @ self.W.T)
        x_tilde = torch.tanh(pre)
        self.x_batch = (1 - self.alpha) * self.x_batch + self.alpha * x_tilde
        bias = torch.ones((U_batch.shape[0], 1), device=DEVICE)
        return torch.cat([bias, self.x_batch], dim=1)

    def reset_batch(self, batch_size):
        self.x_batch = torch.zeros((batch_size, self.Nx), device=DEVICE)

    def train(self, U_trajs, Y_trajs, washout=20):
        """U_trajs:[N,T,Nu], Y_trajs:[N,T,Ny]。軌道ごとにリザーバーを走らせて状態を集める。"""
        # 正規化統計は全データから
        flat_u = U_trajs.reshape(-1, self.Nu)
        self.u_mean = flat_u.mean(dim=0, keepdim=True)
        self.u_std = flat_u.std(dim=0, keepdim=True)

        X_list, Y_list = [], []
        N = U_trajs.shape[0]
        for i in range(N):
            X_i = self._run_traj(U_trajs[i])      # [T, 1+Nx]
            # washout: 各軌道の最初の数ステップは状態が安定しないので捨てる
            X_list.append(X_i[washout:])
            Y_list.append(Y_trajs[i][washout:])
        X = torch.cat(X_list, dim=0)              # [N*(T-washout), 1+Nx]
        Y = torch.cat(Y_list, dim=0)
        n, d = X.shape

        reg = self.ridge * torch.eye(d, device=DEVICE)
        self.XtX_inv = torch.linalg.inv(X.T @ X + reg)
        self.W_out = Y.T @ X @ self.XtX_inv
        Y_hat = X @ self.W_out.T
        resid = Y - Y_hat
        dof = max(n - d, 1)
        self.s = torch.sqrt((resid ** 2).sum() / (dof * self.Ny))
        return X, Y

    def predict_batch(self, U_batch):
        X = self._run_batch_step(U_batch)
        action = X @ self.W_out.T
        quad = torch.sum((X @ self.XtX_inv) * X, dim=1)
        pi = self.s * torch.sqrt(torch.clamp(1.0 + quad, min=0.0))
        return action, pi


def main():
    print("[INFO] Loading per-trajectory PPO demos...")
    data = np.load("/workspace/cesn_plus/data/demos_ppo_traj.npz")
    U = torch.tensor(data["obs"], dtype=torch.float32, device=DEVICE)      # [N,T,Nu]
    Y = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)   # [N,T,Ny]
    N, T, input_dim = U.shape
    action_dim = Y.shape[2]
    print(f"[INFO] trajectories={N}, steps={T}, obs_dim={input_dim}, action_dim={action_dim}")

    model = CESNPlus(input_dim, action_dim,
                     reservoir_size=args_cli.reservoir_size,
                     spectral_radius=args_cli.spectral_radius,
                     leaking_rate=args_cli.leaking_rate,
                     input_scaling=args_cli.input_scaling,
                     ridge_param=args_cli.ridge_param,
                     connectivity=args_cli.connectivity)
    start = time.time()
    X_tr, Y_tr = model.train(U, Y, washout=args_cli.washout)
    train_time = time.time() - start
    fit_err = torch.norm(X_tr @ model.W_out.T - Y_tr, dim=1).mean().item()
    print("======================================")
    print(f"[RESULT] CESN+ Training time: {train_time:.4f} sec")
    print(f"[INFO]  Train action fit error (L2): {fit_err:.4f}")
    print("======================================")

    task = "Isaac-Reach-Franka-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    model.reset_batch(args_cli.num_envs)

    ee_idx = base_env.scene["robot"].body_names.index("panda_hand")
    errors, pis = [], []
    print("[INFO] Evaluating CESN+...")
    for step in range(args_cli.num_steps):
        action, pi = model.predict_batch(obs)
        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]
        robot = base_env.scene["robot"]
        cmd = base_env.command_manager.get_command("ee_pose")
        ee_pos = robot.data.body_pos_w[:, ee_idx, :] - base_env.scene.env_origins
        dist = torch.norm(ee_pos - cmd[:, :3], dim=1)
        errors.append(dist.mean().item())
        pis.append(pi.mean().item())

    final_error = torch.tensor(errors[-50:]).mean().item()
    success = (torch.tensor(errors[-50:]) < 0.05).float().mean().item()
    print("======================================")
    print(f"[RESULT] CESN+ (trajectory-trained)")
    print(f"  Reservoir size:       {args_cli.reservoir_size}")
    print(f"  Leaking rate:         {args_cli.leaking_rate}")
    print(f"  Spectral radius:      {args_cli.spectral_radius}")
    print(f"  Training time:        {train_time:.4f} sec")
    print(f"  Train fit error (L2): {fit_err:.4f}")
    print(f"  Mean position error:  {final_error*100:.2f} cm")
    print(f"  Success rate (<5cm):  {success*100:.1f} %")
    print(f"  Mean prediction PI:   {torch.tensor(pis[-50:]).mean().item():.4f}")
    print("======================================")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
