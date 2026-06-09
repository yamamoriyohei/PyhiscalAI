import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CESN+ on Stack task (NVIDIA 1000 demos).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--reservoir_size", type=int, default=1000)
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
                 spectral_radius=0.95, leaking_rate=0.3, connectivity=0.1, ridge=1e-4):
        self.Nx = reservoir_size
        self.Nu = input_dim
        self.Ny = action_dim
        self.alpha = leaking_rate
        self.ridge = ridge
        g = torch.Generator(device=DEVICE).manual_seed(42)
        self.W_in = (torch.rand((self.Nx, 1 + self.Nu), generator=g, device=DEVICE) * 2 - 1)
        W = torch.rand((self.Nx, self.Nx), generator=g, device=DEVICE) * 2 - 1
        mask = (torch.rand((self.Nx, self.Nx), generator=g, device=DEVICE) < connectivity).float()
        W = W * mask
        eig = torch.max(torch.abs(torch.linalg.eigvals(W)))
        self.W = W / eig * spectral_radius
        self.W_out = torch.zeros((action_dim, 1 + self.Nx), device=DEVICE)
        self.u_mean = torch.zeros((1, input_dim), device=DEVICE)
        self.u_std = torch.ones((1, input_dim), device=DEVICE)

    def _run_reservoir(self, U):
        # U: [T, Nu] 1軌道。状態系列 X:[T,Nx] を返す
        T = U.shape[0]
        x = torch.zeros(self.Nx, device=DEVICE)
        X = torch.empty((T, self.Nx), device=DEVICE)
        Un = (U - self.u_mean[0]) / (self.u_std[0] + 1e-6)
        for t in range(T):
            u = torch.cat([torch.ones(1, device=DEVICE), Un[t]])
            x_tilde = torch.tanh(self.W_in @ u + self.W @ x)
            x = (1 - self.alpha) * x + self.alpha * x_tilde
            X[t] = x
        return X

    def fit(self, obs_traj, act_traj, washout=20):
        # obs_traj:[N,T,Nu], act_traj:[N,T,Ny]
        N, T, _ = obs_traj.shape
        flat = obs_traj.reshape(-1, self.Nu)
        self.u_mean = flat.mean(0, keepdim=True)
        self.u_std = flat.std(0, keepdim=True)
        Xs, Ys = [], []
        for i in range(N):
            X = self._run_reservoir(obs_traj[i])     # [T,Nx]
            Xb = torch.cat([torch.ones((T, 1), device=DEVICE), X], dim=1)  # [T,1+Nx]
            Xs.append(Xb[washout:])
            Ys.append(act_traj[i][washout:])
        Xall = torch.cat(Xs, 0)   # [M,1+Nx]
        Yall = torch.cat(Ys, 0)   # [M,Ny]
        A = Xall.T @ Xall + self.ridge * torch.eye(1 + self.Nx, device=DEVICE)
        B = Xall.T @ Yall
        self.W_out = torch.linalg.solve(A, B).T   # [Ny,1+Nx]
        pred = Xall @ self.W_out.T
        return torch.norm(pred - Yall, dim=1).mean().item()

    def reset_state(self, batch):
        self.x_batch = torch.zeros((batch, self.Nx), device=DEVICE)

    def step(self, U_batch):
        Un = (U_batch - self.u_mean) / (self.u_std + 1e-6)
        ones = torch.ones((U_batch.shape[0], 1), device=DEVICE)
        pre = torch.cat([ones, Un], 1) @ self.W_in.T + self.x_batch @ self.W.T
        x_tilde = torch.tanh(pre)
        self.x_batch = (1 - self.alpha) * self.x_batch + self.alpha * x_tilde
        Xb = torch.cat([ones, self.x_batch], 1)
        return Xb @ self.W_out.T


def main():
    print("[INFO] Loading NVIDIA Stack demos (1000 trajectories)...")
    data = np.load("/workspace/cesn_plus/data/demos_stack_traj.npz", allow_pickle=True)
    obs = torch.tensor(data["obs"], dtype=torch.float32, device=DEVICE)     # [1000,200,94]
    act = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)  # [1000,200,7]
    N, T, obs_dim = obs.shape
    act_dim = act.shape[2]
    print(f"[INFO] trajectories={N}, steps={T}, obs_dim={obs_dim}, act_dim={act_dim}")

    # 行動正規化（評価時に戻す）
    a_mean = act.reshape(-1, act_dim).mean(0, keepdim=True)
    a_std = act.reshape(-1, act_dim).std(0, keepdim=True) + 1e-6

    # メモリ対策: 学習に使う軌道数を制限（全1000は重い場合あり）
    n_train = min(N, 300)
    obs_tr = obs[:n_train]
    act_tr = (act[:n_train] - a_mean) / a_std

    model = CESNPlus(obs_dim, act_dim, reservoir_size=args_cli.reservoir_size)
    print(f"[INFO] Training CESN+ on {n_train} trajectories...")
    start = time.time()
    fit_err = model.fit(obs_tr, act_tr, washout=20)
    train_time = time.time() - start
    print("======================================")
    print(f"[RESULT] CESN+ Training time: {train_time:.4f} sec")
    print(f"[INFO]  Train action fit error (L2, normalized): {fit_err:.4f}")
    print("======================================")

    # ===== 評価 =====
    task = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    obs_dict, _ = env.reset()

    # policy観測を環境順に連結する関数
    order = ["actions", "joint_pos", "joint_vel", "object",
             "cube_positions", "cube_orientations", "eef_pos", "eef_quat", "gripper_pos"]
    def flatten_obs(od):
        p = od["policy"]
        return torch.cat([p[k] for k in order], dim=1)  # [num_envs,94]

    model.reset_state(args_cli.num_envs)
    success_any = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=DEVICE)
    grasp1_any = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=DEVICE)
    stack1_any = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=DEVICE)
    grasp2_any = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=DEVICE)

    print("[INFO] Evaluating CESN+ on Stack...")
    for step in range(args_cli.num_steps):
        cur = flatten_obs(obs_dict)
        a_norm = model.step(cur)
        actions = a_norm * a_std + a_mean
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        # 成功・サブタスク達成を累積（一度でも達成したら記録）
        succ = base_env.termination_manager.get_term("success")
        success_any |= succ
        st = obs_dict.get("subtask_terms", {})
        if "grasp_1" in st: grasp1_any |= st["grasp_1"].bool()
        if "stack_1" in st: stack1_any |= st["stack_1"].bool()
        if "grasp_2" in st: grasp2_any |= st["grasp_2"].bool()

    print("======================================")
    print(f"[RESULT] CESN+ on Stack (NVIDIA 1000 demos)")
    print(f"  Reservoir size:       {args_cli.reservoir_size}")
    print(f"  Train trajectories:   {n_train}")
    print(f"  Training time:        {train_time:.4f} sec")
    print(f"  Train fit error:      {fit_err:.4f}")
    print(f"  --- 達成率（一度でも達成したenv割合）---")
    print(f"  grasp_1 (1個目把持):  {grasp1_any.float().mean()*100:.1f} %")
    print(f"  stack_1 (1個目設置):  {stack1_any.float().mean()*100:.1f} %")
    print(f"  grasp_2 (2個目把持):  {grasp2_any.float().mean()*100:.1f} %")
    print(f"  SUCCESS (完全成功):   {success_any.float().mean()*100:.1f} %")
    print("======================================")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
