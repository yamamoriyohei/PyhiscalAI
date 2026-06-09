import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DiT + Reservoir History Token (Plan A) on Stack.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--infer_steps", type=int, default=4)
parser.add_argument("--res_dim", type=int, default=512)
parser.add_argument("--hist_k", type=int, default=16)
parser.add_argument("--n_train", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.util, time, math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

DEVICE = "cuda"

# 本物のGR00T DiT（無変更でロード）
spec = importlib.util.spec_from_file_location(
    "dit", "/workspace/Isaac-GR00T/gr00t/model/modules/dit.py")
dit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dit_mod)
DiT = dit_mod.DiT


class FixedReservoir(nn.Module):
    """非学習リザーバー。観測履歴[B,K,obs]を1状態[B,res_dim]に圧縮。"""
    def __init__(self, input_dim, res_dim, leak=0.3, spectral=0.95):
        super().__init__()
        self.res_dim = res_dim
        self.leak = leak
        g = torch.Generator().manual_seed(0)
        W_in = (torch.rand(res_dim, input_dim, generator=g) * 2 - 1) * 0.1
        W = torch.rand(res_dim, res_dim, generator=g) * 2 - 1
        mask = (torch.rand(res_dim, res_dim, generator=g) < 0.1).float()
        W = W * mask
        eig = torch.max(torch.abs(torch.linalg.eigvals(W)))
        W = W / eig * spectral
        self.W_in = nn.Parameter(W_in, requires_grad=False)
        self.W_res = nn.Parameter(W, requires_grad=False)

    def forward(self, hist):  # hist:[B,K,obs]
        B, K, _ = hist.shape
        r = torch.zeros(B, self.res_dim, device=hist.device)
        for t in range(K):
            upd = torch.tanh(hist[:, t, :] @ self.W_in.T + r @ self.W_res.T)
            r = (1 - self.leak) * r + self.leak * upd
        return r


class DiTHistPolicy(nn.Module):
    """
    案A: DiTの核(cross-attention)は無変更。
    条件トークンを2本にする: [現在観測トークン, リザーバー履歴トークン]
    """
    def __init__(self, obs_dim, act_dim, hidden=256, n_heads=4, head_dim=64,
                 layers=4, res_dim=512):
        super().__init__()
        self.act_dim = act_dim
        self.inner = n_heads * head_dim
        # 現在観測 -> トークン
        self.cur_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.inner))
        # リザーバー -> 履歴トークン
        self.reservoir = FixedReservoir(obs_dim, res_dim)
        self.hist_proj = nn.Sequential(
            nn.Linear(res_dim, hidden), nn.SiLU(), nn.Linear(hidden, self.inner))
        # type embedding（現在/履歴の区別）
        self.type_emb = nn.Parameter(torch.randn(2, self.inner) * 0.02)
        # 行動入力
        self.act_in = nn.Linear(act_dim, self.inner)
        # DiT本体（無変更）
        self.dit = DiT(num_attention_heads=n_heads, attention_head_dim=head_dim,
                       output_dim=self.inner, num_layers=layers,
                       cross_attention_dim=self.inner)
        self.act_out = nn.Linear(self.inner, act_dim)

    def forward(self, cur_obs, hist_obs, noisy_act, t):
        # cur_obs:[B,obs], hist_obs:[B,K,obs], noisy_act:[B,act], t:[B]
        c_cur = self.cur_encoder(cur_obs) + self.type_emb[0]      # [B,inner]
        r = self.reservoir(hist_obs)                              # [B,res_dim]
        c_hist = self.hist_proj(r) + self.type_emb[1]             # [B,inner]
        cond = torch.stack([c_cur, c_hist], dim=1)                # [B,2,inner] 条件2本
        h = self.act_in(noisy_act).unsqueeze(1)                   # [B,1,inner]
        out = self.dit(hidden_states=h, encoder_hidden_states=cond, timestep=t)
        return self.act_out(out.squeeze(1))


def build_history(obs_traj, K):
    """obs_traj:[N,T,obs] -> 各(n,t)で過去Kステップ履歴。
    返り: flat_cur[N*T,obs], flat_hist[N*T,K,obs]"""
    N, T, D = obs_traj.shape
    # 先頭をpadで埋める（t<Kのとき）
    pad = obs_traj[:, :1, :].repeat(1, K - 1, 1)         # [N,K-1,obs]
    ext = torch.cat([pad, obs_traj], dim=1)              # [N,T+K-1,obs]
    hist = torch.empty((N, T, K, D), device=obs_traj.device)
    for t in range(T):
        hist[:, t] = ext[:, t:t + K, :]                  # [N,K,obs]
    cur = obs_traj.reshape(N * T, D)
    hist = hist.reshape(N * T, K, D)
    return cur, hist


def main():
    print("[INFO] Loading NVIDIA Stack demos...")
    data = np.load("/workspace/cesn_plus/data/demos_stack_traj.npz", allow_pickle=True)
    obs = torch.tensor(data["obs"], dtype=torch.float32, device=DEVICE)
    act = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)
    N, T, obs_dim = obs.shape
    act_dim = act.shape[2]
    n_train = min(N, args_cli.n_train)
    obs, act = obs[:n_train], act[:n_train]
    K = args_cli.hist_k
    print(f"[INFO] trajectories={n_train}, steps={T}, obs_dim={obs_dim}, act_dim={act_dim}, hist_K={K}")

    o_mean = obs.reshape(-1, obs_dim).mean(0, keepdim=True)
    o_std = obs.reshape(-1, obs_dim).std(0, keepdim=True) + 1e-6
    a_mean = act.reshape(-1, act_dim).mean(0, keepdim=True)
    a_std = act.reshape(-1, act_dim).std(0, keepdim=True) + 1e-6
    obs_norm = (obs - o_mean) / o_std                     # [n,T,obs]

    cur_all, hist_all = build_history(obs_norm, K)        # [n*T,obs], [n*T,K,obs]
    act_all = ((act - a_mean) / a_std).reshape(-1, act_dim)
    Ntot = cur_all.shape[0]

    model = DiTHistPolicy(obs_dim, act_dim, hidden=args_cli.hidden, res_dim=args_cli.res_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] DiT+Hist trainable params: {n_params:,}")

    print(f"[INFO] Training {args_cli.epochs} epochs...")
    bs = 512
    start = time.time()
    for epoch in range(args_cli.epochs):
        idx = torch.randint(0, Ntot, (bs,), device=DEVICE)
        cur, hist, x1 = cur_all[idx], hist_all[idx], act_all[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(bs, device=DEVICE)
        xt = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
        v_pred = model(cur, hist, xt, (t * 1000).long())
        loss = ((v_pred - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 2000 == 0:
            print(f"  epoch {epoch+1}/{args_cli.epochs} loss={loss.item():.4f}")
    train_time = time.time() - start

    @torch.no_grad()
    def sample_action(cur_b, hist_b, steps):
        x = torch.randn((cur_b.shape[0], act_dim), device=DEVICE)
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((cur_b.shape[0],), s / steps, device=DEVICE)
            x = x + dt * model(cur_b, hist_b, x, (t * 1000).long())
        return x

    with torch.no_grad():
        sm = torch.randint(0, Ntot, (2000,), device=DEVICE)
        pred = sample_action(cur_all[sm], hist_all[sm], args_cli.infer_steps)
        fit_err = torch.norm((pred - act_all[sm]), dim=1).mean().item()
    dcur = torch.randn(64, obs_dim, device=DEVICE)
    dhist = torch.randn(64, K, obs_dim, device=DEVICE)
    for _ in range(10): sample_action(dcur, dhist, args_cli.infer_steps)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(100): sample_action(dcur, dhist, args_cli.infer_steps)
    torch.cuda.synchronize(); lat_ms = (time.perf_counter() - t0) * 1000 / 100

    print("======================================")
    print(f"[RESULT] DiT+Hist Train: {train_time:.2f}s  fit(norm): {fit_err:.4f}  latency(64env): {lat_ms:.2f}ms")
    print("======================================")

    # 評価: 各envで観測履歴をローリングバッファで保持
    task = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    obs_dict, _ = env.reset()

    order = ["actions", "joint_pos", "joint_vel", "object",
             "cube_positions", "cube_orientations", "eef_pos", "eef_quat", "gripper_pos"]
    def flatten_obs(od):
        p = od["policy"]; return torch.cat([p[k] for k in order], dim=1)

    ne = args_cli.num_envs
    cur0 = (flatten_obs(obs_dict) - o_mean) / o_std
    hist_buf = cur0.unsqueeze(1).repeat(1, K, 1)          # [ne,K,obs] 初期は現在で充填

    succ = torch.zeros(ne, dtype=torch.bool, device=DEVICE)
    g1 = torch.zeros_like(succ); s1 = torch.zeros_like(succ); g2 = torch.zeros_like(succ)
    print("[INFO] Evaluating DiT+Hist on Stack...")
    for step in range(args_cli.num_steps):
        cur = (flatten_obs(obs_dict) - o_mean) / o_std
        # ローリング更新
        hist_buf = torch.cat([hist_buf[:, 1:, :], cur.unsqueeze(1)], dim=1)
        actions = sample_action(cur, hist_buf, args_cli.infer_steps) * a_std + a_mean
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        succ |= base_env.termination_manager.get_term("success")
        st = obs_dict.get("subtask_terms", {})
        if "grasp_1" in st: g1 |= st["grasp_1"].bool()
        if "stack_1" in st: s1 |= st["stack_1"].bool()
        if "grasp_2" in st: g2 |= st["grasp_2"].bool()

    print("======================================")
    print(f"[RESULT] DiT + Reservoir History Token (Plan A) on Stack")
    print(f"  hist_K: {K}  Params: {n_params:,}  Train: {train_time:.2f}s  Latency: {lat_ms:.2f}ms  fit: {fit_err:.4f}")
    print(f"  grasp_1: {g1.float().mean()*100:.1f}%  stack_1: {s1.float().mean()*100:.1f}%  grasp_2: {g2.float().mean()*100:.1f}%  SUCCESS: {succ.float().mean()*100:.1f}%")
    print("======================================")
    print("[COMPARE] DiT(baseline): grasp_1 98.4%, stack_1 15.6%, latency 18.31ms")
    print("[COMPARE] CESN-DiT(full replace): grasp_1 65.6%, latency 13.28ms")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
