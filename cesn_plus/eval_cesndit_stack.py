import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CESN-DiT (Proposed) on Stack task.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--infer_steps", type=int, default=4)
parser.add_argument("--res_dim", type=int, default=512)
parser.add_argument("--n_train", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

DEVICE = "cuda"


class FixedReservoir(nn.Module):
    def __init__(self, input_dim, res_dim, leak_rate=0.1):
        super().__init__()
        self.res_dim = res_dim
        self.leak_rate = leak_rate
        self.W_in = nn.Parameter(torch.randn(res_dim, input_dim) * 0.1, requires_grad=False)
        self.W_res = nn.Parameter(torch.randn(res_dim, res_dim) * 0.9 / math.sqrt(res_dim), requires_grad=False)

    def forward(self, obs_history):
        B, T, _ = obs_history.shape
        r = torch.zeros(B, self.res_dim, device=obs_history.device)
        for t in range(T):
            update = torch.tanh(obs_history[:, t, :] @ self.W_in.T + r @ self.W_res.T)
            r = (1 - self.leak_rate) * r + self.leak_rate * update
        return r


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CESNDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        xm = modulate(self.norm1(x), sh1, sc1)
        a, _ = self.attn(xm, xm, xm, need_weights=False)
        x = x + g1.unsqueeze(1) * a
        xm2 = modulate(self.norm2(x), sh2, sc2)
        x = x + g2.unsqueeze(1) * self.mlp(xm2)
        return x


class CESNDiTPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, n_heads=4, layers=4, res_dim=512):
        super().__init__()
        self.act_dim = act_dim
        self.reservoir = FixedReservoir(obs_dim, res_dim)
        self.res_proj = nn.Sequential(nn.Linear(res_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.act_embed = nn.Linear(act_dim, hidden)
        self.blocks = nn.ModuleList([CESNDiTBlock(hidden, n_heads) for _ in range(layers)])
        self.norm_out = nn.LayerNorm(hidden, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(hidden, hidden * 2)
        self.proj_out_2 = nn.Linear(hidden, act_dim)
        nn.init.constant_(self.proj_out_2.weight, 0)
        nn.init.constant_(self.proj_out_2.bias, 0)

    def forward(self, obs, x, t):
        if obs.dim() == 2: obs = obs.unsqueeze(1)
        r = self.reservoir(obs)
        c = self.res_proj(r) + self.time_embed(t)
        if x.dim() == 2: x = x.unsqueeze(1)
        h = self.act_embed(x)
        for b in self.blocks: h = b(h, c)
        shift, scale = self.proj_out_1(F.silu(c)).chunk(2, dim=-1)
        h = modulate(self.norm_out(h), shift, scale)
        return self.proj_out_2(h).squeeze(1)


def main():
    print("[INFO] Loading NVIDIA Stack demos...")
    data = np.load("/workspace/cesn_plus/data/demos_stack_traj.npz", allow_pickle=True)
    obs = torch.tensor(data["obs"], dtype=torch.float32, device=DEVICE)
    act = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)
    N, T, obs_dim = obs.shape
    act_dim = act.shape[2]
    n_train = min(N, args_cli.n_train)
    obs, act = obs[:n_train], act[:n_train]
    print(f"[INFO] trajectories={n_train}, steps={T}, obs_dim={obs_dim}, act_dim={act_dim}")

    o_mean = obs.reshape(-1, obs_dim).mean(0, keepdim=True)
    o_std = obs.reshape(-1, obs_dim).std(0, keepdim=True) + 1e-6
    a_mean = act.reshape(-1, act_dim).mean(0, keepdim=True)
    a_std = act.reshape(-1, act_dim).std(0, keepdim=True) + 1e-6
    obs_n = ((obs - o_mean) / o_std).reshape(-1, obs_dim)
    act_n = ((act - a_mean) / a_std).reshape(-1, act_dim)
    Ntot = obs_n.shape[0]

    model = CESNDiTPolicy(obs_dim, act_dim, hidden=args_cli.hidden, res_dim=args_cli.res_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] CESN-DiT trainable params: {n_params:,}")

    print(f"[INFO] Training CESN-DiT {args_cli.epochs} epochs...")
    bs = 512
    start = time.time()
    for epoch in range(args_cli.epochs):
        idx = torch.randint(0, Ntot, (bs,), device=DEVICE)
        o, x1 = obs_n[idx], act_n[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(bs, 1, device=DEVICE)
        xt = (1 - t) * x0 + t * x1
        v_pred = model(o, xt, t)
        loss = ((v_pred - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 2000 == 0:
            print(f"  epoch {epoch+1}/{args_cli.epochs} loss={loss.item():.4f}")
    train_time = time.time() - start

    @torch.no_grad()
    def sample_action(o_batch, steps):
        x = torch.randn((o_batch.shape[0], act_dim), device=DEVICE)
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((o_batch.shape[0], 1), s / steps, device=DEVICE)
            x = x + dt * model(o_batch, x, t)
        return x

    with torch.no_grad():
        sm = torch.randint(0, Ntot, (2000,), device=DEVICE)
        pred = sample_action(obs_n[sm], args_cli.infer_steps)
        fit_err = torch.norm((pred - act_n[sm]), dim=1).mean().item()
    dummy = torch.randn(64, obs_dim, device=DEVICE)
    for _ in range(10): sample_action(dummy, args_cli.infer_steps)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(100): sample_action(dummy, args_cli.infer_steps)
    torch.cuda.synchronize(); lat_ms = (time.perf_counter() - t0) * 1000 / 100

    print("======================================")
    print(f"[RESULT] CESN-DiT Train: {train_time:.2f}s  fit(norm): {fit_err:.4f}  latency(64env): {lat_ms:.2f}ms")
    print("======================================")

    task = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    obs_dict, _ = env.reset()

    order = ["actions", "joint_pos", "joint_vel", "object",
             "cube_positions", "cube_orientations", "eef_pos", "eef_quat", "gripper_pos"]
    def flatten_obs(od):
        p = od["policy"]; return torch.cat([p[k] for k in order], dim=1)

    succ = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=DEVICE)
    g1 = torch.zeros_like(succ); s1 = torch.zeros_like(succ); g2 = torch.zeros_like(succ)
    print("[INFO] Evaluating CESN-DiT on Stack...")
    for step in range(args_cli.num_steps):
        cur = (flatten_obs(obs_dict) - o_mean) / o_std
        actions = sample_action(cur, args_cli.infer_steps) * a_std + a_mean
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        succ |= base_env.termination_manager.get_term("success")
        st = obs_dict.get("subtask_terms", {})
        if "grasp_1" in st: g1 |= st["grasp_1"].bool()
        if "stack_1" in st: s1 |= st["stack_1"].bool()
        if "grasp_2" in st: g2 |= st["grasp_2"].bool()

    print("======================================")
    print(f"[RESULT] CESN-DiT (Proposed) on Stack")
    print(f"  Params: {n_params:,}  Train: {train_time:.2f}s  Latency: {lat_ms:.2f}ms  fit: {fit_err:.4f}")
    print(f"  grasp_1: {g1.float().mean()*100:.1f}%  stack_1: {s1.float().mean()*100:.1f}%  grasp_2: {g2.float().mean()*100:.1f}%  SUCCESS: {succ.float().mean()*100:.1f}%")
    print("======================================")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
