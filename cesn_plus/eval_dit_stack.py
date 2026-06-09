import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GR00T DiT on Stack task.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--infer_steps", type=int, default=4)
parser.add_argument("--n_train", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.util
import numpy as np
import torch
import torch.nn as nn
import time
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

DEVICE = "cuda"

spec = importlib.util.spec_from_file_location(
    "dit", "/workspace/Isaac-GR00T/gr00t/model/modules/dit.py")
dit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dit_mod)
DiT = dit_mod.DiT


class DiTActionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, n_heads=4, head_dim=64, layers=4):
        super().__init__()
        self.act_dim = act_dim
        self.inner = n_heads * head_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.inner),
        )
        self.act_in = nn.Linear(act_dim, self.inner)
        self.dit = DiT(num_attention_heads=n_heads, attention_head_dim=head_dim,
                       output_dim=self.inner, num_layers=layers, cross_attention_dim=self.inner)
        self.act_out = nn.Linear(self.inner, act_dim)

    def forward(self, obs, noisy_act, t):
        cond = self.cond_encoder(obs).unsqueeze(1)
        h = self.act_in(noisy_act).unsqueeze(1)
        out = self.dit(hidden_states=h, encoder_hidden_states=cond, timestep=t)
        return self.act_out(out.squeeze(1))


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

    model = DiTActionPolicy(obs_dim, act_dim, hidden=args_cli.hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] DiT policy params: {n_params:,}")

    print(f"[INFO] Training DiT (flow-matching) {args_cli.epochs} epochs...")
    bs = 512
    start = time.time()
    for epoch in range(args_cli.epochs):
        idx = torch.randint(0, Ntot, (bs,), device=DEVICE)
        o, x1 = obs_n[idx], act_n[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(bs, device=DEVICE)
        xt = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
        v_target = x1 - x0
        v_pred = model(o, xt, (t * 1000).long())
        loss = ((v_pred - v_target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 2000 == 0:
            print(f"  epoch {epoch+1}/{args_cli.epochs} loss={loss.item():.4f}")
    train_time = time.time() - start

    @torch.no_grad()
    def sample_action(o_batch, steps):
        x = torch.randn((o_batch.shape[0], act_dim), device=DEVICE)
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((o_batch.shape[0],), s / steps, device=DEVICE)
            x = x + dt * model(o_batch, x, (t * 1000).long())
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
    print(f"[RESULT] DiT Train time: {train_time:.2f}s  fit_err(norm): {fit_err:.4f}  latency(64env): {lat_ms:.2f}ms")
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

    succ_any = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=DEVICE)
    g1 = torch.zeros_like(succ_any); s1 = torch.zeros_like(succ_any); g2 = torch.zeros_like(succ_any)
    print("[INFO] Evaluating DiT on Stack...")
    for step in range(args_cli.num_steps):
        cur = (flatten_obs(obs_dict) - o_mean) / o_std
        actions = sample_action(cur, args_cli.infer_steps) * a_std + a_mean
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        succ_any |= base_env.termination_manager.get_term("success")
        st = obs_dict.get("subtask_terms", {})
        if "grasp_1" in st: g1 |= st["grasp_1"].bool()
        if "stack_1" in st: s1 |= st["stack_1"].bool()
        if "grasp_2" in st: g2 |= st["grasp_2"].bool()

    print("======================================")
    print(f"[RESULT] DiT (GR00T System1) on Stack")
    print(f"  Params: {n_params:,}  Train: {train_time:.2f}s  Latency: {lat_ms:.2f}ms  fit: {fit_err:.4f}")
    print(f"  grasp_1: {g1.float().mean()*100:.1f}%  stack_1: {s1.float().mean()*100:.1f}%  grasp_2: {g2.float().mean()*100:.1f}%  SUCCESS: {succ_any.float().mean()*100:.1f}%")
    print("======================================")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
