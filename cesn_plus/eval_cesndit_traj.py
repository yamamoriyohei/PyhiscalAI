import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CESN-DiT (Proposed Hybrid) on Reach.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--epochs", type=int, default=8000)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--infer_steps", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

DEVICE = "cuda"

# --- CESN-DiT (Reservoir-Conditioned DiT) 実装 ---
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
            obs_t = obs_history[:, t, :]
            update = torch.tanh(obs_t @ self.W_in.T + r @ self.W_res.T)
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
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_mod2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class CESNDiTPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, n_heads=4, layers=4, res_dim=256):
        super().__init__()
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
        c_res = self.res_proj(r)
        c_time = self.time_embed(t)
        c = c_res + c_time
        if x.dim() == 2: x = x.unsqueeze(1)
        h = self.act_embed(x)
        for block in self.blocks: h = block(h, c)
        shift, scale = self.proj_out_1(F.silu(c)).chunk(2, dim=-1)
        h = modulate(self.norm_out(h), shift, scale)
        out = self.proj_out_2(h)
        return out.squeeze(1)

def main():
    print("[INFO] Loading per-trajectory PPO demos...")
    data = np.load("/workspace/cesn_plus/data/demos_noisy_traj.npz")
    obs = torch.tensor(data["obs"], dtype=torch.float32, device=DEVICE)
    act = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)
    N, T_len, obs_dim = obs.shape
    act_dim = act.shape[2]

    o_mean = obs.reshape(-1, obs_dim).mean(0, keepdim=True)
    o_std = obs.reshape(-1, obs_dim).std(0, keepdim=True) + 1e-6
    a_mean = act.reshape(-1, act_dim).mean(0, keepdim=True)
    a_std = act.reshape(-1, act_dim).std(0, keepdim=True) + 1e-6
    obs_n = ((obs - o_mean) / o_std).reshape(-1, obs_dim)
    act_n = ((act - a_mean) / a_std).reshape(-1, act_dim)
    Ntot = obs_n.shape[0]

    model = CESNDiTPolicy(obs_dim, act_dim, hidden=args_cli.hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 勾配が必要なパラメータ（学習対象）のみをカウント
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] CESN-DiT policy trainable params: {n_params:,}")

    print(f"[INFO] Training CESN-DiT (flow-matching) for {args_cli.epochs} epochs...")
    bs = 512
    start = time.time()
    for epoch in range(args_cli.epochs):
        idx = torch.randint(0, Ntot, (bs,), device=DEVICE)
        o = obs_n[idx]
        x1 = act_n[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(bs, 1, device=DEVICE)
        xt = (1 - t) * x0 + t * x1
        v_target = x1 - x0
        v_pred = model(o, xt, t)
        loss = ((v_pred - v_target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 1000 == 0:
            print(f"  epoch {epoch+1}/{args_cli.epochs} loss={loss.item():.4f}")
    train_time = time.time() - start

    @torch.no_grad()
    def sample_action(o_batch, steps):
        B = o_batch.shape[0]
        x = torch.randn((B, act_dim), device=DEVICE)
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((B, 1), s / steps, device=DEVICE)
            v = model(o_batch, x, t)
            x = x + dt * v
        return x

    with torch.no_grad():
        sample = torch.randint(0, Ntot, (2000,), device=DEVICE)
        pred = sample_action(obs_n[sample], args_cli.infer_steps)
        fit_err = torch.norm((pred - act_n[sample]) * a_std, dim=1).mean().item()

    # レイテンシ計測
    dummy_o = torch.randn(1, obs_dim, device=DEVICE)
    for _ in range(10): sample_action(dummy_o, args_cli.infer_steps) # warmup
    lat_start = time.perf_counter()
    for _ in range(100): sample_action(dummy_o, args_cli.infer_steps)
    lat_ms = (time.perf_counter() - lat_start) * 1000 / 100

    print("======================================")
    print(f"[RESULT] CESN-DiT Training time: {train_time:.4f} sec")
    print(f"[INFO]  Train action fit error (L2): {fit_err:.4f}")
    print(f"[INFO]  Inference Latency: {lat_ms:.2f} ms")
    print("======================================")

    env_cfg = parse_env_cfg("Isaac-Reach-Franka-v0", device="cuda:0", num_envs=args_cli.num_envs)
    # バグ回避パッチ
    try: env_cfg.commands.ee_pose.debug_vis = False
    except: pass
    for _attr in ["robot", "table", "ground", "light"]:
        try: getattr(env_cfg.scene, _attr).debug_vis = False
        except: pass
        
    env = gym.make("Isaac-Reach-Franka-v0", cfg=env_cfg)
    base_env = env.unwrapped
    obs_dict, _ = env.reset()
    cur_obs = obs_dict["policy"]
    ee_idx = base_env.scene["robot"].body_names.index("panda_hand")

    errors = []
    print("[INFO] Evaluating CESN-DiT...")
    for step in range(args_cli.num_steps):
        cur_n = (cur_obs - o_mean) / o_std
        actions = sample_action(cur_n, args_cli.infer_steps) * a_std + a_mean
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        cur_obs = obs_dict["policy"]
        robot = base_env.scene["robot"]
        cmd = base_env.command_manager.get_command("ee_pose")
        ee_pos = robot.data.body_pos_w[:, ee_idx, :] - base_env.scene.env_origins
        dist_e = torch.norm(ee_pos - cmd[:, :3], dim=1)
        errors.append(dist_e.mean().item())

    final_error = torch.tensor(errors[-50:]).mean().item()
    success = (torch.tensor(errors[-50:]) < 0.05).float().mean().item()
    print("======================================")
    print(f"[RESULT] CESN-DiT (Proposed Hybrid) Accuracy Evaluation")
    print(f"  Trainable params:     {n_params:,}")
    print(f"  Inference steps:      {args_cli.infer_steps}")
    print(f"  Inference Latency:    {lat_ms:.2f} ms")
    print(f"  Training time:        {train_time:.4f} sec")
    print(f"  Train fit error (L2): {fit_err:.4f}")
    print(f"  Mean position error:  {final_error*100:.2f} cm")
    print(f"  Success rate (<5cm):  {success*100:.1f} %")
    print("======================================")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
