import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GR00T DiT (flow-matching) on Reach, vs CESN+/CNMP.")
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

import importlib.util
import numpy as np
import torch
import torch.nn as nn
import time
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

DEVICE = "cuda"

# --- 本物のGR00T DiTを直接ロード ---
spec = importlib.util.spec_from_file_location(
    "dit", "/workspace/Isaac-GR00T/gr00t/model/modules/dit.py")
dit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dit_mod)
DiT = dit_mod.DiT

class DiTActionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, n_heads=4, head_dim=32, layers=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.inner = n_heads * head_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.inner),
        )
        self.act_in = nn.Linear(act_dim, self.inner)
        self.dit = DiT(
            num_attention_heads=n_heads,
            attention_head_dim=head_dim,
            output_dim=self.inner,
            num_layers=layers,
            cross_attention_dim=self.inner,
        )
        self.act_out = nn.Linear(self.inner, act_dim)

    def forward(self, obs, noisy_act, t):
        cond = self.cond_encoder(obs).unsqueeze(1)
        h = self.act_in(noisy_act).unsqueeze(1)
        out = self.dit(hidden_states=h, encoder_hidden_states=cond, timestep=t)
        return self.act_out(out.squeeze(1))

def main():
    print("[INFO] Loading per-trajectory PPO demos...")
    data = np.load("/workspace/cesn_plus/data/demos_noisy_traj.npz")
    obs = torch.tensor(data["obs"], dtype=torch.float32, device=DEVICE)
    act = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)
    N, T, obs_dim = obs.shape
    act_dim = act.shape[2]

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
    print(f"[INFO] DiT policy total params: {n_params:,}")

    print(f"[INFO] Training DiT (flow-matching) for {args_cli.epochs} epochs...")
    bs = 512
    start = time.time()
    for epoch in range(args_cli.epochs):
        idx = torch.randint(0, Ntot, (bs,), device=DEVICE)
        o = obs_n[idx]
        x1 = act_n[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(bs, device=DEVICE)
        xt = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
        v_target = x1 - x0
        t_disc = (t * 1000).long()
        v_pred = model(o, xt, t_disc)
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
            t = torch.full((B,), s / steps, device=DEVICE)
            t_disc = (t * 1000).long()
            v = model(o_batch, x, t_disc)
            x = x + dt * v
        return x

    with torch.no_grad():
        sample = torch.randint(0, Ntot, (2000,), device=DEVICE)
        pred = sample_action(obs_n[sample], args_cli.infer_steps)
        fit_err = torch.norm((pred - act_n[sample]) * a_std, dim=1).mean().item()

    # --- レイテンシ計測の追加 ---
    dummy_o = torch.randn(1, obs_dim, device=DEVICE)
    for _ in range(10): sample_action(dummy_o, args_cli.infer_steps) # warmup
    lat_start = time.perf_counter()
    for _ in range(100): sample_action(dummy_o, args_cli.infer_steps)
    lat_ms = (time.perf_counter() - lat_start) * 1000 / 100

    print("======================================")
    print(f"[RESULT] DiT Training time: {train_time:.4f} sec")
    print(f"[INFO]  Train action fit error (L2): {fit_err:.4f}")
    print(f"[INFO]  Inference Latency: {lat_ms:.2f} ms")
    print("======================================")

    task = "Isaac-Reach-Franka-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    
    # 可視化マーカーを無効化（libGLU/レンダリング呼び出しを回避）
    try:
        env_cfg.commands.ee_pose.debug_vis = False
    except Exception:
        pass
    for _attr in ["robot", "table", "ground", "light"]:
        try:
            getattr(env_cfg.scene, _attr).debug_vis = False
        except Exception:
            pass

    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    obs_dict, _ = env.reset()
    cur_obs = obs_dict["policy"]
    ee_idx = base_env.scene["robot"].body_names.index("panda_hand")

    errors = []
    print("[INFO] Evaluating DiT...")
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
    print(f"[RESULT] DiT (GR00T System1) Accuracy Evaluation")
    print(f"  Total params:         {n_params:,}")
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
