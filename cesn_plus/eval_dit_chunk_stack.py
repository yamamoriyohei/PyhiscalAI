import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GR00T DiT with action-chunk on Stack.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--infer_steps", type=int, default=4)
parser.add_argument("--chunk", type=int, default=16)
parser.add_argument("--n_train", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.util, time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

DEVICE = "cuda"

spec = importlib.util.spec_from_file_location(
    "dit", "/workspace/Isaac-GR00T/gr00t/model/modules/dit.py")
dit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dit_mod)
DiT = dit_mod.DiT


class DiTChunkPolicy(nn.Module):
    """
    GR00T本来の使い方: 1観測 -> 未来Hステップの行動チャンクを一括生成。
    hidden_states を [B, H, inner] にして、DiTがH本の行動トークンを同時にdenoise。
    """
    def __init__(self, obs_dim, act_dim, chunk=16, hidden=256, n_heads=4, head_dim=64, layers=4):
        super().__init__()
        self.act_dim = act_dim
        self.chunk = chunk
        self.inner = n_heads * head_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.inner))
        self.act_in = nn.Linear(act_dim, self.inner)
        self.dit = DiT(num_attention_heads=n_heads, attention_head_dim=head_dim,
                       output_dim=self.inner, num_layers=layers,
                       cross_attention_dim=self.inner,
                       max_num_positional_embeddings=chunk + 8)
        self.act_out = nn.Linear(self.inner, act_dim)

    def forward(self, obs, noisy_chunk, t):
        # obs:[B,obs], noisy_chunk:[B,H,act], t:[B]
        cond = self.cond_encoder(obs).unsqueeze(1)        # [B,1,inner]
        h = self.act_in(noisy_chunk)                      # [B,H,inner]
        out = self.dit(hidden_states=h, encoder_hidden_states=cond, timestep=t)  # [B,H,inner]
        return self.act_out(out)                          # [B,H,act]


def build_chunks(act_traj, H):
    """act_traj:[N,T,act] -> 各時刻tから未来Hステップのチャンク。
    返り: cur_obs_idx用に (N,T)、chunk targets [N,T,H,act]"""
    N, T, A = act_traj.shape
    # 末尾は最後の行動を繰り返してパディング
    pad = act_traj[:, -1:, :].repeat(1, H, 1)             # [N,H,act]
    ext = torch.cat([act_traj, pad], dim=1)               # [N,T+H,act]
    chunks = torch.empty((N, T, H, A), device=act_traj.device)
    for t in range(T):
        chunks[:, t] = ext[:, t:t + H, :]
    return chunks


def main():
    print("[INFO] Loading NVIDIA Stack demos...")
    data = np.load("/workspace/cesn_plus/data/demos_stack_traj.npz", allow_pickle=True)
    obs = torch.tensor(data["obs"], dtype=torch.float32, device=DEVICE)
    act = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)
    N, T, obs_dim = obs.shape
    act_dim = act.shape[2]
    n_train = min(N, args_cli.n_train)
    obs, act = obs[:n_train], act[:n_train]
    H = args_cli.chunk
    print(f"[INFO] traj={n_train}, steps={T}, obs={obs_dim}, act={act_dim}, chunk_H={H}")

    o_mean = obs.reshape(-1, obs_dim).mean(0, keepdim=True)
    o_std = obs.reshape(-1, obs_dim).std(0, keepdim=True) + 1e-6
    a_mean = act.reshape(-1, act_dim).mean(0, keepdim=True)
    a_std = act.reshape(-1, act_dim).std(0, keepdim=True) + 1e-6
    obs_n = (obs - o_mean) / o_std                        # [n,T,obs]
    act_n = (act - a_mean) / a_std                        # [n,T,act]

    chunks = build_chunks(act_n, H)                       # [n,T,H,act]
    obs_flat = obs_n.reshape(-1, obs_dim)                 # [n*T,obs]
    chunk_flat = chunks.reshape(-1, H, act_dim)           # [n*T,H,act]
    Ntot = obs_flat.shape[0]

    model = DiTChunkPolicy(obs_dim, act_dim, chunk=H, hidden=args_cli.hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] DiT-chunk params: {n_params:,}")

    print(f"[INFO] Training {args_cli.epochs} epochs...")
    bs = 256
    start = time.time()
    for epoch in range(args_cli.epochs):
        idx = torch.randint(0, Ntot, (bs,), device=DEVICE)
        o = obs_flat[idx]                                 # [bs,obs]
        x1 = chunk_flat[idx]                              # [bs,H,act]
        x0 = torch.randn_like(x1)
        t = torch.rand(bs, device=DEVICE)
        xt = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
        v_pred = model(o, xt, (t * 1000).long())
        loss = ((v_pred - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 2000 == 0:
            print(f"  epoch {epoch+1}/{args_cli.epochs} loss={loss.item():.4f}")
    train_time = time.time() - start

    @torch.no_grad()
    def sample_chunk(o_batch, steps):
        B = o_batch.shape[0]
        x = torch.randn((B, H, act_dim), device=DEVICE)
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((B,), s / steps, device=DEVICE)
            x = x + dt * model(o_batch, x, (t * 1000).long())
        return x   # [B,H,act] 正規化スケール

    with torch.no_grad():
        sm = torch.randint(0, Ntot, (1000,), device=DEVICE)
        pred = sample_chunk(obs_flat[sm], args_cli.infer_steps)
        fit_err = torch.norm((pred - chunk_flat[sm]).reshape(-1, act_dim), dim=1).mean().item()
    dummy = torch.randn(64, obs_dim, device=DEVICE)
    for _ in range(5): sample_chunk(dummy, args_cli.infer_steps)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50): sample_chunk(dummy, args_cli.infer_steps)
    torch.cuda.synchronize(); lat_ms = (time.perf_counter() - t0) * 1000 / 50

    print("======================================")
    print(f"[RESULT] DiT-chunk Train: {train_time:.2f}s  fit(norm): {fit_err:.4f}  latency(64env,1chunk): {lat_ms:.2f}ms")
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

    ne = args_cli.num_envs
    succ = torch.zeros(ne, dtype=torch.bool, device=DEVICE)
    g1 = torch.zeros_like(succ); s1 = torch.zeros_like(succ); g2 = torch.zeros_like(succ)

    # action chunkを生成し、H ステップ実行してから次のchunkを生成（receding horizon）
    print(f"[INFO] Evaluating DiT-chunk on Stack (chunk={H})...")
    step = 0
    while step < args_cli.num_steps:
        cur = (flatten_obs(obs_dict) - o_mean) / o_std
        chunk_pred = sample_chunk(cur, args_cli.infer_steps) * a_std + a_mean  # [ne,H,act]
        # チャンクを順に実行
        for h in range(H):
            if step >= args_cli.num_steps:
                break
            actions = chunk_pred[:, h, :]
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            succ |= base_env.termination_manager.get_term("success")
            st = obs_dict.get("subtask_terms", {})
            if "grasp_1" in st: g1 |= st["grasp_1"].bool()
            if "stack_1" in st: s1 |= st["stack_1"].bool()
            if "grasp_2" in st: g2 |= st["grasp_2"].bool()
            step += 1

    print("======================================")
    print(f"[RESULT] DiT with action-chunk (H={H}) on Stack")
    print(f"  Params: {n_params:,}  Train: {train_time:.2f}s  Latency: {lat_ms:.2f}ms  fit: {fit_err:.4f}")
    print(f"  grasp_1: {g1.float().mean()*100:.1f}%  stack_1: {s1.float().mean()*100:.1f}%  grasp_2: {g2.float().mean()*100:.1f}%  SUCCESS: {succ.float().mean()*100:.1f}%")
    print("======================================")
    print("[COMPARE] DiT 1-step: grasp_1 98.4%, stack_1 15.6%, grasp_2 1.6%, SUCCESS 0.0%")
    print("[COMPARE] Plan-A 1-step+hist: grasp_1 92.2%, stack_1 25.0%, grasp_2 21.9%, SUCCESS 0.0%")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
