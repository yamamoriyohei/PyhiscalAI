import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fair CNMP (obs->act regression) vs CESN+.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=300)
parser.add_argument("--epochs", type=int, default=8000)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--n_max_context", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn
import time
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

DEVICE = "cuda"


class CNMP(nn.Module):
    """
    CNMP (Seker et al., RSS 2019) 式(1)(2)(3) の構造を保持。
    条件付き回帰として使用:
      コンテキスト点 = (観測obs_i, 行動act_i) のペア  -> Encoder -> r_i  (式1)
      Aggregator: r = mean(r_i)                                        (式2)
      Decoder: (r, 観測target) -> (mu_act, sigma_act)                 (式3)
    CESN+ と同じ「観測->行動」写像。出力は行動7次元のみ。
    """
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # Encoder: (obs, act) -> 表現r
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # Decoder: (r, obs_target) -> (mu, log_sigma) of action
        self.decoder = nn.Sequential(
            nn.Linear(hidden + obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * act_dim),
        )

    def forward(self, ctx_obs, ctx_act, tgt_obs):
        r = self.encoder(torch.cat([ctx_obs, ctx_act], dim=-1))  # [Nc,hidden] 式(1)
        r = r.mean(dim=0, keepdim=True)                          # 式(2)
        r_rep = r.repeat(tgt_obs.shape[0], 1)
        out = self.decoder(torch.cat([r_rep, tgt_obs], dim=-1))  # 式(3)
        mu, log_sigma = out[:, :self.act_dim], out[:, self.act_dim:]
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-4
        return mu, sigma


def main():
    print("[INFO] Loading per-trajectory PPO demos...")
    data = np.load("/workspace/cesn_plus/data/demos_ppo_traj.npz")
    obs = torch.tensor(data["obs"], dtype=torch.float32, device=DEVICE)     # [N,T,32]
    act = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)  # [N,T,7]
    N, T, obs_dim = obs.shape
    act_dim = act.shape[2]
    print(f"[INFO] trajectories={N}, steps={T}, obs_dim={obs_dim}, act_dim={act_dim}")

    # 正規化
    o_mean = obs.reshape(-1, obs_dim).mean(0, keepdim=True)
    o_std = obs.reshape(-1, obs_dim).std(0, keepdim=True) + 1e-6
    a_mean = act.reshape(-1, act_dim).mean(0, keepdim=True)
    a_std = act.reshape(-1, act_dim).std(0, keepdim=True) + 1e-6
    obs_n = (obs - o_mean) / o_std    # [N,T,32]
    act_n = (act - a_mean) / a_std    # [N,T,7]

    model = CNMP(obs_dim, act_dim, hidden=args_cli.hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    print(f"[INFO] Training CNMP for {args_cli.epochs} epochs...")
    start = time.time()
    for epoch in range(args_cli.epochs):
        i = np.random.randint(N)
        o_i = obs_n[i]    # [T,32]
        a_i = act_n[i]    # [T,7]
        n_ctx = np.random.randint(1, args_cli.n_max_context + 1)
        ctx_ids = np.random.choice(T, n_ctx, replace=False)
        tgt_ids = np.random.choice(T, 64, replace=False)
        mu, sigma = model(o_i[ctx_ids], a_i[ctx_ids], o_i[tgt_ids])
        dist = torch.distributions.Normal(mu, sigma)
        loss = -dist.log_prob(a_i[tgt_ids]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 1000 == 0:
            print(f"  epoch {epoch+1}/{args_cli.epochs} loss={loss.item():.4f}")
    train_time = time.time() - start

    # fit error（行動再現, 実スケール）
    with torch.no_grad():
        errs = []
        for i in range(N):
            ctx_ids = np.random.choice(T, args_cli.n_max_context, replace=False)
            mu, _ = model(obs_n[i][ctx_ids], act_n[i][ctx_ids], obs_n[i])
            pred = mu * a_std + a_mean
            errs.append(torch.norm(pred - act[i], dim=1).mean().item())
        fit_err = float(np.mean(errs))
    print("======================================")
    print(f"[RESULT] CNMP Training time: {train_time:.4f} sec")
    print(f"[INFO]  Train action fit error (L2): {fit_err:.4f}")
    print("======================================")

    # 評価: 学習データから固定コンテキスト集合を作り、現在観測をtargetにして行動予測
    # (各stepで観測が変わる=targetが動く。contextは学習軌道の代表点を使う)
    task = "Isaac-Reach-Franka-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=env_cfg)
    base_env = env.unwrapped
    obs_dict, _ = env.reset()
    cur_obs = obs_dict["policy"]
    ee_idx = base_env.scene["robot"].body_names.index("panda_hand")

    # 固定コンテキスト: 全軌道からランダムにn_max_context点を集める（代表的な obs->act 対応）
    flat_o = obs_n.reshape(-1, obs_dim)
    flat_a = act_n.reshape(-1, act_dim)
    ctx_pick = np.random.choice(flat_o.shape[0], args_cli.n_max_context, replace=False)
    ctx_obs = flat_o[ctx_pick]   # [Nc,32]
    ctx_act = flat_a[ctx_pick]   # [Nc,7]
    # エンコーダ表現は固定なので前計算
    with torch.no_grad():
        r_fixed = model.encoder(torch.cat([ctx_obs, ctx_act], dim=-1)).mean(0, keepdim=True)  # [1,hidden]

    errors = []
    print("[INFO] Evaluating CNMP...")
    for step in range(args_cli.num_steps):
        cur_n = (cur_obs - o_mean) / o_std        # [num_envs,32]
        with torch.no_grad():
            r_rep = r_fixed.repeat(args_cli.num_envs, 1)
            out = model.decoder(torch.cat([r_rep, cur_n], dim=-1))
            mu = out[:, :act_dim]
            actions = mu * a_std + a_mean
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
    print(f"[RESULT] CNMP Accuracy Evaluation (fair: obs->act)")
    print(f"  Hidden size:          {args_cli.hidden}")
    print(f"  Training epochs:      {args_cli.epochs}")
    print(f"  Training time:        {train_time:.4f} sec")
    print(f"  Train fit error (L2): {fit_err:.4f}")
    print(f"  Mean position error:  {final_error*100:.2f} cm")
    print(f"  Success rate (<5cm):  {success*100:.1f} %")
    print("======================================")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
