import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay expert demos from their initial states.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

DEVICE = "cuda"


def main():
    data = np.load("/workspace/cesn_plus/data/demos_stack_traj.npz", allow_pickle=True)
    act = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)  # [1000,200,7]
    init_q = torch.tensor(data["init_robot_q"], dtype=torch.float32, device=DEVICE)   # [64,9]
    init_c1 = torch.tensor(data["init_cube1"], dtype=torch.float32, device=DEVICE)    # [64,7]
    init_c2 = torch.tensor(data["init_cube2"], dtype=torch.float32, device=DEVICE)
    init_c3 = torch.tensor(data["init_cube3"], dtype=torch.float32, device=DEVICE)
    ne = args_cli.num_envs
    print(f"[INFO] replaying {ne} demos from their recorded initial states")

    task = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=ne)
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    obs, _ = env.reset()

    # --- お手本の初期状態を環境に書き込む ---
    robot = base.scene["robot"]
    origins = base.scene.env_origins   # [ne,3] 各envの原点

    # ロボット関節を設定
    q = init_q[:ne].clone()            # [ne,9]
    qd = torch.zeros_like(q)
    robot.write_joint_state_to_sim(q, qd)

    # キューブのposeを設定（env originを加算してワールド座標へ）
    def set_cube(name, pose_local):
        cube = base.scene[name]
        pose = pose_local[:ne].clone()       # [ne,7] (xyz+quat)
        pose[:, :3] = pose[:, :3] + origins  # ローカル->ワールド
        vel = torch.zeros((ne, 6), device=DEVICE)
        cube.write_root_pose_to_sim(pose)
        cube.write_root_velocity_to_sim(vel)

    for nm, pl in [("cube_1", init_c1), ("cube_2", init_c2), ("cube_3", init_c3)]:
        try:
            set_cube(nm, pl)
        except Exception as e:
            print(f"[WARN] set {nm} failed: {e}")

    # 物理を1ステップ進めて反映
    base.sim.step()
    base.scene.update(dt=base.physics_dt)

    succ = torch.zeros(ne, dtype=torch.bool, device=DEVICE)
    g1 = torch.zeros_like(succ); s1 = torch.zeros_like(succ); g2 = torch.zeros_like(succ)
    for t in range(args_cli.num_steps):
        a = act[:ne, t, :]
        obs, r, term, trunc, info = env.step(a)
        succ |= base.termination_manager.get_term("success")
        st = obs.get("subtask_terms", {})
        if "grasp_1" in st: g1 |= st["grasp_1"].bool()
        if "stack_1" in st: s1 |= st["stack_1"].bool()
        if "grasp_2" in st: g2 |= st["grasp_2"].bool()

    print("======================================")
    print(f"[RESULT] Expert demo replay from recorded initial states ({ne} envs)")
    print(f"  grasp_1: {g1.float().mean()*100:.1f}%")
    print(f"  stack_1: {s1.float().mean()*100:.1f}%")
    print(f"  grasp_2: {g2.float().mean()*100:.1f}%")
    print(f"  SUCCESS: {succ.float().mean()*100:.1f}%")
    print("======================================")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
