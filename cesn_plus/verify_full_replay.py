import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(); args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np, torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
DEVICE = "cuda"

def main():
    data = np.load("/workspace/cesn_plus/data/demos_stack_full.npz", allow_pickle=True)
    act = torch.tensor(data["action"], dtype=torch.float32, device=DEVICE)
    init_q = torch.tensor(data["init_robot_q"], dtype=torch.float32, device=DEVICE)
    init_c1 = torch.tensor(data["init_cube1"], dtype=torch.float32, device=DEVICE)
    init_c2 = torch.tensor(data["init_cube2"], dtype=torch.float32, device=DEVICE)
    init_c3 = torch.tensor(data["init_cube3"], dtype=torch.float32, device=DEVICE)
    T_max = int(data["T_max"])
    ne = args_cli.num_envs
    print(f"[INFO] full-length replay, T_max={T_max}, {ne} envs")

    task = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=ne)
    env = gym.make(task, cfg=cfg)
    base = env.unwrapped
    obs, _ = env.reset()
    robot = base.scene["robot"]
    origins = base.scene.env_origins

    q = init_q[:ne].clone(); qd = torch.zeros_like(q)
    robot.write_joint_state_to_sim(q, qd)
    for nm, pl in [("cube_1", init_c1), ("cube_2", init_c2), ("cube_3", init_c3)]:
        cube = base.scene[nm]
        pose = pl[:ne].clone(); pose[:, :3] = pose[:, :3] + origins
        cube.write_root_pose_to_sim(pose)
        cube.write_root_velocity_to_sim(torch.zeros((ne, 6), device=DEVICE))
    base.sim.step(); base.scene.update(dt=base.physics_dt)

    succ = torch.zeros(ne, dtype=torch.bool, device=DEVICE)
    g1 = torch.zeros_like(succ); s1 = torch.zeros_like(succ); g2 = torch.zeros_like(succ)
    for t in range(T_max):
        obs, r, term, trunc, info = env.step(act[:ne, t, :])
        succ |= base.termination_manager.get_term("success")
        st = obs.get("subtask_terms", {})
        if "grasp_1" in st: g1 |= st["grasp_1"].bool()
        if "stack_1" in st: s1 |= st["stack_1"].bool()
        if "grasp_2" in st: g2 |= st["grasp_2"].bool()

    print("======================================")
    print(f"[RESULT] FULL-LENGTH expert replay (T_max={T_max}, {ne} envs)")
    print(f"  grasp_1: {g1.float().mean()*100:.1f}%  stack_1: {s1.float().mean()*100:.1f}%  grasp_2: {g2.float().mean()*100:.1f}%  SUCCESS: {succ.float().mean()*100:.1f}%")
    print("  [200-step replay was: grasp_1 100%, stack_1 95.3%, grasp_2 100%, SUCCESS 1.6%]")
    print("======================================")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
