import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate trained PPO accuracy on Reach.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner


def main():
    task = "Isaac-Reach-Franka-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")

    env = gym.make(task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)

    obs, _ = env.reset()
    base_env = env.unwrapped

    errors = []
    print("[INFO] Evaluating PPO...")
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            outputs = runner.agent.act(states=obs, observations=obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
        obs, reward, terminated, truncated, info = env.step(actions)

        # エンドエフェクタ位置と目標位置の誤差を計算
        cmd = base_env.command_manager.get_command("ee_pose")  # [num_envs, 7]
        robot = base_env.scene["robot"]
        ee_idx = robot.body_names.index("panda_hand")
        ee_pos = robot.data.body_pos_w[:, ee_idx, :] - base_env.scene.env_origins
        target_pos = cmd[:, :3]
        dist = torch.norm(ee_pos - target_pos, dim=1)
        errors.append(dist.mean().item())

    errors_t = torch.tensor(errors)
    final_error = torch.tensor(errors[-50:]).mean().item()
    print("======================================")
    print(f"[RESULT] PPO Accuracy Evaluation")
    print(f"  Mean position error (last 50 steps): {final_error*100:.2f} cm")
    print(f"  Success rate (<5cm): {(torch.tensor(errors[-50:]) < 0.05).float().mean().item()*100:.1f} %")
    print("======================================")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
