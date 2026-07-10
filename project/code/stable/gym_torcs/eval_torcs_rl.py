import argparse
import random
import sys

import numpy as np
import torch

from gym_torcs import TorcsEnv
from rl_agent import PPOAgent, flatten_low_dim_observation, load_checkpoint_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO TORCS agent.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument(
        "--enable-low-progress-termination",
        action="store_true",
        help="Keep the environment's training-style low-progress early termination during evaluation.",
    )
    parser.add_argument(
        "--terminal-judge-start",
        type=int,
        default=None,
        help="Override the step index where low-progress termination becomes active.",
    )
    parser.add_argument(
        "--termination-limit-progress",
        type=float,
        default=None,
        help="Override the minimum progress threshold used by low-progress termination.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="checkpoints/torcs_ppo_latest_best_eval.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--init-log-std", type=float, default=-3.0)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--torcs-path", type=str, default=None)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--debug-env", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=25)
    parser.add_argument(
        "--continue-after-off-track",
        action="store_true",
        help="Apply the off-track penalty but do not terminate/reset the episode.",
    )
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate():
    args = parse_args()
    set_seed(args.seed)

    env = TorcsEnv(
        vision=False,
        throttle=True,
        gear_change=False,
        port=args.port,
        auto_start=args.auto_start,
        torcs_command=args.torcs_path,
        kill_on_shutdown=args.auto_start,
        debug=args.debug_env,
        debug_interval=args.debug_interval,
        terminate_on_off_track=not args.continue_after_off_track,
    )

    # Training benefits from aborting stalled episodes early, but evaluation
    # should usually allow the policy to complete the launch and opening lap.
    if args.enable_low_progress_termination:
        if args.terminal_judge_start is not None:
            env.terminal_judge_start = int(args.terminal_judge_start)
        if args.termination_limit_progress is not None:
            env.termination_limit_progress = float(args.termination_limit_progress)
    else:
        env.terminal_judge_start = int(args.max_steps) + 1

    agent = None

    try:
        completed_episodes = 0
        results = []

        while completed_episodes < args.episodes:
            observation = env.reset(relaunch=False)
            state = flatten_low_dim_observation(observation, env.client.S.d)

            if agent is None:
                state_dim = state.shape[0]
                action_dim = env.action_space.shape[0]
                agent = PPOAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    lr=args.lr,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_epsilon=args.clip_epsilon,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    entropy_coef=args.entropy_coef,
                    value_coef=args.value_coef,
                    target_kl=args.target_kl,
                    max_grad_norm=args.max_grad_norm,
                    init_log_std=args.init_log_std,
                    device=args.device,
                )
                payload = torch.load(args.model_path, map_location=agent.device)
                load_report, metadata = load_checkpoint_payload(agent.model, None, payload)
                agent.model.eval()
                print(f"Loaded weights from {args.model_path}")
                print(
                    "Transfer summary | "
                    f"exact={len(load_report['loaded'])} | "
                    f"partial={len(load_report['partial'])} | "
                    f"skipped={len(load_report['skipped'])}"
                )
                if load_report["partial"]:
                    print("Partially transferred:", "; ".join(load_report["partial"]))
                if load_report["skipped"]:
                    print("Skipped:", "; ".join(load_report["skipped"]))
                if metadata:
                    print(
                        "Checkpoint metadata | "
                        f"update_idx={metadata.get('update_idx', 0)} | "
                        f"episode_count={metadata.get('episode_count', 0)} | "
                        f"env_steps={metadata.get('env_steps', 0)}"
                    )
                print(
                    "Eval termination config | "
                    f"terminal_judge_start={env.terminal_judge_start} | "
                    f"termination_limit_progress={env.termination_limit_progress:.2f} | "
                    f"low_progress_enabled={args.enable_low_progress_termination}"
                )

            total_reward = 0.0
            progress_values = []
            angle_abs_values = []
            track_pos_abs_values = []
            steer_values = []
            accel_values = []
            brake_values = []
            steps = 0
            termination_reason = "max_steps"
            final_info = {}

            for _ in range(args.max_steps):
                action_info = agent.act(state, deterministic=True)
                next_observation, reward, done, info = env.step(action_info["action"])
                final_info = dict(info)
                raw_obs = env.client.S.d
                state = flatten_low_dim_observation(next_observation, raw_obs)
                total_reward += reward
                steps += 1
                progress_values.append(float(raw_obs["speedX"]) * np.cos(float(raw_obs["angle"])))
                angle_abs_values.append(abs(float(raw_obs["angle"])))
                track_pos_abs_values.append(abs(float(raw_obs["trackPos"])))
                steer_values.append(float(info.get("applied_steer", action_info["action"][0])))
                accel_values.append(float(info.get("applied_accel", action_info["action"][1])))
                brake_values.append(float(info.get("applied_brake", action_info["action"][2])))

                if done:
                    termination_reason = info.get("termination_reason") or "done"
                    break

            if steps < args.min_steps:
                print(f"Warm-up reset episode ignored | steps={steps} | reward={total_reward:.2f}")
                continue

            completed_episodes += 1
            episode_metrics = {
                "steps": steps,
                "reward": total_reward,
                "progress": float(np.mean(progress_values)) if progress_values else 0.0,
                "abs_angle": float(np.mean(angle_abs_values)) if angle_abs_values else 0.0,
                "abs_track_pos": float(np.mean(track_pos_abs_values)) if track_pos_abs_values else 0.0,
                "mean_steer": float(np.mean(steer_values)) if steer_values else 0.0,
                "mean_accel": float(np.mean(accel_values)) if accel_values else 0.0,
                "mean_brake": float(np.mean(brake_values)) if brake_values else 0.0,
                "termination_reason": termination_reason,
                "final_track_pos": float(final_info.get("track_pos", 0.0)),
                "final_track_min": float(final_info.get("track_min", 0.0)),
                "final_track_center": float(final_info.get("track_center", 0.0)),
            }
            results.append(episode_metrics)
            print(
                f"Eval episode {completed_episodes}/{args.episodes} | "
                f"steps={episode_metrics['steps']} | "
                f"reward={episode_metrics['reward']:.2f} | "
                f"progress={episode_metrics['progress']:.2f} | "
                f"abs_angle={episode_metrics['abs_angle']:.3f} | "
                f"abs_trackPos={episode_metrics['abs_track_pos']:.3f} | "
                f"mean_steer={episode_metrics['mean_steer']:.3f} | "
                f"mean_accel={episode_metrics['mean_accel']:.3f} | "
                f"mean_brake={episode_metrics['mean_brake']:.3f} | "
                f"final_trackPos={episode_metrics['final_track_pos']:.3f} | "
                f"final_track_min={episode_metrics['final_track_min']:.1f} | "
                f"final_track_center={episode_metrics['final_track_center']:.1f} | "
                f"termination={episode_metrics['termination_reason']}"
            )

        if results:
            avg_steps = float(np.mean([item["steps"] for item in results]))
            avg_reward = float(np.mean([item["reward"] for item in results]))
            avg_progress = float(np.mean([item["progress"] for item in results]))
            avg_abs_angle = float(np.mean([item["abs_angle"] for item in results]))
            avg_abs_track_pos = float(np.mean([item["abs_track_pos"] for item in results]))
            avg_mean_steer = float(np.mean([item["mean_steer"] for item in results]))
            avg_mean_accel = float(np.mean([item["mean_accel"] for item in results]))
            avg_mean_brake = float(np.mean([item["mean_brake"] for item in results]))
            print(
                f"Summary | avg_steps={avg_steps:.1f} | "
                f"avg_reward={avg_reward:.2f} | "
                f"avg_progress={avg_progress:.2f} | "
                f"avg_abs_angle={avg_abs_angle:.3f} | "
                f"avg_abs_trackPos={avg_abs_track_pos:.3f} | "
                f"avg_mean_steer={avg_mean_steer:.3f} | "
                f"avg_mean_accel={avg_mean_accel:.3f} | "
                f"avg_mean_brake={avg_mean_brake:.3f}"
            )
    finally:
        env.end()


if __name__ == "__main__":
    evaluate()
