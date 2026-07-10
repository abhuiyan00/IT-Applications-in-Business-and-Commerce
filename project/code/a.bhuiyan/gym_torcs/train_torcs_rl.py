import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gym_torcs import TorcsEnv
from rl_agent import (
    PPOAgent,
    RolloutBuffer,
    build_state,
    load_checkpoint_payload,
)

DEFAULT_RACING_LINE = str(Path(__file__).resolve().parent / "lines" / "corkscrew.npz")


def parse_args():
    parser = argparse.ArgumentParser(description="PPO trainer for steering-only TORCS.")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--min-train-steps", type=int, default=5)
    parser.add_argument("--relaunch-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--init-log-std", type=float, default=-2.5)
    parser.add_argument("--save-path", type=str, default="checkpoints/corkscrew_ppo_latest.pt")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--racing-line", type=str, default=DEFAULT_RACING_LINE,
                        help="Path to a generated racing-line .npz. Use 'none' to disable "
                             "(falls back to steering-only baseline).")
    parser.add_argument("--throttle", action="store_true",
                        help="Force throttle/brake control even without a racing line.")
    parser.add_argument("--launch-warmup-target", type=float, default=90.0,
                        help="km/h to spool up to before the agent takes over each episode.")
    parser.add_argument("--no-launch-warmup", action="store_true",
                        help="Disable the high-speed launch warmup (start from standstill).")
    parser.add_argument("--no-assist", action="store_true",
                        help="Disable the racing-line baseline path-follower (pure-agent control).")
    parser.add_argument("--residual-scale", type=float, default=None,
                        help="Agent residual authority on top of the baseline (0=pure baseline, "
                             "1=full agent). Sets steer/accel/brake scales. Default 0.5.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--torcs-path", type=str, default=None)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="Disable the per-step telemetry recorder (on by default).")
    parser.add_argument("--debug-env", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=300)
    args = parser.parse_args()
    if args.racing_line and args.racing_line.lower() == "none":
        args.racing_line = None
    sys.argv = [sys.argv[0]]
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_checkpoint_paths(save_path):
    latest_path = Path(save_path)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if latest_path.suffix:
        best_train_path = latest_path.with_name(f"{latest_path.stem}_best_train{latest_path.suffix}")
        best_eval_path = latest_path.with_name(f"{latest_path.stem}_best_eval{latest_path.suffix}")
    else:
        best_train_path = latest_path.with_name(f"{latest_path.name}_best_train")
        best_eval_path = latest_path.with_name(f"{latest_path.name}_best_eval")
    return latest_path, best_train_path, best_eval_path


def compute_episode_score(mean_progress, mean_abs_track_pos, mean_abs_angle, steps):
    return (
        mean_progress
        - 3.0 * mean_abs_track_pos
        - 2.0 * mean_abs_angle
        + 0.01 * steps
    )


def collect_eval_metrics(env, agent, episodes, max_steps):
    results = []
    for _ in range(episodes):
        observation = env.reset(relaunch=False)
        state = build_state(observation, env.client.S.d, env.racing_line)
        total_reward = 0.0
        progress_values = []
        angle_abs_values = []
        track_pos_abs_values = []
        steer_values = []
        steps = 0

        for _ in range(max_steps):
            action_info = agent.act(state, deterministic=True)
            next_observation, reward, done, _ = env.step(action_info["action"])
            raw_obs = env.client.S.d
            state = build_state(next_observation, raw_obs, env.racing_line)
            total_reward += reward
            steps += 1
            progress_values.append(float(raw_obs["speedX"]) * np.cos(float(raw_obs["angle"])))
            angle_abs_values.append(abs(float(raw_obs["angle"])))
            track_pos_abs_values.append(abs(float(raw_obs["trackPos"])))
            steer_values.append(abs(float(action_info["action"][0])))
            if done:
                break

        results.append(
            {
                "steps": steps,
                "reward": total_reward,
                "progress": float(np.mean(progress_values)) if progress_values else 0.0,
                "abs_angle": float(np.mean(angle_abs_values)) if angle_abs_values else 0.0,
                "abs_track_pos": float(np.mean(track_pos_abs_values)) if track_pos_abs_values else 0.0,
                "mean_steer": float(np.mean(steer_values)) if steer_values else 0.0,
            }
        )

    avg_steps = float(np.mean([item["steps"] for item in results])) if results else 0.0
    avg_reward = float(np.mean([item["reward"] for item in results])) if results else 0.0
    avg_progress = float(np.mean([item["progress"] for item in results])) if results else 0.0
    avg_abs_angle = float(np.mean([item["abs_angle"] for item in results])) if results else 0.0
    avg_abs_track_pos = float(np.mean([item["abs_track_pos"] for item in results])) if results else 0.0
    avg_mean_steer = float(np.mean([item["mean_steer"] for item in results])) if results else 0.0
    score = compute_episode_score(avg_progress, avg_abs_track_pos, avg_abs_angle, avg_steps)

    return {
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "avg_progress": avg_progress,
        "avg_abs_angle": avg_abs_angle,
        "avg_abs_track_pos": avg_abs_track_pos,
        "avg_mean_steer": avg_mean_steer,
        "score": score,
    }


def save_checkpoint(path, agent, args, update_idx, env_steps, episode_count, best_train_score, best_eval_score):
    payload = {
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "update_idx": update_idx,
        "env_steps": env_steps,
        "episode_count": episode_count,
        "best_train_score": best_train_score,
        "best_eval_score": best_eval_score,
        "args": vars(args),
    }
    torch.save(payload, path)


def build_env(args):
    throttle = bool(args.throttle or args.racing_line)
    return TorcsEnv(
        vision=False,
        throttle=throttle,
        gear_change=False,
        port=args.port,
        auto_start=args.auto_start,
        torcs_command=args.torcs_path,
        kill_on_shutdown=args.auto_start,
        debug=args.debug_env,
        debug_interval=args.debug_interval,
        racing_line_path=args.racing_line,
        launch_warmup_enabled=not args.no_launch_warmup,
        launch_warmup_target=args.launch_warmup_target,
        baseline_assist=not args.no_assist,
        residual_scale=args.residual_scale,
        telemetry=not args.no_telemetry,
    )


def train():
    args = parse_args()
    set_seed(args.seed)
    latest_path, best_train_path, best_eval_path = build_checkpoint_paths(args.save_path)

    env = build_env(args)
    agent = None

    # Persistent episode log so off-track / termination causes are quantifiable
    # after the fact (stdout-only summaries were lost between runs).
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_log_path = log_dir / f"train_episodes_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    episode_log = open(episode_log_path, "w", newline="")
    episode_writer = csv.writer(episode_log)
    episode_writer.writerow([
        "update", "episode", "steps", "reward", "termination_reason",
        "mean_progress", "mean_abs_track_pos", "mean_abs_angle", "mean_steer",
        "crash_s",
    ])
    print(f"Episode log: {episode_log_path}")

    try:
        observation = env.reset(relaunch=False)
        state = build_state(observation, env.client.S.d, env.racing_line)
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
            init_log_std=args.init_log_std,
            target_kl=args.target_kl,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
        )

        start_update = 0
        episode_count = 0
        env_steps = 0
        best_train_score = None
        best_eval_score = None

        if args.load_path:
            payload = torch.load(args.load_path, map_location=agent.device)
            load_report, metadata = load_checkpoint_payload(agent.model, agent.optimizer, payload)
            start_update = int(metadata.get("update_idx", 0))
            episode_count = int(metadata.get("episode_count", 0))
            env_steps = int(metadata.get("env_steps", 0))
            best_train_score = metadata.get("best_train_score")
            best_eval_score = metadata.get("best_eval_score")
            print(f"Loaded checkpoint from {args.load_path}")
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

        first_episode_mean = None
        first_episode_action = None

        for update_idx in range(start_update, args.updates):
            buffer = RolloutBuffer()
            rollout_episode_metrics = []
            current_episode_reward = 0.0
            current_episode_steps = 0
            current_progress_values = []
            current_angle_abs_values = []
            current_track_pos_abs_values = []
            current_steer_values = []
            episode_crash_s = None
            relaunch = (update_idx % args.relaunch_every == 0) and update_idx != start_update

            if relaunch:
                observation = env.reset(relaunch=True)
                state = build_state(observation, env.client.S.d, env.racing_line)

            while len(buffer) < args.rollout_size:
                action_info = agent.act(state, deterministic=False)
                if first_episode_mean is None:
                    first_episode_mean = float(action_info["mean"][0])
                    first_episode_action = float(action_info["action"][0])

                next_observation, reward, done, info = env.step(action_info["action"])
                raw_obs = env.client.S.d
                next_state = build_state(next_observation, raw_obs, env.racing_line)

                buffer.add(
                    state=state,
                    action=action_info["action"],
                    reward=reward,
                    done=done,
                    value=action_info["value"],
                    log_prob=action_info["log_prob"],
                    episode_id=episode_count,
                )

                current_episode_reward += reward
                current_episode_steps += 1
                env_steps += 1
                progress_value = float(raw_obs["speedX"]) * np.cos(float(raw_obs["angle"]))
                current_progress_values.append(progress_value)
                current_angle_abs_values.append(abs(float(raw_obs["angle"])))
                current_track_pos_abs_values.append(abs(float(raw_obs["trackPos"])))
                current_steer_values.append(float(action_info["action"][0]))

                state = next_state

                termination_reason = info.get("termination_reason")
                if info.get("crash_s") is not None:
                    episode_crash_s = float(info["crash_s"])
                if current_episode_steps >= args.max_steps and not done:
                    termination_reason = "max_steps"

                if done or current_episode_steps >= args.max_steps:
                    if current_episode_steps >= args.min_train_steps:
                        mean_progress = float(np.mean(current_progress_values)) if current_progress_values else 0.0
                        mean_abs_angle = float(np.mean(current_angle_abs_values)) if current_angle_abs_values else 0.0
                        mean_abs_track_pos = float(np.mean(current_track_pos_abs_values)) if current_track_pos_abs_values else 0.0
                        mean_steer = float(np.mean(current_steer_values)) if current_steer_values else 0.0
                        score = compute_episode_score(
                            mean_progress,
                            mean_abs_track_pos,
                            mean_abs_angle,
                            current_episode_steps,
                        )
                        rollout_episode_metrics.append(
                            {
                                "steps": current_episode_steps,
                                "reward": current_episode_reward,
                                "mean_progress": mean_progress,
                                "mean_abs_angle": mean_abs_angle,
                                "mean_abs_track_pos": mean_abs_track_pos,
                                "mean_steer": mean_steer,
                                "score": score,
                                "first_mean": first_episode_mean if first_episode_mean is not None else 0.0,
                                "first_action": first_episode_action if first_episode_action is not None else 0.0,
                                "termination_reason": termination_reason if termination_reason is not None else "unknown",
                            }
                        )
                        episode_writer.writerow([
                            update_idx + 1,
                            episode_count,
                            current_episode_steps,
                            f"{current_episode_reward:.3f}",
                            termination_reason if termination_reason is not None else "unknown",
                            f"{mean_progress:.3f}",
                            f"{mean_abs_track_pos:.4f}",
                            f"{mean_abs_angle:.4f}",
                            f"{mean_steer:.4f}",
                            f"{episode_crash_s:.1f}" if episode_crash_s is not None else "",
                        ])
                        episode_log.flush()
                        episode_count += 1

                    observation = env.reset(relaunch=False)
                    state = build_state(observation, env.client.S.d, env.racing_line)
                    current_episode_reward = 0.0
                    current_episode_steps = 0
                    current_progress_values = []
                    current_angle_abs_values = []
                    current_track_pos_abs_values = []
                    current_steer_values = []
                    episode_crash_s = None
                    first_episode_mean = None
                    first_episode_action = None

            last_value = agent.value(state)
            update_metrics = agent.update(buffer, last_value=last_value)

            rollout_mean_progress = float(np.mean([item["mean_progress"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_abs_angle = float(np.mean([item["mean_abs_angle"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_abs_track_pos = float(np.mean([item["mean_abs_track_pos"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_steer = float(np.mean([item["mean_steer"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_reward = float(np.mean([item["reward"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_steps = float(np.mean([item["steps"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_score = compute_episode_score(
                rollout_mean_progress,
                rollout_mean_abs_track_pos,
                rollout_mean_abs_angle,
                rollout_mean_steps,
            )
            first_mean = float(rollout_episode_metrics[0]["first_mean"]) if rollout_episode_metrics else 0.0
            first_action = float(rollout_episode_metrics[0]["first_action"]) if rollout_episode_metrics else 0.0

            termination_counts = {}
            for item in rollout_episode_metrics:
                reason = item.get("termination_reason", "unknown")
                termination_counts[reason] = termination_counts.get(reason, 0) + 1

            termination_summary = ", ".join(
                f"{key}:{value}" for key, value in sorted(termination_counts.items())
            ) if termination_counts else "none"
            is_best_train = best_train_score is None or rollout_score > best_train_score
            if is_best_train:
                best_train_score = rollout_score
                save_checkpoint(
                    best_train_path,
                    agent,
                    args,
                    update_idx + 1,
                    env_steps,
                    episode_count,
                    best_train_score,
                    best_eval_score,
                )
                # Good run -> persist the temp corner-speed memory alongside the
                # best checkpoint (bad runs' panic slow-downs are discarded).
                env.commit_corner_memory()

            print(
                f"Update {update_idx + 1}/{args.updates} | "
                f"rollout_steps={len(buffer)} | "
                f"episodes={len(rollout_episode_metrics)} | "
                f"mean_reward={rollout_mean_reward:.2f} | "
                f"mean_progress={rollout_mean_progress:.2f} | "
                f"mean_abs_angle={rollout_mean_abs_angle:.3f} | "
                f"mean_abs_trackPos={rollout_mean_abs_track_pos:.3f} | "
                f"mean_steer={rollout_mean_steer:.3f} | "
                f"first_mean={first_mean:.3f} | "
                f"first_action={first_action:.3f} | "
                f"termination={termination_summary} | "
                f"score={rollout_score:.3f} | "
                f"policy_loss={update_metrics['policy_loss']:.4f} | "
                f"value_loss={update_metrics['value_loss']:.4f} | "
                f"entropy={update_metrics['entropy']:.4f} | "
                f"approx_kl={update_metrics['approx_kl']:.5f} | "
                f"clip_fraction={update_metrics['clip_fraction']:.3f} | "
                f"epochs_ran={update_metrics['epochs_ran']} | "
                f"early_stop={update_metrics['stopped_early']}"
            )
            if is_best_train:
                print(f"New best_train checkpoint | score={best_train_score:.3f} | path={best_train_path}")

            if args.eval_interval > 0 and (update_idx + 1) % args.eval_interval == 0:
                eval_metrics = collect_eval_metrics(
                    env=env,
                    agent=agent,
                    episodes=args.eval_episodes,
                    max_steps=args.eval_max_steps,
                )
                is_best_eval = best_eval_score is None or eval_metrics["score"] > best_eval_score
                if is_best_eval:
                    best_eval_score = eval_metrics["score"]
                    save_checkpoint(
                        best_eval_path,
                        agent,
                        args,
                        update_idx + 1,
                        env_steps,
                        episode_count,
                        best_train_score,
                        best_eval_score,
                    )
                print(
                    f"EVAL update={update_idx + 1} | "
                    f"avg_steps={eval_metrics['avg_steps']:.1f} | "
                    f"avg_reward={eval_metrics['avg_reward']:.2f} | "
                    f"avg_progress={eval_metrics['avg_progress']:.2f} | "
                    f"avg_abs_angle={eval_metrics['avg_abs_angle']:.3f} | "
                    f"avg_abs_trackPos={eval_metrics['avg_abs_track_pos']:.3f} | "
                    f"avg_mean_steer={eval_metrics['avg_mean_steer']:.3f} | "
                    f"score={eval_metrics['score']:.3f}"
                )
                if is_best_eval:
                    print(f"New best_eval checkpoint | score={best_eval_score:.3f} | path={best_eval_path}")

                observation = env.reset(relaunch=False)
                state = build_state(observation, env.client.S.d, env.racing_line)

            save_checkpoint(
                latest_path,
                agent,
                args,
                update_idx + 1,
                env_steps,
                episode_count,
                best_train_score,
                best_eval_score,
            )

        print(f"Latest checkpoint: {latest_path}")
        if best_train_score is not None:
            print(f"Best train checkpoint: {best_train_path} | best_train_score={best_train_score:.3f}")
        if best_eval_score is not None:
            print(f"Best eval checkpoint: {best_eval_path} | best_eval_score={best_eval_score:.3f}")
    finally:
        episode_log.close()
        env.end()


if __name__ == "__main__":
    train()
