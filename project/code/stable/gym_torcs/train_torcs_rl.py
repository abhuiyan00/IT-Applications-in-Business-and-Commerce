import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from gym_torcs import TorcsEnv
from rl_agent import (
    PPOAgent,
    RolloutBuffer,
    flatten_low_dim_observation,
    load_checkpoint_payload,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO trainer for TORCS.")
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
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--init-log-std", type=float, default=-1.0)
    parser.add_argument("--save-path", type=str, default="checkpoints/torcs_ppo_latest.pt")
    parser.add_argument("--load-path", type=str, default=None)
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
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-max-steps", type=int, default=300)
    args = parser.parse_args()
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
        state = flatten_low_dim_observation(observation, env.client.S.d)
        total_reward = 0.0
        progress_values = []
        angle_abs_values = []
        track_pos_abs_values = []
        steer_values = []
        accel_values = []
        brake_values = []
        steps = 0
        termination_reason = "max_steps"

        for _ in range(max_steps):
            action_info = agent.act(state, deterministic=True)
            next_observation, reward, done, info = env.step(action_info["action"])
            raw_obs = env.client.S.d
            state = flatten_low_dim_observation(next_observation, raw_obs)
            total_reward += reward
            steps += 1
            progress_values.append(float(raw_obs["speedX"]) * np.cos(float(raw_obs["angle"])))
            angle_abs_values.append(abs(float(raw_obs["angle"])))
            track_pos_abs_values.append(abs(float(raw_obs["trackPos"])))
            steer_values.append(float(action_info["action"][0]))
            accel_values.append(float(action_info["action"][1]))
            brake_values.append(float(action_info["action"][2]))
            if done:
                termination_reason = info.get("termination_reason") or "done"
                break

        results.append(
            {
                "steps": steps,
                "reward": total_reward,
                "progress": float(np.mean(progress_values)) if progress_values else 0.0,
                "abs_angle": float(np.mean(angle_abs_values)) if angle_abs_values else 0.0,
                "abs_track_pos": float(np.mean(track_pos_abs_values)) if track_pos_abs_values else 0.0,
                "mean_steer": float(np.mean(steer_values)) if steer_values else 0.0,
                "mean_accel": float(np.mean(accel_values)) if accel_values else 0.0,
                "mean_brake": float(np.mean(brake_values)) if brake_values else 0.0,
                "termination_reason": termination_reason,
            }
        )

    avg_steps = float(np.mean([item["steps"] for item in results])) if results else 0.0
    avg_reward = float(np.mean([item["reward"] for item in results])) if results else 0.0
    avg_progress = float(np.mean([item["progress"] for item in results])) if results else 0.0
    avg_abs_angle = float(np.mean([item["abs_angle"] for item in results])) if results else 0.0
    avg_abs_track_pos = float(np.mean([item["abs_track_pos"] for item in results])) if results else 0.0
    avg_mean_steer = float(np.mean([item["mean_steer"] for item in results])) if results else 0.0
    avg_mean_accel = float(np.mean([item["mean_accel"] for item in results])) if results else 0.0
    avg_mean_brake = float(np.mean([item["mean_brake"] for item in results])) if results else 0.0
    score = compute_episode_score(avg_progress, avg_abs_track_pos, avg_abs_angle, avg_steps)

    return {
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "avg_progress": avg_progress,
        "avg_abs_angle": avg_abs_angle,
        "avg_abs_track_pos": avg_abs_track_pos,
        "avg_mean_steer": avg_mean_steer,
        "avg_mean_accel": avg_mean_accel,
        "avg_mean_brake": avg_mean_brake,
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
    return TorcsEnv(
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


def train():
    args = parse_args()
    set_seed(args.seed)
    latest_path, best_train_path, best_eval_path = build_checkpoint_paths(args.save_path)

    env = build_env(args)
    agent = None

    try:
        observation = env.reset(relaunch=False)
        state = flatten_low_dim_observation(observation, env.client.S.d)
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
            if metadata.get("optimizer_loaded"):
                print("Optimizer state restored from checkpoint.")
            else:
                print(
                    "Optimizer state not restored"
                    f" ({metadata.get('optimizer_skipped_reason', 'unknown_reason')})."
                )

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
            current_raw_steer_values = []
            current_steer_values = []
            current_raw_accel_values = []
            current_raw_brake_values = []
            current_accel_values = []
            current_brake_values = []
            relaunch = (update_idx % args.relaunch_every == 0) and update_idx != start_update

            if relaunch:
                observation = env.reset(relaunch=True)
                state = flatten_low_dim_observation(observation, env.client.S.d)

            while len(buffer) < args.rollout_size:
                action_info = agent.act(state, deterministic=False)
                if first_episode_mean is None:
                    first_episode_mean = np.asarray(action_info["mean"], dtype=np.float32).copy()
                    first_episode_action = np.asarray(action_info["action"], dtype=np.float32).copy()

                next_observation, reward, done, info = env.step(action_info["action"])
                raw_obs = env.client.S.d
                next_state = flatten_low_dim_observation(next_observation, raw_obs)

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
                current_raw_steer_values.append(float(action_info["action"][0]))
                current_steer_values.append(float(info.get("applied_steer", action_info["action"][0])))
                current_raw_accel_values.append(float(action_info["action"][1]))
                current_raw_brake_values.append(float(action_info["action"][2]))
                current_accel_values.append(float(info.get("applied_accel", action_info["action"][1])))
                current_brake_values.append(float(info.get("applied_brake", action_info["action"][2])))

                state = next_state

                termination_reason = info.get("termination_reason")         
                if current_episode_steps >= args.max_steps and not done:
                    termination_reason = "max_steps"

                if done or current_episode_steps >= args.max_steps:
                    if current_episode_steps >= args.min_train_steps:
                        mean_progress = float(np.mean(current_progress_values)) if current_progress_values else 0.0
                        mean_abs_angle = float(np.mean(current_angle_abs_values)) if current_angle_abs_values else 0.0
                        mean_abs_track_pos = float(np.mean(current_track_pos_abs_values)) if current_track_pos_abs_values else 0.0
                        mean_raw_steer = float(np.mean(current_raw_steer_values)) if current_raw_steer_values else 0.0
                        mean_steer = float(np.mean(current_steer_values)) if current_steer_values else 0.0
                        mean_raw_accel = float(np.mean(current_raw_accel_values)) if current_raw_accel_values else 0.0
                        mean_raw_brake = float(np.mean(current_raw_brake_values)) if current_raw_brake_values else 0.0
                        mean_accel = float(np.mean(current_accel_values)) if current_accel_values else 0.0
                        mean_brake = float(np.mean(current_brake_values)) if current_brake_values else 0.0
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
                                "mean_raw_steer": mean_raw_steer,
                                "mean_steer": mean_steer,
                                "mean_raw_accel": mean_raw_accel,
                                "mean_raw_brake": mean_raw_brake,
                                "mean_accel": mean_accel,
                                "mean_brake": mean_brake,
                                "score": score,
                                "first_mean": first_episode_mean.copy() if first_episode_mean is not None else np.zeros(action_dim, dtype=np.float32),
                                "first_action": first_episode_action.copy() if first_episode_action is not None else np.zeros(action_dim, dtype=np.float32),
                                "termination_reason": termination_reason if termination_reason is not None else "unknown",
                            }
                        )
                        episode_count += 1

                    observation = env.reset(relaunch=False)
                    state = flatten_low_dim_observation(observation, env.client.S.d)
                    current_episode_reward = 0.0
                    current_episode_steps = 0
                    current_progress_values = []
                    current_angle_abs_values = []
                    current_track_pos_abs_values = []
                    current_raw_steer_values = []
                    current_steer_values = []
                    current_raw_accel_values = []
                    current_raw_brake_values = []
                    current_accel_values = []
                    current_brake_values = []
                    first_episode_mean = None
                    first_episode_action = None

            last_value = agent.value(state)
            update_metrics = agent.update(buffer, last_value=last_value)

            rollout_mean_progress = float(np.mean([item["mean_progress"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_abs_angle = float(np.mean([item["mean_abs_angle"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_abs_track_pos = float(np.mean([item["mean_abs_track_pos"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_raw_steer = float(np.mean([item["mean_raw_steer"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_steer = float(np.mean([item["mean_steer"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_reward = float(np.mean([item["reward"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_steps = float(np.mean([item["steps"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_raw_accel = float(np.mean([item["mean_raw_accel"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_raw_brake = float(np.mean([item["mean_raw_brake"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_accel = float(np.mean([item["mean_accel"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_mean_brake = float(np.mean([item["mean_brake"] for item in rollout_episode_metrics])) if rollout_episode_metrics else 0.0
            rollout_score = compute_episode_score(
                rollout_mean_progress,
                rollout_mean_abs_track_pos,
                rollout_mean_abs_angle,
                rollout_mean_steps,
            )
            first_mean = rollout_episode_metrics[0]["first_mean"] if rollout_episode_metrics else np.zeros(action_dim, dtype=np.float32)
            first_action = rollout_episode_metrics[0]["first_action"] if rollout_episode_metrics else np.zeros(action_dim, dtype=np.float32)
            first_mean_str = np.array2string(np.asarray(first_mean), precision=3, suppress_small=True)
            first_action_str = np.array2string(np.asarray(first_action), precision=3, suppress_small=True)

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

            print(
                f"Update {update_idx + 1}/{args.updates} | "
                f"rollout_steps={len(buffer)} | "
                f"episodes={len(rollout_episode_metrics)} | "
                f"mean_reward={rollout_mean_reward:.2f} | "
                f"mean_progress={rollout_mean_progress:.2f} | "
                f"mean_abs_angle={rollout_mean_abs_angle:.3f} | "
                f"mean_abs_trackPos={rollout_mean_abs_track_pos:.3f} | "
                f"raw_steer={rollout_mean_raw_steer:.3f} | "
                f"mean_steer={rollout_mean_steer:.3f} | "
                f"raw_accel={rollout_mean_raw_accel:.3f} | "
                f"raw_brake={rollout_mean_raw_brake:.3f} | "
                f"mean_accel={rollout_mean_accel:.3f} | "
                f"mean_brake={rollout_mean_brake:.3f} | "
                f"first_mean={first_mean_str} | "
                f"first_action={first_action_str} | "
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
                print(
                    f"EVAL update={update_idx + 1} | "
                    f"avg_steps={eval_metrics['avg_steps']:.1f} | "
                    f"avg_reward={eval_metrics['avg_reward']:.2f} | "
                    f"avg_progress={eval_metrics['avg_progress']:.2f} | "
                    f"avg_abs_angle={eval_metrics['avg_abs_angle']:.3f} | "
                    f"avg_abs_trackPos={eval_metrics['avg_abs_track_pos']:.3f} | "
                    f"avg_mean_steer={eval_metrics['avg_mean_steer']:.3f} | "
                    f"avg_mean_accel={eval_metrics['avg_mean_accel']:.3f} | "
                    f"avg_mean_brake={eval_metrics['avg_mean_brake']:.3f} | "
                    f"score={eval_metrics['score']:.3f}"
                )
                if is_best_eval:
                    print(f"New best_eval checkpoint | score={best_eval_score:.3f} | path={best_eval_path}")

                observation = env.reset(relaunch=False)
                state = flatten_low_dim_observation(observation, env.client.S.d)

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
        env.end()


if __name__ == "__main__":
    train()
