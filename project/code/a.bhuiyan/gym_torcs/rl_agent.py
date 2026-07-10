import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def _as_array(value):
    array = np.asarray(value, dtype=np.float32)
    return np.atleast_1d(array)


def flatten_low_dim_observation(observation, raw_state):
    """Build a single flat state vector for low-dimensional steering-only PPO."""
    features = [
        _as_array(observation.speedX),
        _as_array(observation.track),
        _as_array(raw_state["angle"] / np.pi),
        _as_array(raw_state["trackPos"]),
        _as_array(raw_state.get("prevAppliedSteer", 0.0)),
    ]
    return np.concatenate(features, dtype=np.float32)


def flatten_racingline_observation(observation, raw_state, line):
    """State vector for racing-line PPO.

    The agent sees tracking errors against the precomputed line plus a short
    look-ahead preview, so it only has to learn vehicle dynamics, not the path.
    All terms are normalized to roughly [-1, 1].
    """
    speed_ms = float(raw_state["speedX"]) / 3.6
    angle = float(raw_state["angle"])
    track_pos = float(raw_state["trackPos"])
    s = float(raw_state.get("distFromStart", 0.0))
    q = line.query(s)

    cross_track_error = track_pos * line.half_width - q["offset"]
    heading_error = math.atan2(
        math.sin(angle - q["dpsi"]), math.cos(angle - q["dpsi"]))
    speed_delta = q["vmax"] - speed_ms

    v_ref = line.V_REF
    kappa_scale = 50.0
    # Reach far enough to brake for distant corners: scrubbing top speed to a
    # hairpin needs ~200 m, far past the old 70 m horizon.
    preview = line.preview(s, [25.0, 60.0, 120.0, 200.0])
    # Worst upcoming target speed within the braking horizon (negative = must
    # slow down soon); the single most important anti-overshoot signal. Reach
    # further so the policy can read a corner well before the baseline brakes.
    min_vmax_ahead = line.min_vmax_ahead(s, 250.0)
    brake_cue = (min_vmax_ahead - speed_ms) / v_ref

    features = [
        _as_array(speed_ms / v_ref),
        _as_array(cross_track_error / line.half_width),
        _as_array(heading_error / (np.pi / 2.0)),
        _as_array(speed_delta / v_ref),
        _as_array(q["kappa_rl"] * kappa_scale),
        _as_array([k * kappa_scale for k, _ in preview]),
        _as_array([(v - speed_ms) / v_ref for _, v in preview]),
        _as_array(brake_cue),
        _as_array(observation.track),  # 19 rangefinder beams (already /200)
        _as_array(raw_state.get("prevAppliedSteer", 0.0)),
        _as_array(raw_state.get("prevAppliedAccel", 0.0)),
        _as_array(raw_state.get("prevAppliedBrake", 0.0)),
    ]
    return np.concatenate(features, dtype=np.float32)


def build_state(observation, raw_state, racing_line=None):
    """Pick the racing-line state builder when a line is loaded, else baseline."""
    if racing_line is not None:
        return flatten_racingline_observation(observation, raw_state, racing_line)
    return flatten_low_dim_observation(observation, raw_state)


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, init_log_std=-2.5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

        # Neutral steering at start makes PPO much less likely to collapse immediately.
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, state_tensor):
        hidden = self.backbone(state_tensor)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.episode_ids = []

    def add(self, state, action, reward, done, value, log_prob, episode_id):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.episode_ids.append(int(episode_id))

    def __len__(self):
        return len(self.rewards)


def load_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    loaded_keys = []
    partial_keys = []
    skipped_keys = []

    for key, source_tensor in state_dict.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue

        target_tensor = model_state[key]
        if source_tensor.shape == target_tensor.shape:
            model_state[key] = source_tensor
            loaded_keys.append(key)
            continue

        if source_tensor.ndim == target_tensor.ndim and source_tensor.ndim in (1, 2):
            slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source_tensor.shape, target_tensor.shape))
            patched_tensor = target_tensor.clone()
            patched_tensor[slices] = source_tensor[slices]
            model_state[key] = patched_tensor
            partial_keys.append(f"{key}: {tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}")
            continue

        skipped_keys.append(f"{key}: {tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}")

    model.load_state_dict(model_state)
    return {
        "loaded": loaded_keys,
        "partial": partial_keys,
        "skipped": skipped_keys,
    }


def load_checkpoint_payload(model, optimizer, payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        load_report = load_compatible_state_dict(model, payload["model_state_dict"])
        if optimizer is not None and payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        metadata = {
            "update_idx": int(payload.get("update_idx", 0)),
            "env_steps": int(payload.get("env_steps", 0)),
            "episode_count": int(payload.get("episode_count", 0)),
            "best_train_score": payload.get("best_train_score"),
            "best_eval_score": payload.get("best_eval_score"),
        }
        return load_report, metadata

    load_report = load_compatible_state_dict(model, payload)
    return load_report, {}


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=8,
        minibatch_size=256,
        entropy_coef=1e-3,
        value_coef=0.5,
        init_log_std=-2.5,
        target_kl=0.02,
        max_grad_norm=0.5,
        device=None,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCriticNet(
            state_dim=state_dim,
            action_dim=action_dim,
            init_log_std=init_log_std,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _distribution(self, state_tensor):
        mean, value = self.model(state_tensor)
        log_std = torch.clamp(self.model.log_std, -4.0, -0.5)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist, mean, value

    def act(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, mean, value = self._distribution(state_tensor)

        raw_action = mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return {
            "action": action.squeeze(0).detach().cpu().numpy(),
            "log_prob": float(log_prob.squeeze(0).detach().cpu().item()),
            "value": float(value.squeeze(0).detach().cpu().item()),
            "entropy": float(entropy.squeeze(0).detach().cpu().item()),
            "mean": mean.squeeze(0).detach().cpu().numpy(),
        }

    def value(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, value = self._distribution(state_tensor)
        return float(value.squeeze(0).item())

    def evaluate_actions(self, states, actions):
        dist, mean, values = self._distribution(states)
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_actions = torch.atanh(clipped_actions)
        log_prob = dist.log_prob(raw_actions) - torch.log(1 - clipped_actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, values, mean

    def compute_returns_and_advantages(self, buffer, last_value):
        advantages = np.zeros(len(buffer), dtype=np.float32)
        returns = np.zeros(len(buffer), dtype=np.float32)

        gae = 0.0
        next_value = float(last_value)
        for index in reversed(range(len(buffer))):
            non_terminal = 1.0 - buffer.dones[index]
            delta = buffer.rewards[index] + self.gamma * next_value * non_terminal - buffer.values[index]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[index] = gae
            returns[index] = gae + buffer.values[index]
            next_value = buffer.values[index]

        return advantages, returns

    def update(self, buffer, last_value=0.0):
        if len(buffer) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "return_mean": 0.0,
                "advantage_mean": 0.0,
                "stopped_early": False,
                "epochs_ran": 0,
            }

        advantages, returns = self.compute_returns_and_advantages(buffer, last_value=last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.as_tensor(np.asarray(buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(buffer.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(buffer.log_probs), dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        batch_size = states.shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)

        policy_losses = []
        value_losses = []
        entropy_values = []
        approx_kls = []
        clip_fractions = []
        stopped_early = False
        epochs_ran = 0

        for epoch_idx in range(self.ppo_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            epoch_kls = []

            for start in range(0, batch_size, minibatch_size):
                batch_indices = permutation[start:start + minibatch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]

                new_log_probs, entropy, values, _ = self.evaluate_actions(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * batch_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon)
                        .float()
                        .mean()
                    )

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropy_values.append(float(entropy_bonus.detach().cpu().item()))
                approx_kls.append(float(approx_kl.detach().cpu().item()))
                clip_fractions.append(float(clip_fraction.detach().cpu().item()))
                epoch_kls.append(float(approx_kl.detach().cpu().item()))

            epochs_ran += 1
            if epoch_kls and np.mean(epoch_kls) > self.target_kl:
                stopped_early = True
                break

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "return_mean": float(np.mean(returns)),
            "advantage_mean": float(np.mean(advantages)),
            "stopped_early": stopped_early,
            "epochs_ran": epochs_ran,
        }
