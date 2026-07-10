"""Offline smoke test: no TORCS connection, just dims + reward + constraints."""
import numpy as np
from gym_torcs import TorcsEnv
from rl_agent import build_state, PPOAgent

LINE = "lines/corkscrew.npz"

env = TorcsEnv(throttle=True, auto_start=False, racing_line_path=LINE)
print("action_space dim :", env.action_space.shape[0])
print("racing_line loaded:", env.racing_line is not None, "len(m)=", round(env.racing_line.length, 1))

# Synthetic raw TORCS observation dict (mid-track, slight offset, 20 m/s).
raw = {
    "focus": [-1] * 5,
    "speedX": 72.0, "speedY": 0.0, "speedZ": 0.0,   # km/h
    "opponents": [200.0] * 36,
    "rpm": 5000.0,
    "track": [10.0] * 19,
    "wheelSpinVel": [0.0] * 4,
    "angle": 0.05, "trackPos": -0.2, "distFromStart": 1500.0,
    "damage": 0.0, "gear": 3,
    "prevAppliedSteer": 0.1, "prevAppliedAccel": 0.6, "prevAppliedBrake": 0.0,
}
obs = env.make_observaton(raw)
state = build_state(obs, raw, env.racing_line)
print("state dim        :", state.shape[0])
print("state finite     :", bool(np.isfinite(state).all()))

reward, metrics = env._racingline_reward(raw)
print("reward           :", round(reward, 4))
print("metrics          :", {k: round(v, 3) for k, v in metrics.items()})

# Steer constraints: large request at high speed should be clipped + rate limited.
env.last_applied_steer = 0.0
s1 = env._apply_steer_constraints(1.0, 72.0)
s2 = env._apply_steer_constraints(1.0, 72.0)
print("steer step1/step2:", round(s1, 3), round(s2, 3), "(rate-limited, speed-scaled)")

# PPO agent accepts this state/action dim.
agent = PPOAgent(state_dim=state.shape[0], action_dim=env.action_space.shape[0], device="cpu")
info = agent.act(state)
print("agent action     :", np.round(info["action"], 3))
print("SMOKE_OK")
