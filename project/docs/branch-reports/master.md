# Branch Snapshot — `master`

**Repo:** Szymon1905/AI_racer_pytorch
**Tip commit:** `96e3f43` — "Merge branch 'master'" (Przemyslaw Wdowczyk, 2026-04-25)
**Role:** Canonical baseline. All other branches diverge from here. This report is the
reference; per-branch reports describe only their deltas vs this.

---

## 1. What the project is

An AI racing agent trained with **PPO (Proximal Policy Optimization)** in **PyTorch** to
drive a car around a track in **TORCS** (The Open Racing Car Simulator). The agent talks to
TORCS over the **SCR UDP** protocol (the `snakeoil3` client). Built "with help of IBM"
(per ReadMe).

Continuous control: the policy outputs **steer + accel + brake** (3-D action, `throttle=True`,
`gear_change=False` → gear auto-selected by speed). Vision is off; the agent uses
low-dimensional sensors (track rangefinder beams, speeds, rpm, angle, trackPos).

## 2. Top-level layout

```
master/
├── ReadMe.md             project README (install / train / eval)
├── commands.txt          quick command crib
├── requirements.txt      pinned deps (UTF-16 encoded)
├── .gitignore            (single entry)
├── AI-Slop-Racers-main/  TORCS_Python_Windows_Troubleshooting_Guide.docx
├── torcs/                TORCS game binaries + config (wtorcs.exe, tracks, drivers)
└── gym_torcs/            ← the actual project code
    ├── gym_torcs.py          TorcsEnv: gym-like env wrapper + reward (640 lines)
    ├── rl_agent.py           PPO agent, ActorCritic net, rollout buffer (288 lines)
    ├── train_torcs_rl.py     training loop / CLI (433 lines)
    ├── eval_torcs_rl.py      deterministic evaluation / CLI (215 lines)
    ├── snakeoil3_gym.py      SCR UDP client (570 lines)
    ├── sample_agent.py       trivial example agent
    ├── example_experiment.py example loop
    ├── practice.xml          race config
    ├── autostart.sh          Linux GUI automation
    ├── README.md             upstream gym-torcs README
    ├── checkpoints/          torcs_ppo_latest{,_best_eval,_best_train}.pt (~892 KB each)
    └── vtorcs-RL-color/      vendored TORCS source (C/C++, the bulk of 6843 files)
```

Total repo ≈ 6843 files; almost all are the vendored TORCS simulator. **Meaningful
project code lives in `gym_torcs/*.py`** — that is what differs across branches.

## 3. Dependencies (`requirements.txt`, UTF-16)
`filelock==3.28.0`, `fsspec==2026.3.0`, `Jinja2==3.1.6`, `MarkupSafe==3.0.3`,
`mpmath==1.3.0`, `networkx==3.4.2`, `numpy==2.2.6`, `sympy==1.14.0`, **`torch==2.11.0`**,
`typing_extensions==4.15.0`.

## 4. Core code

### 4.1 `gym_torcs.py` — `TorcsEnv`

Gym-like wrapper. Key class constants (the reward/termination tuning knobs):

| Constant | Value | Meaning |
|---|---|---|
| `terminal_judge_start` | 150 | step after which low-progress kills the episode |
| `termination_limit_progress` | 5 | km/h progress floor |
| `default_speed` | 150 | speed normalizer for obs |
| `progress_reward_scale` | 10.0 | divides forward-progress reward |
| `track_pos_penalty_gain` | 1.0 | lateral-position penalty |
| `track_pos_nonlinear_gain` | 0.6 | `abs_track_pos**1.5` penalty |
| `track_pos_edge_threshold` / `..._edge_penalty_gain` | 0.3 / 0.8 | edge penalty |
| `angle_penalty_gain` | 1.0 | heading penalty |
| `steer_smoothness_penalty_gain` | 0.01 | jerk penalty |
| `steer_magnitude_penalty` | 0.01 | |
| `damage_penalty` / `off_track_penalty` | 200 / 800 | terminal penalties |
| `brake_enable_speed` | 35.0 | brake disabled below this |
| `pedal_overlap_threshold` / `accel/brake_overlap_limit` | 0.2 / 0.12 | anti accel+brake |
| `off_track_track_pos_limit` | 1.05 | `|trackPos|` past this = off track |
| `reconnect_each_episode` | True | new UDP client every episode |

**Helper functions (module level):**
- `summarize_track_sensors(track)` → left/center/right means, curve hint, center bias.
- `track_heading_hint(track)` → estimated heading vs track from beam geometry, with a
  straight-confidence deadband so straights don't induce a constant steer bias.
- `curve_proximity_factor(track)` → 0..1 how close the center beam sees a wall.
- `brake_zone_score(track, speed)` → product of curve strength × distance × speed, used to
  reward braking into corners.

**`step(u)`** pipeline: map action → apply to TORCS (`client.R.d`), auto-gear by speed
(`_gear_for_speed`), step the sim, read new obs, compute reward, judge termination.

**Reward (master's "straightness" formulation):**
```
reward  = progress / 10                      # progress = speedX*cos(angle)
        + 0.08*straightness*(speed/100)      # straightness = symmetry*open_forward*heading_align
        + 0.06*straightness*accel
        + 0.15*max(progress_delta, 0)
        - track_pos penalties (linear + ^1.5 + edge)
        - angle penalty + angle/track_pos improvement bonuses
        - steer smoothness/magnitude penalties
        + 0.10*accel  - 0.08*accel*angle_excess - 0.08*accel*abs_track_pos
        + 0.25*brake*brake_zone + 0.10*brake_zone*brake*decel   # reward braking into corners
        - 0.12*brake*(1-brake_zone) - 0.30*accel*brake          # punish needless brake / overlap
        - damage_penalty (on damage increase)
```
**Termination:** off-track (`|trackPos|>1.05`, −800), low progress after step 150 (−5),
backward (`cos(angle)<0`, reward set −10).

**`agent_to_torcs(u)`**: steer = `clip(u[0])`; accel/brake = positive halves of `u[1]`,`u[2]`
scaled, with brake disabled below 35 km/h and pedal-overlap clamps. Gear auto by `_gear_for_speed`
(1→2→3→4→5→6 at 30/60/95/140/185 km/h).

**`make_observaton`** normalizes: speeds /default_speed(150), track & focus & opponents /200,
rpm raw, wheelSpinVel raw.

Robust reset/connection handling: `_wait_for_valid_reset_state`, `_create_client` (retries),
process kill via `taskkill` on Windows / `pkill` on Linux.

### 4.2 `rl_agent.py` — PPO

- **`flatten_low_dim_observation(obs, raw_state)`** → state vector:
  `[speedX, speedY, rpm/1e4, track[19], wheelSpinVel/100, angle/π, trackPos,
  prevAppliedSteer, prevAppliedAccel, prevAppliedBrake]`.
- **`ActorCriticNet`**: shared 2-layer Tanh MLP (`backbone`, hidden 256) → `policy_head` +
  `value_head`; learnable `log_std` (init −3.0). Policy head zero-init (neutral start →
  avoids early collapse).
- **`PPOAgent`**: tanh-squashed Gaussian, GAE(λ=0.95, γ=0.99), clipped surrogate
  (ε=0.2), value loss (coef 0.5), entropy bonus (1e-3), `target_kl=0.01` early stop,
  grad clip 0.5, Adam lr 3e-5. `log_std` clamped to [−4, −0.5].
- **`load_compatible_state_dict` / `load_checkpoint_payload`**: shape-tolerant checkpoint
  transfer (partial loads, skips optimizer when shapes change).

### 4.3 `train_torcs_rl.py`
PPO loop: collect `rollout-size` (1024) steps across episodes → `agent.update`. Tracks
per-rollout mean reward/progress/angle/trackPos/steer/accel/brake. Saves three checkpoints
(`latest`, `best_train`, `best_eval`); periodic deterministic eval (`--eval-interval` 10).
Score = `mean_progress − 3*|trackPos| − 2*|angle| + 0.01*steps`.

Key default args: `--updates 200 --rollout-size 1024 --max-steps 300 --lr 3e-5
--init-log-std -2.0 --target-kl 0.01 --eval-interval 10`.

### 4.4 `eval_torcs_rl.py`
Loads a checkpoint, runs deterministic episodes. By default **disables** low-progress
termination (sets `terminal_judge_start = max_steps+1`) so the policy can complete a lap.
Flags to re-enable / override termination thresholds.

### 4.5 `snakeoil3_gym.py`
Classic SCR UDP client (`Client`, `ServerState`, `DriverAction`). Sensor angles string,
parse server state, send actuator string. Default sensor list includes `stucktimer`,
`targetSpeed`. Synchronous `recvfrom` with timeout. (Branches `parallel`/`headless` rework
this heavily.)

## 5. Checkpoints present
`checkpoints/torcs_ppo_latest.pt`, `_best_eval.pt`, `_best_train.pt` (~892 KB each).
State dim drives net size; these match the master `backbone`-style net.

## 6. Findings / notes
- Master is the **shared, conservative baseline**: speed-capped (150), center-of-track
  hugging reward, gear auto by speed, no gear learning.
- The reward is dominated by "straightness" shaping + progress; braking only rewarded inside
  detected brake zones.
- Commit history shows the team enabled brake/throttle control (`7eb8a94`) and added brake
  zone detection (`8e0733a`) shortly before this merge.
- See sibling reports: speed pushes [[160speed]] / [[160speed+stablesteer]] / [[stable]],
  gear experiments [[gear_strategy_max_jerk]] / [[headless]], infra [[parallel]] /
  [[headless]], docs [[Reward_func_improvement]], one-liner fix [[piasecki_fix1]], and the
  full racing-line rewrite [[a.bhuiyan]].
