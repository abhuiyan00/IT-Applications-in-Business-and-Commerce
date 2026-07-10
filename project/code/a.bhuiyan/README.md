# AI Racer (TORCS + PyTorch PPO)

RL driver for **TORCS** over SCR UDP. A precomputed **racing line** handles the
*path*; a deterministic **baseline** handles known-good driving; **PPO** learns
only the residual vehicle dynamics on top. PyTorch, with help from IBM.

Default track: **corkscrew** (road, big elevation ±12%), port **3001**.

---

## 1. Install
```bash
python -m venv RACERvenv
# Windows: RACERvenv\Scripts\activate   |   Mac/Linux: source RACERvenv/bin/activate
python -m pip install --upgrade pip && pip install -r requirements.txt
```

## 2. Launch TORCS (manual on Windows)
```bash
torcs/wtorcs.exe
```
**Race → Practice → New Race**, `scr_server` driver, port 3001. Leave on blue
"waiting" screen, then run train/eval from `gym_torcs/`.

## 3. Train / Evaluate
```bash
# train fresh (no --load-path). BIG --max-steps = full laps, far fewer GUI restarts.
python train_torcs_rl.py --port 3001 --max-steps 6000 --eval-max-steps 4000 --relaunch-every 40

# evaluate best checkpoint
python eval_torcs_rl.py --port 3001 --model-path checkpoints/corkscrew_ppo_latest_best_eval.pt
```
`--racing-line none` = steering-only baseline. `--no-assist` = pure PPO.
New track: `python racing_line.py --track-xml <path> --out lines/<name>.npz`.

> ⚠️ Small `--max-steps` (default 300) resets the episode every 300 steps →
> reconnect each episode → TORCS keeps restarting AND the car never finishes a
> fast lap. Use a big value.

---

## How it works (3 layers, PPO learns only the last)

1. **Offline racing line** (`racing_line.py`) — XML → centerline → min-curvature
   line `offset(s)` + speed profile `vmax(s)` + **elevation `grade(s)`**. The speed
   profile is **grade-aware**: gravity adds braking uphill / subtracts downhill, so
   downhill corners get a lower, earlier-braked entry automatically. Saved to
   `lines/corkscrew.npz` (re-run once after editing this file).

2. **Launch warmup** (`_launch_warmup`) — each episode spools to ~90 km/h before
   the agent takes over → learns from driving at speed, not a standstill.

3. **Deterministic baseline** (`_baseline_action`):
   - **Steering** — heading + cross-track + curvature feed-forward toward the line;
     deadband on straights (no micro-jerk); mid-lane settle after corners.
   - **Slow-in / fast-out** — `safe_speed` per-distance braking envelope brakes for
     each corner at its TRUE distance (look-ahead grows with speed); throttle early
     when the corner opens.
   - **Hill logic** — from the **offline grade** (noise-free), not the jittery
     speedZ sensor: brakes earlier for a descent ahead, adds throttle uphill.
   - **rpm gearbox** — six gears, shifts on engine **rpm** (up near the limiter,
     down below the torque band) with hysteresis + a hold timer → no hunting. No
     per-car speed table.

4. **Self-learning corner memory** — per-node speed-scale. Crash → slow that corner
   (and its curvature look-alikes) DOWN; clean fast pass → speed it UP. Similar
   corners learn together → faster convergence. Temp in RAM, saved only on a best
   checkpoint. Delete `lines/*_corner_memory.npy` to reset.

5. **Residual PPO** (`rl_agent.py`, `train_torcs_rl.py`) — policy outputs a bounded
   correction (`±0.5`, `--residual-scale`) on top of the baseline. Reward rewards
   speed-along-line, punishes off-line / off-speed / off-track. Calm exploration.

**Diagnostics** — every `off_track` logs its `crash_s` (distFromStart) to the CSV,
so repeat killer corners are obvious. `--debug-env` prints per-step speed, rpm,
grade, gear, reward.

## Known struggles during development
The project did not fully solve the **S-curve** on corkscrew, and the car did not
finish a clean full lap in the earlier runs. The main issues we kept chasing were:

- The car repeatedly lost time or went off-track in two hotspot regions around
   `s≈225-237 m` and `s≈386-397 m`.
- A gearbox bug could leave the car stuck in a tall gear with too little torque,
   which made it slow out of corners and behave as if it was "stuck at 6".
- Another tuning bug pinned the engine near the limiter instead of upshifting on
   time, which kept the car revving too high without translating that into better
   speed.
- Small `--max-steps` settings caused TORCS to reset too often, which made the
   car restart from the launch state before it had a chance to complete a fast
   lap.
- The corner-memory and hill-speed logic were added to reduce those failures, but
   the S-curve remained the hardest section to stabilize.

---

## Command-line arguments
| Arg | Meaning |
|-----|---------|
| `--updates` | Learning cycles. More = better, slower. |
| `--rollout-size` | Frames per update. Larger = more stable. |
| `--max-steps` | Steps per episode before reset. **Big = fewer restarts, full laps.** |
| `--residual-scale` | Agent authority over baseline (0 = pure baseline, 1 = full agent). |
| `--no-assist` | Disable baseline (pure PPO). |
| `--load-path` / `--save-path` | Resume / save checkpoint. |
| `--debug-env [--debug-interval N]` | Per-step telemetry (speed, rpm, grade, gear). |

Checkpoints: `checkpoints/corkscrew_ppo_latest{,_best_train,_best_eval}.pt`.
Episode metrics: `gym_torcs/logs/train_episodes_*.csv` (incl. `crash_s`).
