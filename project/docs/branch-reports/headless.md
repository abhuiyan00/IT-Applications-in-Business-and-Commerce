# Branch Snapshot — `headless`

**Tip commit:** `ab4b328` — "adjust the gear reward" (Przemysław Piasecki, 2026-06-14)
**Base:** shared `d9310d0` "initial work for headless training" with [[parallel]]; later
**merges [[stable]]** (`0d523c9` "integrate changes from stable gym") and parallel changes.
Recent line: `74eed5d` jerk calc → `7999b57` optimum rpm reward → `ab4b328` gear reward.
**Goal:** train without the GUI (headless TORCS launched from a templated config) **and** add
gear/rpm-aware driving on top of the stable-gym reward.
**Diff vs master:** `gym_torcs.py` (big), `rl_agent.py`, `train_torcs_rl.py`, `eval_torcs_rl.py`,
`snakeoil3_gym.py`, new `quickrace_template.xml`, new `raceconfig.py`, new `quickrace2.xml`,
`scr_server.xml` (open-wheel).

---

## 1. Summary
The most feature-complete of the team's (non-a.bhuiyan) branches: it combines the **stable-gym
reward + split actor/critic net** (from [[stable]]) with **headless self-launching TORCS** (from
the [[parallel]] lineage, but single-instance/synchronous), and adds **gear control + an rpm-band
reward** so the agent learns to keep the engine in its powerband. The latest commits tune that
gear/rpm reward.

## 2. Code changes vs [[master]]

### `rl_agent.py`
Identical to [[stable]]: split `actor`/`critic` MLPs, `init_log_std −1.2`, accel-bias
`policy_head.bias[1]=0.5`, `backbone.*`→`actor/critic` checkpoint migration, `lr 1e-4`,
`entropy_coef 3e-3`, `target_kl 0.03`, `log_std` clamp `[-3,0]`.

### `gym_torcs.py`
- Adopts the **stable-gym danger/time-to-edge reward** (drift penalty, `danger` brake reward,
  safe-zone vs corner speed shaping, anti-overlap `−0.5*accel*brake`, etc.).
- Consts: `default_speed 150 → 400`, `other_speed 50`, `off_track_penalty 800 → 2000`,
  `target_speed 200`, `corner_speed 80`, `brake_enable_speed 80`. New ctor args `race_config`,
  `terminate_on_off_track`.
- New helper **`next_corner_hint(track)`** → (direction, strength) of the upcoming corner from
  weighted beam geometry; used to bias a desired track position toward the corner's inside
  (`prev_desired_track_pos` low-pass).
- **rpm-band gear reward** (the headline feature, tuned by the tip commit):
  ```python
  gear = int(obs["gear"]); rpm = obs["rpm"]
  if rpm < 17000:        rpm_reward = (rpm - 10000) / 7000
  elif rpm < 19000 or gear == 7:  rpm_reward = 1.0       # 17k–19k = optimum band
  else:                  rpm_reward = (27000 - rpm) / 8000
  if gear == 1 and rpm_reward < 0: rpm_reward = 0.0       # don't punish launch gear
  reward += (gear * 0.2) * rpm_reward                     # higher gears at good rpm pay more
  ```
- A `calculate_jerk_reward` method exists (from the gear lineage) but is **commented out** of
  the step reward in favor of the rpm-band reward.
- `reset` refactored into `stop()` (tear down + null the client) + `reset()`; client now fully
  recreated each episode (no `reconnect_each_episode` short-circuit).
- `agent_to_torcs`: hard pedal mutual-exclusion (keep larger of accel/brake); gear index `3`.
- speedY/Z normalized by `other_speed` (50).

### `train_torcs_rl.py`
- `from raceconfig import get_templated_config` (single config, `${id}=1`).
- `build_env(args, config)` with `gear_change=True`, `race_config`, `terminate_on_off_track`.
- `train()` builds a **single headless env on port 3002** from the templated config
  (graphical env commented out); `--lr 1e-4`, `--init-log-std −1.0`, `--continue-after-off-track`
  flag; logs raw + applied steer/accel/brake.
- Calls `headless_env.stop()` after each update (clean per-rollout teardown).

### `eval_torcs_rl.py`
- `gear_change=True`, `--continue-after-off-track`, `terminate_on_off_track` wired in.

### `snakeoil3_gym.py`
- `race_config` support + `setup_torcs` (synchronous `subprocess.Popen`, `wine wtorcs.exe -r
  <config>` on Linux); `shutdown` terminates the owned process; `__del__` → shutdown.
  (Synchronous — unlike [[parallel]]'s async client.)

### `raceconfig.py` (new)
`get_templated_config(id=1, path)` → single templated NamedTemporaryFile quick-race XML.

### Config / new files
- `gym_torcs/quickrace_template.xml` (corkscrew; `laps=100` here vs parallel's 2).
- `torcs/config/raceman/quickrace2.xml` (2-driver corkscrew quick race).
- `scr_server.xml` cars `car1-trb1 → car1-ow1` (open-wheel), all 9 driver slots.

## 3. Artifacts
Standard three checkpoints (~892 KB).

## 4. Findings
- **Integration branch:** stable-gym reward + headless launch + gear/rpm powerband reward. The
  three latest commits (`74eed5d`/`7999b57`/`ab4b328`) are iterations specifically on the gear
  reward — the team was converging on rpm-band shaping (optimum 17k–19k rpm) rather than the
  earlier raw-jerk idea from [[gear_strategy_max_jerk]].
- Single-instance + synchronous client (contrast [[parallel]], which is multi-instance async but
  keeps master's reward). headless = "best reward, runs without a screen, learns gears".
- The rpm thresholds (17k–27k) are large — consistent with the open-wheel `car1-ow1` car's
  high-revving engine set in `scr_server.xml`.
- Compared to the data-driven engine model in [[a.bhuiyan]] (`car_spec.py` parses the real
  torque curve), headless's rpm bands are hand-tuned constants.
