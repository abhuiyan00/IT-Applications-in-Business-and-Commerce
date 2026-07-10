# Branch Snapshot — `160speed`

**Tip commit:** `d28c6b7` — "140 speed reached but not complete lap, 3/4" (Szymon Borzdyński, 2026-04-30)
**Base:** forks [[master]] (shares history up to `8e80a47` "Added eval command").
**Goal:** push top speed higher (toward 160 km/h) while keeping the car on track.
**Diff vs master:** 2 code files (`gym_torcs.py`, `snakeoil3_gym.py`) + many checkpoint folders.

---

## 1. Summary
A speed-focused experiment. Raises the observation speed normalizer and reward weights so the
policy is paid more for going fast, and adds a crude "wall ahead → brake" reward. The commit
message records the honest result: **reached ~140 km/h but only completed ¾ of a lap** — i.e.
faster but not yet able to finish.

## 2. Code changes vs [[master]]

### `gym_torcs.py`
- `default_speed` **150 → 200.0** (changes obs speed normalization; policy sees more headroom
  above 150 km/h).
- `track_pos_edge_penalty_gain` **0.8 → 0.95** (old value left commented) — punish edge-riding
  a bit harder to offset the higher speed.
- Reward "straightness" terms **bumped up**:
  ```
  reward += 0.40 * straightness * (speed/100)   # was 0.08
  reward += 0.20 * straightness * accel          # was 0.06
  ```
  (a commented "# original" block notes prior 0.30/0.06 values — the team had been iterating.)
- New **wall-check braking shaping**:
  ```python
  if center_distance < 70.0 and speed > 70.0:
      reward += 0.40 * brake_value   # reward braking when a wall is close & fast
      reward -= 0.40 * accel_value   # punish staying on the gas
  ```
- Added a `# TODO try adjusting the gear` marker above `_gear_for_speed` (gear still auto by speed).

### `snakeoil3_gym.py`
- Sensor list trimmed: `'stucktimer'` and `'targetSpeed'` commented out (no longer requested
  from the server). Cosmetic/perf; doesn't change the RL state.

## 3. Artifacts
Many checkpoint subfolders carried along (experiment snapshots), e.g.
`checkpoints/torcs_ppo_latest.pt`, `_best_train.pt`, `torcs_best_przemo.pt`, and folders
`nice 4/`, `old/`, `old 2/`, `old 3/`, `prez/`. These are saved weights from different runs,
~862–892 KB each.

## 4. Findings
- **Net effect:** stronger incentive for speed (higher straightness weights + higher
  `default_speed`), partial braking awareness near walls.
- **Outcome (from commit msg):** 140 km/h reached, lap not completed (3/4). Speed up, stability
  not yet solved — motivates the sibling [[160speed+stablesteer]] which adds steering-stability
  shaping on top of this.
- No agent/training hyperparameter changes; purely env-reward + obs-scale tuning.
