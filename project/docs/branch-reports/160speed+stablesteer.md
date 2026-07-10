# Branch Snapshot — `160speed+stablesteer`

**Tip commit:** `e73e9b1` — "does 140 speed, struggles on s turn, still too close to sides"
(Szymon Borzdyński, 2026-05-01)
**Base:** builds directly on [[160speed]] (`d28c6b7`).
**Goal:** keep the higher speed of [[160speed]] but make the steering more stable and the
throttle smarter in corners.
**Diff vs master:** `gym_torcs.py`, `snakeoil3_gym.py` + checkpoint folders.

---

## 1. Summary
Iteration on top of [[160speed]]. Adds steering-smoothness pressure and a traction-control-style
throttle/angle coupling so the car doesn't snap-steer or floor the gas mid-corner. Commit
message result: **does 140 km/h, but struggles on the S-turn and rides too close to the sides.**

## 2. Code changes vs [[master]] (superset of [[160speed]])

### `gym_torcs.py`
Everything from [[160speed]] (`default_speed=200`, edge gain 0.95, straightness 0.40/0.20,
wall-check brake reward, TODO gear) **plus**:

- `steer_smoothness_penalty_gain` **0.01 → 0.1** (old commented). 10× harder penalty on steering
  jerk → smoother, less nervous steering.
- Reworked accel/angle coupling ("TC" = traction control):
  ```python
  # old 0.10*accel and 0.08 angle/trackpos couplings commented out
  accel_alignment = max(0.0, 1.0 - 2.5*abs(angle))   # only reward gas when pointing straight
  reward += 0.15 * accel_value * accel_alignment
  angle_excess = max(0.0, abs(angle) - 0.05)          # tighter threshold than master's 0.10
  reward -= 0.30 * accel_value * angle_excess          # stronger punish for gas-in-corner
  ```
  Net: gas is rewarded only when nearly straight, and penalized much harder when the car is
  angled — discouraging power-on understeer/oversteer.

### `snakeoil3_gym.py`
Same sensor-list trim as [[160speed]] (`stucktimer`, `targetSpeed` commented out).

## 3. Artifacts
Checkpoint folders incl. an extra `nice 5/` set (`torcs_best_przemo.pt`, `torcs_ppo_latest.pt`,
`_best_train.pt`) beyond [[160speed]]'s, plus the carried-over `nice 4/`, `old/`, `old 2/`,
`old 3/`, `prez/`.

## 4. Findings
- This is the **stability patch** for the speed push: same speed target, but smoother steering
  (10× jerk penalty) and corner-aware throttle (TC term).
- **Outcome (commit msg):** holds 140 km/h; remaining problems are the **S-turn** and **riding
  too close to track edges** — exactly what the stronger edge penalty (0.95) and steering
  smoothness were meant to fix, so not fully solved.
- Like [[160speed]], no PPO-agent hyperparameter changes; all tuning is in the env reward.
- The "S-turn is hardest" finding recurs independently in [[a.bhuiyan]] (corkscrew S-curve).
