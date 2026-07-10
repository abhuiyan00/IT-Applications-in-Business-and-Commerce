# Branch Snapshot — `gear_strategy_max_jerk`

**Tip commit:** `1a2b326` — "WIP - gear change" (Gabriel Bekier, 2026-05-02)
**Base:** [[master]] tip `96e3f43` (includes brake-zone detection).
**Goal:** let the **agent control the gear** and shape reward around **jerk** (rate of change
of acceleration) to encourage well-timed shifts.
**Diff vs master:** `gym_torcs.py` only (+55 / −6). Work-in-progress.

---

## 1. Summary
First experiment to move gear out of the speed-based auto table and into the **action space**
(`gear_change=True`), plus a **jerk-based reward** that pays the car for increasing acceleration
(a successful upshift) and punishes flat/zero-jerk acceleration regions (a sign it's time to
shift). Debug printing is turned on. Marked "WIP".

## 2. Code changes vs [[master]] (`gym_torcs.py`)

### Constructor defaults
- `gear_change=False → True` (agent now picks gear).
- `debug=False → True` (verbose per-step logging on by default).
- New state fields: `self.prev_speedX = 0.0`, `self.prev_accel = 0.0`, `self.jerk_threshold = 1`.
- Same three reset to 0 in `reset()`.

### Gear handling in `step`
```python
if self.gear_change is True:
    requested_gear = int(np.clip(round(float(this_action['gear'])), 1, 7))
    if requested_gear < 1: requested_gear = 1
    action_torcs['gear'] = requested_gear
```
(Previously just passed `this_action['gear']`. Now clamped to 1..7.)

### New reward term — jerk bonus
```python
def calculate_jerk_reward(self, current_speed, prev_speed, accel_applied):
    current_accel = current_speed - prev_speed
    jerk = current_accel - self.prev_accel
    reward = 0
    if accel_applied > 0.2 and current_speed > 5.0:
        if abs(jerk) < self.jerk_threshold:
            reward = -0.5            # punish flat accel (no shift happening)
        elif jerk > 0:
            reward = jerk * 15.0     # reward rising acceleration (good shift)
    self.prev_accel = current_accel
    return reward
```
Added into the step reward as `reward += jerk_bonus` right after the straightness terms.

### Action mapping
`agent_to_torcs` gear discretization widened from 6 to 7 gears:
```python
gear = int(np.clip(np.round(((gear_signal + 1.0) / 2.0) * 6.0) + 1, 1, 7))  # was *5.0 ... 1,6
```

### Debug
A verbose `--- TORCS STEP DEBUG ---` block printing step, action (steer/accel/gear), physics
(speed/accel/jerk), reward, and a "ZERO JERK REGION DETECTED (Time to shift?)" warning, gated on
`self.gear_change`.

## 3. Artifacts
Standard three checkpoints (~892 KB).

## 4. Findings
- **Concept:** "max jerk" = reward the *positive* acceleration spikes that a correct upshift
  produces; penalize being stuck at constant acceleration with the throttle down.
- The jerk signal here is derived from raw `speedX` differences (`current_accel = Δspeed`), so
  it's noisy — a known limitation of doing this off the UDP speed sensor.
- Work-in-progress and single-commit; `jerk_threshold = 1` is a coarse first guess.
- The **gear/rpm idea is carried much further** in [[headless]] (rpm-band reward) and in
  [[a.bhuiyan]] (full data-driven engine torque/shift map via `car_spec.py`).
