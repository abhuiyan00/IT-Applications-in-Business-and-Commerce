# Branch Snapshot — `stable`

**Tip commit:** `57941d6` — "improved lap time" (Przemyslaw Wdowczyk, 2026-06-04)
**Base:** [[master]] line at `8703ee8`; commits `d636136` "stable gym" → `09ef6e3` "current best"
→ `57941d6` "improved lap time".
**Goal:** a hardened, faster, "best-so-far" gym — separate actor/critic networks, a
physics-based slow-in/fast-out reward, and an off-track-continue eval mode.
**Diff vs master:** `gym_torcs.py` (heavy reward rewrite), `rl_agent.py` (net + hypers),
`train_torcs_rl.py` (raw-action logging + flag), `eval_torcs_rl.py` (flag) + checkpoint copies.

---

## 1. Summary
The team's consolidated "good" branch. Three big ideas: (1) **separate actor and critic
backbones** with higher exploration, (2) a **time-to-edge / danger** reward that brakes for
corners and chases a high target speed on straights, (3) an option to **apply the off-track
penalty without ending the episode** (for evaluation). Net result per commit log: improved lap
time over the baseline.

## 2. Code changes vs [[master]]

### `rl_agent.py` — network + hyperparameters
- `ActorCriticNet`: master's single shared `backbone` is **split into independent `actor` and
  `critic` MLPs** (each 256×2 Tanh). `policy_head`/`value_head` read their own trunk. This
  decouples policy and value learning.
- `init_log_std` **−3.0 → −1.2** (much more initial exploration).
- New: if `action_dim >= 2`, bias the **accel** policy output positive: `policy_head.bias[1]=0.5`
  → the car launches/accelerates instead of dithering at start.
- `load_compatible_state_dict` extended: **migrates old `backbone.*` checkpoints** into both
  `actor.*` and `critic.*` so master-trained weights warm-start the new split net.
- PPOAgent defaults: `lr 3e-5 → 1e-4`, `entropy_coef 1e-3 → 3e-3`, `init_log_std −3 → −1.2`,
  `target_kl 0.01 → 0.03`. `log_std` clamp widened `[-4,-0.5]` → `[-3.0, 0.0]` (allows larger
  action std).

### `gym_torcs.py` — reward rewrite
- **Removed** the `track_heading_hint` helper and master's "straightness" reward block entirely.
- New class consts: `default_speed 150 → 600`, `other_speed = 50` (speedY/Z now normalized by
  50 instead of default_speed), `off_track_penalty 800 → 10000`, `target_speed = 360`,
  `corner_speed = 120`, `brake_enable_speed 35 → 80`. New ctor arg `terminate_on_off_track=True`.
- **Reward built from a "danger / time-to-edge" model:**
  ```
  reward  = progress/10
          - drift_penalty            # 0.3 * (|speedY|-5)+ * speed_factor
          + 1.5 * danger * brake     # danger = clip((2.4 - time_to_edge)/2.4)
          if safe_zone (danger<0.2):
              - (target_speed - speed)+/target_speed       # punish being slow
              - 50 * (1 - accel)+                           # punish lifting off on straights
          else (cornering):
              - 2*(corner_speed - speed)+/corner_speed
              - 1.5 * overspeed^1.5  - 3*drift_penalty      # punish carrying too much speed in
          - 0.8*brake (if speed<target)                     # don't brake when slow
          + 0.05*brake*brake_need - 0.25*brake*(1-brake_need)
          + 0.25*accel*stable (or 0.05 above target)        # gas only when stable
          + 0.10*progress_delta
          - track_pos / angle penalties + improvement bonuses (as master)
          - steer smoothness/magnitude
          - 0.50*accel*brake                                # strong anti-overlap
  ```
  where `time_to_edge = lookahead_distance / (speed/3.6)`, lookahead = max beam in a sector
  around center.
- Off-track now respects `terminate_on_off_track`: applies −10000 always, but only ends the
  episode if the flag is set.
- `agent_to_torcs`: pedal-overlap handling replaced — if both accel & brake >0, **keep the
  larger and zero the other** (hard mutual exclusion). Gear index uses `3` (room for a 4-dim
  action incl. gear) though gear stays auto here.

### `train_torcs_rl.py`
- `--lr` default `3e-5 → 1e-4`, `--init-log-std −2.0 → −1.0`.
- New `--continue-after-off-track` flag wired to `terminate_on_off_track=not flag`.
- Logs **raw** (pre-clip) steer/accel/brake means alongside applied means.

### `eval_torcs_rl.py`
- New `--continue-after-off-track` flag → evaluate full laps even when the car briefly leaves
  the track (penalty applied, episode continues).

## 3. Artifacts
Checkpoints are **larger (~862 KB → but with extra copies)**; carries multiple timestamped
copy folders (`checkpoints copy 1_31_18/`, `..._34_56/`, etc.), each with the standard trio at
~1.8 MB (the split actor+critic net is bigger). Latest trio at `checkpoints/`.

## 4. Findings
- Most architecturally mature of the speed branches: **separate actor/critic**, **warm-start
  migration** from old checkpoints, and a **physically-motivated braking reward** (brake when
  time-to-edge is short, sprint to `target_speed` on straights).
- `default_speed=600` is a normalization choice (keeps high speeds in a sane obs range), not a
  literal speed cap.
- `--continue-after-off-track` is the practical evaluation trick that lets a still-imperfect
  policy be scored over a whole lap. This flag and the off-track-continue logic were inherited
  by [[headless]].
- This branch is the "stable gym" that [[headless]] explicitly merges in (`0d523c9`).
