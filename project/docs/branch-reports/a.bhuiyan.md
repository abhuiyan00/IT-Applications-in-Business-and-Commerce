# Branch Snapshot — `a.bhuiyan`

**Tip commit:** `6a0598b` — "docs: document corkscrew development struggles" (abhuiyan00, 2026-06-17)
**Base:** [[master]] line at `25201e8`. Commits: `0b40e7b` docs → `b6c6411` "expand Torcs
training and runtime pipeline" → `cb50351` checkpoints+telemetry → `6a0598b` docs.
**Goal:** a fundamentally different architecture — a **precomputed racing line + deterministic
baseline driver**, with PPO learning only a **bounded residual correction** on top.
**Diff vs master:** the largest by far — `gym_torcs.py` almost fully rewritten (≈938 changed
lines), new modules `racing_line.py`, `car_spec.py`, `analyze_telemetry.py`, `_smoke.py`,
rewrites of `rl_agent.py`/`train_torcs_rl.py`/`eval_torcs_rl.py`, new docs (`README.md`,
`guide.txt`), generated artifacts (`lines/corkscrew.npz`, corner memory), 12 telemetry/episode
CSV logs, new corkscrew checkpoints, and `practice.xml` set to corkscrew.

---

## 1. Summary — the "3 layers, PPO learns only the last" design
From the README:
1. **Offline racing line** (`racing_line.py`): parse track XML → integrate centerline →
   minimum-curvature relaxation → grade-aware speed profile `vmax(s)`. Saved as
   `lines/corkscrew.npz`.
2. **Launch warmup** (`_launch_warmup`): spool to ~90 km/h before the agent takes over, so it
   learns from speed, not standstill.
3. **Deterministic baseline** (`_baseline_action`): heading + cross-track + curvature
   feed-forward steering toward the line; slow-in/fast-out braking envelope (`safe_speed`);
   grade-aware hill logic; rpm-based 6-speed gearbox with hysteresis.
4. **Self-learning corner memory**: per-node speed scale; crash → slow that corner (and
   curvature look-alikes) down, clean pass → speed up. Saved on best checkpoint.
5. **Residual PPO** (`rl_agent.py`/`train_torcs_rl.py`): policy outputs a bounded ±0.5
   correction (`--residual-scale`) on top of the baseline action.

Default track **corkscrew** (road, ±12% elevation), port 3001.

## 2. New module: `racing_line.py` (offline generator + runtime loader, pure numpy/stdlib)

**Why XML, not `tTrackSeg`:** `tTrackSeg` is a TORCS C++ runtime struct, invisible over SCR
UDP. The identical geometry is parsed offline from `torcs/tracks/road/<track>/<track>.xml`.

Pipeline:
- **`_load_track_xml_root`** — strips DOCTYPE / undefined entities (`&default-surfaces;`) so the
  stdlib parser reads the geometry.
- **`parse_track_xml`** → `(width, segments)`. Segments are straights (`lg`, grade) or arcs
  (`arc`, `r0`, `r1` for spirals/clothoids, grade). Reads longitudinal **grade %**.
- **`integrate_centerline(segments, ds=2)`** → uniformly arc-length-sampled
  `s, x, y, psi, kappa, grade, z(elevation), length`. Distributes the small 2D loop-closure gap
  (from ignoring grade in the horizontal projection) linearly along `s`.
- **`relax_racing_line(center, width, margin=1.5, iters=800, relax=0.2)`** — periodic Laplacian
  smoothing of the line within a corridor `±(width/2 − margin)`, projected back each iter.
  Heading derived analytically from the offset gradient (robust to the seam gap); curvature via
  wrapped finite differences, then smoothed (avoids single-node spikes).
- **`build_speed_profile(kappa_rl, ds, mu=1.1, a_accel=5, a_brake=8, v_top=55, grip=0.7, grade)`**
  — cornering cap `sqrt(a_lat/|kappa|)` then **grade-aware** forward/backward passes (gravity
  `g*slope` adds to braking uphill, subtracts from accel; reverse downhill), wrapped for the
  closed loop. → `vmax(s)` in m/s.
- **`generate(...)`** saves the `.npz` (`s,x,y,psi_center,psi_rl,offset,kappa_*,grade,z,vmax,
  length,width,ds,s_offset`) and prints loop-closure %, offset range, grade range, speed range;
  optional matplotlib plot of centerline vs racing line.

**Runtime `RacingLine` loader** — indexes by `distFromStart` (= s, mod length):
- `query(s)` → offset, psi_center, psi_rl, dpsi, kappa_rl, vmax.
- `preview(s, distances)` → (kappa, vmax) at look-aheads.
- `min_vmax_ahead`, **`safe_speed(s, a_brake, horizon)`** (per-distance braking envelope
  `min_j sqrt(vmax_j^2 + 2*a_brake*d_j)` — brakes for a near hairpin now, ignores a distant one),
  `grade_at`, `min_grade_ahead`, `max_abs_kappa_ahead`. `V_REF = 80 m/s` normalization.

## 3. New module: `car_spec.py` (data-driven engine/driveline model)

Parses the **configured car's** XML (resolved practice.xml → scr_server.xml → car XML) so gear
selection follows the engine's real power map — no hardcoded driveline.
- `resolve_configured_car(torcs_root)` — walks the config chain to find the car name + XML.
- **`CarSpec.from_xml`** parses: forward **gear ratios** + efficiencies, **final drive**, rear
  **tyre radius** (rim + sidewall), **engine torque curve** (rpm→Nm data points), **rev limiter**,
  tickover. Fails loud if a usable model can't be parsed (caller falls back to rpm/speed gearbox).
- Kinematics: `rpm_at(speed,gear)`, `speed_at`, `engine_torque(rpm)` (interp curve),
  `wheel_force(speed,gear)` (0 past limiter).
- **`optimal_gear(speed)`** = lowest gear keeping rpm under the shift ceiling
  (`shift_frac=0.93 × limiter`) → max wheel force while never bouncing the limiter (fixes the
  "stuck in gear 1 / stuck at 6" bug). `upshift_speeds_kmh()` for inspection.

## 4. New module: `analyze_telemetry.py` (offline race-engineer analysis)

Reads per-step telemetry CSVs (written by the env) with pure `csv`+`numpy`:
- **Grip calibration** from measured lateral-g (`v^2*|kappa|`) on-line in corners → suggests a
  `racing_line --grip` value.
- **Off-track hotspots**: clusters `crash_s` locations (`cluster_1d`, 30 m gap) → the corners
  that actually cost laps, with vmax there.
- **Speed audit** by 50 m bin (under/over target, slip) and **curvature clusters**
  (straight/gentle/medium/tight/hairpin).
- **`write_corner_memory`**: builds a data-derived per-node speed-scale from crash clusters,
  saved where the env auto-loads it (`lines/<track>_corner_memory.npy`).

## 5. `gym_torcs.py` rewrite (method map)

New ctor args incl. `racing_line_path`, `launch_warmup_enabled/target`, `baseline_assist`,
`residual_scale`, `telemetry`. Loads `RacingLine` and (best-effort) `CarSpec`. New methods:
- `_launch_warmup(client)` — spool-up before agent control.
- `_baseline_action(raw_obs)` — the deterministic driver (steering FF + `safe_speed` braking +
  hill logic + gearbox), ≈120 lines.
- `_gear_select(speed, rpm)` — rpm/CarSpec-based gearbox with hysteresis + hold timer.
- Corner memory: `_corner_memory_file`, `_load_corner_memory`, `_build_kappa_clusters`,
  `_nudge_corner_scale`, `_register_clean_pass`, `_register_offtrack`, `commit_corner_memory`.
- `_apply_steer_constraints(desired_steer, speed_kmh)` — speed-scaled steer clip + rate limit.
- `_racingline_reward(obs, ...)` — speed-along-line reward, off-line / off-speed / off-track
  penalties.
- `agent_to_torcs(u)` — combines baseline + bounded **residual** policy output.
- `_telemetry_init` / `_telemetry_write` — per-step CSV logging (speed, rpm, grade, gear,
  `e_y`, `kappa_rl`, `lat_accel`, `crash_s`, `off_track`, termination, …).

## 6. PPO / training changes
- `rl_agent.py`: adds `flatten_racingline_observation` (≈33-dim tracking-error state) +
  `build_state` selector; PPO core as before (separate config).
- `train_torcs_rl.py` new args: `--racing-line` (default corkscrew.npz; `none` = steer-only
  baseline), `--throttle`, `--launch-warmup-target 90`, `--no-launch-warmup`, `--no-assist`
  (pure PPO), `--residual-scale`, `--no-telemetry`. Defaults: `--lr 1e-4`, `--target-kl 0.02`,
  `--init-log-std −2.5`, `--save-path checkpoints/corkscrew_ppo_latest.pt`.
- `eval_torcs_rl.py`: racing-line auto-enabled, `build_state` wired in.

## 7. Artifacts
- `lines/corkscrew.npz` (generated line, ~length 3650 m, 12 m wide) + `corkscrew_corner_memory.npy`.
- Checkpoints `corkscrew_ppo_latest{,_best_eval,_best_train}.pt` (~906–907 KB).
- 12 log files under `gym_torcs/logs/` — telemetry CSVs (30k+/21k/7k rows) and
  `train_episodes_*.csv` (incl. `crash_s`).
- `guide.txt` (359 lines): full design rationale + math for the 5-phase pipeline and a
  calibration/gotchas checklist (units km/h vs m/s, trackPos sign calibration, distFromStart
  origin/wrap, loop-closure gate, gear-1 hardcode warning).

## 8. Findings / documented struggles (from README "Known struggles")
- **Did not fully solve the corkscrew S-curve**; no clean full lap in earlier runs.
- Repeated off-tracks in two hotspots: **`s≈225–237 m`** and **`s≈386–397 m`**.
- A **gearbox bug** could leave the car stuck in a tall gear with too little torque ("stuck at
  6"); another pinned the engine near the limiter instead of upshifting in time — the
  `car_spec.py` optimal-gear logic + `shift_frac=0.93` ceiling are the fix.
- Small `--max-steps` caused TORCS to reset too often (car restarts from launch before finishing
  a fast lap) → README strongly recommends a **big `--max-steps`** (e.g. 6000).
- Corner-memory + grade-aware hill speed were added specifically to reduce those failures; the
  S-curve remained the hardest section.

## 9. How it relates to the other branches
- By far the most ambitious: only branch with an **offline racing line**, a **data-driven car
  model**, **telemetry logging + offline analysis**, and **residual RL**. Where [[stable]] /
  [[headless]] reshape the *reward* and [[parallel]] scales *throughput*, this branch changes the
  *control architecture* (baseline + residual) and the *track* (corkscrew).
- Shares the corkscrew/open-wheel direction with [[headless]]/[[parallel]] but reaches the
  gearbox problem from the engine-physics side rather than hand-tuned rpm bands.
