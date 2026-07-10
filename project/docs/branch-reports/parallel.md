# Branch Snapshot — `parallel`

**Tip commit:** `c90cdd6` — "recover from a timeout" (Przemysław Piasecki, 2026-05-17)
**Base:** [[master]] `96e3f43`; line `d9310d0` "initial work for headless training" →
`34613e8` "parallel execution support" → ... → `c90cdd6`.
**Goal:** train against **many TORCS instances at once** by making the env/client fully
**asyncio-based**, for a large rollout-throughput speedup.
**Diff vs master:** `gym_torcs.py`, `snakeoil3_gym.py`, `train_torcs_rl.py` (async rewrite),
new `quickrace_template.xml`, new `raceconfig.py`, new `torcs/config/raceman/quickrace2.xml`,
and `scr_server.xml` (cars → open-wheel).

---

## 1. Summary
An infrastructure branch (not a reward experiment). It converts the blocking SCR client and the
env step/reset into **`async` coroutines** and rewrites the training loop to drive **9 parallel
headless TORCS instances** concurrently via an `asyncio.Queue`, each launched from a templated
race config on its own UDP port. Includes timeout recovery so a stuck instance is reset rather
than killing the run.

## 2. Code changes vs [[master]]

### `snakeoil3_gym.py` — async client
- New imports: `platform`, `from asyncio import get_running_loop, sleep, wait_for`.
- `Client.__init__` gains `race_config`; if set, **launches its own TORCS** via `setup_torcs`
  (`wine wtorcs.exe -r <config>` on Linux, `cwd=../torcs/`, stdout to DEVNULL).
- New async factory `Client.create(...)` → builds the client then `await setup_connection()`.
- `setup_connection` and `get_servers_input` become **`async`**, using non-blocking
  `loop.sock_sendto` / `loop.sock_recvfrom` wrapped in `wait_for(..., 1)`; socket set
  non-blocking (`settimeout(0.0)`); `TimeoutError` replaces `socket.error` handling. Auto-relaunch
  on failure is **disabled** (commented) since each client owns its TORCS process.
- `shutdown` now kills the owned TORCS process; `__del__` calls `shutdown`.

### `gym_torcs.py` — async env
- `server_wait_loops 5 → 10`; new ctor arg `race_config`.
- `step`, `reset`, `_wait_for_valid_reset_state`, `_create_client` all become **`async`**;
  `time.sleep` → `await asyncio.sleep`; client created via `await snakeoil3.Client.create(...)`.
- `reset` split into `async stop()` (tear down client) + `async reset()`.
- Reward logic itself **unchanged from master** (this branch is about concurrency, not shaping).
  A commented `print("race time:", obs['curLapTime'])` left as a speedup probe.
- `import` quirk: `from asyncio import sleep` at top.

### `train_torcs_rl.py` — parallel loop
- `train()` becomes `async`, launched with `asyncio.run(train())`.
- `build_envs(args, start, end)` builds N envs from `get_templated_configs` on ports
  `args.port + i`; `train()` builds **envs 1..9** (`build_envs(args, 1, 10)`).
- Per-instance state held in a `RunData` class; a shared `asyncio.Queue` collects step results.
  `do_step` / `do_reset_step` coroutines push `(action_info, *result, i, was_reset)`; the main
  loop pulls from the queue until the rollout buffer is full, schedules the next step per
  instance, and **on TimeoutError resets that instance and continues** ("recover from a timeout").
- Eval cancels all in-flight step tasks, stops all envs, then evaluates on `envs[-1]`.

### Config / new files
- **`raceconfig.py`** — `get_templated_configs(start, end, path)`: fills `${id}` in the template
  via `string.Template`, writes N `NamedTemporaryFile` XMLs (kept alive so they aren't deleted).
- **`gym_torcs/quickrace_template.xml`** — quick-race template, track **corkscrew**, focused
  driver `${id}`, scr_server module.
- **`torcs/config/raceman/quickrace2.xml`** — 2-driver quick race (corkscrew, 2 laps).
- **`scr_server.xml`** — all scr_server cars changed `car1-trb1 → car1-ow1` (open-wheel).

## 3. Artifacts
Standard three checkpoints (~892 KB).

## 4. Findings
- Pure **throughput / scaling** work: ~9× environment concurrency via asyncio, each instance a
  self-launched headless TORCS, with templated per-port race configs.
- Shares origin and several config files (`quickrace_template.xml`, `raceconfig.py`,
  `quickrace2.xml`, open-wheel `scr_server.xml`) with [[headless]] — they branched from the same
  `d9310d0` "initial work for headless training". [[headless]] kept it single-instance but added
  the reward/gear work; `parallel` kept master's reward but went multi-instance async.
- Caveat: `gear_change=False` in `build_envs` here (gear auto), unlike [[headless]].
- Reward unchanged → don't expect different driving behavior, just faster data collection.
