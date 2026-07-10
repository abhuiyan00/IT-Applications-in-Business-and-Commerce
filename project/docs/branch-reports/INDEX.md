# AI_racer_pytorch — Branch Snapshots

Detailed per-branch reports for **Szymon1905/AI_racer_pytorch**. The project trains a **PPO
agent (PyTorch) to drive in TORCS** over the SCR UDP protocol. All meaningful code lives in
`gym_torcs/*.py`; the rest of the ~6843 files are the vendored TORCS simulator.

Each `.md` documents that branch's tip commit, base, goal, full code-level diff vs `master`,
artifacts (checkpoints/lines/logs), and findings. Read **[master.md](master.md)** first — it is
the shared baseline every other report diffs against.

| Branch | File | Tip | One-line essence |
|---|---|---|---|
| master | [master.md](master.md) | `96e3f43` | Baseline PPO env+agent; straightness+progress reward, auto-gear by speed. |
| 160speed | [160speed.md](160speed.md) | `d28c6b7` | Speed push (default_speed 200, bigger speed reward, wall-brake). 140 km/h, ¾ lap. |
| 160speed+stablesteer | [160speed+stablesteer.md](160speed+stablesteer.md) | `e73e9b1` | 160speed + 10× steer-smoothness + traction-control throttle. Struggles on S-turn. |
| Reward_func_improvement | [Reward_func_improvement.md](Reward_func_improvement.md) | `351fa47` | Docs only — adds README eval section (no code change). |
| a.bhuiyan | [a.bhuiyan.md](a.bhuiyan.md) | `6a0598b` | Full rewrite: offline racing line + data-driven car model + baseline & residual PPO + telemetry. |
| gear_strategy_max_jerk | [gear_strategy_max_jerk.md](gear_strategy_max_jerk.md) | `1a2b326` | WIP: agent-controlled gear + jerk-based shift reward. |
| headless | [headless.md](headless.md) | `ab4b328` | Stable-gym reward + headless TORCS + gear/rpm-band reward. |
| parallel | [parallel.md](parallel.md) | `c90cdd6` | Asyncio rewrite: ~9 parallel headless TORCS instances for throughput. |
| piasecki/fix1 | [piasecki_fix1.md](piasecki_fix1.md) | `537e7f9` | One-line fix: `--updates` relative to resumed checkpoint. |
| stable | [stable.md](stable.md) | `57941d6` | Separate actor/critic, danger/time-to-edge reward, off-track-continue eval. Improved lap time. |

## Lineage at a glance
- **Speed line:** master → [160speed] → [160speed+stablesteer]; separately master → [stable].
- **Infra line:** `d9310d0` "initial work for headless training" → [parallel] (async, multi-instance)
  and → [headless] (single-instance + stable reward + gear/rpm). [headless] also merges [stable].
- **Gear ideas:** [gear_strategy_max_jerk] (raw jerk) → [headless] (rpm band) → [a.bhuiyan]
  (data-driven torque/shift map).
- **Standalone:** [piasecki/fix1] (bugfix), [Reward_func_improvement] (docs), [a.bhuiyan] (new architecture).

## Recurring findings across branches
- The **S-curve / corkscrew S-section** is repeatedly the hardest part (160speed+stablesteer, a.bhuiyan).
- **Gearing** is a persistent problem: stuck-in-gear / engine-on-limiter bugs drove the jerk →
  rpm-band → data-driven-shift-map progression.
- **Episode length matters:** too-small `--max-steps` causes constant TORCS resets before a lap
  completes (called out explicitly in a.bhuiyan's README).
