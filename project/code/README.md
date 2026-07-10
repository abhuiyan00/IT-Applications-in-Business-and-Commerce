# AI Racer — code

Two agents from the [AI-Slop Racers](../README.md) project are included here:

| Folder | What it is | Result |
|--------|-----------|--------|
| [`stable/`](stable/) ⭐ | Separate actor/critic networks + the time-to-edge **danger reward**. | The best agent — a clean **1:31.53** lap, top speed ≈ 245 km/h. |
| [`a.bhuiyan/`](a.bhuiyan/) | Offline **racing line** + deterministic baseline, with PPO learning only a bounded **residual**. | A different control architecture; reduced (not eliminated) S-curve errors. |

Both share the same layout — the team's reinforcement-learning code lives in `gym_torcs/`:

```text
<variant>/
├── README.md            the branch's own notes (stable: ReadMe; a.bhuiyan: README + guide.txt)
├── requirements.txt     pinned Python dependencies
├── commands.txt         quick command crib
└── gym_torcs/
    ├── gym_torcs.py         TorcsEnv — env wrapper, reward shaping, termination   (team)
    ├── rl_agent.py          PPO agent + actor/critic network                       (team)
    ├── train_torcs_rl.py    training loop / CLI                                    (team)
    ├── eval_torcs_rl.py     deterministic evaluation / CLI                         (team)
    ├── snakeoil3_gym.py     SCR UDP client                                         (SCR/upstream)
    ├── practice.xml         race configuration
    ├── checkpoints/         trained model weights (*.pt)
    ├── LICENSE, README.md   upstream gym_torcs licence + readme  (MIT © N. Yoshida)
    └── …                    a.bhuiyan adds racing_line.py, car_spec.py,
                             analyze_telemetry.py, lines/, logs/
```

The environment wrapper began as **[gym_torcs](https://github.com/ugo-nama-kun/gym_torcs)**
(MIT © 2016 Naoto Yoshida); the reward, agent, training, and evaluation code were substantially
rewritten by the team. The upstream licence is preserved in `gym_torcs/LICENSE`.

---

## Prerequisite — TORCS with the SCR server

The agent drives an external **TORCS** process over the SCR UDP protocol, so you need a TORCS
build that includes the **`scr_server`** bots. This ≈ 1 GB third-party simulator is **not**
committed here (it is not the team's work). Get it one of these ways:

- The original project repository vendors the full patched simulator:
  **[github.com/Szymon1905/AI_racer_pytorch](https://github.com/Szymon1905/AI_racer_pytorch)**
  (Windows users can run the prebuilt `wtorcs.exe`).
- Or build TORCS + the **Simulated Car Racing** patch from the
  [SCR Championship](https://arxiv.org/abs/1304.1672) distribution.

A Windows setup/troubleshooting guide is in
[`../docs/TORCS_Windows_Troubleshooting_Guide.docx`](../docs/TORCS_Windows_Troubleshooting_Guide.docx)
(firewall rules for the UDP port are the usual gotcha).

---

## Setup

```bash
python -m venv RACERvenv
# Windows:            RACERvenv\Scripts\activate
# macOS / Linux:      source RACERvenv/bin/activate
python -m pip install --upgrade pip
pip install -r stable/requirements.txt          # (or a.bhuiyan/requirements.txt)
```

A CUDA GPU is optional but speeds up training a lot; the device is auto-selected.

## Run

1. **Start TORCS** and open a **Practice** race with the **`scr_server`** driver, leaving it on
   the blue “waiting” screen (Corkscrew circuit, port **3001**).
2. From the variant's `gym_torcs/` folder:

```bash
# evaluate the best checkpoint (deterministic, full laps)
python eval_torcs_rl.py --model-path checkpoints/torcs_ppo_latest_best_eval.pt --episodes 10 --max-steps 5000

# train (start fresh by omitting --load-path). A large --max-steps = full laps, fewer resets.
python train_torcs_rl.py --updates 1500 --rollout-size 4096 --max-steps 100000 \
    --load-path checkpoints/torcs_ppo_latest.pt --save-path checkpoints/torcs_ppo_latest.pt
```

> **Use a large `--max-steps`.** A small value resets the episode before the car can finish a
> lap, so TORCS keeps restarting and the agent never sees a full fast lap.

The `a.bhuiyan` variant defaults to the Corkscrew racing line and adds `--residual-scale`
(agent authority over the baseline) and `--no-assist` (pure PPO); regenerate its line with
`python racing_line.py`. See [`a.bhuiyan/README.md`](a.bhuiyan/README.md) and
[`a.bhuiyan/guide.txt`](a.bhuiyan/guide.txt) for the full design.

---

Released under the [MIT License](../../LICENSE). Upstream `gym_torcs` retains its own MIT licence
in `gym_torcs/LICENSE`.
