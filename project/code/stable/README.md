# AI racer 

## Description

An AI racing project built with PyTorch with help of IBM. The goal is to train an AI agent using Reinforcement Learning to navigate a track in the TORCS (The Open Racing Car Simulator) environment as fast and efficiently as possible.

## Installation

```bash
python -m venv RACERvenv
# Activate the environment (Windows: RACERvenv\Scripts\activate | Mac/Linux: source RACERvenv/bin/activate)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
1. Starting the simulator

Before running any scripts, you must launch the TORCS game client:
```bash
torcs/wtorcs.exe
```

2. Training \ Evaluation

This project has 2 main modes of operation: Training and Evaluation.

Training (train_torcs_rl.py): In this mode, the AI explores the track, takes actions, and receives rewards or penalties based on its performance. The aim is to maximize the cumulative reward. Based on the data it collects, the AI updates its neural network weights to learn the optimal driving strategy.

Evaluation (eval_torcs.py / Evaluation mode): In this mode, the AI's learning is paused. It relies entirely on a previously saved model (weights) to drive the car from the /checkpoints directory. Its purpose is to evaluate the performance of the trained weights.

3. Running the Training Script

To start training the AI, run the following command:
```bash
python gym_torcs/train_torcs_rl.py --updates 1500 --max-steps 100000 --rollout-size 4096 --load-path checkpoints/torcs_ppo_latest.pt --save-path checkpoints/torcs_ppo_latest.pt
```
If there are no weights available, remove the --load-path and --save-path arguments from the command to train them first.

4. Command line arguments

--updates 1500: Number of Learning Cycles. This defines how many times the algorithm will update the neural network's weights. Each update occurs after the agent collects a batch of driving data (defined by --rollout-size) and processes it to improve its driving strategy. More updates lead to better performance but also require more time. Train on cuda compatible GPU if possible to speed up the process.

--max-steps 100000: Episode Length. The maximum number of actions (frames/steps) the car is allowed to take during a single training session or episode before the environment resets.

--rollout-size 4096: Batch Size for Learning. The agent will collect 4,096 frames of driving data (states, actions, rewards) before pausing to calculate how to improve and update its network. A larger rollout size provides a more stable learning update but requires more memory.

--load-path checkpoints/...: Resume Training. The file path to a previously saved .pt (PyTorch) weights file. This allows the AI to pick up exactly where it left off instead of relearning how to drive from scratch.

--save-path checkpoints/...: Save Progress. The file path where the script will save the newly updated neural network weights during and after the training session.
