# Racing Car DQN with MetaDrive

A reinforcement learning project that trains a **Deep Q-Network (DQN)** agent to drive a vehicle in the [MetaDrive](https://github.com/metadriverse/metadrive) simulator.

The agent receives MetaDrive's vector observation, chooses from six discrete driving actions, and learns a driving policy using experience replay, epsilon-greedy exploration, a target network, and a custom reward function focused on lane keeping, safe speed, smooth control, and reaching the destination.
Note: This project is far from being perfect. It has some major issues

## Overview

The project converts MetaDrive's continuous vehicle controls into a small discrete action space so that a DQN can be used for control.

The network maps a **259-dimensional observation** to Q-values for **6 possible actions**:

| Action | Steering | Throttle |
|---|---|---|
| Right + Accelerate | Right | Accelerate |
| Right + Brake | Right | Brake / reduced throttle |
| Left + Accelerate | Left | Accelerate |
| Left + Brake | Left | Brake / reduced throttle |
| Straight + Accelerate | Straight | Accelerate |
| Straight + Brake | Straight | Brake / reduced throttle |

During training, the selected discrete action is converted into the continuous control format expected by MetaDrive.

## Features

- Deep Q-Network implemented with PyTorch
- MetaDrive driving environment
- Six-action discrete control space
- Experience replay buffer
- Epsilon-greedy exploration
- Target Q-network
- Soft target-network updates
- Gradient clipping
- Custom reward shaping
- CPU and CUDA support
- Saved PyTorch checkpoint for evaluation

## Project Structure

```text
racing-car/
├── model.py           # DQN neural-network architecture
├── training.py        # Training loop, replay buffer, agent and reward function
├── test.py            # Checkpoint loading and rendered evaluation
├── race.pth           # Trained model checkpoint
├── environment.yaml   # Conda environment export
└── README.md
```

For a public GitHub repository, generated folders such as `__pycache__/` should be excluded with `.gitignore`.

## DQN Architecture

The Q-network in `model.py` is a fully connected neural network:

```text
Input observation: 259
        │
        ▼
Linear(259, 128)
        │
       ReLU
        │
        ▼
Linear(128, 128)
        │
       ReLU
        │
        ▼
Linear(128, 6)
        │
        ▼
Q-value for each action
```

The action with the highest predicted Q-value is selected during exploitation.

## Reward Function

Training uses a custom reward function in addition to MetaDrive's step reward.

The reward encourages the agent to:

- maintain a reasonable driving speed,
- stay completely inside the lane,
- avoid large changes between consecutive controls,
- avoid crashes,
- stay on the road,
- avoid excessive speed, and
- reach the destination.

Large penalties are applied for crashes, leaving the road, and excessive speed, while reaching the destination receives a large positive reward.

This reward shaping is intended to encourage safer and smoother driving instead of optimizing only forward progress.

## Training Configuration

The current implementation uses approximately the following DQN settings:

| Parameter | Value |
|---|---:|
| Observation size | 259 |
| Number of actions | 6 |
| Hidden-layer size | 128 |
| Replay-buffer capacity | 10,000 |
| Batch size | 64 |
| Discount factor (`gamma`) | 0.99 |
| Learning rate | 0.0001 |
| Target-update coefficient (`tau`) | 0.005 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.01 |
| Default training steps | 200,000 |
| Loss | Mean Squared Error |
| Optimizer | Adam |

The code automatically uses CUDA when a compatible GPU is available.

## Requirements

The current development environment uses Python 3.10 and includes:

- PyTorch
- NumPy
- Pandas
- MetaDrive

Versions recorded in the supplied environment include:

```text
Python 3.10
MetaDrive 1.4.35
PyTorch 2.12.0
NumPy 2.2.6
Pandas 2.3.3
```

A clean environment can be created with Conda or `venv`.

### Example with Conda

```bash
conda create -n racing-car python=3.10
conda activate racing-car
pip install metadrive==1.4.35 torch numpy pandas
```

> The repository currently contains an exported `environment.yaml`. For portability, it is recommended to keep only direct project dependencies in that file and remove machine-specific environment prefixes before publishing it.

## Training

Run the training script from the project directory:

```bash
python training.py
```

The default number of training steps is:

```text
200000
```

A different number can be supplied through the command line:

```bash
python training.py --train_steps 100000
```

For example:

```bash
python training.py --train_steps 500000
```

After training, the Q-network weights are saved as:

```text
race.pth
```

## Evaluation

The project includes `test.py` as the current rendered evaluation script.

It loads `race.pth`, creates a MetaDrive environment with rendering enabled, predicts the best discrete action from the current observation, converts it to MetaDrive controls, and executes the policy in the simulator.

After cleaning the evaluation script so that it uses the single `DQNNetwork` implementation from `model.py`, it can be run with:

```bash
python test.py
```

For a cleaner public repository, renaming `test.py` to `evaluate.py` is recommended:

```bash
python evaluate.py
```

## Experience Replay

Transitions are stored in a replay buffer with a maximum capacity of 10,000 samples.

Each transition contains:

```text
(state, action, reward, next_state, terminated, truncated)
```

Mini-batches are sampled from this buffer and used to update the Q-network.

The current implementation gives newer transitions higher sampling probability. This differs from standard uniform DQN replay and can be treated as an experimental recency-prioritized replay strategy.

## Target Network

Two neural networks are maintained during training:

- **Q-network** — optimized using gradient descent.
- **Target network** — used to calculate Bellman targets.

The target network is updated using a soft update:

```text
target = tau × q_network + (1 - tau) × target
```

with:

```text
tau = 0.005
```

Using a separate target network helps stabilize DQN training.

## Exploration

The agent uses an epsilon-greedy strategy.

At the beginning of training, epsilon is high, so the agent frequently explores random actions. As training progresses, epsilon decreases toward a minimum value of `0.01`, allowing the learned Q-network to control more of the agent's behavior.

## Model Checkpoint

The included:

```text
race.pth
```

contains the saved state dictionary of the trained Q-network.

Because the checkpoint is small, it can be stored directly in the GitHub repository. For substantially larger checkpoints, Git LFS, GitHub Releases, or an external model repository would be preferable.

## Possible Improvements

Future versions of the project could include:

- reproducible random seeds,
- separate `train.py` and `evaluate.py` entry points,
- configurable hyperparameters,
- TensorBoard or Weights & Biases logging,
- episode reward and success-rate plots,
- systematic evaluation over multiple random seeds,
- uniform or standard prioritized experience replay,
- Double DQN,
- Dueling DQN,
- better checkpoint management,
- reward-function ablation experiments, and
- a demonstration GIF or video in this README.

## Repository Notes

Before publishing the project, it is recommended to:

1. remove `__pycache__/`,
2. add a `.gitignore`,
3. clean unused imports and commented-out legacy code,
4. remove the machine-specific `prefix` from `environment.yaml`,
5. keep the DQN architecture defined only in `model.py`, and
6. clean or rename `test.py` so evaluation runs directly from the shared model implementation.

## About

This project is an experimental implementation of a DQN-based driving agent for the MetaDrive simulator. It is intended for reinforcement-learning experimentation and educational/research use.
