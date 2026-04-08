# An Empirical Study into the Robustness and Efficiency of LLM-Guided Reinforcement Learning
Final project for CSC2515.

# Overview

This project investigates the robustness and efficiency of reinforcement learning algorithms when guided by large language models (LLMs). We explore how reward sparsity, feedback noise and query frequency influence it's performance.

# Sparsity and Noise

For data training, see 
```bash
pokeagent/train_dqn_sparsity_noise.py
```

We provide a shell script to automate this process.
```bash
run pokeagent/run_experiments_dqn_sparsity_noise.sh
```

For plotting, refer to:
```bash
data_analysis.py
```

# LLM Frequency

## Setup & Execution
Before running the experiment, make sure you are logged into Weights & Biases:
```bash
wandb login
```

Then launch training with configurable parameters: 
```bash
python pokeagent/local_llm_qlearn.py
```

