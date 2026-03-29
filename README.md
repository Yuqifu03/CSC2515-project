# An Empirical Study into the Robustness and Efficiency of LLM-Guided Reinforcement Learning
Final project for CSC2515.

# Overview

This project investigates the robustness and efficiency of reinforcement learning algorithms when guided by large language models (LLMs). We explore how reward sparsity, feedback noise and query frequency influence it's performance.

# LLM Frequency

## Local LLM Configuration

We use a locally deployed language model for feedback generation:

- Model: ```deepseek-coder-v2:16b```

## Setup & Execution
Before running the experiment, make sure you are logged into Weights & Biases:
```bash
wandb login
```

Then launch training with configurable parameters: 
```bash
python pokeagent/local_llm_qlearn.py \
  ++train_iterations=<num_iterations> \
  ++feedback_frequency=<frequency>
```
