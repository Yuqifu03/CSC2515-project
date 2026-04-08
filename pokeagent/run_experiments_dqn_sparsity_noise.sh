#!/bin/bash
# run_experiments_dqn.sh

# Ensure we're in the right directory and use the virtual environment
cd "$(dirname "$0")"
if [ -d "../venv" ]; then
  source ../venv/bin/activate
fi

mkdir -p results

EPISODES=1000
FREQ=100
BATCH_INTERVAL=20

echo "=== DQN Experiment 1: Sparsity (Level 0, 1) ==="
for mode in baseline llm-code llm-direct; do
  for sparsity in 0 1; do
    echo "Running $mode with sparsity $sparsity..."
    python train_dqn_sparsity_noise.py --mode $mode --sparsity-level $sparsity --episodes $EPISODES --llm-freq $FREQ --batch-interval $BATCH_INTERVAL --output-csv results/exp1_${mode}_s${sparsity}.csv
  done
done

echo "=== DQN Experiment 2: Robustness (Noise type) ==="
for mode in llm-code llm-direct; do
  for noise in gaussian logical; do
    echo "Running $mode with $noise noise..."
    python train_dqn_sparsity_noise.py --mode $mode --sparsity-level 0 --noise-type $noise --noise-std 0.5 --episodes $EPISODES --llm-freq $FREQ --batch-interval $BATCH_INTERVAL --output-csv results/exp2_${mode}_${noise}.csv
  done
done

echo "=== DQN Experiment 3: Frequency ==="
for freq in 0 50 200; do
  echo "Running llm-code with freq $freq..."
  python train_dqn_sparsity_noise.py --mode llm-code --sparsity-level 0 --episodes $EPISODES --llm-freq $FREQ --output-csv results/exp3_llmcode_f${freq}.csv
done

echo "DQN Experiments complete. Run 'python plot_results_dqn.py' to generate result charts."
