# Hierarchical Semantic RL (HSRL)

Official implementation of the paper:<br>
**“Hierarchical Semantic RL: Tackling the Problem of Dynamic Action Spaces for RL-based Recommendation.”**

## Overview

HSRL is a reinforcement learning framework for recommender systems that addresses dynamic and high-dimensional action spaces.

The framework introduces a Semantic Action Space (SAS), where each item is represented by a compact hierarchical Semantic Identifier (SID). It combines:

- Semantic Identifiers (SIDs)
- Hierarchical Policy Network (HPN)
- Multi-Level Critic (MLC)

## Setup

### 0. Pretrain the User Response Model

Modify `train_env.sh` or `train_env_ml.sh` and run:

```bash
bash train_env.sh
```

### 1. Build the Semantic Codebook

```bash
cd dataset
python build_codebook.py
python build_item2sid.py
```

### 2. Training

Available scripts:

```bash
bash train_ddpg.sh
bash train_superddpg.sh
bash train_supervise.sh
bash train_sid_rl4rs.sh
```

To resume from a checkpoint:

```bash
--n_iter ${PREVIOUS_N_ITER} ${N_ITER}
```

### 3. Evaluation & Analysis

Run testing:

```bash
bash test_sid.sh
```

Visualize and analyze results with `result_analysis.ipynb`.
