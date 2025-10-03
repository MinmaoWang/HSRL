# HSRL
Official implementation of the paper “Hierarchical Semantic RL: Tackling the Problem of Dynamic Action Spaces for RL-based Recommendation”.


This repository provides code, processing scripts for HSRL — a novel reinforcement learning framework for recommender systems. HSRL introduces Semantic Action Space (SAS), which maps item-level actions to compact, hierarchical semantic identifiers (SIDs). With a Hierarchical Policy Network (HPN) and a Multi-level Critic (MLC), HSRL stabilizes training, improves interpretability, and scales RL to large-scale recommendation scenarios.

🔹 Key Features:

* **Problem**: RL-based recommendation struggles with **vast & dynamic action spaces**, hindering stable policy learning.
* **Solution**: Propose **Hierarchical Semantic RL (HSRL)** with a **fixed Semantic Action Space (SAS)**.

  * Encode items as **Semantic Identifiers (SIDs)**, invertible without loss.
  * **Hierarchical Policy Network (HPN)**: coarse-to-fine generation + residual state modeling.
  * **Multi-level Critic (MLC)**: token-level value estimates for fine-grained credit assignment.
* **Results**:

  * Outperforms SOTA on **public benchmarks** and a **large-scale industrial dataset**.
  * **Online A/B test (7 days)**: **+18.421% CVR** with only **+1.251% cost**.
* **Impact**: Validates **semantic action modeling** as a scalable, interpretable paradigm for RL-based recommendation.
