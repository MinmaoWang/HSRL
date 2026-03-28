# Hierarchical Semantic RL (HSRL)

Official implementation of the paper:<br>
**“Hierarchical Semantic RL: Tackling the Problem of Dynamic Action Spaces for RL-based Recommendation.”**

---

## 🚀 Overview

**HSRL** is a novel reinforcement learning framework for recommender systems that addresses one of the most critical bottlenecks in RL-based recommendation — the **dynamic and high-dimensional action space**.

HSRL introduces a **Semantic Action Space (SAS)**, where each item is represented by a compact, hierarchical **Semantic Identifier (SID)**.
Through this design, HSRL achieves **structured decision-making**, **stable training**, and **interpretable long-term optimization**.

---

## 🌟 Key Features

| Category            | Description                                                                                                                                                                                                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Problem**         | RL-based recommenders face **huge and constantly changing action spaces**, making policy learning unstable and inefficient.                                                                                                                                                                |
| **Solution**        | **Hierarchical Semantic Reinforcement Learning (HSRL)** defines a fixed **Semantic Action Space (SAS)** to decouple policy learning from item dynamics.                                                                                                                                    |
| **Core Components** | • **Semantic Identifiers (SIDs):** Compact, invertible item representations.<br>• **Hierarchical Policy Network (HPN):** Coarse-to-fine token generation with residual state modeling.<br>• **Multi-Level Critic (MLC):** Token-level value estimation for fine-grained credit assignment. |
| **Results**         | • Consistently outperforms SOTA methods on public and industrial datasets.<br>• **7-day online A/B test:** +18.421% CVR gain with only +1.251% cost increase.                                                                                                                              |
| **Impact**          | Demonstrates **semantic action modeling** as a **scalable and interpretable paradigm** for large-scale RL-based recommendation.                                                                                                                                                            |

---



## ⚙️ Setup

### 0. Pretrain the User Response Model (Environment Component)

Modify `train_env.sh`:

* Update dataset paths:

  ```bash
  data_path=/your/path/to/dataset/
  output_path=/your/path/to/output/
  ```
* Configure model arguments (`X ∈ {RL4RS, ML1M}`):

  ```bash
  --model {X}UserResponse \
  --reader {X}DataReader \
  --train_file ${data_path}{X}_b_train.csv \
  --val_file ${data_path}{X}_b_test.csv
  ```
* Set `model_path` and `log_path`.

Run:

```bash
bash train_env.sh
```

---

### 1. Build the Semantic Codebook

```bash
cd dataset
python build_codebook.py
python build_item2sid.py
```

---

### 2. Training

#### 2.1 Available Scripts

| Task                        | Script                    |
| --------------------------- | ------------------------- |
| DDPG                        | `bash train_ddpg.sh`      |
| BehaviorDDPG                | `bash train_superddpg.sh` |
| Offline Supervised Learning | `bash train_supervise.sh` |
| HSRL                        | `bash train_sid_rl4rs.sh` |

#### 2.2 Continue Training

To resume from a checkpoint:

```bash
--n_iter ${PREVIOUS_N_ITER} ${N_ITER}
```

---

### 3. Evaluation & Analysis

Run testing:

```bash
bash test.sh
```

Visualize and analyze results:

```bash
result_analysis.ipynb
```


## 🧩 Acknowledgements

The code implementation and framework details of this project are built upon the work presented in the following paper. For more in-depth information regarding the framework architecture and implementation specifics, please refer directly to the original publication:

🔗 Exploration and Regularization of the Latent Action Space in Recommendation ([arXiv:2302.03431](https://arxiv.org/pdf/2302.03431))
