#!/usr/bin/env bash
set -e

output_path="output/rl4rs/"
log_name="rl4rs_user_env_lr0.001_reg0.0003"


N_ITER=20000
GAMMA=0.9
TOPK=1
EMPTY=0
MAX_STEP=20
REG=0.00001
NOISE=0.01
EP_BS=32
BS=64
SCORER="SIDPolicy_credit"
CRITIC_LR=0.001
ACTOR_LR=0.00001
TEMPER_SWEET_POINT=0.9
ADV=0
SEED=17




SAVE_DIR="output_v1_credit_train/rl4rs/agents/sid_SIDPolicy_credit_actor0.001_critic0.001_entropy0.1_1000_niter50100_reg0.00001_ep0_noise0.01_bs64_epbs32_advantage0_step20_seed7/"



mkdir -p "$SAVE_DIR"

args=(
  --env_class RL4RSEnvironment_GPU
  --policy_class "$SCORER"
  --critic_class Token_Critic
  --agent_class A2C_SID_rl4rs
  --facade_class SIDFacade_credit       
  --seed "$SEED"
  --cuda 0
  --env_path "${output_path}env/${log_name}.env"
  --urm_log_path "${output_path}env/log/${log_name}.model.log"
  --data_split test
  --max_step_per_episode "$MAX_STEP"
  --initial_temper "$MAX_STEP"
  --temper_sweet_point "$TEMPER_SWEET_POINT"
  --reward_func mean_with_cost
  --sasrec_n_layer 2
  --sasrec_d_model 32
  --sasrec_n_head 4
  --sasrec_dropout 0.1
  --critic_hidden_dims 256 64
  --slate_size 9
  --buffer_size 100000
  --start_timestamp 0                    
  --noise_var 0                           
  --empty_start_rate "$EMPTY"
  --save_path "${SAVE_DIR}model"         
  --episode_batch_size "$EP_BS"
  --batch_size "$BS"
  --actor_lr "$ACTOR_LR"
  --critic_lr "$CRITIC_LR"
  --actor_decay "$REG"
  --critic_decay "$REG"
  --target_mitigate_coef 0.01
  --advantage_bias "$ADV"
  --entropy_coef 0
  --gamma "$GAMMA"
  --n_iter "$N_ITER"
  --initial_greedy_epsilon 0      
  --final_greedy_epsilon 0
  --elbow_greedy 0
  --check_episode 10
  --topk_rate "$TOPK"
)

python test.py "${args[@]}" > "${SAVE_DIR}test_log"
echo "DONE. See:"
echo "  Log:    ${SAVE_DIR}test_log"
echo "  Report: ${SAVE_DIR}model.report"