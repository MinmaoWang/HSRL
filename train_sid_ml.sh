mkdir -p output

# ml1m environment

mkdir -p output/ml1m/
mkdir -p output/ml1m/env/
mkdir -p output/ml1m/env/log/
mkdir -p output/ml1m/agents/

output_path="output/ml1m/"

echo $output_path
env_path="output/ml1m/"
log_name="ml1m_user_env_lr0.001_reg0.0003"
SID_PATH="dataset/ml1m/sid_index_item2sid_64.pkl"

N_ITER=50000
CONTINUE_ITER=0
GAMMA=0.9
TOPK=1
EMPTY=0

MAX_STEP=20
INITEP=0
REG=0.001
NOISE=0.01
ELBOW=0.5
EP_BS=32
BS=64
SEED=17
SCORER="SIDPolicy_credit"
CRITIC_LR=0.001
BEHAVE_LR=0
TEMPER_SWEET_POINT=0.9
ADV=0





for REG in 0.00001
do
    for time in 1 2 3
    do
        for HCOEF in 0.1
        do
            for ACTOR_LR in 0.00005 # 0.00003
            do
                for SEED in 7 # 11 13 17 19 23
                do
                    for behavior_lr in 0.001 0.01
                    do
                    
                        mkdir -p ${output_path}agents/sid_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_entropy${HCOEF}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_advantage${ADV}_step${MAX_STEP}_seed${SEED}_times${time}_behavior${behavior_lr}/
    
                        python train_ddpg.py\
                            --env_class ML1MEnvironment_GPU\
                            --policy_class ${SCORER}\
                            --critic_class Token_Critic\
                            --agent_class A2C_SID_ml1m\
                            --facade_class SIDFacade_credit\
                            --behavior_lr ${behavior_lr}\
                            --seed ${SEED}\
                            --cuda 0\
                            --env_path ${env_path}env/${log_name}.env\
                            --max_step_per_episode ${MAX_STEP}\
                            --initial_temper ${MAX_STEP}\
                            --temper_sweet_point ${TEMPER_SWEET_POINT}\
                            --reward_func mean_with_cost\
                            --urm_log_path ${env_path}env/log/${log_name}.model.log\
                            --sasrec_n_layer 2\
                            --sasrec_d_model 32\
                            --sasrec_n_head 4\
                            --sasrec_dropout 0.1\
                            --critic_hidden_dims 256 64\
                            --slate_size 9\
                            --buffer_size 100000\
                            --start_timestamp 2000\
                            --noise_var ${NOISE}\
                            --empty_start_rate ${EMPTY}\
                            --save_path ${output_path}agents/sid_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_entropy${HCOEF}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_advantage${ADV}_step${MAX_STEP}_seed${SEED}_times${time}_behavior${behavior_lr}/model\
                            --episode_batch_size ${EP_BS}\
                            --batch_size ${BS}\
                            --actor_lr ${ACTOR_LR}\
                            --critic_lr ${CRITIC_LR}\
                            --actor_decay ${REG}\
                            --critic_decay ${REG}\
                            --target_mitigate_coef 0.01\
                            --advantage_bias ${ADV}\
                            --entropy_coef ${HCOEF}\
                            --gamma ${GAMMA}\
                            --n_iter ${N_ITER}\
                            --initial_greedy_epsilon ${INITEP}\
                            --final_greedy_epsilon ${INITEP}\
                            --elbow_greedy ${ELBOW}\
                            --check_episode 10\
                            --topk_rate ${TOPK}\
                            --item2sid ${SID_PATH}\
                            > ${output_path}agents/sid_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_entropy${HCOEF}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_advantage${ADV}_step${MAX_STEP}_seed${SEED}_times${time}_behavior${behavior_lr}/log
                    done
                done
            done
        done
    done
done
