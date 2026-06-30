mkdir -p output

# ml1m environment

mkdir -p output/ml1m/
mkdir -p output/ml1m/env
mkdir -p output/ml1m/env/log

data_path="dataset/ml1m/"
output_path="output/ml1m/"


for REG in 0.0001 0.0003 0.001
do
    for LR in 0.0003 0.001 0.003
    do
        python train_env.py\
            --model ML1MUserResponse\
            --reader ML1MDataReader\
            --train_file ${data_path}train.tsv\
            --val_file ${data_path}test.tsv\
            --user_meta_file ${data_path}users_meta.npy\
            --item_meta_file ${data_path}items_meta.npy\
            --data_separator '\t'\
            --meta_data_separator ' '\
            --loss 'bce'\
            --l2_coef ${REG}\
            --lr ${LR}\
            --epoch 2\
            --seed 19\
            --model_path ${output_path}env/ml1m_user_env_lr${LR}_reg${REG}.model\
            --max_seq_len 50\
            --n_worker 4\
            --feature_dim 16\
            --hidden_dims 256\
            --attn_n_head 2\
            > ${output_path}env/log/ml1m_user_env_lr${LR}_reg${REG}.model.log
    done
done