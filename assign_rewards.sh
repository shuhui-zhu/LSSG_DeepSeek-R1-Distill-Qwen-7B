MODEL_NAME=im_deepseekQwen7B

#CONDA_BASE=$(conda info --base)
#source $CONDA_BASE/etc/profile.d/conda.sh
#conda activate test

PYTHONPATH=. python3 tools/assign_rewards.py \
    --input_data_path data/self_play_results/${MODEL_NAME}_sampling_all_words_results.json \
    --output_data_path data/train_lssg_data_${MODEL_NAME}.json \
    --sft_data_path data/alpaca_train.json
