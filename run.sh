#!/bin/bash
#SBATCH --partition=donut-default
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=48:00:00

source /nas/home/siyiguo/anaconda3/envs/damf_env/bin/activate topic_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

# pip install emoji

python src/topic_modeling_multi.py
# python src/utils/print_tweets.py