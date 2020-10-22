#!/bin/bash -l                                                                                    
#SBATCH --output=/mnt/lustre/users/%u/%j.out
#SBATCH --job-name=alphavr
# #SBATCH --gres=gpu
#SBATCH --ntasks=10
#SBATCH --mem=20000
#SBATCH --time=6-12:00
# #SBATCH --constrain=v100
#SBATCH --constrain=skylake

ulimit -n 4096
git status
cat global_parameters.py

# module load libs/cuda
python ./rainbow_hac/train.py --id='hac_fed_large_15_variance' --active-scheduler --active-accesspoint --previous-action-observable --history-length-accesspoint=2 --history-length-scheduler=1 --architecture='canonical_4uav_61obv_3x3_mix' --action-selection='greedy' --data-reinforce --evaluation-interval=500 --evaluation-episodes=20000 --federated-round='20'