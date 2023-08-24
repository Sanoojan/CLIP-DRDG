#!/bin/bash
#SBATCH --job-name=DR_baseline
#SBATCH --gres gpu:2
#SBATCH --nodes 1
#SBATCH --partition=ai702
#SBATCH --reservation=ai702

nvidia-smi


for lr in  0.00005 
do
    for dataset in DR
    do
        for init in clip_full
        do
            for command in delete_incomplete launch
            do
                python -m domainbed.scripts.sweep $command\
                    --data_dir=DATASET_PATH \
                    --output_dir=./domainbed/Outputs/Eye_resnet-check-lr${lr}\
                    --command_launcher multi_gpu\
                    --algorithms ERM \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 1 \
                    --hparams """{\"lr\":${lr}}"""\
                    --skip_confirmation  
            done > Outs/ERM_check.out
        done
    done
done

