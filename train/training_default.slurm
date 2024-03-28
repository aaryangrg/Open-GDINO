#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/train/finetuning.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2    # And two GPU
#SBATCH --cpus-per-task=18           # Two cores per task
#SBATCH --job-name=Train
#SBATCH --constraint=gmem32

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python -m torch.distributed.launch  --nproc_per_node 2 ../main.py \
        --output_dir ../results/train/finetuning/ \
        -c ../config/cfg_odvg_train.py \
        --datasets ../config/coco_od_new.json \
        --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \

    

    