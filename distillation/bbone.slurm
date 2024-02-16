#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/OPEN-GDINO/outs/test.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1      # And two GPU
#SBATCH --cpus-per-task=10            # Two cores per task
#SBATCH --job-name=GDinoTest

source /home/aaryang/anaconda3/bin/activate
conda activate gdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python3 ./test.py --config_file ../config/cfg_coco.py \
    --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
    