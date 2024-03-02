#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/eval/pretrained_swint_480_360_crop.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=320Eval
#SBATCH --constraint=gmem16

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

nvidia-smi

python3 ../main.py \
    --config_file ../config/cfg_coco.py \
    --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
    --output_dir ../results/eval/pretrained_swint_480_360_crop/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 10
