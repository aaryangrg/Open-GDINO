#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/eval/pretrained.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1    # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=GDinoTest
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate gdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

cuda_path=$(which nvcc | xargs dirname | xargs dirname)
export CUDA_HOME=$cuda_path


python3 ../test.py \
    --config_file ../config/cfg_coco.py \
    --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
    --output_dir ../results/eval/pretrained/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \

    

    