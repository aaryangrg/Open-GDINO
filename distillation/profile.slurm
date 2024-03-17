#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/profile/swin_b_320_480_1333.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=Profile
#SBATCH --constraint=gmem32

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

nvidia-smi

python3 ../custom_eval.py \
    --config_file ../config/cfg_coco_swin_b.py \
    --pretrain_model_path ../weights/groundingdino_swinb_cogcoor.pth \
    --output_dir ../results/profile/swinb/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 10 \
    --custom_transform resize \
    --custom_res 320

python3 ../custom_eval.py \
    --config_file ../config/cfg_coco_swin_b.py \
    --pretrain_model_path ../weights/groundingdino_swinb_cogcoor.pth \
    --output_dir ../results/profile/swinb/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 10 \
    --custom_transform resize \
    --custom_res 480

python3 ../custom_eval.py \
    --config_file ../config/cfg_coco_swin_b.py \
    --pretrain_model_path ../weights/groundingdino_swinb_cogcoor.pth \
    --output_dir ../results/profile/swinb/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 6 \
    --custom_transform resize \
    --custom_res 1333