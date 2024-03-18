#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/profile/swin_both_detectron.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=Profile
#SBATCH --constraint=gmem24

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

nvidia-smi

echo "SWIN-T X 320"
python3 ../custom_eval.py \
    --config_file ../config/cfg_coco.py \
    --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
    --output_dir ../results/profile/swint/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 1 \
    --custom_transform resize \
    --custom_res 320

echo "SWIN-T X 480"
python3 ../custom_eval.py \
    --config_file ../config/cfg_coco.py \
    --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
    --output_dir ../results/profile/swint/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 1 \
    --custom_transform resize \
    --custom_res 480

echo "SWIN-T X 1333"
python3 ../custom_eval.py \
    --config_file ../config/cfg_coco.py \
    --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
    --output_dir ../results/profile/swint/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 1 \
    --custom_transform resize \
    --custom_res 1333

echo "SWIN-B X 320"
python3 ../custom_eval.py \
    --config_file ../config/cfg_coco_swin_b.py \
    --pretrain_model_path ../weights/groundingdino_swinb_cogcoor.pth \
    --output_dir ../results/profile/swinb/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 1 \
    --custom_transform resize \
    --custom_res 320

echo "SWIN-B X 480"
python3 ../custom_eval.py \
    --config_file ../config/cfg_coco_swin_b.py \
    --pretrain_model_path ../weights/groundingdino_swinb_cogcoor.pth \
    --output_dir ../results/profile/swinb/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 1 \
    --custom_transform resize \
    --custom_res 480

echo "SWIN-B X 1333"
python3 ../custom_eval.py \
    --config_file ../config/cfg_coco_swin_b.py \
    --pretrain_model_path ../weights/groundingdino_swinb_cogcoor.pth \
    --output_dir ../results/profile/swinb/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 1 \
    --custom_transform resize \
    --custom_res 1333