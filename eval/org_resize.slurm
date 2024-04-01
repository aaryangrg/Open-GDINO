#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/eval/org_swint_custom_subset_eval.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=ORRes
#SBATCH --constraint=gmem16

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

nvidia-smi

# python3 ../custom_eval.py \
#     --config_file ../config/cfg_coco.py \
#     --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
#     --output_dir ../results/eval/org_swint_lowered_iou_0.40/ \
#     --datasets ../config/coco_od_datasets.json \
#     --eval \
#     --eval_batch 6 \
#     # --custom_transform resize \
#     # --custom_res 320

# python3 ../custom_eval.py \
#     --config_file ../config/cfg_coco.py \
#     --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
#     --output_dir ../results/eval/org_swint_lowered_iou_0.40/ \
#     --datasets ../config/coco_od_datasets.json \
#     --eval \
#     --eval_batch 10 \
#     --custom_transform resize \
#     --custom_res 320

echo "ORIGINAL MODEL"
python3 ../custom_eval.py \
    --config_file ../config/cfg_coco.py \
    --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
    --output_dir ../results/eval/org_swint_lowered_iou_0.40/ \
    --datasets ../config/coco_od_new.json \
    --eval \
    --eval_batch 10 \
    --custom_transform resize \
    --custom_res 480

echo "FINE-TUNED MODEL"
python3 ../custom_eval.py \
    --config_file ../config/cfg_coco.py \
    --pretrain_model_path ../weights/gdinot-coco-ft.pth \
    --output_dir ../results/eval/org_swint_lowered_iou_0.40/ \
    --datasets ../config/coco_od_new.json \
    --eval \
    --eval_batch 10 \
    --custom_transform resize \
    --custom_res 480