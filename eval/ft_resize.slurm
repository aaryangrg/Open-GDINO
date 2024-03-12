#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/eval/miou_0.40/ft_swint_lowered_iou_0.40.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=FTRes
#SBATCH --constraint=gmem16

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

nvidia-smi

python3 ../custom_eval.py \
    --config_file ../config/cfg_coco.py \
    --pretrain_model_path ../weights/gdinot-coco-ft.pth \
    --output_dir ../results/eval/ft_swint_lowered_iou_0.40/ \
    --datasets ../config/coco_od_datasets.json \
    --eval \
    --eval_batch 6 \
    # --custom_transform resize \
    # --custom_res 320
