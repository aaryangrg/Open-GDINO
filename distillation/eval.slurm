#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/mixed_precision_effvit.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=MxdPrec
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python3 -m torch.distributed.launch  --nproc_per_node 1 ../test.py \
    --config_file ../config/cfg_odvg.py \
    --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
    --output_dir ../results/eval/pretrained/ \
    --datasets ../config/coco_od_datasets.json \
    --config ../effvit/configs/cls/imagenet/b1.yaml \
    --path ../experiments/effvit/backbone_train_mixed/ \
    --distributed  true \
    --fp16 true \
