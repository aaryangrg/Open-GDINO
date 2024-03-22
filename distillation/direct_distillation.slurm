#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/backbone_distillation/test.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2   # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=BackDistill
#SBATCH --constraint=gmem24

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

nvidia-smi

python3 -m torch.distributed.launch  --nproc_per_node 2  ../direct_distillation.py \
    --config_file ../config/cfg_odvg.py \
    --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
    --output_dir ../results/backbone_distillation/test/ \
    --datasets ../config/coco_od_datasets.json \
    --config ../effvit/configs/cls/imagenet/b1.yaml \
    --path ../experiments/backbone_distillation/test/ \
    --distributed  true \
    --fp16 \
    --eval_batch_size 10 \
    --effvit_model swint \
    --custom_transforms resize \
    --custom_res 480 

# output_dir --> GDINO logs
# path --> model checkpoints + Effvit logs