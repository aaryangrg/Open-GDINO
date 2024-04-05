#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/backbone_distillation/test_res_bn_stats.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=FPDist
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=99
nvidia-smi

python3 -m torch.distributed.launch  --nproc_per_node 1 --master_port=29533 ../direct_distillation.py \
    --config_file ../config/cfg_odvg.py \
    --pretrain_model_path ../weights/gdinot-coco-ft.pth \
    --output_dir ../results/backbone_distillation/test_res_bn_stats/ \
    --datasets ../config/coco_od_new.json \
    --config ../effvit/configs/cls/imagenet/b1.yaml \
    --path ../experiments/backbone_distillation/test_res_bn_stats/ \
    --distributed  true \
    --fp16 \
    --eval_batch_size 1 \
    --effvit_model swint \
    --custom_transforms resize \
    --custom_res 480 \
    --kd_loss kld \
    --pretrained_patch_embed true

# output_dir --> GDINO logs
# path --> model checkpoints + Effvit logs
