#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/backbone_distillation/task_l2/v3_b24_480res_datasubset_effvit_configs_1_to_1_augmentations.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2   # And two GPU
#SBATCH --cpus-per-task=18           # Two cores per task
#SBATCH --job-name=V3L2Task
#SBATCH --constraint=gmem48

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=99
nvidia-smi

python3 -m torch.distributed.launch  --nproc_per_node 2 --master_port=29533 ../direct_distillation.py \
    --config_file ../config/cfg_odvg.py \
    --pretrain_model_path ../weights/gdinot-coco-ft.pth \
    --output_dir ../results/backbone_distillation/task_l2/v3_b24_480res_datasubset_effvit_configs_1_to_1_augmentations/ \
    --datasets ../config/coco_od_new.json \
    --config ../effvit/configs/cls/imagenet/b1.yaml \
    --path ../experiments/backbone_distillation/task_l2/v3_b24_480res_datasubset_effvit_configs_1_to_1_augmentations/ \
    --distributed  true \
    --fp16 \
    --eval_batch_size 18 \
    --effvit_model swint \
    --custom_transforms resize \
    --custom_res 480 \
    --kd_loss l2 \
    --pretrained_patch_embed true \
    --with_task_loss true \
    --model_version_tb  effvit-v3 \
    --experiment_name_tb  task_l2_1_to_1_b24_480res_datasubset_effvit_configs_augmentations \
    --train_batch_size 12

# output_dir --> GDINO logs
# path --> model checkpoints + Effvit logs