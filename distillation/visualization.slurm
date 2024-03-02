#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/visualizations/pretrained_swint_480_360_crop.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2   # And two GPU
#SBATCH --cpus-per-task=8           # Two cores per task
#SBATCH --job-name=VisDino
#SBATCH --constraint=gmem11

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

python ../tools/inference_on_a_image.py \
  -c ../tools/GroundingDINO_SwinT_OGC.py \
  -p ../weights/groundingdino_swint_ogc.pth \
  -i /home/c3-0/datasets/COCO/val2017000000227187.jpg \
  -t "bird" \
  -o ../visualizations/ \
  --output_file_name bird_banana_480_360_center_crop.jpg

