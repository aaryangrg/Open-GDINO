#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/outs/odvg.out
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --job-name=GDinoTest

source /home/aaryang/anaconda3/bin/activate
conda activate gdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python3 ../tools/coco2odvg.py --input /home/c3-0/datasets/COCO/annotations/instances_train2017.json \
    --output ../config/coco2017_train_odvg.jsonl \