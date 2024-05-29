#!/bin/sh
# Below, is the queue
#PBS -q ai
#PBS -j oe
# Number of cores
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:25:00
#PBS -P 12003789
#PBS -N yolo_train
# Start of commands
cd $PBS_O_WORKDIR
module load singularity/3.10.0
image="/app/apps/containers/pytorch/pytorch_23.05_py3.sif"
singularity run --nv -B /scratch,/app,/data $image python3 train_voc_at.py --workers 8 --device 0 --batch-size 48 --data data/voc.yaml --img 640 640 --cfg cfg/training/yolov7-voc.yaml --weights '' --name yolov7-train-tap-at4 --noise_path data/TAP-def-noise.pkl --hyp data/hyp.scratch.p5.yaml --augment --epochs 100 --pgd-radius 4 --pgd-steps 3 --pgd-step-size 2 --pgd-random-start