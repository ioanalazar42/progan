#!/bin/bash
echo "Model name: $1"
echo "Grid size: $2"
echo "Num runs: $3"
echo "Command:"
echo "python generate_images.py --model_file_name=$1 --grid_size=$2"
for run in `seq 1 $3`
do
  python generate_images.py --model_file_name=$1 --grid_size=$2
done
