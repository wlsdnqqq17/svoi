#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_number>"
    exit 1
fi
EXPERIMENT_NUM=$1

echo "=== Image file transfer in progress ==="
if [ -f "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.png" ]; then
    scp "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.png" "gpu-server:Gen3DSR/imgs/${EXPERIMENT_NUM}.png"
    echo "Image sent: ${EXPERIMENT_NUM}_before.png -> ${EXPERIMENT_NUM}.png"
else
    echo "Error: dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.png file not found."
    exit 1
fi

AVAILABLE_GPU=1
echo "Running with GPU $AVAILABLE_GPU..."

ssh gpu-server bash << EOF
cd Gen3DSR/src

# conda initialization (multiple paths)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "conda initialization script not found, searching for conda in PATH..."
    export PATH=\$PATH:/opt/miniconda3/bin:/home/\$USER/miniconda3/bin
fi

conda activate gen121-310

# CUDA compiler compatibility settings
export TORCH_CUDA_ARCH_LIST="8.6"
export NVCC_APPEND_FLAGS="--allow-unsupported-compiler"
export CUDA_NVCC_FLAGS="--allow-unsupported-compiler"
export TORCH_NVCC_FLAGS="--allow-unsupported-compiler"

CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU python run.py --config ./configs/image.yaml \\
    scene.attributes.img_path='../imgs/${EXPERIMENT_NUM}.png' \\
    scene.save_dir='../out/${EXPERIMENT_NUM}'
EOF

mkdir -p input/${EXPERIMENT_NUM}

echo "Copying result files to local..."

scp gpu-server:/node_data/urp25sp_kong/Gen3DSR/out/${EXPERIMENT_NUM}/reconstruction/full_scene.glb /Users/jinwoo/Documents/work/svoi/input/${EXPERIMENT_NUM}/

scp gpu-server:/node_data/urp25sp_kong/Gen3DSR/out/${EXPERIMENT_NUM}/c2w.npy /Users/jinwoo/Documents/work/svoi/input/${EXPERIMENT_NUM}/

scp gpu-server:/node_data/urp25sp_kong/Gen3DSR/out/${EXPERIMENT_NUM}/K.npy /Users/jinwoo/Documents/work/svoi/input/${EXPERIMENT_NUM}/

scp gpu-server:/node_data/urp25sp_kong/Gen3DSR/out/${EXPERIMENT_NUM}/depth_map.npy /Users/jinwoo/Documents/work/svoi/input/${EXPERIMENT_NUM}/

echo "All tasks completed!"
