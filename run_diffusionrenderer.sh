#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_number>"
    exit 1
fi
EXPERIMENT_NUM=$1

echo "=== Image file transfer in progress ==="
ssh gpu-server "mkdir -p diffusion-renderer/inference_input_dir/${EXPERIMENT_NUM}"
if [ -f "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_after.png" ]; then
    scp "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_after.png" "gpu-server:diffusion-renderer/inference_input_dir/${EXPERIMENT_NUM}/frame000.png"
else
    echo "Error: File dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_after.png not found"
    exit 1
fi
# scp "out/l${EXPERIMENT_NUM}/gt.png" "gpu-server:diffusion-renderer/inference_input_dir/l${EXPERIMENT_NUM}/frame000.png"
ssh gpu-server "cd diffusion-renderer/inference_input_dir/${EXPERIMENT_NUM} && for i in \$(seq -w 1 23); do cp frame000.png frame0\${i}.png; done"
# ssh gpu-server "cd diffusion-renderer/inference_input_dir/l${EXPERIMENT_NUM} && for i in \$(seq -w 1 23); do cp frame000.png frame0\${i}.png; done"
if [ -f "input/${EXPERIMENT_NUM}/global.exr" ]; then
    scp "input/${EXPERIMENT_NUM}/global.exr" "gpu-server:diffusion-renderer/examples/hdri/"
else
    echo "Error: File input/${EXPERIMENT_NUM}/global.exr not found"
    exit 1
fi
# scp "input/l${EXPERIMENT_NUM}/global.exr" "gpu-server:diffusion-renderer/examples/hdri/"



AVAILABLE_GPU=7
echo "Running with GPU $AVAILABLE_GPU..."

ssh gpu-server bash << EOF
cd diffusion-renderer

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

conda activate diffusion_renderer

export TORCH_CUDA_ARCH_LIST="8.6"
export NVCC_APPEND_FLAGS="--allow-unsupported-compiler"
export CUDA_NVCC_FLAGS="--allow-unsupported-compiler"
export TORCH_NVCC_FLAGS="--allow-unsupported-compiler"

CUDA_VISIBLE_DEVICES=1 python inference_svd_rgbx.py --config configs/rgbx_inference.yaml
CUDA_VISIBLE_DEVICES=1 python inference_svd_xrgb.py --config configs/xrgb_inference.yaml
EOF

mkdir -p out/${EXPERIMENT_NUM}

echo "Copying result files to local..."

scp gpu-server:/node_data/urp25sp_kong/diffusion-renderer/out/output_relighting/${EXPERIMENT_NUM}/0000.0000.env0000.png /Users/jinwoo/Documents/work/svoi/out/${EXPERIMENT_NUM}/result3.png

echo "All tasks completed!"
