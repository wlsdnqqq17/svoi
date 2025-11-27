#!/bin/bash

echo -n "Enter experiment number (e.g. 001, 002, ...): "
read -r EXPERIMENT_NUM

echo "=== Image file transfer in progress ==="
if [ -f "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.jpg" ]; then
    scp "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.jpg" "gpu-server:StyleLight/assets/wild2/${EXPERIMENT_NUM}.jpg"
    echo "Image sent: ${EXPERIMENT_NUM}_before.jpg"
else
    echo "Error: dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.jpg file not found."
    exit 1
fi

echo "=== GPU usage check ==="
GPU_INFO=$(ssh gpu-server "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
echo "$GPU_INFO"

AVAILABLE_GPU=$(ssh gpu-server "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1")
echo "Recommended GPU: $AVAILABLE_GPU"

echo -n "Running with GPU $AVAILABLE_GPU? (y/n): "
read -r CONFIRM

if [[ "$CONFIRM" == "y" ]] || [[ "$CONFIRM" == "Y" ]] || [[ "$CONFIRM" == "yes" ]] || [[ "$CONFIRM" == "YES" ]]; then
    echo "GPU $AVAILABLE_GPU is running..."

    ssh gpu-server bash << EOF
cd StyleLight

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

conda activate StyleLight

CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU python test_lighting.py

PYTHONPATH=/node_data/urp25sp_kong/StyleLight:/node_data/urp25sp_kong/StyleLight/skylibs  python evaluation/tonemap.py --testdata assets/checkpoints_without_light_mask_both_finetuned --out_dir out

EOF

    if [ $? -eq 0 ]; then
        echo "=== GPU server processing completed ==="

        mkdir -p input/${EXPERIMENT_NUM}

        echo "Copying result HDR file to local..."
        scp gpu-server:/node_data/urp25sp_kong/StyleLight/data/tone/out/${EXPERIMENT_NUM}_test.exr /Users/jinwoo/Documents/work/svoi/input/${EXPERIMENT_NUM}/sl.exr

        if [ $? -eq 0 ]; then
            echo "HDR file copied: input/${EXPERIMENT_NUM}/sl.exr"
            echo "All tasks completed!"
        else
            echo "Error: Failed to copy HDR file."
            exit 1
        fi
    else
        echo "Error: An error occurred while processing on the GPU server."
        exit 1
    fi
else
    echo "Execution cancelled."
    exit 1
fi
