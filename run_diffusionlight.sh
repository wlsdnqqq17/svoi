#!/bin/bash

echo -n "Enter experiment number (e.g. 001, 002, ...): "
read -r EXPERIMENT_NUM

if [ -f "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.png" ]; then
    scp "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.png" "gpu-server:DiffusionLight/imgs/${EXPERIMENT_NUM}.png"
    echo "Image sent: ${EXPERIMENT_NUM}_before.png"
elif [ -f "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.jpg" ]; then
    scp "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.jpg" "gpu-server:DiffusionLight/imgs/${EXPERIMENT_NUM}.png"
    echo "Image sent: ${EXPERIMENT_NUM}_before.jpg"
else
    echo "Error: Neither dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.png nor dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.jpg file found."
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
cd DiffusionLight

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    export PATH=\$PATH:/opt/miniconda3/bin:/home/\$USER/miniconda3/bin
fi

conda activate diffusionlight

CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU python padding.py imgs/${EXPERIMENT_NUM}.png
CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU python inpaint.py --dataset imgs/${EXPERIMENT_NUM} --output_dir out/${EXPERIMENT_NUM} --no_torch_compile
CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU python ball2envmap.py --ball_dir out/${EXPERIMENT_NUM}/square --envmap_dir out/${EXPERIMENT_NUM}/envmap
python exposure2hdr.py --input_dir out/${EXPERIMENT_NUM}/envmap --output_dir out/${EXPERIMENT_NUM}/hdr

echo "DiffusionLight Success"
EOF

    if [ $? -eq 0 ]; then
        
        mkdir -p input/${EXPERIMENT_NUM}
        
        echo "Copying HDR file to local..."
        scp gpu-server:/node_data/urp25sp_kong/DiffusionLight/out/${EXPERIMENT_NUM}/hdr/padded_${EXPERIMENT_NUM}.exr /Users/jinwoo/Documents/work/svoi/input/${EXPERIMENT_NUM}/global.exr
        
        if [ $? -eq 0 ]; then
            echo "HDR file copied to input/${EXPERIMENT_NUM}/global.hdr"
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