#!/bin/bash

# 실험 번호 입력 받기
echo -n "실험 번호를 입력하세요 (예: 001, 002, ...): "
read -r EXPERIMENT_NUM

# 로컬 이미지를 GPU 서버로 전송
echo "=== 이미지 파일 전송 중 ==="
if [ -f "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.png" ]; then
    scp "dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.png" "gpu-server:DiffusionLight/imgs/${EXPERIMENT_NUM}.png"
    echo "이미지 전송 완료: ${EXPERIMENT_NUM}_before.png"
else
    echo "오류: dataset/${EXPERIMENT_NUM}/${EXPERIMENT_NUM}_before.png 파일을 찾을 수 없습니다."
    exit 1
fi

# GPU 서버에서 GPU 상태 확인
echo "=== GPU 사용 현황 확인 ==="
GPU_INFO=$(ssh gpu-server "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
echo "$GPU_INFO"

# 사용 가능한 GPU 자동 선택 (메모리 사용량이 가장 적은 GPU)
AVAILABLE_GPU=$(ssh gpu-server "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1")
echo "추천 GPU: $AVAILABLE_GPU"

# 사용자 확인 (로컬에서 처리)
echo -n "GPU $AVAILABLE_GPU 를 사용하여 실행하시겠습니까? (y/n): "
read -r CONFIRM

if [[ "$CONFIRM" == "y" ]] || [[ "$CONFIRM" == "Y" ]] || [[ "$CONFIRM" == "yes" ]] || [[ "$CONFIRM" == "YES" ]]; then
    echo "GPU $AVAILABLE_GPU 로 실행을 시작합니다..."
    
    # GPU 서버에서 실행
    ssh gpu-server bash << EOF
cd DiffusionLight

# conda 초기화 (여러 경로 시도)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "conda 초기화 스크립트를 찾을 수 없습니다. PATH에서 conda를 찾습니다..."
    export PATH=\$PATH:/opt/miniconda3/bin:/home/\$USER/miniconda3/bin
fi

conda activate dl121-311

echo "=== 1단계: Padding 실행 ==="
CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU python padding.py imgs/${EXPERIMENT_NUM}.png

echo "=== 2단계: Inpainting 실행 ==="
CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU python inpaint.py --dataset imgs/${EXPERIMENT_NUM} --output_dir out/${EXPERIMENT_NUM} --no_torch_compile

echo "=== 3단계: Ball to envmap 변환 ==="
CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU python ball2envmap.py --ball_dir out/${EXPERIMENT_NUM}/square --envmap_dir out/${EXPERIMENT_NUM}/envmap

echo "=== 4단계: Exposure to HDR 변환 ==="
python exposure2hdr.py --input_dir out/${EXPERIMENT_NUM}/envmap --output_dir out/${EXPERIMENT_NUM}/hdr

echo "=== 5단계: EXR to HDR 변환 ==="
python exr2hdr.py --input_dir out/${EXPERIMENT_NUM}/hdr

echo "모든 DiffusionLight 처리가 완료되었습니다!"
EOF

    if [ $? -eq 0 ]; then
        echo "=== GPU 서버에서 처리 완료 ==="
        
        # 로컬 저장 폴더 생성
        mkdir -p input/${EXPERIMENT_NUM}
        
        # 결과 HDR 파일을 로컬로 복사
        echo "결과 HDR 파일을 로컬로 복사 중..."
        scp gpu-server:/node_data/urp25sp_kong/DiffusionLight/out/${EXPERIMENT_NUM}/hdr/padded_${EXPERIMENT_NUM}.hdr /Users/jinwoo/Documents/work/svoi/input/${EXPERIMENT_NUM}/global.hdr
        
        if [ $? -eq 0 ]; then
            echo "HDR 파일 복사 완료: input/${EXPERIMENT_NUM}/global.hdr"
            echo "모든 작업이 완료되었습니다!"
        else
            echo "오류: HDR 파일 복사에 실패했습니다."
            exit 1
        fi
    else
        echo "오류: GPU 서버에서 처리 중 오류가 발생했습니다."
        exit 1
    fi
else
    echo "실행이 취소되었습니다."
    exit 1
fi
