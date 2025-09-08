#!/bin/bash

echo -n "실험 번호를 입력하세요 (예: 001, 002, ...): "
read -r EXPERIMENT_NUM

python src/download_assets.py --dataset_id $EXPERIMENT_NUM
python src/put_plane.py --dataset_id $EXPERIMENT_NUM