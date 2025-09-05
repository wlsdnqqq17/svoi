#!/bin/bash

echo -n "실험 번호를 입력하세요 (예: 001, 002, ...): "
read -r EXPERIMENT_NUM

python src/gen_fenv.py $EXPERIMENT_NUM