#!/bin/bash

X=$1
Y=$2
Z=$3
orig_x=$4
orig_y=$5
width=$6
height=$7
folder_name=$8
nx=$9
ny=${10}
nz=${11}

python src/gen_fenv.py "$X" "$Y" "$Z" "$folder_name" "$nx" "$ny" "$nz"
python src/insert_object.py "$X" "$Y" "$Z" "$width" "$height" "$folder_name" "$nx" "$ny" "$nz"
