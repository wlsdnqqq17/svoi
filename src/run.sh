#!/bin/bash

X=$1
Y=$2
Z=$3
orig_x=$4
orig_y=$5
width=$6
height=$7
folder_name=$8

python gen_fenv.py "$X" "$Y" "$Z" "$folder_name"
blender --background --python insert_object.py -- "$orig_x" "$orig_y" "$width" "$height" "$folder_name"
