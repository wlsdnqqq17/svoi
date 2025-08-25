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

# Check if envmap.hdr already exists, if not run gen_fenv.py
if [ ! -f "out/$folder_name/envmap.hdr" ]; then
    echo "envmap.hdr not found, running gen_fenv.py..."
    python src/gen_fenv.py "$X" "$Y" "$Z" "$folder_name" "$nx" "$ny" "$nz"
else
    echo "envmap.hdr already exists, skipping gen_fenv.py"
fi

# Check if result.png already exists, if not run insert_object.py
if [ ! -f "out/$folder_name/result.png" ]; then
    echo "result.png not found, running insert_object.py..."
    python src/insert_object.py "$X" "$Y" "$Z" "$width" "$height" "$folder_name" "$nx" "$ny" "$nz"
else
    echo "result.png already exists, skipping insert_object.py"
fi
