#!/bin/bash

source /net/student33/data2/hendrix/miniconda3/etc/profile.d/conda.sh

conda activate tf_py

python /net/student33/data2/hendrix/git/neural_deprojection/neural_deprojection/models/identify_medium_SCD/generate_voxel_data.py
