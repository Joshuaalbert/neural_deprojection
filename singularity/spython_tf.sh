#!/bin/bash
echo "Ensure your repo is in '$HOME/git/'"
export SINGULARITY_BINDPATH="/data"
singularity exec --nv $HOME/git/neural_deprojection/singularity/tensorflow_gpu.simg python "$@"