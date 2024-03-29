#!/bin/tcsh -f
#PBS -S /bin/tcsh
#PBS -j oe
#PBS -q para
#PBS -l nodes=1:ppn=48
#PBS -l walltime=336:00:00

# mail alert at start, end and abortion of execution
#PBS -m bea
# send mail to this address
#PBS -M hendrix@strw.leidenuniv.nl
#
cd /net/student33/data2/hendrix/
setenv PATH /net/student33/data2/hendrix/
#
/net/student33/data2/hendrix/git/neural_deprojection/neural_deprojection/models/identify_medium_SCD/run_generate_voxel_data.sh