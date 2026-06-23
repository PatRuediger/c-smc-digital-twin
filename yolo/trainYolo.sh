#!/bin/bash
#SBATCH -t 1:00:00             # time limit set to 1 hour
#SBATCH --mem=32G               # 32G of memory are reserved
#SBATCH -J yolov11              # the job is named yolov11
#SBATCH -o jobname.%j.out	# Output: %j expands to jobid
#SBATCH -e jobname.%j.err	# Error: %j expands to jobid
#SBATCH --mail-type=END      # an email is send at the end of the job
#SBATCH -n 2                 # 2 processors to be used
#SBATCH --gres=gpu:v100        # Get a volta GPU
#SBATCH -N 1                 # 1 node is used

module purge				# don't inherit, use clean environment
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
module load cudnn
module load nvidia
conda activate pyTorchYOLO

HOST=`hostname`
echo " Slurm scheduled it on node $HOST"
python3 modelTrainYolo.py

conda deactivate