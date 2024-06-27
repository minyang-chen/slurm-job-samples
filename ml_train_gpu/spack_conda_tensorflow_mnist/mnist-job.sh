#!/bin/sh

#SBATCH --partition=slurmgpu
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
### SBATCH --gres=gpu:nvidia:1
### SBATCH --gpus=1

date
echo $CUDA_VISIBLE_DEVICES
hostname
nvidia-smi
cd /home/admin/ml_train/tensorflow
sbcast -f mnist-train.py /tmp/mnist-train.py

sudo su
bash /tools/spack/share/spack/setup-env.sh
/tools/spack/bin/spack --version

eval `/tools/spack/bin/spack load --sh   miniconda3` 

## New Environment
##/tools/spack/bin/spack install miniconda3
##/tools/spack/bin/spack load miniconda3

conda init
. ~/.bashrc
conda deactivate

conda create -n tf-gpu tensorflow-gpu -y
conda activate tf-gpu

##source /etc/profile.d/lmod.sh
##export LD_LIBRARY_PATH=/jhpce/shared/jhpce/core/JHPCE_tools/3.0/lib:/usr/local/lib/python3.9/site-packages/nvidia/#cudnn/lib:/jhpce/shared/jhpce/core/conda/miniconda3-23.3.1/envs/cudatoolkit-11.8.0/lib

START=$(date +%s)
srun python /tmp/mnist-train.py
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
date