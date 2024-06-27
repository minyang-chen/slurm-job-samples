#!/bin/bash

# slurm script to train a PyTorch deep learning model on GPU Node

#SBATCH -J mnist-train-job             # Job name
#SBATCH -p slurmgpu                    # Partition (queue) to submit to
#SBATCH -c 10                          # Number of CPU cores 
#SBATCH -N 1                           # Request one node 
#SBATCH -t 1:00:00                     # Max job time: 1 day 
#SBATCH -G 1                           # Max GPUs per user: 2
#SBATCH -o mnist-train.output          # Output file

date
# Create virtual environment
sudo apt install python3.9-venv

if [ ! -d "vmnist" ]; then
  echo "vmnist virtual environment does not exist."
  python -m venv vmnist
  echo "activate new virtual environment"  
  source vmnist/bin/activate
  # Install dependencies
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install transformers
  pip install -U scikit-learn
  pip install pytorch-lightning
  source vmnist/bin/activate
else
  echo "activate existing virtual environment"
  source vmnist/bin/activate
fi

echo "Check cuda availability"
START=$(date +%s)

srun hostname
srun nvidia-smi
srun python -c "import torch; print(torch.cuda.is_available()); print (torch.cuda.device_count())"

echo "Execute the Python script with multiple workers (for data loading)"
echo "Training job start"

srun python mnist-train.py ${SLURM_CPUS_PER_TASK}
deactivate

echo "Training job finished!"
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
date