# `mnist-gpu-demo`: Training on Tensorflow MNist job on Slurm GPU Node

This repository contains the code and scripts required to train a simple MNist model for hand writing character recognition.

## Table of Contents

- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Training Script Details](#training-script-details)
- [Usage](#usage)
- [Monitoring Training](#monitoring-training)
- [Output](#output)
- [Authors](#authors)

## Repository Structure
- `tf.keras.datasets.mnist`: dataset of labeled hand writing characters.  
- `mnist-job.sh`: Slurm script for training the BERT model to be run on the Yens.
- `mnist-train.py`: Python script to train the model on Yen's GPU node.

## Training Script Details
The training script `train-finbert.py` performs the following:

- Defines a custom dataset to read in financial news texts and their labels.
- Utilizes the BERT model from the `transformers` library for sentiment analysis.
- Trains the model using PyTorch Lightning's Trainer on Yen's GPU.

## Usage
Modify the Slurm script to include your email address. Slurm will report useful metrics via email such as queue time, runtime, CPU and RAM utilization and will alert you if the job has failed.

Submit the Slurm script to initiate the model training on `gpu` partition on the Yens:

```bash
$ sbatch train-finbert-job.sh
```

Monitor the training progress by checking the Slurm queue for your username:

```bash
$ squeue -u $USER
$ sacct -j 3
```

## Monitoring Training
Instructions for monitoring GPU utilization and other training metrics.

login to compute nodes
```bash
$ nvtop
```

## Output
After the training is complete, check the output file `finBERT-train.out` for training and evaluation metrics:

```bash
$ cat train-finbert.output
```

# Credits 

Original Authors: Stanford GSB DARC Team (gsb_darcresearch@stanford.edu) 
```bash
$ https://github.com/gsbdarc/yens-gpu-demo.git 
```
