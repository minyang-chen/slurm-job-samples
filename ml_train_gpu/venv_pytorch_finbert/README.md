# `finbert-gpu-demo`: Training a Pytorch Bert Model on Slurm GPU Node

This repository contains the code and scripts required to train a BERT model for sentiment analysis on financial news using GSB <a href="https://rcpedia.stanford.edu/topicGuides/yenGPU.html" target="_blank">Slurm GPU node</a>. The BERT model is trained on a <a href="https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/" target="_blank">Kaggle dataset</a> of financial phrase bank. The project utilizes the `transformers` library along with `PyTorch Lightning` for streamlined model training and evaluation.

## Table of Contents

- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Training Script Details](#training-script-details)
- [Usage](#usage)
- [Monitoring Training](#monitoring-training)
- [Output](#output)
- [Authors](#authors)

## Repository Structure
- `Sentences_AllAgree.txt`: dataset of 2,264 financial news headlines labeled with a sentiment (<a href="https://arxiv.org/abs/1307.5336" target="_blank">Malo et al., 2014</a>).  
- `finbert-job.sh`: Slurm script for training the BERT model to be run on the Yens.
- `finbert-train.py`: Python script to train the model on Yen's GPU node.

## Training Script Details
The pre-check script `check-cuda.py` performs scan of the GPU configuration on job host.

The training script `finbert-train.py` performs the following:

- Defines a custom dataset to read in financial news texts and their labels.
- Utilizes the BERT model from the `transformers` library for sentiment analysis.
- Trains the model using PyTorch Lightning's Trainer on Yen's GPU.

## Usage
Modify the Slurm script to include your email address. Slurm will report useful metrics via email such as queue time, runtime, CPU and RAM utilization and will alert you if the job has failed.

Submit the Slurm script to initiate the model training on `gpu` partition on the Yens:

```bash
$ sbatch finbert-job.sh
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

# Credits Original Author

Authors: Stanford GSB DARC Team (gsb_darcresearch@stanford.edu) 
```bash
$ https://github.com/gsbdarc/yens-gpu-demo.git 
```
