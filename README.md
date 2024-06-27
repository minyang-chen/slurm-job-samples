# slurm-job-samples
Repository for Slurm Job Samples encapsulate GPU resources.

### List of ML training jobs

[venv_pytorch_finBert](ml_train_gpu/venv_pytorch_finbert)

[conda_tensorflow_mnist](ml_train_gpu/spack_conda_tensorflow_mnist)

## Check Slurm Cluster Partition and Node Status
```
$ sinfo -s
$ scontrol show node <nodename>
$ scontrol show partition <partitionname>
```

## Slurm Batch Job submission
```
$ sbatch  <myjob.sh>

On success expect return <job-id>
```

## Slurm Interactive Session for Debugging 
Using interactive session with a time limit that will launch a bash shell to run job step by step
```
$ srun -t 2:00:00 --partition=gpu-queue --nodes=1 --ntasks-per-node=1 --gpus=1 --pty bash -i
```

## Cancel a running job 
```
$scancel job_id

# verify cancellation
#squeue
```

## Slurm Job Monitor and Status

Slurm commands communicate with the slurmctld and slurmd daemons to retrieve or modify information about nodes, jobs, and partitions.
```
$squeue
```
Beyond using squeue to check the status of a job, here are a few other non-basic ways to get more information about your jobs.

Check the output/error logs in the files you specified in your sbatch script:
```
#SBATCH --output=my_output.txt
#SBATCH --error=my_errors.txt
```
For more detailed info about the state of a job:
```
# sacct provides accounting data for all jobs (running or terminated)

$sacct
$sacct -j <job_id>
$sacct -o ALL -j <jobid>

# detailed information about the status of a specific job including starting/end time, cpus used, task id etc.

$scontrol show job <job_id>
$scontrol show --details job <jobid>

$sstat -j 11600 -o maxrss,maxvmsize,maxdiskwrite

```

### Slurm GPU resource scheduling
see here for more details: [GPU Resource Scheduling](docs/gpu_resource_scheduling.md)

### Slurm Job Resources Request Tips

```
# General
#SBATCH --nodes=2                 # Number of compute nodes to use for the job
#SBATCH --ntasks=4                # Number of tasks (processes) to run
#SBATCH --gres=gpu:1              # What general resources to use per node
#SBATCH --mem=32G                 # How much memory to use
#SBATCH --mem-per-cpu=4G          # How much memory to use per cpu
#SBATCH --time=2:00:00            # Time limit for the job
#SBATCH --partition=general       # Which partition the job should use
```
```
# CPU specific
#SBATCH --cpus-per-task=2         # Number of CPU cores to use for each task
```
```
# GPU specific 
#SBATCH --gpus=1                  # How many gpus to use for an entire job
#SBATCH --gpus-per-node=1         # How many gpus to use per node
#SBATCH --gpus-per-task=1         # How many gpus to use per task
#SBATCH --gpus-per-socket=1       # How many gpus to use per socket
#SBATCH --constraint=gpu_type     # What type of gpu to use
#SBATCH --constraint=gmem24G      # only use 24G of GPU memory
```

Note:
Generally it is recommended to use --gpus-per-node in most cases 
combined with --ntasks-per-gpu as all tasks in your job will be guaranteed 
to have access to a GPU.
