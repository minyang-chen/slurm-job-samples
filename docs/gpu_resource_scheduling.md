## GPU resource scheduling in Slurm

### Simple GPU scheduling with exclusive node access
Slurm supports scheduling GPUs as a consumable resource just like memory and disk. However, configuration can be simplified when not allowing multiple jobs per compute node.

Scheduling GPUs without making use of GRES (Generic REsource Scheduling) is to create partitions or queues for logical groups of GPUs.

For example, grouping nodes with RTX6000 GPUs into a RTX6000 partition:

```console
$ sinfo -s
PARTITION AVAIL  TIMELIMIT   NODES(A/I/O/T)  NODELIST
RTX6000     up   infinite         4/9/3/16  node[1-3,5-8]
```

Partition configuration via Slurm configuration file `slurm.conf`:

```console
NodeName=node[1-3,5-8]
PartitionName=RTX6000 Default=NO DefaultTime=01:00:00 State=UP Nodes=node[1-3,5-8]
```
### Scheduling resources at the per GPU level

Slurm can be made aware of GPUs as a consumable resource to allow jobs to request any number of GPUs.

This feature requires job accounting to be enabled first; for more info, see: https://slurm.schedmd.com/accounting.html

The Slurm configuration file needs parameters set to enable cgroups for resource management and GPU resource scheduling:

`slurm.conf`:

```console
# General
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup

# Scheduling
SelectType=select/cons_res
SelectTypeParameters=CR_Core_Memory

# Logging and Accounting
AccountingStorageTRES=gres/gpu
DebugFlags=CPU_Bind,gres                # show detailed information in Slurm logs about GPU binding and affinity
JobAcctGatherType=jobacct_gather/cgroup
```

Partition information in `slurm.conf` defines the available GPUs for each resource:

```console
# Partitions
GresTypes=gpu
NodeName=slurm-node-0[0-1] Gres=gpu:2 CPUs=10 Sockets=1 CoresPerSocket=10 ThreadsPerCore=1 RealMemory=30000 State=UNKNOWN
PartitionName=compute Nodes=ALL Default=YES MaxTime=48:00:00 DefaultTime=04:00:00 MaxNodes=2 State=UP DefMemPerCPU=3000
```

Cgroups require a seperate configuration file:

`cgroup.conf`:

```console
CgroupAutomount=yes 
CgroupReleaseAgentDir="/etc/slurm/cgroup" 

ConstrainCores=yes 
ConstrainDevices=yes
ConstrainRAMSpace=yes
#TaskAffinity=yes
```

GPU resource scheduling requires a configuration file to define the available GPUs and their CPU affinity

`gres.conf`:

```console
Name=gpu File=/dev/nvidia0 CPUs=0-4
Name=gpu File=/dev/nvidia1 CPUs=5-9
```

Running jobs utilizing GPU resources requires the `--gres` flag; for example, to run a job requiring a single GPU:

```console
$ srun --gres=gpu:1 nvidia-smi
$ srun -p slurmgpu --gres=gpu:1 nvidia-smi
```

In order to enforce proper CPU:GPU affinity (i.e. for performance reasons), use the flag `--gres-flags=enforce-binding`

> --gres-flags=enforce-binding
If set, the only CPUs available to the job will be those bound to the selected GRES (i.e. the CPUs identified in the gres.conf file will be strictly enforced rather than advisory). This option may result in delayed initiation of a job. For example a job requiring two GPUs and one CPU will be delayed until both GPUs on a single socket are available rather than using GPUs bound to separate sockets, however the application performance may be improved due to improved communication speed. Requires the node to be configured with more than one socket and resource filtering will be performed on a per-socket basis. This option applies to job allocations.


### Kernel configuration

Using memory cgroups to restrict jobs to allocated memory resources requires setting kernel parameters

On Ubuntu systems this is configurable via `/etc/default/grub`

> GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"






