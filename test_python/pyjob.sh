#!/bin/bash
#
#SBATCH --job-name=pytest
#SBATCH --output=pytest_result.txt
#SBATCH --error=pytest_errors.txt

#
#SBATCH --ntasks=6
#sbcast -f pytest.py /tmp/pytest.py
srun python3 /tmp/pytest.py
date
