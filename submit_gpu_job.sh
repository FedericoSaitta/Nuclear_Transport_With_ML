#!/bin/bash --login
#SBATCH -p gpuL                    # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU(s)
#SBATCH --ntasks-per-node=12          # Number of tasks per node
#SBATCH -t 4-0                     # Wallclock time limit
#SBATCH --mail-type=ALL                   # Email notifications
#SBATCH --mail-user=abel.castanedarodriguez@student.manchester.ac.uk
#SBATCH -J gpu_job                     # Job name
#SBATCH -o logs/job_%j.out                # Standard output log
#SBATCH -e logs/job_%j.err                # Standard error log

# Run your Python script
python sweep_train.py
