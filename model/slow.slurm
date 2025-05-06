#!/bin/bash
#SBATCH --job-name=text2midi    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ms1438@princeton.edu

module purge
module load anaconda3/2024.10
conda activate venv
python train.py

