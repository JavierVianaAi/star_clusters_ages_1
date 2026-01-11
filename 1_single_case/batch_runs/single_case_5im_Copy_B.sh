#!/bin/bash
#SBATCH --job-name="single_case_5im_Copy_B"
#SBATCH --partition=sched_mit_mki_preempt_r8      # PRIORITY: sched_mit_mki_r8     NON_PRIORITY: sched_mit_mki_preempt_r8
#SBATCH --time=06:00:00                     
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=100GB                          # Satori limit 1100GB   
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=vianajr@mit.edu
#SBATCH --output=/home/vianajr/cluster_ages_1/1_single_case/batch_runs/slurm_outs/single_case_5im_Copy_B.out

# Code below this line:
date +"%T"
python3 /home/vianajr/cluster_ages_1/1_single_case/batch_runs/single_case_5im_Copy_B.py

