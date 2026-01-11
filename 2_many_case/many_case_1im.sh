#!/bin/bash
#SBATCH --job-name="many_case_1im"
#SBATCH --partition=sched_mit_mki_r8      # PRIORITY: sched_mit_mki_r8     NON_PRIORITY: sched_mit_mki_preempt_r8
#SBATCH --time=2-00:00:00                     
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=100GB                             # Satori limit 1100GB   
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=vianajr@mit.edu
#SBATCH -o /home/vianajr/cluster_ages_1/2_many_case/slurm_outs/many_case_1im.out

# Code below this line:
date +"%T"
python3 /home/vianajr/cluster_ages_1/2_many_case/many_case_1im.py

