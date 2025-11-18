#!/bin/bash
#SBATCH --time= 1-00:05:00   #Format is d-hh:mm:ss
#SBATCH --account=_________  #Replace with your account name
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=24
#SBATCH --output=gen_alg_%j_out.txt
#SBATCH --error=gen_alg_%j_err.txt
#SBATCH --job-name=gen_alg
#SBATCH --mail-user=_________ #Put your email here if you would like to receive mail regarding the progress of your job. Otherwise, delete
#SBATCH --mail-type=ALL

module purge
module load python/3.13
module load scipy-stack

python -m pip install networkx
python -m pip install pulp
python dummycode.py
