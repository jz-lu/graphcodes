#!/bin/bash 
#SBATCH -n 2 #Request 12 tasks (cores)
#SBATCH -t 0-12:00 #Request runtime of 12 hours
#SBATCH -p sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=8000 #Request 4G of memory per CPU
#SBATCH -o out_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e err_%j.txt #redirect errors to error_JOBID.txt
#SBATCH --mail-type=END #Mail when job ends
#SBATCH --mail-user=lujz@mit.edu #email recipient

source activate myenv
echo "Hello World"
python main.py
echo "Done!"
