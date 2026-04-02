#!/bin/bash
#SBATCH --job-name=Diff-CNOP
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gpu015
#SBATCH -t 72:00:00
#SBATCH --output=/public/home/qinbo/Diff-CNOP_clean/Diff-CNOP_%j.log
#SBATCH --mem=100G

ulimit -s unlimited
# load environment
module purge
module load compiler/cuda/11.4
# source /public/home/qinbo/cau/bin/activate
echo "==========================="
echo "Job ID: $SLURM_JOB_NAME"
echo "Job name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of processors: $SLURM_NTASKS"
echo "Task is running on the following nodes:"
echo $SLURM_JOB_NODELIST
echo "==========================="
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export OMP_NUM_THREADS=1

echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

# pip list
python -u /public/home/qinbo/Diff-CNOP_clean/Diff-CNOP_sampling.py