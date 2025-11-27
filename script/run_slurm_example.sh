#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --account=your_account_here
#SBATCH --mail-user=your_email_here
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

#module purge
#module load gcc/8.3.0
module load python/3.9.21
#module load cuda/10
#module load cudnn/7

python main_crossformer.py \
  --data custom_graph \
  --root_path ./datasets/ \
  --file_list $(ls ./datasets/*.csv | xargs -n 1 basename | tr '\n' ' ') \
  --data_dim 16 \
  --in_len 96 \
  --out_len 24 \
  --seg_len 6 \
  --win_size 2 \
  --factor 10 \
  --topk 3 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 3 \
  --dropout 0.2 \
  --train_epochs 20 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --checkpoints ./checkpoints/ \
  --save_pred \
  --use_gpu True \
  --itr 1