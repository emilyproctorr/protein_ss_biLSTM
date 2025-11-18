#!/bin/bash
#SBATCH --output=logs/log_lstm_%j_%a.log
#SBATCH --partition=example
#SBATCH --time=0-06:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu
#SBATCH --array=1-16

source /lstm_env/bin/activate

CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" config.txt)
read HIDDEN NUM_LAYERS DROPOUT LR <<< "$CONFIG"

OUTDIR="output_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTDIR"

echo "job:              $SLURM_ARRAY_TASK_ID"
echo "hidden_dim:       $HIDDEN"
echo "num_layers:       $NUM_LAYERS"
echo "dropout:          $DROPOUT"
echo "learning_rate:    $LR"
echo "output directory: $OUTDIR"

python3 LSTM.py --mode "train" \
        --batch_size 32 \
        --hidden_dim $HIDDEN \
        --num_layers $NUM_LAYERS \
        --learning_rate $LR \
        --num_epochs 150 \
        --dropout $DROPOUT \
        --num_outputs 3 \
        --seed 42 \
        --output_folder "$OUTDIR"

