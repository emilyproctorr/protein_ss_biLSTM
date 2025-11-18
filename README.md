# Protein Secondary Structure Prediction using Bidirectional LSTM

Authors: <i>Emily Proctor and David Noblett</i>

A PyTorch implementation of a bidirectional LSTM used to predict per-residue secondary structure assignment given primary amino acid sequence. The model predicts either 3-class (general) or 9-class (specific) secondary structure assignment using DSSP (Database of Secondary Structure Assignments) labels. This model supports both training and inference of protein sequences.

---

### Configurable parameters:
- batch size
- hidden dimension size
- number of layers
- learning rate
- number of training epochs
- dropout rate

### Training

- Training uses a curated dataset of ~25,000 protein sequences with residue-level DSSP secondary structure labels.
- The sequences are processed into tensors with padding and masking to account for sequences of variable length.
- Early stopping is used to halt training when validation performance stops improving.
- The model with the best validation balanced accuracy is restored and saved in weights file.
- Jobs were executed on an HPC cluster using SLURM.

Example SLURM script to run training for multiple combinations of hyperparaparameters: [run_scripts/run_lstm.sh](run_scripts/run_lstm.sh)

**Example training command:**
<pre>
python3 LSTM.py --mode "train" \
        --batch_size 32 \
        --hidden_dim 256 \
        --num_layers 3 \
        --learning_rate 0.0005 \
        --num_epochs 150 \
        --dropout 0.2 \
        --num_outputs 3 \
        --seed 42 \
        --output_folder "outdir"
</pre>

### Inference

- Inference can be performed using any previously trained model checkpoint.
- Given a FASTA file, the model outputs per-residue predicted secondary structure in FASTA format (single-line) using either 3-class or 9-class DSSP labeling for each protein sequence.

Example FASTA input file: [data/input_sequences.fasta](data/input_sequences.fasta)

**Example inference command:**
<pre>
python3 LSTM.py --mode "inference" \
        --batch_size 32 \
        --hidden_dim 256 \
        --num_layers 3 \
        --learning_rate 0.0005 \
        --num_epochs 150 \
        --dropout 0.2 \
        --num_outputs 3 \
        --seed 42 \
        --output_folder "outdir"
</pre>

### Output class information

| **9-class label** | **secondary structure (detailed)**      | **3-class label** | **secondary structure (common)** |
|-------------------|-----------------------------------------|-------------------|----------------------------------|
| H                 | α-helix                                 | H                 | α-helix                          |
| G                 | 3₁₀-helix                               | H                 | α-helix                          |
| I                 | π-helix                                 | H                 | α-helix                          |
| E                 | Extended β-strand (β-sheet)             | S                 | β-sheet                          |
| B                 | Isolated β-bridge                       | S                 | β-sheet                          |
| T                 | Hydrogen-bonded turn                    | R                 | Other (random coil, loop, bend)  |
| S                 | Bend                                    | R                 | Other (random coil, loop, bend)  |
| P                 | Poly-proline II (κ-helix)               | R                 | Other (random coil, loop, bend)  |
| 0                 | Unknown / Loop                          | R                 | Other (random coil, loop, bend)  |

### Output files

| **Mode**    | **File**                | **Description** |
|-------------|-------------------------|-----------------|
| Training    | `training_log`          | Per-epoch training & validation metrics (loss, balanced accuracy, precision, recall, F1) |
|             | `test_results.txt`      | Final test set performance |
|             | `lstm_weights.pt`       | Saved model weights of final epoch|
| Inference   | `lstm_predictions.fasta` | FASTA file containing predicted per-residue secondary structure labels |

Example output folder with above files: [results/model_output_folder/](results/model_output_folder/)

### Example performance

The dataset of ~25,000 sequences was split into 80% training, 10% validation, and 10% testing. The data below shows representative performance from one trained model configuration (not an exhaustive hyperparameter search) and included as an example of model behavior.

| Split | Balanced Accuracy | Precision | Recall | F1 |
|-------|------------------|-----------|--------|----|
| Train | 0.94 | 0.95 | 0.95 | 0.95 |
| Validation | 0.86 | 0.86 | 0.86 | 0.86 |
| Test | 0.85 | 0.86 | 0.86 | 0.86 |

Raw data files: [results/model_output_folder/](results/model_output_folder/)

### Plotting

Training and validation metrics were saved at every epoch and visualized to evaluates model performance. I have included a script that constructs plots comparing key metrics, including loss, balanced accuracy, precision, and recall, between the training and validation sets over each epoch.

Plotting script: [src/plot_training.py](src/plot_training.py)   
Example plot: [results/train_val_curves.pdf](results/train_val_curves.pdf)


