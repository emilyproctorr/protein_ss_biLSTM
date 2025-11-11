import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

# command line arguments
parser = argparse.ArgumentParser(description='Train an bidirectional LSTM model for protein secondary structure prediction.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for LSTM.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs for training.')
parser.add_argument('--log_file', type=str, default="training_log")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for LSTM layers.')
parser.add_argument('--num_outputs', type=int, default=9, help='Number of output classes for DSSP labels.')
args = parser.parse_args()


########## DATASET HANDLING ##########

# This section loads the dataset from a json file. The json file contains protein sequences and their corresponding secondary structure labels.
# The sequences are converted to indices based on a predefined mapping, and the labels are converted to indices based on the DSSP labels.

# The AminoAcidDataset class is used to create a custom dataset for amino acid sequences and their corresponding labels.

# The pad_and_mask function pads the sequences and labels to the same length.
# It also generates a mask for each sequence to indicate which positions are valid (non-padded) in the sequences and labels.

######################################

# load and process the dataset
def read_dataset_file():
    with open('sequences_test_25000.json', 'r') as file: 
        ss_dataset = json.load(file)
    
    print(f"Loaded {len(ss_dataset)} sequences from the dataset.\n", flush=True)

    sequences = []
    labels = []
    for id, info_dict in ss_dataset.items():
        seq = info_dict['sequence']
        ss = info_dict['ss']

        # make sure sequence and label are same length
        if len(seq) != len(ss): 
            print(f"Sequence and label length mismatch in {id}\n", flush=True)
            continue 

        # make sure all amino acids and labels are valid
        if not all(aa in index_to_aa for aa in seq) or not all(label in dssp_to_index for label in ss):
            print(f"Invalid sequence or label in {id}\n", flush=True)
            continue

        seq_indices = [index_to_aa[aa] for aa in seq] # each amino acid is converted to index value
        ss_indices = [dssp_to_index[label] for label in ss] # each dssp label is converted to index value
        
        if seq_indices not in sequences:
            sequences.append(seq_indices) # store sequence
            labels.append(ss_indices) # store dssp labeled sequence
    
    print(f"Processed {len(sequences)} valid sequences and labels.\n", flush=True)

    return sequences, labels

# custom dataset class for amino acid sequences and labels
class AminoAcidDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    
# pad sequences and generate masks
def pad_and_mask(batch):
    sequences, labels = zip(*batch) # unzip the batch into sequences and labels, separating the list of sequences and list of labels into two separate variables
    lengths = [len(seq) for seq in sequences] # list of lengths of each sequence
    max_len = max(lengths) # find the maximum length of the sequences in the batch

    padded_seqs = torch.full((len(sequences), max_len), 0, dtype=torch.long)
    padded_labels = torch.full((len(labels), max_len), 0, dtype=torch.long)
    mask = torch.zeros(len(labels), max_len, dtype=torch.bool) # tensor will act as a mask, indicating which positions in the sequences and labels are valid (True) and which are padded (False)

    for index, (seq, label) in enumerate(zip(sequences, labels)):
        length = len(seq)
        padded_seqs[index, :length] = seq # copies the sequence into the correct row of padded tensor up to actual seq length
        padded_labels[index, :length] = label # same as above but for labels
        mask[index, :length] = 1 # set the mask to 1 for valid (non padded) positions)

    return padded_seqs.to(device), padded_labels.to(device), mask.to(device) # move all tensors to correct device

########## METRIC COMPUTATION ##########

# This section defines a function to compute various evaluation metrics for the model's predictions.

########################################

def compute_metrics(preds, labels):

    return {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
        "f1": f1_score(labels, preds, average="micro", zero_division=0),
    }

########## MODEL DEFINITION ##########

# This section defines the birdirecitonal LSTM model architecture for protein secondary structure prediction.

######################################

class LSTMModel(nn.Module): # inheriting from pytorch nn.Module class, class represents a neural network model based on an LSTM (Long Short-Term Memory) architecture
    def __init__(self, 
                 input_dim, # size of the input vocab (number of amino acids)
                 hidden_dim, # size of the hidden state in the LSTM
                 output_dim, # size of the output vocab (number of DSSP labels)
                 num_layers, # number of LSTM layers
                 dropout): # dropout rate
        super().__init__() # call the constructor of the parent class (nn.Module)

        # embedding layer to convert input indices to dense vectors
            # hidden_dim = size of the embedding vectors
            # discrete -> continuous, each row represents an amino acid
            # randomly initialized during training
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0) # padding_idx is the index of the padding token (0), so that the embedding for padding is a zero vector, cant be negative bc you are specifying where it is in tensor
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True) # batch_first=True ensures that the input and output tensors have the shape [batch_size, seq_length, hidden_dim]
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # FCL to convert hidden dimension output to num output classes, *2 because bidirectional LSTM

    def forward(self, x): # x = [batch_size, seq_length]
        x = self.embedding(x) # convert indices to dense vectors
        lstm_out, _ = self.lstm(x) # pass through LSTM layer, lstm_out = [batch_size, seq_length, hidden_dim]
        output = self.fc(lstm_out) # pass through fully connected layer, output = [batch_size, seq_length, output_dim]
        return output
    
########## TRAINING AND EVALUATION ##########

# This section defines the training and evaluation process for the LSTM model.
# The train_model function trains the model on the training dataset over multiple epochs and evaluates the model on the validation dataset.
# The evaluate_model function evaluates the model on the test dataset and computes metrics.

#############################################

def train_model(batch_size=32, hidden_dim=64, num_layers=2, learning_rate=0.001, num_epochs=150, dropout=0.2, end_training_threshold=5, num_outputs=9):
    sequences, labels = read_dataset_file() # two lists -> sequences and labels

    # split dataset into train (80%) and temp (20%)
    train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    # split temp dataset into validation (50%) and test (50%)
    val_seqs, test_seqs, val_labels, test_labels = train_test_split(temp_seqs, temp_labels, test_size=0.5, random_state=42)
    
    train_dataset = AminoAcidDataset(train_seqs, train_labels)
    val_dataset = AminoAcidDataset(val_seqs, val_labels)
    test_dataset = AminoAcidDataset(test_seqs, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_and_mask)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_and_mask)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_and_mask)

    model = LSTMModel(input_dim=num_aa, hidden_dim=hidden_dim, output_dim=num_dssp, num_layers=num_layers, dropout=dropout).to(device)
    loss_func = nn.CrossEntropyLoss() # loss function for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # update weights using gradient descent

    log_file = open(f"{args.log_file}", "w")
    log_file.write(f"batch_size: {train_dataloader.batch_size}, num_epochs: {num_epochs}, num_layers: {num_layers}, hidden_dim: {hidden_dim}, learning_rate: {learning_rate}, dropout: {dropout}, num_outputs: {num_outputs}\n")
    log_file.flush()

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    # fore each epoch
    for epoch in range(num_epochs):
        
        start_time = time.time()

        model.train() # activate training mode
        total_loss = 0.0 # initialize total loss for the epoch
        all_preds = []
        all_labels = []

        # TRAINING STEP
        for batch_sequences, batch_labels, mask in train_dataloader:
            batch_sequences, batch_labels, mask = batch_sequences.to(device), batch_labels.to(device), mask.to(device)
            optimizer.zero_grad() # clear gradients
            outputs = model(batch_sequences)  # forward pass, outputs = [batch_size, seq_length, num_dssp]
            outputs = outputs.view(-1, num_dssp) # reshape to [batch_size * seq_length, num_dssp] for loss calculation (flattens)
            batch_labels = batch_labels.view(-1)
            mask = mask.view(-1)

            outputs = outputs[mask] # apply mask to filter out the padded positions
            batch_labels = batch_labels[mask]

            loss = loss_func(outputs, batch_labels) # calculate loss
            loss.backward() # compute gradients
            optimizer.step() # update weights

            total_loss += loss.item() 

            _, predicted = torch.max(outputs, dim=1)  # get the predicted class (index of max logit)
            all_preds.extend(predicted.cpu().numpy()) # store predictions
            all_labels.extend(batch_labels.cpu().numpy()) # store labels

        avg_loss = total_loss / len(train_dataloader) # calculate average loss for the epoch
        train_metrics = compute_metrics(all_preds, all_labels) # calculate training metrics
        end_time = time.time()
        
        # VALIDATION STEP
        model.eval()
        val_all_preds = []
        val_all_labels = []
        with torch.no_grad(): # no gradient calculation for validation
            for val_sequences, val_labels, val_mask in val_dataloader:
                val_sequences, val_labels, val_mask = val_sequences.to(device), val_labels.to(device), val_mask.to(device)
                val_outputs = model(val_sequences)
                val_outputs = val_outputs.view(-1, num_dssp) # reshape to [batch_size * seq_length, num_dssp]
                val_labels = val_labels.view(-1) # flatten labels
                val_mask = val_mask.view(-1) # apply mask to filter out the padded positions

                val_outputs = val_outputs[val_mask] # apply mask to filter out the padded positions
                val_labels = val_labels[val_mask]

                _, val_predicted = torch.max(val_outputs, dim=1) # get the predicted class (index of max logit)
                val_all_preds.extend(val_predicted.cpu().numpy()) # store predictions
                val_all_labels.extend(val_labels.cpu().numpy()) # store labels

        val_metrics = compute_metrics(val_all_preds, val_all_labels) # calculate validation metrics
        val_bal_acc = val_metrics['balanced_accuracy'] # grab balanced accuracy score

        # log training and validation metrics
        log_file.write(
            f"Epoch {epoch+1}/{num_epochs}, Time: {end_time - start_time:.2f}s, Loss: {avg_loss:.4f}\n"
            f"Training Metrics: BalAcc {train_metrics['balanced_accuracy']:.4f}, Prec {train_metrics['precision']:.4f}, Recall {train_metrics['recall']:.4f}, F1 {train_metrics['f1']:.4f}\n"
            f"Validation Metrics: BalAcc {val_metrics['balanced_accuracy']:.4f}, Prec {val_metrics['precision']:.4f}, Recall {val_metrics['recall']:.4f}, F1 {train_metrics['f1']:.4f}\n\n"
        )
        log_file.flush()

        # early stopping check
        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= end_training_threshold:
                print(f"Early stopping triggered at epoch {epoch+1}. Best Val Balanced Acc: {best_val_acc:.4f}")
                break
    
    log_file.close()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, test_dataloader

# TESTING STEP
def evaluate_model(model, dataloader, loss_func, device, num_dssp):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    # same process as above but no gradient calculation
    with torch.no_grad():
        for sequences, labels, mask in dataloader:
            sequences, labels, mask = sequences.to(device), labels.to(device), mask.to(device) 
            outputs = model(sequences)
            outputs = outputs.view(-1, num_dssp)
            labels = labels.view(-1)
            mask = mask.view(-1)

            outputs = outputs[mask]
            labels = labels[mask]

            loss = loss_func(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics

# main function to run the training and evaluation    
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n", flush=True)

    aa_options = "ACDEFGHIKLMNPQRSTVWY"
    index_to_aa = {aa: index + 1 for index, aa in enumerate(aa_options)}
    index_to_aa['PAD'] = 0
    num_aa = len(aa_options) + 1

    num_outputs = args.num_outputs

    # common secondary structure labels
    if num_outputs == 3:
        dssp_to_index = {
            'H': 0, 'G': 0, 'I': 0,
            'E': 1, 'B': 1,
            'T': 2, 'S': 2, 'P': 2, '0': 2
        }
        num_dssp = 3
    # dssp labels -> more detailed classes
    elif num_outputs == 9:
        dssp_to_index = {'H': 0, "G": 1, "I": 2, "E": 3, "B": 4, "T": 5, "S": 6, "P": 7, "0": 8}
        num_dssp = 9

    # train the model
    model, test_dataloader = train_model(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        dropout=args.dropout,
        num_outputs=args.num_outputs
    )

    # evaluate model
    loss_func = nn.CrossEntropyLoss()
    test_loss, test_metrics = evaluate_model(model, test_dataloader, loss_func, device, num_dssp)

    # log final test results
    log_file = open(f"{args.log_file}", "a")

    log_file.write(
        f"\nFinal Test Evaluation:\n"
        f"Test Loss: {test_loss:.4f}\n"
        f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}\n"
        f"Precision: {test_metrics['precision']:.4f}\n"
        f"Recall: {test_metrics['recall']:.4f}\n"
        f"F1 Score: {test_metrics['f1']:.4f}\n"
    )
    log_file.close()
