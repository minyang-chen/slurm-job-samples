"""
Financial News Sentiment Analysis with BERT

Description:
    This script trains a BERT model for sentiment analysis on financial news data. The sentiments can be positive, neutral, or negative. After training, the script evaluates the model on a test set and prints out the accuracies.

Usage:
    This script is designed to be run via the Slurm scheduler on one GPU. 
    Sample Slurm command:

    sbatch train-finbert-job.slurm

Dependencies:
    - torch
    - transformers
    - pytorch_lightning
    - sklearn

Author:
    Natalya Rapstine, nrapstin@stanford.edu

Date of Creation:
   Oct. 10, 2023

Last Modified:
   Oct. 16, 2023 

Notes:
    Ensure you have the required dependencies installed and have set up the data path correctly.
    The script leverages GPU acceleration if available, else falls back to CPU.
"""

import torch, time, sys
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification 
from torch.optim import AdamW
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score

class FinancialNewsDataset(Dataset):
    """
    Define a dataset
    """ 
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class NewsClassifier(pl.LightningModule):
    """
    Define a PyTorch Lightning model
    """
    def __init__(self):
        super(NewsClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3) # 3 labels: Positive, Neutral & Negative
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.loss = torch.nn.CrossEntropyLoss()
        self.outputs = []
        self.test_outputs = []

    def forward(self, ids, mask):
        return self.model(ids, mask)[0]

    def training_step(self, batch, batch_nb):
        ids, mask, labels = batch['ids'], batch['mask'], batch['label']
        outputs = self(ids, mask)
        return {'loss': self.loss(outputs, labels)}


    def validation_step(self, batch, batch_nb):
        ids, mask, labels = batch['ids'], batch['mask'], batch['label']
        outputs = self(ids, mask)
        loss = self.loss(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        val_acc = torch.tensor(accuracy_score(predicted.cpu(), labels.cpu()), dtype=torch.float32)
        self.log('val_loss', loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=False, on_step=True, on_epoch=True)
        # Store outputs in an instance attribute
        output = {'val_loss': loss, 'val_acc': val_acc}
        self.outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.outputs]).mean()
        print(f"Epoch {self.current_epoch} - avg_val_loss: {avg_loss:.4f}, avg_val_acc: {avg_acc:.4f}")
        # Clear the stored outputs for the next validation run
        self.outputs = []

    def test_step(self, batch, batch_nb):
        ids, mask, labels = batch['ids'], batch['mask'], batch['label']
        outputs = self(ids, mask)
        _, predicted = torch.max(outputs, 1)
        test_acc = torch.tensor(accuracy_score(predicted.cpu(), labels.cpu()))
        self.log('test_acc', test_acc, prog_bar=False)
        # Store outputs in an instance attribute
        output = {'test_acc': test_acc}
        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        avg_test_acc = torch.stack([x['test_acc'] for x in self.test_outputs]).mean()
        print(f"Test accuracy: {avg_test_acc:.4f}")
        # Clear the stored outputs for the next test run
        self.test_outputs = []

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=2e-5)


def read_financial_phrasebank(file_path):
    """
    Read and Process the Dataset
    """
    # file was saved in'ISO-8859-1' encoding (also known as 'latin1')
    with open(file_path, 'r', encoding = 'ISO-8859-1') as f:
        lines = f.readlines()
    
    sentences, sentiments = [], []
    for line in lines:
        # split sentence and its label
        # The sentence and sentiment are separated with @ symbol
        sentence, sentiment = line.split("@")
        sentences.append(sentence.strip())
        sentiments.append(sentiment.strip())
    
    # Convert sentiments to integers: 0-negative, 1-neutral, 2-positive
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    labels = [label_mapping[s] for s in sentiments]

    return sentences, labels

def split_data(texts, labels, train_ratio = 0.7, val_ratio = 0.2):
    """
    Split Data into Train, Validation, and Test
    """
    total_size = len(texts)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Randomly split data
    data = list(zip(texts, labels))
    train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size], )
    
    train_texts, train_labels = zip(*train_data)
    val_texts, val_labels = zip(*val_data)
    test_texts, test_labels = zip(*test_data)
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# 1. Check for GPU availability and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# accept command line arguments
# set the number of workers to use in DataLoader here from the command line.
NUM_WORKERS = int(sys.argv[1])

print("DEVICE:",device)
print("NUM_WORKERS:",NUM_WORKERS)

# 2. Read and Process the Dataset 
# Load the dataset
file_path = f"Sentences_AllAgree.txt"

# Get sentences and labels
texts, labels = read_financial_phrasebank(file_path)

# 3. Split into training / validation / test sets
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(texts, labels)

# 4. Create DataLoaders
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = FinancialNewsDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = FinancialNewsDataset(val_texts, val_labels, tokenizer, max_length=128)
test_dataset = FinancialNewsDataset(test_texts, test_labels, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size = 16, num_workers = NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size = 16, num_workers = NUM_WORKERS)

# 5. Define a BERT model and move to the device
model = NewsClassifier().to(device)

# specify the number of GPUs to use with the gpus argument 
trainer = pl.Trainer(max_epochs = 3, accelerator = 'gpu' if torch.cuda.is_available() else 'cpu', devices = 1 if torch.cuda.is_available() else None)

# 6. Before we train the BERT model, let's evaluate the model's baseline performance on test set
# Evaluate the untrained model on the test set

print(f"Untrainted model accuracy:")
trainer.test(model, test_loader)

# 7. Train the model

start_time = time.time()  # Start timing here

trainer.fit(model, train_loader, val_loader)

end_time = time.time()  # End timing here
print(f"Training time: {end_time - start_time:.2f} seconds")

# 8. Evaluate the trained on test set
start_time = time.time()  # Start timing here

print(f"Trained model accuracy:")
trainer.test(model, test_loader)

end_time = time.time()  # End timing here
print(f"Testing time: {end_time - start_time:.2f} seconds")
print(f"Training and testing complete!")
