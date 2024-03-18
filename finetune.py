import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DebertaForSequenceClassification, DebertaTokenizerFast, AdamW, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# save the model in the cache directory
cache_dir = "huggingface/hub"

# Load tokenizer and dataset
tokenizer = DebertaTokenizerFast.from_pretrained('./data/debertaTokenizer')

class LipoDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(filepath, 'r') as file:
            next(file)  # Skip header
            for line in file:
                mol = line.strip().split(',')
                selfies = mol[0]
                prop = mol[-1]
                encoding = tokenizer(selfies, truncation=True, padding='max_length', max_length=max_length, add_special_tokens=True)
                self.data.append((encoding, float(prop)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding, lipo = self.data[idx]
        return {'input_ids': torch.tensor(encoding['input_ids']),
                'attention_mask': torch.tensor(encoding['attention_mask']),
                'labels': torch.tensor(lipo)}

# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 50

# Load dataset
dataset = LipoDataset('data/esol.csv', tokenizer, max_length=MAX_LENGTH)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.3, random_state=1337)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define model and optimizer
model_pretrained = DebertaForSequenceClassification.from_pretrained('results/checkpoint-3900', num_labels=1).cuda()  # Regression head
optimizer_pretrained = AdamW(model_pretrained.parameters(), lr=LEARNING_RATE)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results-finetune',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=100,
    save_total_limit=1,
    overwrite_output_dir=True,
)

# Define a function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    mse = mean_squared_error(labels, preds)
    return {'mse': mse}

# Trainer for fine-tuning with pretrained weights
trainer_pretrained = Trainer(
    model=model_pretrained,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Pass the compute_metrics function here
)

# Fine-tune with pretrained weights
trainer_pretrained.train()


# Fine-tune without pretrained weights
model_no_pretrain = DebertaForSequenceClassification(config=DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=1).config).cuda()  # Regression head
optimizer_no_pretrain = AdamW(model_no_pretrain.parameters(), lr=LEARNING_RATE)

# Trainer for fine-tuning without pretrained weights
trainer_no_pretrain = Trainer(
    model=model_no_pretrain,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Pass the compute_metrics function here
)

trainer_no_pretrain.train()

# Evaluate models
metrics_pretrained = trainer_pretrained.evaluate()
metrics_no_pretrain = trainer_no_pretrain.evaluate()

# Plotting
labels = ['Pretrained', 'No Pretrained']
mse_values = [metrics_pretrained['eval_mse'], metrics_no_pretrain['eval_mse']]

plt.bar(labels, mse_values, color=['blue', 'green'])
plt.ylabel('Mean Squared Error')
plt.title('Comparison of Models with and without Pretraining')
plt.savefig('comparison.png')
