from transformers import (AutoModelForMaskedLM,DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, DebertaTokenizerFast )
from torch.utils.data import Dataset, DataLoader

# save the model in the cache directory
cache_dir = "huggingface/hub"

model = AutoModelForMaskedLM.from_pretrained('microsoft/deberta-base',  cache_dir=cache_dir).cuda()
tokenizer = DebertaTokenizerFast.from_pretrained('./data/debertaTokenizer')

# Define a custom Dataset class to load and preprocess the data
class LipoPreTrainDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        with open(file_path, 'r') as file:
            self.lines = file.readlines()
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.lines)
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.lines[idx], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {key: tensor[0] for key, tensor in encoding.items()}

# Load data
dataset_path = "./data/pretrain.txt"  # Specify the path to your dataset
dataset = LipoPreTrainDataset(dataset_path, tokenizer)

# DataLoader
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=data_collator)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=64,
    save_steps=100,
    save_total_limit=1,
    resume_from_checkpoint=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Training
trainer.train()
