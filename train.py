import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# Create results folder
os.makedirs("results", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("quora.csv")


df = df[['question1', 'question2', 'is_duplicate']].dropna()

df = df.sample(5000, random_state=42)

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    list(zip(df['question1'], df['question2'])),
    df['is_duplicate'].tolist(),
    test_size=0.1
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(data):
    return tokenizer(
        [x[0] for x in data],
        [x[1] for x in data],
        padding=True,
        truncation=True,
        max_length=64
    )

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_strategy="epoch",
    logging_steps=200
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Training...")
trainer.train()

print("✅ Done! Model saved in ./results")