from torch.utils.data import DataLoader, Dataset

import torch

from llama import LLaMa
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer

import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np

device = 'cpu'
PAD_ID = 2
MAX_SEQ_LEN = 1024
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')

def tokenize(batch):
    return tokenizer(batch['text'])

class MyDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.tokenized_data = dataset.map(tokenize, batched=True, remove_columns=['text'])['input_ids']

    def __getitem__(self, index):
        return self.tokenized_data[index]

    def __len__(self):
        return len(self.tokenized_data)

def collate_fn(batch):
    flatten = []
    for text in batch:
        flatten += text
    target_size = len(batch) * MAX_SEQ_LEN
    if len(flatten) < target_size:
        flatten += [PAD_ID] * (target_size - len(flatten))
    else:
        flatten = flatten[:target_size]
    
    flatten = torch.Tensor(flatten).to(torch.long).view(len(batch), MAX_SEQ_LEN)
    return flatten


def compute_loss(criterion, logits: torch.Tensor, labels: torch.Tensor, pad_id):
    logits = logits.reshape(-1, 32000)
    labels = labels.view(-1)

    loss = criterion(logits, labels, ignore_index=pad_id)
    return loss

def train_epoch(model, criterion, optimizer, train_loader, epoch, pad_id):
    loss_log = []
    model.train()
    
    for data in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        data = data.to(device) # batch_size, seq_len

        optimizer.zero_grad()
        out = model(data) # batch_size, seq_len, vocab_size
        loss = compute_loss(criterion, out[:, :-1], data[:, 1:].clone(), pad_id)
        loss_log.append(loss.item())
        
        loss.backward()
        optimizer.step()

    return loss_log

def train(model, criterion, optimizer, n_epochs, train_loader, pad_id):
    print(model)
    for epoch in range(n_epochs):
        
        train_loss = train_epoch(model, criterion, optimizer, train_loader, epoch, pad_id)
        print(f"Train loss: {np.mean(train_loss)}")


def main():
    login(token="")
    dataset_id = "ashaba1in/small_openwebtext"
    dataset = load_dataset(dataset_id)

    print("Dataset size", len(dataset['train']['text']))
    # print(dataset['train']['text'][0])
    tokenizer_id ="mistralai/Mistral-7B-v0.1"
    

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    cropped_dataset = dataset['train'].select(range(1000)) 
    MAX_SEQ_LEN = 1024
    PAD_ID = tokenizer.eos_token_id


    my_data = MyDataset(cropped_dataset)
    train_dataloader = DataLoader(my_data, batch_size=32, collate_fn=collate_fn) 
    device = 'cpu' 


    vocab_size = tokenizer.vocab_size
        # LLaMA(n_heads, emb_dim, vocab_size, n_layers, norm_eps, max_seq_len, max_batch_size)
    model = LLaMA(8, 768, vocab_size, 3, 1e-8, MAX_SEQ_LEN, 32)
    criterion = F.cross_entropy
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    n_epochs = 5
    train(model, criterion, optimizer, n_epochs, train_dataloader, PAD_ID)

if __name__ == "__main__":
    main()
