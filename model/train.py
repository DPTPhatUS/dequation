import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data.my_datasets import MathBridge
from model.translator import TeX2Eng

def parse_args():
    parser = argparse.ArgumentParser(description='Train the TeX2Eng model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--train_dataset', type=str, default='train[:1000]')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    return parser.parse_args()

def collate_fn(batch, tokenizer, device):
    inputs = [item['equation'] for item in batch]
    targets = [item['spoken_English'] for item in batch]

    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
    targets = tokenizer(targets, padding=True, truncation=True, return_tensors='pt')

    return {
        'input_ids': inputs['input_ids'].to(device),
        'attention_mask': inputs['attention_mask'].to(device),
        'labels': targets['input_ids'].to(device),
        'decoder_attention_mask': targets['attention_mask'].to(device)
    }

def train(args):
    print('Training started...')
    print(f'Using device: {args.device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Dataset split: {args.train_dataset}')

    tokenizer = AutoTokenizer.from_pretrained('aaai25withanonymous/MathBridge_T5_small')
    train_dataset = MathBridge(split=args.train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.device)
    )

    model = TeX2Eng('google-t5/t5-small', tokenizer).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            loss, _ = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                decoder_attention_mask=batch['decoder_attention_mask']
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.epochs}] finished with Avg Loss: {avg_loss:.4f}')

        checkpoint_path = os.path.join(args.checkpoint_dir, f'tex2eng_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

if __name__ == '__main__':
    args = parse_args()
    train(args)
    print('Training completed.')