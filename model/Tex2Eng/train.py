import os
import argparse
from tqdm import tqdm
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data.my_dataset import MyDataset
from model.Tex2Eng.translator import Tex2Eng

def parse_args():
    parser = argparse.ArgumentParser(description='Train the Tex2Eng model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='train')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_epoch', type=int, help='Epoch to resume training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()

def collate_fn(batch):
    return {k: [item[k] for item in batch] for k, _ in batch[0].items()}

def train(args):
    tokenizer = AutoTokenizer.from_pretrained('aaai25withanonymous/MathBridge_T5_small')

    train_dataset = MyDataset(split=args.dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    model = Tex2Eng(model_name='google-t5/t5-small', tokenizer=tokenizer).to(args.device)
    if args.resume_epoch:
        path = os.path.join(args.checkpoint_dir, f'Tex2Eng_epoch_{args.resume_epoch}.pth')
        
        checkpoint = torch.load(path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
                
        model.load_state_dict(state_dict)
                

    if args.device == 'cuda' and args.num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    if args.resume_epoch and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print('\nTraining started...')
    print(f'Using device: {args.device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Dataset size: {len(train_dataset)}')

    model.train()
    start_epoch = args.resume_epoch if args.resume_epoch else 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        total_loss = 0.0
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{start_epoch + args.epochs}", disable=args.verbose, unit='batch'):
            inputs = tokenizer(batch['equation'], padding=True, truncation=True, return_tensors="pt")
            targets = tokenizer(batch['spoken_English'], padding=True, truncation=True, return_tensors="pt")
            loss, _ = model(
                input_ids=inputs['input_ids'].to(args.device),
                attention_mask=inputs['attention_mask'].to(args.device),
                labels=targets['input_ids'].to(args.device),
                decoder_attention_mask=targets['attention_mask'].to(args.device)
            )

            if args.device == 'cuda' and args.num_gpus > 1:
                loss = loss.mean()  # Average loss across GPUs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if args.verbose:
                print(f'Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{start_epoch + args.epochs}, Avg Loss: {avg_loss:.4f}')

        checkpoint_path = os.path.join(args.checkpoint_dir, f'Tex2Eng_epoch_{epoch+1}.pth')
        checkpoint_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
        }
        torch.save(checkpoint_dict, checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

if __name__ == '__main__':
    args = parse_args()
    train(args)
    print('\nTraining completed.')