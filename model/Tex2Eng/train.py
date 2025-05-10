import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data.my_datasets import MathBridge
from model.Tex2Eng.translator import Tex2Eng

def parse_args():
    parser = argparse.ArgumentParser(description='Train the Tex2Eng model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='train[:1000]')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
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
    tokenizer = AutoTokenizer.from_pretrained('aaai25withanonymous/MathBridge_T5_small')

    train_dataset = MathBridge(split=args.dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.device)
    )

    model = Tex2Eng(model_name='google-t5/t5-small', tokenizer=tokenizer).to(args.device)
    if args.device == 'cuda' and args.num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print('\nTraining started...')
    print(f'Using device: {args.device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Dataset size: {len(train_dataset)}')

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
        print(f'Epoch [{epoch+1}/{args.epochs}], Avg Loss: {avg_loss:.4f}')

        checkpoint_path = os.path.join(args.checkpoint_dir, f'Tex2Eng_epoch_{epoch+1}.pth')
        state_dict = model.state_dict()
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        torch.save(state_dict, checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

if __name__ == '__main__':
    args = parse_args()
    train(args)
    print('\nTraining completed.')