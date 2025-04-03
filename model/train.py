import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data.my_datasets import MathBridge
from model.translator import TeX2Eng
from utils.bleu_score import bleu_score
import os

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained('aaai25withanonymous/MathBridge_T5_small')

def collate_fn(batch):
    inputs = [item['equation'] for item in batch]
    targets = [item['spoken_English'] for item in batch]

    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
    targets = tokenizer(targets, padding=True, truncation=True, return_tensors='pt')

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    targets = {k: v.to(DEVICE) for k, v in targets.items()}

    collated = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids'],
        'decoder_attention_mask': targets['attention_mask']
    }

    return collated

train_dataset = MathBridge(split='train[:1000]')
val_dataset = MathBridge(split='validation[:125]')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = TeX2Eng('google-t5/t5-small', tokenizer).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def train():
    print('Training started...')
    print(f'Using device: {DEVICE}')
    print(f'Batch size: {BATCH_SIZE}')
    print(f'Learning rate: {LEARNING_RATE}')
    print(f'Number of epochs: {EPOCHS}')
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_idx, data_dict in enumerate(train_loader):
            loss, _ = model(
                input_ids=data_dict['input_ids'],
                attention_mask=data_dict['attention_mask'],
                labels=data_dict['labels'],
                decoder_attention_mask=data_dict['decoder_attention_mask']
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        lr_scheduler.step()

        print(f'Epoch [{epoch+1}/{EPOCHS}], Average Loss: {running_loss / len(train_loader):.4f}')

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')

        evaluate()

def evaluate():
    print('Evaluating...')

    model.eval()
    with torch.no_grad():
        # BLEU score for the entire validation set
        all_predicted = []
        all_targets = []
        for batch_idx, data_dict in enumerate(val_loader):
            outputs = model.generate(
                input_ids=data_dict['input_ids'],
                attention_mask=data_dict['attention_mask'],
                max_length=128,
                num_beams=10,
                early_stopping=True
            )

            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            target_text = tokenizer.batch_decode(data_dict['labels'], skip_special_tokens=True)

            all_predicted.extend(generated_text)
            all_targets.extend(target_text)

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'Batch [{batch_idx+1}/{len(val_loader)}], Predicted: {generated_text}, Target: {target_text}')

        bleu = bleu_score(all_predicted, all_targets)
        print(f'Overall BLEU score: {bleu:.4f}')
            
if __name__ == '__main__':
    train()