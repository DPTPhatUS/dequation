import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import argparse

from data.my_datasets import MathBridge
from model.Tex2Eng.translator import Tex2Eng
from utils.bleu_score import CorpusBLEU

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the Tex2Eng model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='validation[:128]')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_path', type=str, default='checkpoints/Tex2Eng_epoch_1.pth')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--early_stopping', action='store_true')
    return parser.parse_args()

def collate_fn(batch, tokenizer, device):
    inputs = [item['equation'] for item in batch]
    targets = [item['spoken_English'] for item in batch]

    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
    targets = tokenizer(targets, padding=True, truncation=True, return_tensors='pt')

    inputs = {k: v.to(device) for k, v in inputs.items()}
    targets = {k: v.to(device) for k, v in targets.items()}

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids'],
        'decoder_attention_mask': targets['attention_mask']
    }

def evaluate(model, tokenizer, dataloader, max_length, num_beams, early_stopping):
    bleu = CorpusBLEU()

    model.eval()
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(dataloader):
            outputs = model.generate(
                input_ids=data_dict['input_ids'],
                attention_mask=data_dict['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping
            )

            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            target_text = tokenizer.batch_decode(data_dict['labels'], skip_special_tokens=True)

            for gen, target in zip(generated_text, target_text):
                bleu.add(tokenizer.tokenize(gen), [tokenizer.tokenize(target)])

            if batch_idx % 1 == 0:
                print(f'Batch [{batch_idx+1}/{len(dataloader)}], Predicted: {generated_text}, Target: {target_text}')

    score = bleu.compute()
    print(f'\nBLEU score: {score:.4f}')

if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained('aaai25withanonymous/MathBridge_T5_small')

    dataset = MathBridge(split=args.dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.device)
    )

    print('Starting evaluation...')
    print(f'Device: {args.device}, Dataset size: {len(dataset)}')
    print(f'Model path: {args.model_path}')

    model = Tex2Eng('google-t5/t5-small', tokenizer).to(args.device)

    # Load the state dictionary
    state_dict = torch.load(args.model_path, map_location=args.device)
    # Remove "module." prefix if necessary
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    evaluate(model, tokenizer, dataloader, args.max_length, args.num_beams, args.early_stopping)