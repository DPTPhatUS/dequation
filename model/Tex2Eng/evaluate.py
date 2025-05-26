import os
import re
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from data.my_dataset import MyDataset
from model.Tex2Eng.translator import Tex2Eng

from metrics.bleu import CorpusBLEU
from metrics.meteor import METEOR

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the Tex2Eng model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='validation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    return parser.parse_args()


def collate_fn(batch):
    return {k: [item[k] for item in batch] for k, _ in batch[0].items()}

def extract_epoch(filename):
    match = re.search(r'Tex2Eng_epoch_(\d+)\.pth', filename)
    if match:
        return int(match.group(1))
    else:
        return -1

def evaluate(args):
    tokenizer = AutoTokenizer.from_pretrained('aaai25withanonymous/MathBridge_T5_small')

    dataset = MyDataset(split=args.dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print('\nStarting evaluation...')
    print(f'\nDevice: {args.device}, Dataset size: {len(dataset)}')

    checkpoints = [file for file in os.listdir(args.checkpoint_dir) if file.startswith('Tex2Eng_epoch_') and file.endswith('.pth')]
    checkpoints.sort(key=extract_epoch, reverse=True)
    if not checkpoints:
        print(r'No checkpoints Tex2Eng_epoch_{i}.pth found')
        exit()
    print('Checkpoints: ', checkpoints)

    for checkpoint in checkpoints:
        path = os.path.join(args.checkpoint_dir, checkpoint)

        state_dict = torch.load(path, map_location=args.device)
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

        model = Tex2Eng('google-t5/t5-small', tokenizer).to(args.device)
        model.load_state_dict(state_dict)

        bleu = CorpusBLEU()
        meteor = METEOR()

        print('-'*20)
        print(f'Evaluating: {checkpoint}')

        model.eval()
        with torch.no_grad():
            for data_dict in tqdm(dataloader, disable=args.verbose, unit='batch'):
                inputs = tokenizer(data_dict['equation'], padding=True, truncation=True, return_tensors="pt")
                targets = tokenizer(data_dict['spoken_English'], padding=True, truncation=True, return_tensors="pt")
                outputs = model.generate(
                    input_ids=inputs['input_ids'].to(args.device),
                    attention_mask=inputs['attention_mask'].to(args.device),
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    early_stopping=args.early_stopping
                )

                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                target_text = tokenizer.batch_decode(targets['input_ids'], skip_special_tokens=True)

                for gen, target in zip(generated_text, target_text):
                    hyp = gen.split()
                    refs = [target.split()]

                    bleu.add(hyp, refs)
                    meteor.add(hyp, refs)

                if args.verbose:
                    print(f'Predicted: {generated_text}, Target: {target_text}')

        bleu_score = bleu.compute()
        meteor_score = meteor.compute()
        print(f'BLEU: {bleu_score:.4f}, METEOR: {meteor_score:.4f}')
        
        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
    print('\nEvaluation completed.')