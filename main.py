import sentencepiece as spm
from datasets import load_dataset
from data.my_datasets import MathBridge

# Prepare a text file for training
dataset = MathBridge(split='train')
with open("t5_tokenizer_data.txt", "w", encoding="utf-8") as f:
    for example in dataset:
        f.write(example["equation"] + "\n")  # Adjust field as needed

# Train SentencePiece model (UnigramLM by default)
spm.SentencePieceTrainer.Train(
    input='t5_tokenizer_data.txt',
    model_prefix='tokenizer',
    vocab_size=32128,  # T5 default
    character_coverage=1.0,
    model_type='unigram',
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)