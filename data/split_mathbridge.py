from datasets import load_dataset, DatasetDict

# Load and split 
dataset = load_dataset('aaai25withanonymous/MathBridge')
train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=2112005)
val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=2112005)
final_dataset = DatasetDict({
    'train': train_val_split['train'],
    'validation': val_test_split['train'],
    'test': val_test_split['test']
})

print(final_dataset)

# Push to Huggingface
final_dataset.push_to_hub('Last-Bullet/MathBridge_Splitted')