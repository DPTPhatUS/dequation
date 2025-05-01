# Test trained model
import torch
from model.Tex2Eng.translator import Tex2Eng
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('aaai25withanonymous/MathBridge_T5_small')
model = Tex2Eng('google-t5/t5-small', tokenizer).to('cuda' if torch.cuda.is_available() else 'cpu')

# Load the state dictionary
state_dict = torch.load('checkpoints/Tex2Eng_epoch_6.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu')
# Remove "module." prefix if necessary
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
model.load_state_dict(state_dict)

sample_input = r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}"
inputs = tokenizer(sample_input, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
outputs = model.generate(
    input_ids=inputs['input_ids'], 
    attention_mask=inputs['attention_mask'],
    max_length=50,
    num_beams=1
)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Input: {sample_input}\nOutput: {decoded_output}")