import subprocess

from torch import nn
from transformers import T5ForConditionalGeneration

class Translator:
    def __init__(self):
        pass
    
    def tex_to_eng(self, latex: str) -> str:
        pass

class Tex2Eng(nn.Module):
    def __init__(self, model_name: str, tokenizer):
        super(Tex2Eng, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        
        return outputs.loss, outputs.logits
    
    def generate(self, input_ids, attention_mask, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
class SRE(Translator):
    def __init__(self):
        super().__init__()
        
    def tex_to_eng(self, latex):
        process = subprocess.Popen(
            ["./node_modules/.bin/latex-to-speech", "-d", "clearspeak", "-b", "speech", latex],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"SRE error: {stderr.strip()}")
        
        return stdout.strip()

if __name__ == "__main__":
    latex = r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"
    sre = SRE()
    try:
        english = sre.tex_to_eng(latex)
        print("LaTeX:", latex)
        print("English:", english)
    except Exception as e:
        print("Error:", e)