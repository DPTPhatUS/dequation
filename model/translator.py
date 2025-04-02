from torch import nn
from transformers import T5ForConditionalGeneration

class TeX2Eng(nn.Module):
    def __init__(self, model_name: str, tokenizer):
        super(TeX2Eng, self).__init__()
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