import torch
import cv2 as cv
import argparse
from model.TexTeller.src.models.ocr_model.model.TexTeller import TexTeller
from model.TexTeller.src.models.ocr_model.utils.to_katex import to_katex
from model.TexTeller.src.models.ocr_model.utils.inference import inference as latex_inference
from model.Tex2Eng.translator import SRE, Tex2Eng
from transformers import AutoTokenizer
import regex

def remove_boldsymbol(text):
    pattern = r'\\boldsymbol\s*\{((?:[^{}]+|(?R))*)\}'
    while regex.search(pattern, text):
        text = regex.sub(pattern, r' \1 ', text)

    return text

def parse_args():
    parser = argparse.ArgumentParser(description='Dequate an image')
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--translator', type=str, choices=['tex2eng', 'sre'], default='tex2eng')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()

def main(args):
    print("Using device: ", args.device)
    print("Loading models...")
    latex_rec_model = TexTeller.from_pretrained()
    
    img_path = args.img_path
    img = cv.imread(img_path)
    
    print("Generating LaTeX...")
    res = latex_inference(latex_rec_model, TexTeller.get_tokenizer(), [img], args.device, 4)
    latex = to_katex(res[0])
    latex = remove_boldsymbol(latex)
    print(latex)
    
    print("Translating...")
    if args.translator == 'sre':
        translator = SRE()
        speech = translator.tex_to_eng(latex)
    else:
        tokenizer = AutoTokenizer.from_pretrained('aaai25withanonymous/MathBridge_T5_small')
        
        checkpoint_dict = torch.load('checkpoints/t5-small_latex-spoken/Tex2Eng_epoch_9.pth', map_location=args.device)
        if isinstance(checkpoint_dict, dict) and 'model_state_dict' in checkpoint_dict:
            state_dict = checkpoint_dict['model_state_dict']
        else:
            state_dict = checkpoint_dict

        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

        model = Tex2Eng('google-t5/t5-small', tokenizer).to(args.device)
        model.load_state_dict(state_dict)
        
        model.eval()
        with torch.no_grad():
            input = tokenizer(latex, return_tensors='pt')
            output = model.generate(
                input_ids=input['input_ids'].to(args.device),
                attention_mask=input['attention_mask'].to(args.device),
                max_length=512,
                num_beams=4,
            )
            speech = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            
    print(speech)

if __name__ == "__main__":
    args = parse_args()
    main(args)