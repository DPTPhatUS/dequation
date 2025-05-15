import torch
import cv2 as cv
import argparse
from model.TexTeller.src.models.ocr_model.model.TexTeller import TexTeller
from model.TexTeller.src.models.ocr_model.utils.to_katex import to_katex
from model.TexTeller.src.models.ocr_model.utils.inference import inference as latex_inference
from model.Tex2Eng.translator import SRE
import regex

def remove_boldsymbol(text):
    pattern = r'\\boldsymbol\s*\{((?:[^{}]+|(?R))*)\}'
    while regex.search(pattern, text):
        text = regex.sub(pattern, r' \1 ', text)

    return text

def parse_args():
    parser = argparse.ArgumentParser(description='Dequate an image')
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()

def main(args):
    print("Using device: ", args.device)
    print("Loading models...")
    latex_rec_model = TexTeller.from_pretrained()
    tokenizer = TexTeller.get_tokenizer()
    translator = SRE()
    
    img_path = args.img_path
    img = cv.imread(img_path)
    
    print("Generating LaTeX...")
    res = latex_inference(latex_rec_model, tokenizer, [img], args.device, 4)
    latex = to_katex(res[0])
    latex = remove_boldsymbol(latex)
    print(latex)
    
    print("Translating...")
    speech = translator.tex_to_eng(latex)

    print(speech)

if __name__ == "__main__":
    args = parse_args()
    main(args)