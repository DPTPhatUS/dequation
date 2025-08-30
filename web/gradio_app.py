import gradio as gr
import torch
import cv2 as cv
import numpy as np
from PIL import Image
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.TexTeller.src.models.ocr_model.model.TexTeller import TexTeller
from model.TexTeller.src.models.ocr_model.utils.to_katex import to_katex
from model.TexTeller.src.models.ocr_model.utils.inference import (
    inference as latex_inference,
)
from model.Tex2Eng.translator import SRE, Tex2Eng
from transformers import AutoTokenizer
import regex


def remove_boldsymbol(text):
    pattern = r"\\boldsymbol\s*\{((?:[^{}]+|(?R))*)\}"
    while regex.search(pattern, text):
        text = regex.sub(pattern, r" \1 ", text)
    return text


class MathEquationProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print("Loading models...")

        self.latex_rec_model = TexTeller.from_pretrained()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "aaai25withanonymous/MathBridge_T5_small"
        )

        checkpoint_path = "checkpoints/t5-small_latex-spoken/Tex2Eng_epoch_9.pth"
        if os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)
            if (
                isinstance(checkpoint_dict, dict)
                and "model_state_dict" in checkpoint_dict
            ):
                state_dict = checkpoint_dict["model_state_dict"]
            else:
                state_dict = checkpoint_dict

            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {
                    key.replace("module.", ""): value
                    for key, value in state_dict.items()
                }

            self.tex2eng_model = Tex2Eng("google-t5/t5-small", self.tokenizer).to(
                self.device
            )
            self.tex2eng_model.load_state_dict(state_dict)
            self.tex2eng_model.eval()
        else:
            self.tex2eng_model = None
            print(
                "Warning: Tex2Eng model checkpoint not found. Only SRE will be available."
            )

        self.sre_translator = SRE()

        print("Models loaded successfully!")

    def process_image(self, image, translator_choice):
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
                else:
                    img = img_array
            else:
                img = image

            res = latex_inference(
                self.latex_rec_model, TexTeller.get_tokenizer(), [img], self.device, 4
            )
            latex = to_katex(res[0])
            latex = remove_boldsymbol(latex)

            if translator_choice == "SRE":
                speech = self.sre_translator.tex_to_eng(latex)
            else:
                if self.tex2eng_model is None:
                    return (
                        latex,
                        "Error: Tex2Eng model not available. Please use SRE translator.",
                    )

                with torch.no_grad():
                    input_tokens = self.tokenizer(latex, return_tensors="pt")
                    output = self.tex2eng_model.generate(
                        input_ids=input_tokens["input_ids"].to(self.device),
                        attention_mask=input_tokens["attention_mask"].to(self.device),
                        max_length=512,
                        num_beams=4,
                    )
                    speech = self.tokenizer.batch_decode(
                        output, skip_special_tokens=True
                    )[0]

            return latex, speech

        except Exception as e:
            return (
                f"Error generating LaTeX: {str(e)}",
                f"Error in translation: {str(e)}",
            )


processor = MathEquationProcessor()


def process_equation(image, translator):
    latex, speech = processor.process_image(image, translator)
    return latex, speech


with gr.Blocks(
    title="Mathematical Equation to Speech Converter", theme=gr.themes.Soft()
) as demo:
    gr.Markdown(
        """
    # Mathematical Equation to Speech Converter
    
    Upload an image containing a mathematical equation, and this tool will:
    1. Extract the LaTeX representation from the image
    2. Convert it to natural spoken language
    
    Choose your preferred translation method below.
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Math Equation Image", type="pil", height=300
            )

            translator_choice = gr.Radio(
                choices=["Tex2Eng", "SRE"],
                value="Tex2Eng",
                label="Translation Method",
                info="Tex2Eng: Neural model, SRE: Rule-based system",
            )

            process_btn = gr.Button("Process Equation", variant="primary", size="lg")

        with gr.Column(scale=1):
            latex_output = gr.Textbox(
                label="Generated LaTeX",
                placeholder="LaTeX will appear here...",
                lines=3,
                max_lines=10,
            )

            speech_output = gr.Textbox(
                label="Spoken Description",
                placeholder="Natural language description will appear here...",
                lines=5,
                max_lines=15,
            )

    gr.Markdown("### Examples")
    gr.Markdown(
        "Try uploading images of mathematical equations, formulas, or expressions!"
    )

    process_btn.click(
        fn=process_equation,
        inputs=[image_input, translator_choice],
        outputs=[latex_output, speech_output],
    )

    image_input.change(
        fn=process_equation,
        inputs=[image_input, translator_choice],
        outputs=[latex_output, speech_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
