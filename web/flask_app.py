from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch
import cv2 as cv
import numpy as np
from PIL import Image
import os
import sys
import base64
from io import BytesIO


from model.TexTeller.src.models.ocr_model.model.TexTeller import TexTeller
from model.TexTeller.src.models.ocr_model.utils.to_katex import to_katex
from model.TexTeller.src.models.ocr_model.utils.inference import (
    inference as latex_inference,
)
from model.Tex2Eng.translator import SRE, Tex2Eng
from transformers import AutoTokenizer
import regex

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


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

        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "checkpoints/t5-small_latex-spoken/Tex2Eng_epoch_9.pth",
        )
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

            if translator_choice == "sre":
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_equation():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image file selected"}), 400

        translator = request.form.get("translator", "tex2eng")

        image = Image.open(file.stream)
        latex, speech = processor.process_image(image, translator)

        return jsonify({"latex": latex, "speech": speech, "success": True})

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
