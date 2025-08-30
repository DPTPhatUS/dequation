from model.Tex2Eng.translator import *
from tqdm import tqdm

with open("./data/PRINTED_TEX_230k/final_png_formulas_stripped_3.txt", "r") as file:
    lines = file.readlines()
    translator = SRE()

    with open("output1.txt", "w") as output:
        for line in tqdm(lines[3529:90423], unit="line"):
            translate = translator.tex_to_eng(line)
            output.write(translate + "\n")
