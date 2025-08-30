def clean_latex_spacing(text):
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\":
            result.append(char)
            i += 1
            while i < len(text) and text[i].isalpha():
                result.append(text[i])
                i += 1
            if i < len(text) and text[i] == " ":
                result.append(" ")
                i += 1
                while i < len(text) and text[i] == " ":
                    i += 1
        elif char == " ":
            j = i
            while j < len(text) and text[j] == " ":
                j += 1

            if j < len(text) and text[j] == "\\":
                result.append(" ")
                i = j
            else:
                while i < len(text) and text[i] == " ":
                    i += 1
        else:
            result.append(char)
            i += 1
    return "".join(result)


with open("./data/PRINTED_TEX_230k/final_png_formulas_original.txt", "r") as f:
    content = f.read()

cleaned_content = "\n".join(clean_latex_spacing(line) for line in content.splitlines())

with open("./data/PRINTED_TEX_230k/final_png_formulas_stripped_3.txt", "w") as f:
    f.write(cleaned_content)
