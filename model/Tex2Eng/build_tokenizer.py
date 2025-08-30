import re
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------
# 1. Load dataset
# -------------------------------
# Replace with your dataset path or HF repo name
dataset = load_dataset("path_to_your_dataset")  # has train/validation/test
# columns: "equation" (LaTeX) and "spoken_English"


# -------------------------------
# 2. LaTeX tokenizer
# -------------------------------
def tokenize_latex(formula: str):
    """
    Naive tokenizer for LaTeX math expressions.
    Real MathBERT uses im2markup tokenizer.
    """
    return re.findall(r"[A-Za-z]+|\\[A-Za-z]+|\d+|[^A-Za-z0-9\s]", formula)


# -------------------------------
# 3. Operator Tree (OPT) generator (stub)
# -------------------------------
def build_opt_nodes_stub(formula: str):
    """
    Dummy OPT builder.
    Replace with Tangent-S parser for real use.
    """
    if "c^2 = a^2 + b^2" in formula:
        return ["EQ", "ADD", "SUP", "SUP", "SUP"]
    elif "E = mc^2" in formula:
        return ["EQ", "MUL", "SUP"]
    else:
        return ["EQ"]


# -------------------------------
# 4. Collect new tokens
# -------------------------------
new_tokens = set()

for split in ["train", "validation", "test"]:
    for example in dataset[split]:
        formula = example["equation"]

        # Tokenize LaTeX
        latex_tokens = tokenize_latex(formula)
        new_tokens.update(latex_tokens)

        # OPT nodes
        opt_nodes = build_opt_nodes_stub(formula)
        new_tokens.update(opt_nodes)

new_tokens = sorted(list(new_tokens))
print(f"Collected {len(new_tokens)} new tokens from formulas + OPT")

# -------------------------------
# 5. Load and extend T5 tokenizer
# -------------------------------
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Add new tokens
num_added = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added} new tokens to T5 tokenizer")
print("New vocab size:", len(tokenizer))

# -------------------------------
# 6. Resize model embeddings
# -------------------------------
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.resize_token_embeddings(len(tokenizer))
print("Resized T5 embeddings to match new vocab size.")

# -------------------------------
# 7. Save tokenizer and model
# -------------------------------
save_dir = "./math_t5_tokenizer"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print(f"Tokenizer + model with math tokens saved at: {save_dir}")

# -------------------------------
# 8. Example usage
# -------------------------------
example_formula = r"c^2 = a^2 + b^2"
example_context = (
    "The Pythagorean theorem states that [MATH] relates sides of a triangle."
)

formula_tokens = tokenize_latex(example_formula)
opt_nodes = build_opt_nodes_stub(example_formula)

final_tokens = (
    ["<extra_id_0>"]  # T5 uses <extra_id> for masking, but you can also use <pad>/<s>
    + formula_tokens
    + ["</s>"]
    + tokenizer.tokenize(example_context)
    + ["</s>"]
    + opt_nodes
)

print("Example combined tokens:", final_tokens)
print("Token IDs:", tokenizer.convert_tokens_to_ids(final_tokens))
