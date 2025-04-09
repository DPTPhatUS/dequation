# # from openai import OpenAI

# # client = OpenAI(
# #   base_url="https://openrouter.ai/api/v1",
# #   api_key="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",
# # )

# # completion = client.chat.completions.create(
# #   extra_body={},
# #   model="google/gemini-2.5-pro-exp-03-25:free",
# #   messages=[
# #     {
# #       "role": "user",
# #       "content": [
# #         {
# #           "type": "text",
# #           "text": r"Translate the following LaTeX equation into spoken English without adding any extra explanations: \begin{equation} \snorm{ m^{(\sigma)} }_{H^{s+1}_\kappa} \lesssim \sum_{\ell=1}^\infty \ell^\sigma 2^{\sigma-\ell} \Bigl( 1 + \| q\|_{H^{s}_\kappa}\Bigr)^\sigma \| q \|_{H^{s+\sigma}_\kappa} < \infty \end{equation}"
# #         }
# #       ]
# #     }
# #   ]
# # )
# # print(completion.choices[0].message.content)

# # Test TexTeller tokenizer
# from model.TexTeller.src.models.ocr_model.model.TexTeller import TexTeller
# from transformers import AutoTokenizer

# input = r"\begin{aligned}{T_{Dp-brane}}&{{}=T_{\frac{p}{2},\frac{p}{2}-1}=\frac{(2\pi)^{\frac{p}{2}}\,R^{\frac{p}{2}-1}}{l_{p}^{\frac{3p}{2}}}\,=\,\frac{1}{(2\pi)^{p}\,g_{s}\,l_{s}^{p+1}}\,,}\\{T_{NSp-brane}}&{{}=T_{\frac{p+3}{4},\frac{5-p}{4}}=\frac{(2\pi)^{\frac{9-p}{4}}\,R^{\frac{5-p}{4}}}{l_{p}^{\frac{2p+9}{4}}}\,=\,\frac{1}{(2\pi)^{p}\,g_{s}^{\frac{p-1}{2}}\,l_{s}^{p+1}}\,.}\\\end{aligned}"

# # tokenizer = TexTeller.get_tokenizer()
# tokenizer = AutoTokenizer.from_pretrained('aaai25withanonymous/MathBridge_T5_small')


# tokens = tokenizer.tokenize(input)
# print(tokens)

import sympy as sp
import random

# Define symbols
x, y, z, n = sp.symbols('x y z n')

# Define a list of random mathematical expressions
expressions = [
    sp.sin(x)**2 + sp.cos(x)**2,  # Trigonometric identity
    sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo)),  # Gaussian integral
    sp.diff(sp.tan(x), x),  # Derivative of tangent
    sp.limit((1 + 1/x)**x, x, sp.oo),  # Limit of (1 + 1/x)^x as x -> infinity
    sp.Sum(1/n**2, (n, 1, sp.oo)),  # Infinite sum
    sp.Matrix([[x, y], [y, z]]).det(),  # Determinant of a matrix
    sp.expand((x + y)**3),  # Expand a polynomial
    sp.factor(x**3 + 3*x**2*y + 3*x*y**2 + y**3),  # Factor a polynomial
    sp.solve(x**2 - 2, x),  # Solve a quadratic equation
    sp.series(sp.sin(x), x, 0, 5),  # Taylor series expansion
]

# Randomly select an expression
expr = random.choice(expressions)

# Convert the expression to LaTeX
latex_formula = sp.latex(expr)

# Print the LaTeX formula
print("Random LaTeX Formula:")
print(latex_formula)