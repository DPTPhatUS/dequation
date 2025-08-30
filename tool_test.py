from model.Tex2Eng.TangentCFT.TangentS.math_tan.math_extractor import MathExtractor
from model.Tex2Eng.TangentCFT.TangentS.Tuple_Extraction import latex_math_to_opt_tuples

# sample_input = r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}"
# sample_input = r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}"
# sample_input = r"\int_{a}^{b} f(x) \, dx"
# sample_input = r"\lim_{x \to \infty}, \sin(x), \cos(x), \log(x)"
# sample_input = r"\begin{cases} x^2 & \text{if } x > 0 \\ -x & \text{otherwise} \end{cases}"
# sample_input = r"\pm, \times, \div, \cdot, \propto"
# sample_input = r"\{a, b, c\}, \cup, \cap, \subseteq"
# sample_input = r"\land, \lor, \neg, \implies, \iff"
# sample_input = r"\overline{AB}, \underline{x}"
# sample_input = r"\rightarrow, \leftrightarrow, \mapsto"
# sample_input = r"\binom{n}{k}, \operatorname{mod}, \operatorname{gcd}"
sample_input = r"\sqrt{x}, \sqrt[n]{x}"

opt = MathExtractor.parse_from_tex_opt(sample_input)
print(opt.tostring())

# pairs = latex_math_to_opt_tuples(sample_input)
# print(pairs)
