"""Convert the Comprehensive Analysis Report from Markdown to PDF."""
import markdown
from pathlib import Path
from xhtml2pdf import pisa

project_root = Path(__file__).parent
md_path = project_root / "outputs" / "Comprehensive_Analysis_Report.md"
pdf_path = project_root / "outputs" / "Comprehensive_Analysis_Report.pdf"

md_text = md_path.read_text(encoding="utf-8")

# Convert LaTeX math to readable Unicode text for PDF rendering
import re

# Simple inline math replacements
math_replacements = [
    (r'\$r = 0\.7936\$', 'r = 0.7936'),
    (r'\$r \\approx 0\.79\$', 'r ≈ 0.79'),
    (r'\$p < 0\.05\$', 'p < 0.05'),
    (r'\$p < 0\.001\$', 'p < 0.001'),
    (r'\$p \\ll 0\.001\$', 'p ≪ 0.001'),
    (r'\$p < 10\^{-300}\$', 'p < 0.001'),
    (r'\$R\^2\$', 'R²'),
    (r'\$R\^2 = 0\.031\$', 'R² = 0.031'),
    (r'\$R\^2 = 0\.320\$', 'R² = 0.320'),
    (r'\$K\$', 'K'),
    (r'\$K=2\$', 'K=2'),
    (r'\$K=3\$', 'K=3'),
    (r'\$K=4\$', 'K=4'),
    (r'\$K=5\$', 'K=5'),
    (r'\$K \\in \\{2, 3, 4\\}\$', 'K ∈ {2, 3, 4}'),
    (r'\$n\$', 'n'),
    (r'\$k\$', 'k'),
    (r'\$d\$', 'd'),
    (r'\$H_0\$', 'H₀'),
    (r'\$H_1\$', 'H₁'),
    (r'\$\\eta\^2\$', 'η²'),
    (r'\$r\$', 'r'),
    (r'\$p\$', 'p'),
    (r'\$\\lambda\$', 'λ'),
    (r'\$\\lambda_k\$', 'λₖ'),
    (r'\$\\sigma\^2 < 10\^{-8}\$', 'σ² < 10⁻⁸'),
    (r'\$p = 1\.11 \\times 10\^{-16}\$', 'p = 1.11 × 10⁻¹⁶'),
    (r'\$p = 5\.89 \\times 10\^{-9}\$', 'p = 5.89 × 10⁻⁹'),
    (r'\$p = 5\.35 \\times 10\^{-57}\$', 'p = 5.35 × 10⁻⁵⁷'),
    (r'\$p = 0\.0215\$', 'p = 0.0215'),
    (r'\$\\beta_1 = 0\.0007\$', 'β₁ = 0.0007'),
    (r'\$\\beta_1\$', 'β₁'),
    (r'\$\\beta_0\$', 'β₀'),
    (r'\$n = 14\$', 'n = 14'),
    (r'\$n = 15\$', 'n = 15'),
    (r'\$R\^2_\{marginal\}\$', 'R²ₘₐᵣ'),
    (r'\$R\^2_\{conditional\}\$', 'R²ₒₙ'),
    (r'\$R\^2_m = 0\.168\$', 'R²ₘ = 0.168'),
    (r'\$R\^2_m = 0\.198\$', 'R²ₘ = 0.198'),
    (r'\$R\^2_c = 0\.177\$', 'R²c = 0.177'),
    (r'\$R\^2_c = 0\.427\$', 'R²c = 0.427'),
    (r'\$R\^2\_{marginal\} = 0\.168\$', 'R²marginal = 0.168'),
    (r'\$R\^2\_{conditional\} = 0\.177\$', 'R²conditional = 0.177'),
    (r'\$R\^2\_{marginal\} = 0\.198\$', 'R²marginal = 0.198'),
    (r'\$R\^2\_{conditional\} = 0\.427\$', 'R²conditional = 0.427'),
]

for pattern, replacement in math_replacements:
    md_text = re.sub(pattern, replacement, md_text)

# Convert remaining simple $...$ inline math to plain text
def simplify_math(match):
    """Convert simple LaTeX math to readable text."""
    content = match.group(1)
    content = content.replace('\\text{', '').replace('}', '')
    content = content.replace('\\bar{x}', 'x̄').replace('\\sigma', 'σ')
    content = content.replace('\\mu', 'μ').replace('\\pi', 'π')
    content = content.replace('\\lambda', 'λ').replace('\\phi', 'φ')
    content = content.replace('\\chi^2', 'χ²').replace('\\eta', 'η')
    content = content.replace('\\Delta', 'Δ').replace('\\approx', '≈')
    content = content.replace('\\times', '×').replace('\\cdot', '·')
    content = content.replace('\\leq', '≤').replace('\\geq', '≥')
    content = content.replace('\\ll', '≪').replace('\\gg', '≫')
    content = content.replace('\\neq', '≠').replace('\\pm', '±')
    content = content.replace('\\sum', 'Σ').replace('\\ln', 'ln')
    content = content.replace('\\cos', 'cos').replace('\\sin', 'sin')
    content = content.replace('\\arg\\max', 'argmax')
    content = content.replace('\\bmod', 'mod')
    content = content.replace('\\frac{', '(').replace('^2', '²')
    content = content.replace('\\sqrt{', '√(')
    content = content.replace('\\mathbf{A}', 'A').replace('\\mathbf{', '')
    content = content.replace('\\boldsymbol{', '').replace('\\mathcal{N}', 'N')
    content = content.replace('\\mid', '|').replace('\\sim', '~')
    content = content.replace('\\left(', '(').replace('\\right)', ')')
    content = content.replace('\\left[', '[').replace('\\right]', ']')
    content = content.replace('\\!', '')
    content = content.replace('_{', '_').replace('^{', '^')
    content = content.replace('\\bar{\\bar{x}}', 'x̄')
    content = content.replace('\\overline', '')
    return content

# Handle $$...$$ block math (remove delimiters, keep text)
md_text = re.sub(r'\$\$(.+?)\$\$', lambda m: '\n' + simplify_math(m) + '\n', md_text, flags=re.DOTALL)
# Handle $...$ inline math
md_text = re.sub(r'(?<!\$)\$(?!\$)(.+?)\$', simplify_math, md_text)

# Convert Markdown to HTML
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc"],
)

# Convert relative image paths to absolute paths for xhtml2pdf
def _abs_img(match):
    prefix, src, suffix = match.group(1), match.group(2), match.group(3)
    abs_path = (project_root / src).resolve()
    return f'{prefix}{abs_path}{suffix}'

html_body = re.sub(r'(<img[^>]+src=["\'])([^"\'>]+)(["\'])', _abs_img, html_body)

# Wrap in full HTML with CSS styling
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;
    @frame footer {{
        -pdf-frame-content: footerContent;
        bottom: 0.5cm;
        margin-left: 2cm;
        margin-right: 2cm;
        height: 1cm;
    }}
}}
body {{
    font-family: Times, "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
    text-align: justify;
}}
h1 {{
    font-size: 18pt;
    font-weight: bold;
    text-align: center;
    margin-top: 30pt;
    margin-bottom: 20pt;
    color: #2c3e50;
    page-break-before: avoid;
}}
h2 {{
    font-size: 15pt;
    font-weight: bold;
    margin-top: 24pt;
    margin-bottom: 12pt;
    color: #2c3e50;
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 4pt;
    page-break-after: avoid;
}}
h3 {{
    font-size: 12pt;
    font-weight: bold;
    margin-top: 16pt;
    margin-bottom: 8pt;
    color: #34495e;
    page-break-after: avoid;
}}
h4 {{
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 12pt;
    margin-bottom: 6pt;
}}
p {{
    margin-bottom: 8pt;
}}
table {{
    border-collapse: collapse;
    margin: 12pt 0;
    width: 100%;
    font-size: 10pt;
}}
th {{
    background-color: #ecf0f1;
    font-weight: bold;
    text-align: left;
    padding: 6pt 8pt;
    border: 1px solid #bdc3c7;
}}
td {{
    padding: 4pt 8pt;
    border: 1px solid #bdc3c7;
    vertical-align: top;
}}
tr:nth-child(even) td {{
    background-color: #f9f9f9;
}}
code {{
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 1pt 3pt;
}}
pre {{
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 8pt;
    border: 1px solid #ddd;
    page-break-inside: avoid;
}}
ul, ol {{
    margin-bottom: 8pt;
    padding-left: 20pt;
}}
li {{
    margin-bottom: 4pt;
}}
strong {{
    font-weight: bold;
}}
em {{
    font-style: italic;
}}
img {{
    max-width: 100%;
    height: auto;
    margin: 12pt auto;
    display: block;
    page-break-inside: avoid;
}}
hr {{
    border: none;
    border-top: 2px solid #bdc3c7;
    margin: 20pt 0;
}}
.subtitle {{
    text-align: center;
    font-style: italic;
    font-size: 13pt;
    margin-bottom: 30pt;
    color: #555;
}}
</style>
</head>
<body>
{html_body}
<div id="footerContent">
    <p style="text-align: center; font-size: 9pt; color: #888;">
        Comprehensive Analysis Report — Circadian Activity Patterns of Large Carnivores
    </p>
</div>
</body>
</html>
"""

with open(pdf_path, "wb") as f:
    status = pisa.CreatePDF(html, dest=f)

if status.err:
    print(f"ERROR: PDF generation failed with {status.err} errors")
else:
    print(f"PDF generated successfully: {pdf_path}")
    print(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB")
