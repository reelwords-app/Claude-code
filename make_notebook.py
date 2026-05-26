"""Convert pinn_vs_nn_cmg.py to a structured Jupyter notebook."""
import nbformat as nbf
import re, textwrap

src = open('/home/user/Claude-code/pinn_vs_nn_cmg.py').read()

# Section delimiters match the # ──── headers in the script
SECTIONS = [
    ("## 0  Imports & Setup",         r'import numpy'),
    ("## 1  Load & Preprocess Data",   r'def load_agg'),
    ("## 2  Case-Based Train/Val/Test Split  (SPE-218863-MS)", r'FRAC_TR'),
    ("## 3  Model Architecture",       r'def build_model'),
    ("## 4  Training Loop",            r'EPOCHS\s*='),
    ("## 5  Evaluate — Train / Val / Test", r'nn_pred_tr'),
    ("## 6  Polymer Timing Optimisation", r'scan_ps\s*='),
    ("## 7  Figures (1–8)",            r'plt\.rcParams'),
    ("## 8  BL Concentration & 2-D Optimisation", r'MU_OIL'),
    ("## 9  Summary & Architecture Figures", r'print.*SUMMARY'),
]

lines = src.splitlines()

def find_line(pattern):
    rx = re.compile(pattern)
    for i, ln in enumerate(lines):
        if rx.search(ln):
            return i
    return len(lines)

# Build (start, end) index pairs for each section
boundaries = [(find_line(p), title) for title, p in SECTIONS]
boundaries.append((len(lines), None))

nb = nbf.v4.new_notebook()

# Add a markdown title cell
nb.cells.append(nbf.v4.new_markdown_cell(textwrap.dedent("""\
    # NN vs PINN — Pelican Lake CMG STARS Polymer Flood
    **Reference:** SPE-218863-MS — Effective Production Forecasting and Robust Rate
    Optimization Using Physics-Informed Neural Networks

    **Data split:** Case-based 70 / 20 / 10 (train / validation / test)
    following the 3-D Brugge benchmark protocol from the reference paper.

    **Physics constraint (PINN):** WC monotonicity (dWC/dt ≥ 0 after polymer injection start),
    enforced via domain-wide collocation over the full (t, T_start) space (Ugembe et al. 2026).
    Thermodynamically valid — water saturation is irreversible once polymer flooding begins.
""")))

for k in range(len(SECTIONS)):
    start = boundaries[k][0]
    end   = boundaries[k+1][0]
    title = SECTIONS[k][0]

    block = '\n'.join(lines[start:end]).strip()
    if not block:
        continue

    # Markdown section header
    nb.cells.append(nbf.v4.new_markdown_cell(f'---\n{title}'))
    nb.cells.append(nbf.v4.new_code_cell(block))

# Write notebook
out = '/home/user/Claude-code/pinn_vs_nn_cmg.ipynb'
with open(out, 'w') as f:
    nbf.write(nb, f)
print(f'Notebook written: {out}')
