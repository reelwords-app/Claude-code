"""
Convert paper_pinn_polymer_flood.md to a formatted Word (.docx) document.
Embeds all figures referenced as *(figX_name.png)* in block-quote captions.
"""
import re, os
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

MD_PATH   = '/home/user/Claude-code/paper_pinn_polymer_flood.md'
OUT_PATH  = '/home/user/Claude-code/paper_pinn_polymer_flood.docx'
FIG_DIR   = '/home/user/Claude-code/pinn_cmg_results/'

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
section = doc.sections[0]
section.page_width  = Inches(8.5)
section.page_height = Inches(11)
section.left_margin = section.right_margin = Inches(1.0)
section.top_margin  = section.bottom_margin = Inches(1.0)

# ── Default font ──────────────────────────────────────────────────────────────
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)

def set_heading(level, text):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h

def _inline_runs(p, text, base_size=11, bold=False, italic=False, color=None):
    """Add inline bold/italic markdown segments to paragraph p."""
    segments = re.split(r'(\*\*.*?\*\*|\*[^*].*?[^*]\*|\*[^*]\*)', text)
    for seg in segments:
        if seg.startswith('**') and seg.endswith('**'):
            r = p.add_run(seg[2:-2]); r.bold = True
        elif seg.startswith('*') and seg.endswith('*'):
            r = p.add_run(seg[1:-1]); r.italic = True
        else:
            r = p.add_run(seg)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(base_size)
        if bold:   r.bold   = True
        if italic: r.italic = True
        if color:  r.font.color.rgb = color

def add_para(text, style_name='Normal', bold=False, italic=False,
             align=WD_ALIGN_PARAGRAPH.LEFT, size=None):
    p = doc.add_paragraph(style=style_name)
    p.alignment = align
    _inline_runs(p, text, base_size=size or 11, bold=bold, italic=italic)
    return p

def add_figure_caption(caption_text):
    """Add an indented italic caption paragraph (no border)."""
    p = doc.add_paragraph(style='Normal')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.left_indent  = Inches(0.4)
    p.paragraph_format.right_indent = Inches(0.4)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(8)
    _inline_runs(p, caption_text, base_size=9,
                 color=RGBColor(60, 60, 60), italic=True)
    return p

def add_image(filename):
    """Insert a centered figure image at full text width."""
    path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(path):
        # Fallback: insert a placeholder note
        p = doc.add_paragraph(style='Normal')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f'[Figure: {filename} — file not found]')
        r.font.name = 'Times New Roman'; r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(180, 0, 0)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run()
    run.add_picture(path, width=Inches(6.0))

def add_quote(text):
    """Block-quote: detect figure reference, embed image, then add caption."""
    raw = re.sub(r'^>\s*', '', text).strip()

    # Extract figure filename  *(figN_name.png)*
    fig_match = re.search(r'\*\((fig\d+[a-z0-9_]*\.png)\)\*', raw)
    fig_file  = fig_match.group(1) if fig_match else None

    # Caption text: strip the *(filename)* tag from the end
    if fig_file:
        caption = raw[:fig_match.start()].rstrip('. ')
    else:
        caption = raw

    # Insert image first, then caption below
    if fig_file:
        add_image(fig_file)

    add_figure_caption(caption)

def add_table_from_md(lines):
    """Parse a markdown table and add it as a docx table."""
    rows = []
    for ln in lines:
        if re.match(r'\s*\|[-| :]+\|\s*$', ln):
            continue  # separator row
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        rows.append(cells)
    if not rows:
        return

    n_cols = max(len(r) for r in rows)
    tbl = doc.add_table(rows=len(rows), cols=n_cols)
    tbl.style = 'Table Grid'
    for ri, row_data in enumerate(rows):
        for ci in range(n_cols):
            cell_text = row_data[ci] if ci < len(row_data) else ''
            cell = tbl.cell(ri, ci)
            cell.text = ''
            p = cell.paragraphs[0]
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', cell_text)
            clean = re.sub(r'\*(.*?)\*',    r'\1', clean)
            run = p.add_run(clean)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            if ri == 0:
                run.bold = True
    doc.add_paragraph()

# ── Parse and render markdown ─────────────────────────────────────────────────
with open(MD_PATH, encoding='utf-8') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    ln = lines[i].rstrip('\n')

    # Headings
    if ln.startswith('#### '):
        set_heading(4, ln[5:].strip()); i += 1; continue
    if ln.startswith('### '):
        set_heading(3, ln[4:].strip()); i += 1; continue
    if ln.startswith('## '):
        set_heading(2, ln[3:].strip()); i += 1; continue
    if ln.startswith('# '):
        p = set_heading(1, ln[2:].strip())
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        i += 1; continue

    # Horizontal rule
    if re.match(r'^---+\s*$', ln):
        doc.add_paragraph(); i += 1; continue

    # Block quote (figure captions with embedded images)
    if ln.startswith('>'):
        add_quote(ln); i += 1; continue

    # Markdown table: collect all consecutive table lines
    if ln.startswith('|'):
        tbl_lines = []
        while i < len(lines) and lines[i].startswith('|'):
            tbl_lines.append(lines[i].rstrip('\n'))
            i += 1
        add_table_from_md(tbl_lines)
        continue

    # Numbered list items
    m = re.match(r'^(\d+)\.\s+(.*)', ln)
    if m:
        p = doc.add_paragraph(style='List Number')
        _inline_runs(p, m.group(2), base_size=11)
        i += 1; continue

    # Bullet list items
    if ln.startswith('- ') or ln.startswith('* '):
        p = doc.add_paragraph(style='List Bullet')
        _inline_runs(p, ln[2:], base_size=11)
        i += 1; continue

    # Empty line
    if ln.strip() == '':
        i += 1; continue

    # Regular paragraph
    add_para(ln.strip()); i += 1

doc.save(OUT_PATH)
print(f'Word document written: {OUT_PATH}')
