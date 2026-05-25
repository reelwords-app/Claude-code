"""
Generate static architecture figures for PINN polymer flood paper.
Does NOT require trained models.

Figures produced:
  fig9_voronoi.png      — Voronoi diagram (fine grid + Voronoi)
  fig10_pinn_structure.png — PINN structure diagram
  fig11_network_arch.png   — Network architecture flowchart
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.spatial import Voronoi, Delaunay
import warnings, os

warnings.filterwarnings('ignore')
OUT_DIR = '/home/user/Claude-code/pinn_cmg_results/'
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [PLOT] {name} saved → {path}')


# ─────────────────────────────────────────────────────────────────────
# Helper: bounded Voronoi polygon clipping
# ─────────────────────────────────────────────────────────────────────
def _clip_polygon_to_box(poly, xmin, xmax, ymin, ymax):
    """Sutherland-Hodgman polygon clipping to axis-aligned box."""
    def _inside(p, edge):
        if edge == 'left':   return p[0] >= xmin
        if edge == 'right':  return p[0] <= xmax
        if edge == 'bottom': return p[1] >= ymin
        if edge == 'top':    return p[1] <= ymax

    def _intersect(p1, p2, edge):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        if edge == 'left':
            t = (xmin - p1[0]) / (dx + 1e-15)
        elif edge == 'right':
            t = (xmax - p1[0]) / (dx + 1e-15)
        elif edge == 'bottom':
            t = (ymin - p1[1]) / (dy + 1e-15)
        else:
            t = (ymax - p1[1]) / (dy + 1e-15)
        return [p1[0] + t*dx, p1[1] + t*dy]

    result = list(poly)
    for edge in ['left', 'right', 'bottom', 'top']:
        if not result:
            break
        output = []
        for i, cur in enumerate(result):
            prev = result[i-1]
            if _inside(cur, edge):
                if not _inside(prev, edge):
                    output.append(_intersect(prev, cur, edge))
                output.append(cur)
            elif _inside(prev, edge):
                output.append(_intersect(prev, cur, edge))
        result = output
    return np.array(result) if result else np.empty((0, 2))


# ─────────────────────────────────────────────────────────────────────
# Figure 9 — Voronoi diagram
# ─────────────────────────────────────────────────────────────────────
def make_fig9():
    INJS = np.array([[200, 100], [200, 200]], dtype=float)
    PRDS = np.array([
        [50,  50], [50, 150], [50, 250],
        [350, 50], [350, 150], [350, 250],
        [200, 300], [50, 0], [350, 0]
    ], dtype=float)

    BX, BY = 400, 300   # field boundary

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: Fine-Grid Reservoir Model ──────────────────────
    ax = axes[0]
    nx, ny = 20, 15
    xs = np.linspace(0, BX, nx+1)
    ys = np.linspace(0, BY, ny+1)
    for x in xs:
        ax.axvline(x, color='lightgray', lw=0.5)
    for y in ys:
        ax.axhline(y, color='lightgray', lw=0.5)

    # Well symbols
    for i, (ix, iy) in enumerate(INJS):
        ax.scatter(ix, iy, s=200, marker='s', color='red', zorder=6)
        ax.text(ix+8, iy+8, f'I{i+1}', fontsize=9, color='red', fontweight='bold')
    for i, (px, py) in enumerate(PRDS):
        ax.scatter(px, py, s=120, marker='o', color='black', zorder=6)
        ax.text(px+8, py+5, f'P{i+1}', fontsize=9, color='black')

    ax.set_xlim(-10, BX+10); ax.set_ylim(-10, BY+20)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=11); ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('(a) Fine-Grid Reservoir Model', fontsize=12, fontweight='bold')
    ax.set_facecolor('#f9f9f9')

    # ── Panel B: Voronoi-Diagram Model ──────────────────────────
    ax2 = axes[1]
    margin = 200
    all_pts = np.vstack([INJS, PRDS])
    mirrors = []
    for pt in all_pts:
        mirrors += [
            [pt[0] - 2*margin, pt[1]],
            [pt[0] + 2*margin, pt[1]],
            [pt[0], pt[1] - 2*margin],
            [pt[0], pt[1] + 2*margin],
        ]
    extended = np.vstack([all_pts, mirrors])
    vor = Voronoi(extended)

    n_real = len(all_pts)   # first n_real points are real wells

    # Draw clipped Voronoi polygons
    for k in range(n_real):
        region_idx = vor.point_region[k]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        poly_pts = vor.vertices[region]
        clipped = _clip_polygon_to_box(poly_pts, 0, BX, 0, BY)
        if len(clipped) < 3:
            continue
        is_inj = k < len(INJS)
        color = '#d0e8f8' if is_inj else '#fffde7'
        patch = plt.Polygon(clipped, closed=True,
                            facecolor=color, edgecolor='gray',
                            lw=0.8, alpha=0.85, zorder=1)
        ax2.add_patch(patch)

    # Draw Delaunay triangulation (red lines between real wells)
    tri = Delaunay(all_pts)
    drawn = set()
    for simplex in tri.simplices:
        for i in range(3):
            e = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
            if e not in drawn:
                drawn.add(e)
                p1, p2 = all_pts[e[0]], all_pts[e[1]]
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                         '-', color='red', lw=1.2, alpha=0.8, zorder=3)

    # Field boundary
    rect = plt.Polygon([[0,0],[BX,0],[BX,BY],[0,BY]],
                       closed=True, fill=False, edgecolor='black', lw=2, zorder=5)
    ax2.add_patch(rect)

    # Well symbols
    for i, (ix, iy) in enumerate(INJS):
        ax2.scatter(ix, iy, s=200, marker='s', color='red', zorder=7)
        ax2.text(ix+8, iy+8, f'I{i+1}', fontsize=9, color='red', fontweight='bold')
    for i, (px, py) in enumerate(PRDS):
        ax2.scatter(px, py, s=120, marker='o', color='black', zorder=7)
        ax2.text(px+8, py+5, f'P{i+1}', fontsize=9, color='black')

    ax2.set_xlim(-10, BX+10); ax2.set_ylim(-10, BY+20)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (m)', fontsize=11); ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('(b) Voronoi-Diagram Model', fontsize=12, fontweight='bold')
    ax2.set_facecolor('#f9f9f9')

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(facecolor='#d0e8f8', edgecolor='gray', label='Injector region'),
        mpatches.Patch(facecolor='#fffde7', edgecolor='gray', label='Producer region'),
        Line2D([0],[0], color='red',   lw=1.5, label='Delaunay connections'),
        Line2D([0],[0], color='gray',  lw=0.8, label='Voronoi boundaries'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='red',   markersize=9, label='Injector'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Producer'),
    ]
    ax2.legend(handles=handles, loc='upper right', fontsize=8.5,
               facecolor='white', framealpha=0.9)

    fig.suptitle('Pelican Lake Reservoir: Grid Models\n(Line-drive polymer flood pattern)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig9_voronoi.png')


# ─────────────────────────────────────────────────────────────────────
# Figure 10 — PINN structure
# ─────────────────────────────────────────────────────────────────────
def make_fig10():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')

    def circle(x, y, r=0.28, fc='white', ec='black', lw=1.5, zorder=4):
        c = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder)
        ax.add_patch(c)

    def arrow(x1, y1, x2, y2, color='gray', lw=1.2, head=0.15, zorder=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=f'->', color=color, lw=lw,
                                   mutation_scale=head*100),
                    zorder=zorder)

    # ── Input nodes
    inp_labels = ['t_norm', 'T_start', 'Q_total']
    inp_y      = [7.5, 4.5, 1.5]
    for lbl, y in zip(inp_labels, inp_y):
        circle(1.0, y, fc='#e8f5e9', ec='#388e3c', lw=2)
        ax.text(1.0, y, lbl, ha='center', va='center', fontsize=8.5, fontweight='bold')

    # ── Hidden layer 1
    hl1_y = [7.5, 6.0, 4.5, 3.0, 1.5]
    for y in hl1_y:
        circle(4.0, y, fc='#bbdefb', ec='#1565c0', lw=1.5)

    # ── Hidden layer 2
    for y in hl1_y:
        circle(6.5, y, fc='#bbdefb', ec='#1565c0', lw=1.5)

    # ── Output nodes
    out_labels = ['WC-hat', 'Q_oil-hat']
    out_y      = [6.5, 2.5]
    for lbl, y in zip(out_labels, out_y):
        circle(9.0, y, fc='#fff9c4', ec='#f57f17', lw=2)
        ax.text(9.0, y, lbl, ha='center', va='center', fontsize=8, fontweight='bold')

    # ── Thin gray lines: input → HL1
    for iy in inp_y:
        for hy in hl1_y:
            ax.plot([1.28, 3.72], [iy, hy], color='#bdbdbd', lw=0.5, zorder=1)

    # ── Thin gray lines: HL1 → HL2
    for y1 in hl1_y:
        for y2 in hl1_y:
            ax.plot([4.28, 6.22], [y1, y2], color='#bdbdbd', lw=0.5, zorder=1)

    # ── Thin gray lines: HL2 → output
    for hy in hl1_y:
        for oy in out_y:
            ax.plot([6.78, 8.72], [hy, oy], color='#bdbdbd', lw=0.5, zorder=1)

    # ── Layer labels
    ax.text(1.0,  0.3, 'Input layer',   ha='center', fontsize=10, color='#388e3c', fontweight='bold')
    ax.text(5.25, 0.3, 'Hidden layers', ha='center', fontsize=10, color='#1565c0', fontweight='bold')
    ax.text(9.0,  0.3, 'Output layer',  ha='center', fontsize=10, color='#f57f17', fontweight='bold')

    # ── Physics Parameters box
    pp_x, pp_y, pp_w, pp_h = 10.2, 3.5, 2.6, 4.5
    pp_box = FancyBboxPatch((pp_x, pp_y), pp_w, pp_h,
                            boxstyle='round,pad=0.1',
                            facecolor='#e0f2f1', edgecolor='#00796b', lw=2, zorder=3)
    ax.add_patch(pp_box)
    ax.text(pp_x + pp_w/2, pp_y + pp_h - 0.3,
            'Physics Parameters', ha='center', va='top',
            fontsize=9.5, fontweight='bold', color='#00796b')
    params_text = (
        'k$_{rw}^{max}$ = 0.2918\n'
        'μ$_o$ = 5000 cp\n'
        'S$_{wi}$ = 0.36\n'
        'S$_{or}$ = 0.10\n'
        'μ$_w$(C$_p$) Hand\n'
        'f$_w$ BL'
    )
    ax.text(pp_x + pp_w/2, pp_y + pp_h - 0.75,
            params_text, ha='center', va='top',
            fontsize=9, linespacing=1.6, color='#004d40')

    # ── Data Loss box
    dl_x, dl_y, dl_w, dl_h = 13.0, 6.8, 2.8, 2.2
    dl_box = FancyBboxPatch((dl_x, dl_y), dl_w, dl_h,
                            boxstyle='round,pad=0.1',
                            facecolor='#e3f2fd', edgecolor='#1565c0', lw=2, zorder=3)
    ax.add_patch(dl_box)
    ax.text(dl_x + dl_w/2, dl_y + dl_h - 0.25,
            'Data Loss', ha='center', va='top',
            fontsize=9.5, fontweight='bold', color='#1565c0')
    ax.text(dl_x + dl_w/2, dl_y + dl_h - 0.75,
            r'$\mathcal{L}_D = \frac{1}{N}\sum[(WC-\hat{WC})^2$' + '\n'
            r'$+ (Q - \hat{Q})^2]$',
            ha='center', va='top', fontsize=8.5, color='#0d47a1', linespacing=1.5)

    # ── Physics Loss box
    pl_x, pl_y, pl_w, pl_h = 13.0, 3.5, 2.8, 2.8
    pl_box = FancyBboxPatch((pl_x, pl_y), pl_w, pl_h,
                            boxstyle='round,pad=0.1',
                            facecolor='#fff3e0', edgecolor='#e65100', lw=2, zorder=3)
    ax.add_patch(pl_box)
    ax.text(pl_x + pl_w/2, pl_y + pl_h - 0.25,
            'Physics Loss', ha='center', va='top',
            fontsize=9.5, fontweight='bold', color='#e65100')
    pl_text = (
        r'$\mathcal{L}_P = \sum\mathrm{ReLU}(-\Delta WC)^2$' + '\n'
        r'$+ \sum\mathrm{ReLU}(\Delta Q_{oil})^2$' + '\n\n'
        r'$\mathcal{L} = \mathcal{L}_D + \lambda \cdot \mathcal{L}_P$'
    )
    ax.text(pl_x + pl_w/2, pl_y + pl_h - 0.75,
            pl_text, ha='center', va='top', fontsize=8.5,
            color='#bf360c', linespacing=1.5)

    # ── Arrows: output → param box
    for oy in out_y:
        ax.annotate('', xy=(pp_x, pp_y + pp_h/2),
                    xytext=(9.28, oy),
                    arrowprops=dict(arrowstyle='->', color='#00796b', lw=1.5,
                                   connectionstyle='arc3,rad=0.0'),
                    zorder=5)

    # ── Arrows: output → data loss
    for oy in out_y:
        ax.annotate('', xy=(dl_x, dl_y + dl_h/2),
                    xytext=(9.28, oy),
                    arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.5,
                                   connectionstyle='arc3,rad=-0.15'),
                    zorder=5)

    # ── Arrow: output → physics loss
    for oy in out_y:
        ax.annotate('', xy=(pl_x, pl_y + pl_h/2),
                    xytext=(9.28, oy),
                    arrowprops=dict(arrowstyle='->', color='#e65100', lw=1.5,
                                   connectionstyle='arc3,rad=0.1'),
                    zorder=5)

    # ── Arrow: param box → physics loss
    ax.annotate('', xy=(pl_x + pl_w/2, pl_y + pl_h),
                xytext=(pp_x + pp_w/2, pp_y),
                arrowprops=dict(arrowstyle='->', color='#00796b', lw=1.8,
                                connectionstyle='arc3,rad=0.3'),
                zorder=5)

    ax.set_title('Physics-Informed Neural Network Structure\n'
                 'Pelican Lake CMG STARS Polymer Flood',
                 fontsize=13, fontweight='bold', pad=15)

    fig.tight_layout()
    save(fig, 'fig10_pinn_structure.png')


# ─────────────────────────────────────────────────────────────────────
# Figure 11 — Network architecture flowchart
# ─────────────────────────────────────────────────────────────────────
def make_fig11():
    fig, ax = plt.subplots(figsize=(7, 12))
    ax.set_xlim(0, 7); ax.set_ylim(0, 13)
    ax.axis('off')
    ax.set_facecolor('white')

    BOX_W  = 5.0
    BOX_CX = 3.5  # center x

    def draw_box(cx, y_top, height, label, sublabel=None,
                 fc='#d4e8f0', ec='navy', fontsize=11, subfontsize=9):
        bx = cx - BOX_W/2
        box = FancyBboxPatch((bx, y_top - height), BOX_W, height,
                             boxstyle='round,pad=0.12',
                             facecolor=fc, edgecolor=ec, lw=1.8, zorder=3)
        ax.add_patch(box)
        ty = y_top - height/2
        ax.text(cx, ty + (0.15 if sublabel else 0),
                label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='#1a1a2e', zorder=4)
        if sublabel:
            ax.text(cx, ty - 0.22, sublabel, ha='center', va='center',
                    fontsize=subfontsize, color='#333333', zorder=4)

    def draw_arrow(y_from, y_to):
        ax.annotate('', xy=(BOX_CX, y_to),
                    xytext=(BOX_CX, y_from),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.8,
                                   mutation_scale=18),
                    zorder=5)

    # Layout top → bottom
    # Each block: (y_top, height, label, sublabel, fc, ec)
    blocks = [
        (12.5, 1.0,
         'Input:  t,  T_start,  Q_total',
         '3 input features',
         '#90ee90', '#2e7d32'),

        (11.0, 1.0,
         'Dense Layer 1',
         'Input dim: 3  /  Output dim: 64',
         '#d4e8f0', 'navy'),

        (9.6,  0.7,
         'Tanh Activation',
         '64 → 64',
         '#e8f4f8', '#3a7ebf'),

        (8.5,  1.0,
         'Dense Layer 2',
         'Input dim: 64  /  Output dim: 64',
         '#d4e8f0', 'navy'),

        (7.1,  0.7,
         'Tanh Activation',
         '64 → 64',
         '#e8f4f8', '#3a7ebf'),

        (6.0,  1.0,
         'Dense Layer 3',
         'Input dim: 64  /  Output dim: 64',
         '#d4e8f0', 'navy'),

        (4.6,  0.7,
         'Tanh Activation',
         '64 → 64',
         '#e8f4f8', '#3a7ebf'),

        (3.5,  1.0,
         'Dense Output Layer',
         'Input dim: 64  /  Output dim: 2',
         '#d4e8f0', 'navy'),

        (2.1,  0.7,
         'Sigmoid Activation',
         'Outputs  ∈  (0, 1)',
         '#e8f4f8', '#3a7ebf'),

        (1.0,  0.8,
         'Output:  WC,  Q_oil_norm',
         '2 outputs',
         '#90ee90', '#2e7d32'),
    ]

    arrow_tops  = []
    arrow_bottoms = []

    for i, (ytop, h, lbl, sub, fc, ec) in enumerate(blocks):
        draw_box(BOX_CX, ytop, h, lbl, sub, fc=fc, ec=ec)
        arrow_tops.append(ytop - h)      # bottom of this box
        arrow_bottoms.append(ytop)       # top of this box (for next arrow start)

    # Draw arrows between consecutive boxes
    for i in range(len(blocks) - 1):
        y_from = arrow_tops[i]       # bottom of box i
        y_to   = arrow_bottoms[i+1] - blocks[i+1][1]  # top of box i+1 minus height? no
        # arrow from bottom of box[i] to top of box[i+1]
        bot_i   = blocks[i][0]   - blocks[i][1]    # y_top[i] - h[i]
        top_i1  = blocks[i+1][0]                   # y_top[i+1]
        draw_arrow(bot_i + 0.05, top_i1 - 0.05)

    # Physics constraints annotation on right side
    ann_x = 6.4
    ax.text(ann_x, 7.5, 'Physics\nConstraints:', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#b71c1c',
            bbox=dict(boxstyle='round', fc='#ffebee', ec='#b71c1c', lw=1.5))
    ax.text(ann_x, 6.4,
            'dWC/dt ≥ 0\n(monotone ↑)',
            ha='center', va='center', fontsize=9, color='#c62828',
            bbox=dict(boxstyle='round', fc='white', ec='#ef9a9a', lw=1))
    ax.text(ann_x, 5.5,
            'dOil/dt ≤ 0\n(decline ↓)',
            ha='center', va='center', fontsize=9, color='#c62828',
            bbox=dict(boxstyle='round', fc='white', ec='#ef9a9a', lw=1))

    # Arrow from constraint annotation to main diagram
    ax.annotate('', xy=(BOX_CX + BOX_W/2 + 0.05, 6.0),
                xytext=(ann_x - 0.5, 6.5),
                arrowprops=dict(arrowstyle='<-', color='#b71c1c', lw=1.5),
                zorder=5)

    ax.set_title('PINN Network Architecture\n(3 × 64 Tanh + Sigmoid output)',
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    save(fig, 'fig11_network_arch.png')


# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('[FIG9]  Generating Voronoi diagram ...')
    make_fig9()
    print('[FIG10] Generating PINN structure diagram ...')
    make_fig10()
    print('[FIG11] Generating network architecture flowchart ...')
    make_fig11()
    print('\nAll architecture figures generated successfully.')
