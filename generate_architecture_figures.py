"""
Generate publication-quality architecture figures for PINN polymer flood paper.
Closely matches reference paper style (SPE-218863-MS, Meng et al. 2024).

Figures:
  fig9_voronoi.png          — Fine-grid + Voronoi diagram (ref Fig 2/3)
  fig10_pinn_structure.png  — PINN structure: NN + learnable params + loss boxes (ref Fig 4)
  fig11_network_arch.png    — Network architecture flowchart with dim annotations (ref Fig 5)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi, Delaunay
import warnings, os

warnings.filterwarnings('ignore')
OUT_DIR = '/home/user/Claude-code/pinn_cmg_results/'
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [PLOT] {name}')


# ─────────────────────────────────────────────────────────────────────────────
# Voronoi clipping helper
# ─────────────────────────────────────────────────────────────────────────────
def _clip_poly(poly, xmin, xmax, ymin, ymax):
    def inside(p, e):
        if e == 'L': return p[0] >= xmin
        if e == 'R': return p[0] <= xmax
        if e == 'B': return p[1] >= ymin
        return p[1] <= ymax
    def isect(p1, p2, e):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        if e == 'L':  t = (xmin-p1[0])/(dx+1e-15)
        elif e == 'R':t = (xmax-p1[0])/(dx+1e-15)
        elif e == 'B':t = (ymin-p1[1])/(dy+1e-15)
        else:          t = (ymax-p1[1])/(dy+1e-15)
        return [p1[0]+t*dx, p1[1]+t*dy]
    result = list(poly)
    for e in ['L','R','B','T']:
        if not result: break
        out = []
        for i, cur in enumerate(result):
            prev = result[i-1]
            if inside(cur, e):
                if not inside(prev, e): out.append(isect(prev, cur, e))
                out.append(cur)
            elif inside(prev, e):
                out.append(isect(prev, cur, e))
        result = out
    return np.array(result) if result else np.empty((0,2))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Voronoi diagram  (ref style: Fig 2 / Fig 3)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig9():
    """
    Pelican Lake HP-6 pilot: 5 horizontal wells (P1-I1-P2-I2-P3), each 4593.176 ft = 1400 m long,
    spaced 574.147 ft = 175 m apart in X. Grid: 157 × 10 × 3 cells.
    DY: 1148 ft buffer + 8×574 ft (well region) + 1148 ft buffer = 2100 m total.
    Reference: Ugembe et al., Manuscript_final.pdf, Section 2.2.
    """
    # Dimensions in metres (1 ft = 0.3048 m)
    WELL_LEN  = 4593.176 * 0.3048   # ≈ 1400 m  — horizontal well length (Y direction)
    SPACING   = 574.147  * 0.3048   # ≈ 175 m   — inter-well spacing (X direction)
    BUF_Y     = 1148.294 * 0.3048   # ≈ 350 m   — Y buffer beyond well ends
    BUF_X     = 1.5 * SPACING       # ≈ 262 m   — X buffer beyond outermost wells

    # Total field extents
    # X: 4 gaps × 175 m + 2 × 262 m buffer
    # Y: well_len + 2 × buffer
    BY_start  = 0.0
    BY_end    = BUF_Y + WELL_LEN + BUF_Y   # ≈ 2100 m
    BX_start  = -BUF_X
    BX_end    = 4 * SPACING + BUF_X        # 4 gaps × 175 m + buffer ≈ 962 m

    # Well centre-lines: X positions of each horizontal well
    # Layout (line-drive): P1 – I1 – P2 – I2 – P3
    well_x    = np.array([0, 1, 2, 3, 4]) * SPACING   # 0, 175, 350, 525, 700 m
    well_types= ['P', 'I', 'P', 'I', 'P']             # producer / injector
    well_names= ['P1','I1','P2','I2','P3']
    Y0        = BUF_Y                                   # well start in Y
    Y1        = BUF_Y + WELL_LEN                        # well end in Y

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Panel (a): Fine-Grid Reservoir Model ────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#f0f4ff')

    # Simplified grid: 40 cols × 22 rows (representing 157 × 10 cells)
    nx_show, ny_show = 40, 22
    xs = np.linspace(BX_start, BX_end, nx_show + 1)
    ys = np.linspace(BY_start, BY_end, ny_show + 1)
    for x in xs:
        ax.plot([x, x], [BY_start, BY_end], color='#c5cae9', lw=0.4, zorder=1)
    for y in ys:
        ax.plot([BX_start, BX_end], [y, y], color='#c5cae9', lw=0.4, zorder=1)

    # Draw horizontal wells as thick lines
    for wx, wt, wn in zip(well_x, well_types, well_names):
        col = '#d32f2f' if wt == 'I' else '#1565c0'
        lw  = 3.5 if wt == 'I' else 3.0
        ax.plot([wx, wx], [Y0, Y1], color=col, lw=lw, solid_capstyle='round', zorder=5)
        ax.text(wx + 8, Y1 + 18, wn, ha='center', fontsize=9.5,
                color=col, fontweight='bold', zorder=6)

    # Permeability colour gradient (schematic)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('perm',
               ['#1a237e', '#1565c0', '#4caf50', '#ff9800', '#b71c1c'])
    np.random.seed(42)
    perm = np.random.rand(ny_show, nx_show) * 0.6 + 0.2
    ax.imshow(perm, extent=[BX_start, BX_end, BY_start, BY_end],
              origin='lower', cmap=cmap, alpha=0.30, aspect='auto', zorder=0)

    ax.set_xlim(BX_start - 20, BX_end + 20)
    ax.set_ylim(BY_start - 30, BY_end + 60)
    ax.set_xlabel('X (m)', fontsize=11); ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('(a)  Fine-Grid Reservoir Model\n'
                 '(157 × 10 × 3 cells, 3 layers)', fontsize=11, fontweight='bold', pad=6)
    ax.set_aspect('equal')

    # ── Panel (b): Voronoi-Diagram Model ────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#fafafa')

    # For horizontal wells, sample N_PT points along each well
    N_PT = 30
    y_pts = np.linspace(Y0, Y1, N_PT)

    all_pts  = np.array([[wx, yp] for wx in well_x for yp in y_pts])
    # which original well does each sampled point belong to?
    well_idx = np.array([i for i, wx in enumerate(well_x) for _ in y_pts])

    # Mirror points to bound the Voronoi
    m = max(BX_end - BX_start, BY_end - BY_start)
    mirrors = []
    for pt in all_pts:
        for dx, dy in [(-2*m, 0), (2*m, 0), (0, -2*m), (0, 2*m)]:
            mirrors.append([pt[0]+dx, pt[1]+dy])
    extended = np.vstack([all_pts, mirrors])
    vor = Voronoi(extended)

    # Colours per well type
    fcols = {'P': '#dceefb', 'I': '#e8f5e9'}
    ecols = {'P': '#1565c0', 'I': '#2e7d32'}

    drawn_regions = set()
    for k in range(len(all_pts)):
        wi = well_idx[k]
        wt = well_types[wi]
        reg = vor.regions[vor.point_region[k]]
        if -1 in reg or not reg or id(tuple(reg)) in drawn_regions: continue
        drawn_regions.add(id(tuple(reg)))
        pts = vor.vertices[reg]
        clipped = _clip_poly(pts, BX_start, BX_end, BY_start, BY_end)
        if len(clipped) < 3: continue
        ax2.add_patch(plt.Polygon(clipped, closed=True,
                                  facecolor=fcols[wt], edgecolor='#aaaaaa',
                                  lw=0.6, alpha=0.85, zorder=1))

    # Draw thick Voronoi boundary lines between wells (at X midpoints)
    for i in range(len(well_x) - 1):
        xb = (well_x[i] + well_x[i+1]) / 2
        ax2.plot([xb, xb], [BY_start, BY_end],
                 color='#555555', lw=1.2, ls='--', zorder=3, alpha=0.7)

    # Draw well connections (Delaunay-style — adjacent wells connected)
    for i in range(len(well_x) - 1):
        ymid = (Y0 + Y1) / 2
        ax2.annotate('', xy=(well_x[i+1], ymid), xytext=(well_x[i], ymid),
                     arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.8),
                     zorder=4)

    # Field boundary
    ax2.add_patch(plt.Polygon(
        [[BX_start, BY_start],[BX_end, BY_start],
         [BX_end, BY_end],[BX_start, BY_end]],
        closed=True, fill=False, edgecolor='#222222', lw=2, zorder=5))

    # Draw horizontal wells
    for wx, wt, wn in zip(well_x, well_types, well_names):
        col = '#d32f2f' if wt == 'I' else '#1565c0'
        mk  = 's' if wt == 'I' else 'o'
        lw  = 3.5 if wt == 'I' else 3.0
        ax2.plot([wx, wx], [Y0, Y1], color=col, lw=lw,
                 solid_capstyle='round', zorder=6)
        # midpoint marker
        ymid = (Y0 + Y1) / 2
        ax2.scatter(wx, ymid, s=80, marker=mk, color=col, zorder=8,
                    edgecolors='white', linewidths=1.2)
        ax2.text(wx + 8, Y1 + 18, wn, ha='center', fontsize=9.5,
                 color=col, fontweight='bold', zorder=9)

    # T_i,j label on a connection arrow
    ymid = (Y0 + Y1) / 2
    ax2.text((well_x[0]+well_x[1])/2, ymid + 20, '$T_{i,j}$',
             ha='center', fontsize=10, color='#c62828', fontweight='bold', zorder=7)

    ax2.set_xlim(BX_start - 20, BX_end + 20)
    ax2.set_ylim(BY_start - 30, BY_end + 60)
    ax2.set_xlabel('X (m)', fontsize=11); ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('(b)  Voronoi-Diagram Model\n'
                  '(5 regions: P1–I1–P2–I2–P3)', fontsize=11, fontweight='bold', pad=6)
    ax2.set_aspect('equal')

    # Annotations: well specs
    ax2.text(BX_end - 10, BY_start + 30,
             f'Well length: {WELL_LEN:.0f} m\nSpacing: {SPACING:.0f} m',
             ha='right', fontsize=8.5, color='#333333',
             bbox=dict(boxstyle='round', fc='white', ec='#aaaaaa', lw=1), zorder=9)

    legend_handles = [
        mpatches.Patch(facecolor='#e8f5e9', edgecolor='#aaaaaa', label='Injector Voronoi region'),
        mpatches.Patch(facecolor='#dceefb', edgecolor='#aaaaaa', label='Producer Voronoi region'),
        Line2D([0],[0], color='#d32f2f', lw=3, label='Injector well (horiz.)'),
        Line2D([0],[0], color='#1565c0', lw=3, label='Producer well (horiz.)'),
        Line2D([0],[0], color='#c62828', lw=1.8, label='Inter-well connection ($T_{i,j}$)'),
        Line2D([0],[0], color='#555555', lw=1.2, ls='--', label='Voronoi boundary'),
    ]
    ax2.legend(handles=legend_handles, loc='lower right',
               fontsize=8, facecolor='white', framealpha=0.95, edgecolor='#aaaaaa')

    fig.suptitle('Pelican Lake HP-6 Pilot — Reservoir Grid Models\n'
                 '(5 horizontal wells: P1–I1–P2–I2–P3, well spacing = 175 m)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'fig9_voronoi.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — PINN structure  (closely matches ref Fig 4)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig10():
    fig = plt.figure(figsize=(18, 8))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18); ax.set_ylim(0, 9)
    ax.axis('off')

    # ── helpers ──────────────────────────────────────────────────────────────
    def circ(x, y, r=0.32, fc='white', ec='black', lw=1.8, zorder=4, label=None, fs=8.5):
        c = plt.Circle((x,y), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder)
        ax.add_patch(c)
        if label:
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', zorder=zorder+1)

    def arr(x1, y1, x2, y2, col='#555555', lw=1.4, style='->', rad=0.0):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                   connectionstyle=f'arc3,rad={rad}'),
                    zorder=6)

    def fancy_box(x, y, w, h, fc, ec, title, body_lines,
                  title_color='black', body_color='#1a1a1a',
                  title_fs=10.5, body_fs=9.5):
        ax.add_patch(FancyBboxPatch((x,y), w, h,
                                    boxstyle='round,pad=0.12',
                                    facecolor=fc, edgecolor=ec, lw=2.2, zorder=3))
        ax.text(x+w/2, y+h-0.28, title, ha='center', va='top',
                fontsize=title_fs, fontweight='bold', color=title_color, zorder=4)
        txt = '\n'.join(body_lines)
        ax.text(x+w/2, y+h-0.62, txt, ha='center', va='top',
                fontsize=body_fs, color=body_color, linespacing=1.65,
                zorder=4)

    # ── layer column positions ───────────────────────────────────────────────
    X_IN  = 1.1
    X_H1  = 3.6
    X_H2  = 5.6
    X_H3  = 7.6
    X_OUT = 10.2
    N_H   = 5   # circles per hidden layer (represents 64 neurons)

    inp_ys  = [7.2, 4.5, 1.8]
    hid_ys  = [7.5, 6.0, 4.5, 3.0, 1.5]
    out_ys  = [6.5, 2.5]

    inp_labels = ['t', 'T$_{start}$', 'Q$_{inj}$']
    out_labels = ['$\\widehat{WC}$', '$\\widehat{Q}_{oil}$']

    # input circles
    for lbl, y in zip(inp_labels, inp_ys):
        circ(X_IN, y, fc='#e8f5e9', ec='#2e7d32', lw=2.2, label=lbl, fs=9)

    # hidden layers
    for xh in [X_H1, X_H2, X_H3]:
        for y in hid_ys:
            circ(xh, y, fc='#e3f2fd', ec='#1565c0', lw=1.6)

    # output circles
    for lbl, y in zip(out_labels, out_ys):
        circ(X_OUT, y, fc='#fff9c4', ec='#f57f17', lw=2.2, label=lbl, fs=9.5)

    # connection lines (thin gray)
    for iy in inp_ys:
        for hy in hid_ys:
            ax.plot([X_IN+0.32, X_H1-0.32], [iy, hy], color='#cccccc', lw=0.45, zorder=1)
    for y1 in hid_ys:
        for y2 in hid_ys:
            ax.plot([X_H1+0.32, X_H2-0.32], [y1, y2], color='#cccccc', lw=0.45, zorder=1)
            ax.plot([X_H2+0.32, X_H3-0.32], [y1, y2], color='#cccccc', lw=0.45, zorder=1)
    for hy in hid_ys:
        for oy in out_ys:
            ax.plot([X_H3+0.32, X_OUT-0.32], [hy, oy], color='#cccccc', lw=0.45, zorder=1)

    # ── layer labels ─────────────────────────────────────────────────────────
    for x, lbl, col in [
        (X_IN,  'Input layer',    '#2e7d32'),
        (X_H2,  'Hidden layers',  '#1565c0'),
        (X_OUT, 'Output layer',   '#f57f17'),
    ]:
        ax.text(x, 0.55, lbl, ha='center', fontsize=10.5,
                color=col, fontweight='bold')

    # ── Learnable Parameters box (teal, right of output) ─────────────────────
    fancy_box(11.5, 3.8, 2.5, 4.6,
              fc='#e0f2f1', ec='#00796b',
              title='Learnable Parameters',
              title_color='#00695c',
              body_lines=[
                  '$W_{i,j}$   weights',
                  '$b_i$   biases',
                  '3 × Dense (64)',
                  'Activation: Tanh',
                  'Output: Sigmoid',
              ],
              body_fs=9)

    # ── Arrows: output → learnable params ────────────────────────────────────
    for oy in out_ys:
        arr(X_OUT+0.32, oy, 11.5, 6.0, col='#00796b', lw=1.6, rad=0.0)

    # ── k_rw / k_ro labels (intermediate, like paper) ───────────────────────
    ax.text(14.5, 7.3,
            '$k_{rw}$\n$k_{ro}$\n$\\mu_w(C_p)$\n$f_w$ BL',
            ha='center', va='center', fontsize=10.5,
            color='#1a237e', fontweight='bold')

    arr(14.0, 6.15, 14.0, 6.9, col='#1a237e', lw=1.6)

    # ── Data Loss box (blue) ─────────────────────────────────────────────────
    fancy_box(14.8, 5.5, 3.0, 2.8,
              fc='#e3f2fd', ec='#1565c0',
              title='Data Loss',
              title_color='#0d47a1',
              body_lines=[
                  r'$\hat{q} = WI \frac{k_{rw}}{\mu_w B_w}(P_j - P^{wf})$',
                  r'$\mathcal{L}_D = \frac{1}{N}\sum[(WC - \hat{WC})^2$',
                  r'$\qquad\qquad + (Q_{oil} - \hat{Q}_{oil})^2]$',
              ],
              title_fs=10.5, body_fs=8.5, body_color='#0d47a1')

    # ── Physics Loss box (red) ───────────────────────────────────────────────
    fancy_box(14.8, 1.5, 3.0, 3.6,
              fc='#fff3e0', ec='#e65100',
              title='Physics Loss',
              title_color='#bf360c',
              body_lines=[
                  r'$\mathcal{L}_P = \sum\mathrm{ReLU}(-\Delta WC)^2$',
                  r'$\quad\quad + \sum\mathrm{ReLU}(\Delta Q_{oil})^2$',
                  '',
                  r'$\mathcal{L} = \mathcal{L}_D + \lambda(t)\cdot\mathcal{L}_P$',
                  r'$\lambda(t) = \lambda_{max}\cdot\min(1,\, t/t_{warm})$',
              ],
              title_fs=10.5, body_fs=8.5, body_color='#bf360c')

    # ── Arrow: learnable params → physics loss ───────────────────────────────
    arr(14.0, 3.8, 14.8, 3.3, col='#e65100', lw=1.8, rad=0.0)
    # Arrow: output → data loss
    arr(X_OUT+0.32, 6.5, 14.8, 6.8, col='#1565c0', lw=1.6, rad=-0.2)
    arr(X_OUT+0.32, 2.5, 14.8, 6.0, col='#1565c0', lw=1.6, rad=0.3)
    # Arrow: output → physics loss
    arr(X_OUT+0.32, 6.5, 14.8, 3.1, col='#e65100', lw=1.6, rad=0.3)
    arr(X_OUT+0.32, 2.5, 14.8, 2.8, col='#e65100', lw=1.6, rad=-0.15)

    ax.text(9.0, 8.6,
            'Figure 4 — The structure of proposed PINN model',
            ha='center', va='center', fontsize=12,
            fontweight='bold', color='#1a1a2e')

    save(fig, 'fig10_pinn_structure.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Network architecture flowchart  (closely matches ref Fig 5)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig11():
    fig = plt.figure(figsize=(9, 14))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 9); ax.set_ylim(0, 15)
    ax.axis('off')

    CX = 4.0       # center x of main column
    W  = 4.6       # block width
    DX = 6.9       # x-start of dim annotations column
    DW = 1.9       # dim annotation box width

    # colors
    C_INP  = ('#c8e6c9', '#2e7d32')   # green  — input
    C_DNS  = ('#bbdefb', '#0d47a1')   # blue   — dense
    C_ACT  = ('#e1f5fe', '#0277bd')   # light blue — tanh
    C_SIG  = ('#e8eaf6', '#3949ab')   # indigo — sigmoid
    C_OUT  = ('#c8e6c9', '#2e7d32')   # green  — output

    def block(y_ctr, h, label, fc, ec, bold=True,
              dim_in=None, dim_out=None):
        bx = CX - W/2
        ax.add_patch(FancyBboxPatch((bx, y_ctr-h/2), W, h,
                                    boxstyle='round,pad=0.10',
                                    facecolor=fc, edgecolor=ec,
                                    lw=2.0, zorder=3))
        ax.text(CX, y_ctr, label, ha='center', va='center',
                fontsize=11, fontweight='bold' if bold else 'normal',
                color='#1a1a2e', zorder=4)

        if dim_in is not None:
            # right-side dim annotation box
            ax.add_patch(FancyBboxPatch((DX, y_ctr-h/2+0.04), DW, h-0.08,
                                        boxstyle='round,pad=0.08',
                                        facecolor='white', edgecolor='#aaaaaa',
                                        lw=1.0, zorder=3))
            ax.text(DX + DW/2, y_ctr + 0.13,
                    f'Input dim:  {dim_in}',
                    ha='center', va='center', fontsize=8.5,
                    color='#333333', zorder=4)
            ax.text(DX + DW/2, y_ctr - 0.17,
                    f'Output dim: {dim_out}',
                    ha='center', va='center', fontsize=8.5,
                    color='#333333', zorder=4)

    def darr(y_from, y_to):
        ax.annotate('', xy=(CX, y_to+0.02), xytext=(CX, y_from-0.02),
                    arrowprops=dict(arrowstyle='->', color='#333333',
                                   lw=1.8, mutation_scale=18),
                    zorder=5)

    # ── Blocks (y_center, height, label, style, dims) ────────────────────────
    layout = [
        # y,    h,    label,                     fc/ec,    dim_in, dim_out
        (13.8, 0.75, 'Input:  t,  T_start,  Q_inj', C_INP, '3',   '3'),
        (12.5, 0.75, 'Dense Layer 1',               C_DNS, '3',  '64'),
        (11.2, 0.65, 'Tanh Activation',             C_ACT, '64',  '64'),
        (10.1, 0.75, 'Dense Layer 2',               C_DNS, '64',  '64'),
        ( 8.8, 0.65, 'Tanh Activation',             C_ACT, '64',  '64'),
        ( 7.7, 0.75, 'Dense Layer 3',               C_DNS, '64',  '64'),
        ( 6.4, 0.65, 'Tanh Activation',             C_ACT, '64',  '64'),
        ( 5.3, 0.75, 'Dense Output Layer',          C_DNS, '64',   '2'),
        ( 4.0, 0.65, 'Sigmoid Activation',          C_SIG,  '2',   '2'),
        ( 2.8, 0.75, 'Output:  WC,  Q_oil_norm',   C_OUT,  '2',   '2'),
    ]

    for (y, h, lbl, (fc, ec), di, do) in layout:
        block(y, h, lbl, fc, ec, dim_in=di, dim_out=do)

    # Arrows between consecutive blocks
    for i in range(len(layout)-1):
        y_cur  = layout[i][0]   - layout[i][1]/2    # bottom of current
        y_next = layout[i+1][0] + layout[i+1][1]/2  # top of next
        darr(y_cur, y_next)

    # ── Physics constraint annotation (right margin) ─────────────────────────
    ann_cx = 8.4
    ax.add_patch(FancyBboxPatch((7.6, 5.6), 1.6, 4.8,
                                boxstyle='round,pad=0.1',
                                facecolor='#fff8e1', edgecolor='#f9a825',
                                lw=1.8, zorder=2))
    ax.text(ann_cx, 10.1, 'Physics\nConstraints',
            ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='#e65100', zorder=3)
    ax.text(ann_cx, 9.0,
            'dWC/dt ≥ 0\nWC monotone↑',
            ha='center', va='center', fontsize=8.5,
            color='#bf360c', zorder=3)
    ax.text(ann_cx, 7.7,
            'dQ_oil/dt ≤ 0\nOil decline↓',
            ha='center', va='center', fontsize=8.5,
            color='#bf360c', zorder=3)
    ax.text(ann_cx, 6.5,
            'λ warm-up\n0→0.10\nepoch 0–150',
            ha='center', va='center', fontsize=8,
            color='#555555', zorder=3)

    # Bracket arrow from physics box to tanh layers
    ax.annotate('', xy=(CX+W/2+0.05, 8.8),
                xytext=(7.6, 8.3),
                arrowprops=dict(arrowstyle='<-', color='#e65100', lw=1.6),
                zorder=5)

    ax.text(CX, 14.6,
            'Figure 5 — The architecture of the fully-connected\n'
            'network in the proposed PINN model',
            ha='center', va='center', fontsize=11.5,
            fontweight='bold', color='#1a1a2e')

    save(fig, 'fig11_network_arch.png')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('[FIG9 ] Voronoi diagram ...')
    make_fig9()
    print('[FIG10] PINN structure ...')
    make_fig10()
    print('[FIG11] Network architecture ...')
    make_fig11()
    print('\nDone.')
