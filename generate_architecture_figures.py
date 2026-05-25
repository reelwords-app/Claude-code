"""
Generate publication-quality architecture figures for PINN polymer flood paper.
Matches reference paper style (SPE-218863-MS, Meng et al. 2024).

Figures:
  fig9_voronoi.png          — Fine-grid + Voronoi (Pelican Lake HP-6 geometry)
  fig10_pinn_structure.png  — PINN structure diagram  (ref Fig 4)
  fig11_network_arch.png    — Network architecture flowchart (ref Fig 5)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import warnings, os

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
})
OUT_DIR = '/home/user/Claude-code/pinn_cmg_results/'
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [PLOT] {name}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Voronoi diagram  (Pelican Lake HP-6 actual geometry)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig9():
    """
    5 horizontal wells: P1–I1–P2–I2–P3
    Each 4593.176 ft ≈ 1400 m long (Y-direction)
    Spacing: 574.147 ft ≈ 175 m (X-direction)
    Grid: 157 × 10 × 3 (CMG STARS, Ugembe et al. manuscript)
    """
    SP   = 574.147 * 0.3048    # 175.0 m  — well spacing (X)
    WL   = 4593.176 * 0.3048   # 1400.0 m — well length (Y)
    BUFY = 1148.294 * 0.3048   # 350.1 m  — Y-axis buffer
    BUFX = SP                  # 175.0 m  — X-axis buffer

    # Field extent
    X0, X4 = 0.0, 4 * SP       # 0 to 700 m
    Y0_w, Y1_w = BUFY, BUFY + WL  # well extent in Y

    # Field bounding box (includes buffers)
    FX0, FX1 = -BUFX,   X4 + BUFX
    FY0, FY1 =  0.0,    BUFY + WL + BUFY

    # Well X positions: P1=0, I1=SP, P2=2SP, I2=3SP, P3=4SP
    WX   = [0, SP, 2*SP, 3*SP, 4*SP]
    WTYP = ['P', 'I', 'P', 'I', 'P']
    WNAM = ['P1', 'I1', 'P2', 'I2', 'P3']
    C_I  = '#c62828';  C_P = '#1565c0'   # injector red, producer blue

    fig, axes = plt.subplots(1, 2, figsize=(15, 8),
                              gridspec_kw={'wspace': 0.12})

    # ── (a) Fine-Grid Reservoir Model ────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('white')

    # Heterogeneous permeability background (schematic)
    np.random.seed(7)
    from scipy.ndimage import gaussian_filter
    raw = np.random.randn(80, 30)
    perm = gaussian_filter(raw, sigma=4)
    perm = (perm - perm.min()) / (perm.max() - perm.min())
    from matplotlib.colors import LinearSegmentedColormap
    cmap_perm = LinearSegmentedColormap.from_list(
        'perm', ['#08306b','#2166ac','#4dac26','#f7f700','#d73027'])
    ax.imshow(perm.T, extent=[FX0, FX1, FY0, FY1],
              origin='lower', cmap=cmap_perm, alpha=0.55,
              aspect='auto', zorder=0)

    # Grid lines (simplified 40×22 for visual)
    nx_g, ny_g = 40, 22
    for x in np.linspace(FX0, FX1, nx_g + 1):
        ax.axvline(x, color='#90a4ae', lw=0.3, alpha=0.7, zorder=1)
    for y in np.linspace(FY0, FY1, ny_g + 1):
        ax.axhline(y, color='#90a4ae', lw=0.3, alpha=0.7, zorder=1)

    # Colorbar (permeability scale)
    sm = plt.cm.ScalarMappable(cmap=cmap_perm,
                               norm=plt.Normalize(100, 5000))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=20)
    cbar.set_label('Permeability (md)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Horizontal wells
    for wx, wt, wn in zip(WX, WTYP, WNAM):
        col = C_I if wt == 'I' else C_P
        lw  = 4.5 if wt == 'I' else 4.0
        ax.plot([wx, wx], [Y0_w, Y1_w], color=col, lw=lw,
                solid_capstyle='round', zorder=5)
        ax.text(wx, Y1_w + 22, wn, ha='center', va='bottom',
                fontsize=10, color=col, fontweight='bold', zorder=6)

    ax.set_xlim(FX0 - 10, FX1 + 10)
    ax.set_ylim(FY0 - 20, FY1 + 55)
    ax.set_xlabel('X (m)', fontsize=11, labelpad=6)
    ax.set_ylabel('Y (m)', fontsize=11, labelpad=6)
    ax.set_title('(a)  Fine-Grid Reservoir Model\n'
                 r'157 $\times$ 10 $\times$ 3 cells — heterogeneous $k_h$',
                 fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(labelsize=9)
    ax.set_aspect('equal', adjustable='box')

    # ── (b) Voronoi-Diagram Model ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('white')

    # Voronoi regions = vertical bands (accurate for parallel horizontal wells)
    # Boundaries at midpoints: -BUFX, SP/2, 3SP/2, 5SP/2, 7SP/2, 4SP+BUFX
    boundaries = [FX0, SP/2, 3*SP/2, 5*SP/2, 7*SP/2, FX1]
    fc_map = {'I': '#ffebee', 'P': '#e3f2fd'}
    ec_map = {'I': C_I,       'P': C_P}

    for i, (wt, wn) in enumerate(zip(WTYP, WNAM)):
        xL, xR = boundaries[i], boundaries[i+1]
        rect = mpatches.FancyBboxPatch(
            (xL, FY0), xR - xL, FY1 - FY0,
            boxstyle='square,pad=0',
            facecolor=fc_map[wt], edgecolor='#888888', lw=1.0,
            alpha=0.85, zorder=1)
        ax2.add_patch(rect)
        # Region label (centred)
        ax2.text((xL + xR)/2, (FY0 + FY1)/2,
                 f'{wn}\nregion', ha='center', va='center',
                 fontsize=10, color=ec_map[wt],
                 fontweight='bold', alpha=0.60, zorder=2)

    # Voronoi boundary lines
    for xb in boundaries[1:-1]:
        ax2.axvline(xb, color='#555555', lw=1.4, ls='--', zorder=3, alpha=0.8)

    # Delaunay-style inter-well connections (horizontal arrows at well midpoint)
    ymid = (Y0_w + Y1_w) / 2
    for i in range(len(WX) - 1):
        ax2.annotate('', xy=(WX[i+1], ymid), xytext=(WX[i], ymid),
                     arrowprops=dict(arrowstyle='->', color='#c62828',
                                     lw=1.8, mutation_scale=16),
                     zorder=5)

    # T_ij label on first connection
    ax2.text((WX[0]+WX[1])/2, ymid + 28,
             r'$T_{i,j}$', ha='center', fontsize=11,
             color='#c62828', fontweight='bold', zorder=6)

    # Horizontal wells (on top of Voronoi bands)
    for wx, wt, wn in zip(WX, WTYP, WNAM):
        col = C_I if wt == 'I' else C_P
        mk  = 's' if wt == 'I' else 'o'
        ax2.plot([wx, wx], [Y0_w, Y1_w], color=col, lw=4.5,
                 solid_capstyle='round', zorder=6)
        ax2.scatter(wx, ymid, s=120, marker=mk, color=col,
                    edgecolors='white', linewidths=1.5, zorder=8)
        ax2.text(wx, Y1_w + 22, wn, ha='center', va='bottom',
                 fontsize=10, color=col, fontweight='bold', zorder=9)

    # Field boundary
    for spine in ax2.spines.values():
        spine.set_linewidth(1.8); spine.set_edgecolor('#333333')

    ax2.set_xlim(FX0 - 10, FX1 + 10)
    ax2.set_ylim(FY0 - 20, FY1 + 55)
    ax2.set_xlabel('X (m)', fontsize=11, labelpad=6)
    ax2.set_ylabel('Y (m)', fontsize=11, labelpad=6)
    ax2.set_title('(b)  Voronoi-Diagram Model\n'
                  r'5 drainage regions — connections $T_{i,j}$',
                  fontsize=11, fontweight='bold', pad=10)
    ax2.tick_params(labelsize=9)
    ax2.set_aspect('equal', adjustable='box')

    # ── Shared legend — BELOW both panels ────────────────────────────────────
    leg_handles = [
        Line2D([0],[0], color=C_I, lw=4, solid_capstyle='round',
               label='Injector well (I1, I2)'),
        Line2D([0],[0], color=C_P, lw=4, solid_capstyle='round',
               label='Producer well (P1, P2, P3)'),
        mpatches.Patch(facecolor='#ffebee', edgecolor='#888888',
                       label='Injector Voronoi region'),
        mpatches.Patch(facecolor='#e3f2fd', edgecolor='#888888',
                       label='Producer Voronoi region'),
        Line2D([0],[0], color='#c62828', lw=1.8,
               label=r'Inter-well transmissibility ($T_{i,j}$)'),
        Line2D([0],[0], color='#555555', lw=1.4, ls='--',
               label='Voronoi boundary'),
    ]
    fig.legend(handles=leg_handles, ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, -0.04), fontsize=9.5,
               frameon=True, edgecolor='#cccccc', framealpha=0.95,
               columnspacing=1.2, handlelength=2.0)

    fig.suptitle('Pelican Lake HP-6 Polymer Flood Pilot — Reservoir Models\n'
                 '2 injectors × 3 producers, horizontal wells, 1 400 m length, 175 m spacing',
                 fontsize=12, fontweight='bold', y=1.02)

    save(fig, 'fig9_voronoi.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — PINN structure  (matches ref Fig 4 style)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig10():
    fig = plt.figure(figsize=(20, 9))
    ax  = fig.add_axes([0.01, 0.05, 0.98, 0.88])
    ax.set_xlim(0, 20); ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('white')

    R = 0.38   # circle radius

    # ── helpers ──────────────────────────────────────────────────────────────
    def circ(x, y, r=R, fc='white', ec='#333333', lw=2.0, zorder=4):
        ax.add_patch(plt.Circle((x, y), r, facecolor=fc,
                                edgecolor=ec, lw=lw, zorder=zorder))

    def ctext(x, y, txt, fs=9, fw='normal', col='#1a1a1a', zorder=5):
        ax.text(x, y, txt, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=col, zorder=zorder)

    def conn_lines(xs, ys_from, ys_to, col='#d0d0d0', lw=0.5):
        for yf in ys_from:
            for yt in ys_to:
                ax.plot([xs[0]+R, xs[1]-R], [yf, yt],
                        color=col, lw=lw, zorder=1)

    def box(x, y, w, h, fc, ec, lw=2.0, zorder=3, radius=0.15):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f'round,pad={radius}',
            facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder))

    def arrow(x1, y1, x2, y2, col='#555555', lw=1.6, rad=0.0, zorder=6):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color=col, lw=lw,
                        mutation_scale=18,
                        connectionstyle=f'arc3,rad={rad}'),
                    zorder=zorder)

    # ── Column x-positions ───────────────────────────────────────────────────
    XI  = 1.2                       # input layer
    XH  = [3.5, 5.5, 7.5]          # hidden layers
    XO  = 9.8                       # output layer
    XP  = 11.6                      # learnable params box left edge
    XL  = 14.8                      # loss boxes left edge
    BOX_W_P = 2.5                   # params box width
    BOX_W_L = 4.6                   # loss boxes width

    # ── Node y-positions ─────────────────────────────────────────────────────
    inp_ys = [8.0, 6.5, 5.0, 3.5, 2.0]   # 5 input nodes (3 real + 2 dots)
    hid_ys = [8.2, 7.0, 5.5, 4.0, 2.8, 1.6]  # 6 hidden nodes per layer
    out_ys = [7.2, 5.0, 2.8]              # 3 output nodes

    # ── Layer header labels ───────────────────────────────────────────────────
    for x, lbl in [(XI, 'Input layer'), (XH[1], 'Hidden layers'), (XO, 'Output layer')]:
        ax.text(x, 9.55, lbl, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color='#212121')

    # ── Input nodes + labels (left-side) ─────────────────────────────────────
    inp_labels = ['$Q_1$', '...', '$Q_{nI}$', '$P_1^{wf}$', '$t$']
    for lbl, y in zip(inp_labels, inp_ys):
        circ(XI, y, fc='white', ec='#333333', lw=2.0)
        ax.text(XI - R - 0.05, y, lbl, ha='right', va='center',
                fontsize=10, color='#1a1a1a')

    # ── Hidden layer circles ──────────────────────────────────────────────────
    for xh in XH:
        for y in hid_ys:
            circ(xh, y, fc='white', ec='#333333', lw=1.8)

    # ── Output nodes + labels (right-side) ───────────────────────────────────
    out_labels = ['$\\hat{P}_1$', '$\\hat{S}_{wc}$', '$\\hat{Q}_{oil}$']
    for lbl, y in zip(out_labels, out_ys):
        circ(XO, y, fc='white', ec='#333333', lw=2.0)
        ax.text(XO + R + 0.05, y, lbl, ha='left', va='center',
                fontsize=10, color='#1a1a1a')

    # ── Connection lines (thin gray) ──────────────────────────────────────────
    conn_lines([XI, XH[0]],  inp_ys, hid_ys)
    conn_lines([XH[0], XH[1]], hid_ys, hid_ys)
    conn_lines([XH[1], XH[2]], hid_ys, hid_ys)
    conn_lines([XH[2], XO],  hid_ys, out_ys)

    # ── Learnable Parameters box (teal) ──────────────────────────────────────
    LP_Y = 2.8; LP_H = 5.5
    box(XP, LP_Y, BOX_W_P, LP_H,
        fc='#e0f7fa', ec='#00838f', lw=2.2)
    ax.text(XP + BOX_W_P/2, LP_Y + LP_H - 0.28,
            'Learnable Parameters',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#006064')
    params = [
        r'$T_{i,j}$',
        r'$V^0_{p,i}$',
        r'$WI_i$',
        r"$k'_{rw}$",
        r"$k'_{ro}$",
        r'$S_{wc}$',
        r'$S_{or}$',
    ]
    for k, p in enumerate(params):
        ax.text(XP + BOX_W_P/2, LP_Y + LP_H - 0.85 - k*0.60,
                p, ha='center', va='top',
                fontsize=10.5, color='#004d40', style='italic')

    # ── Physical property labels (between param box and loss boxes) ───────────
    mid_x = (XP + BOX_W_P + XL) / 2
    for k, lbl in enumerate(['$k_{rw}$', '$k_{ro}$',
                              r'$T_{w,i,j}$', r'$T_{o,i,j}$', r'$V_{p,i}$']):
        ax.text(mid_x, 7.6 - k * 0.72, lbl,
                ha='center', va='center', fontsize=10.5,
                color='#212121', fontweight='bold')
    arrow(XP + BOX_W_P + 0.08, 5.55, XL - 0.12, 7.2,
          col='#00838f', lw=1.6)

    # ── Data Loss box (blue) ─────────────────────────────────────────────────
    DL_Y = 5.8; DL_H = 3.5
    box(XL, DL_Y, BOX_W_L, DL_H,
        fc='#e8f4fd', ec='#1565c0', lw=2.2)
    ax.text(XL + BOX_W_L/2, DL_Y + DL_H - 0.28,
            'Data Loss', ha='center', va='top',
            fontsize=11, fontweight='bold', color='#0d47a1')
    dl_lines = [
        r'$\hat{q}_{o} = WI \frac{k_{ro}}{\mu_o B_o}(P_j - P^{wf})$',
        r'$\hat{q}_{w} = WI \frac{k_{rw}}{\mu_w B_w}(P_j - P^{wf})$',
        r'$\mathcal{L}_D = \frac{1}{nP}\sum[(q_o - \hat{q}_o)^2 + (q_w - \hat{q}_w)^2]$',
    ]
    for k, line in enumerate(dl_lines):
        ax.text(XL + BOX_W_L/2, DL_Y + DL_H - 0.82 - k * 0.82,
                line, ha='center', va='top',
                fontsize=8.8, color='#0d47a1')

    # ── Physics Loss box (red) ────────────────────────────────────────────────
    PL_Y = 1.0; PL_H = 4.4
    box(XL, PL_Y, BOX_W_L, PL_H,
        fc='#fff8e1', ec='#e65100', lw=2.2)
    ax.text(XL + BOX_W_L/2, PL_Y + PL_H - 0.28,
            'Physics Loss', ha='center', va='top',
            fontsize=11, fontweight='bold', color='#bf360c')
    pl_lines = [
        r'$\mathcal{L}_P = \sum T_{w,i,j}[\hat{P}_j - \hat{P}_i]$',
        r'$\quad + \hat{q}_{w,i} - \frac{\partial}{\partial t}\!\left(\frac{V_{p,i}\hat{S}_{w,i}}{B_w}\right)$',
        r'$+ \sum T_{o,i,j}[\hat{P}_j - \hat{P}_i]$',
        r'$\quad + \hat{q}_{o,i} - \frac{\partial}{\partial t}\!\left(\frac{V_{p,i}(1-\hat{S}_{w,i})}{B_o}\right)$',
        r'$\mathcal{L} = \mathcal{L}_D + \lambda\,\mathcal{L}_P$',
    ]
    for k, line in enumerate(pl_lines):
        ax.text(XL + BOX_W_L/2, PL_Y + PL_H - 0.82 - k * 0.68,
                line, ha='center', va='top',
                fontsize=8.8, color='#bf360c')

    # ── Arrows from output → param box ───────────────────────────────────────
    for oy in out_ys:
        arrow(XO + R, oy, XP - 0.05, LP_Y + LP_H/2,
              col='#00838f', lw=1.4, rad=0.0)

    # ── Arrows from output → data loss ───────────────────────────────────────
    arrow(XO + R, out_ys[0], XL - 0.05, DL_Y + DL_H * 0.75,
          col='#1565c0', lw=1.5, rad=-0.15)
    arrow(XO + R, out_ys[1], XL - 0.05, DL_Y + DL_H * 0.4,
          col='#1565c0', lw=1.5, rad=0.05)

    # ── Arrows from output → physics loss ────────────────────────────────────
    arrow(XO + R, out_ys[1], XL - 0.05, PL_Y + PL_H * 0.75,
          col='#e65100', lw=1.5, rad=0.15)
    arrow(XO + R, out_ys[2], XL - 0.05, PL_Y + PL_H * 0.40,
          col='#e65100', lw=1.5, rad=-0.05)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(10.0, 9.80,
            'Figure 4 — The structure of proposed PINN model',
            ha='center', va='center', fontsize=13,
            fontweight='bold', color='#1a1a2e')

    save(fig, 'fig10_pinn_structure.png')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Network architecture flowchart  (matches ref Fig 5)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig11():
    fig = plt.figure(figsize=(9, 14))
    ax  = fig.add_axes([0.04, 0.02, 0.92, 0.96])
    ax.set_xlim(0, 9); ax.set_ylim(0, 15)
    ax.axis('off')

    CX = 3.8       # center x of main column
    W  = 4.2       # block width
    DX = 6.4       # x-start of dim annotation column
    DW = 2.2       # dim annotation width

    C_INP = ('#c8e6c9', '#2e7d32')
    C_DNS = ('#bbdefb', '#0d47a1')
    C_ACT = ('#e1f5fe', '#0277bd')
    C_SIG = ('#e8eaf6', '#3949ab')
    C_OUT = ('#c8e6c9', '#2e7d32')

    def block(yc, h, label, fc, ec, dim_in=None, dim_out=None):
        ax.add_patch(FancyBboxPatch(
            (CX - W/2, yc - h/2), W, h,
            boxstyle='round,pad=0.10',
            facecolor=fc, edgecolor=ec, lw=2.0, zorder=3))
        ax.text(CX, yc, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#1a1a2e', zorder=4)
        if dim_in is not None:
            ax.add_patch(FancyBboxPatch(
                (DX, yc - h/2 + 0.04), DW, h - 0.08,
                boxstyle='round,pad=0.06',
                facecolor='white', edgecolor='#90a4ae',
                lw=1.0, zorder=3))
            ax.text(DX + DW/2, yc + 0.13, f'Input dim:  {dim_in}',
                    ha='center', va='center', fontsize=8.5, color='#333', zorder=4)
            ax.text(DX + DW/2, yc - 0.17, f'Output dim: {dim_out}',
                    ha='center', va='center', fontsize=8.5, color='#333', zorder=4)

    def darr(y_from, y_to):
        ax.annotate('', xy=(CX, y_to + 0.02), xytext=(CX, y_from - 0.02),
                    arrowprops=dict(arrowstyle='->', color='#333333',
                                   lw=1.8, mutation_scale=18), zorder=5)

    layout = [
        (13.8, 0.78, 'Input:  t,  T_start,  Q_inj', C_INP, '3',  '3'),
        (12.5, 0.78, 'Dense Layer 1',               C_DNS, '3',  '64'),
        (11.2, 0.65, 'Tanh Activation',             C_ACT, '64', '64'),
        (10.1, 0.78, 'Dense Layer 2',               C_DNS, '64', '64'),
        ( 8.8, 0.65, 'Tanh Activation',             C_ACT, '64', '64'),
        ( 7.7, 0.78, 'Dense Layer 3',               C_DNS, '64', '64'),
        ( 6.4, 0.65, 'Tanh Activation',             C_ACT, '64', '64'),
        ( 5.3, 0.78, 'Dense Output Layer',          C_DNS, '64',  '2'),
        ( 4.0, 0.65, 'Sigmoid Activation',          C_SIG,  '2',  '2'),
        ( 2.8, 0.78, 'Output:  WC,  Q_oil_norm',   C_OUT,  '2',  '2'),
    ]

    for (y, h, lbl, (fc, ec), di, do) in layout:
        block(y, h, lbl, fc, ec, dim_in=di, dim_out=do)

    for i in range(len(layout) - 1):
        bot = layout[i][0] - layout[i][1]/2
        top = layout[i+1][0] + layout[i+1][1]/2
        darr(bot, top)

    # Physics constraint sidebar
    ax.add_patch(FancyBboxPatch(
        (0.2, 5.5), 1.9, 5.0,
        boxstyle='round,pad=0.12',
        facecolor='#fff8e1', edgecolor='#f9a825', lw=1.8, zorder=2))
    ax.text(1.15, 10.2, 'Physics\nConstraints',
            ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='#e65100', zorder=3)
    ax.text(1.15, 9.0, 'WC monotone\ndWC/dt ≥ 0',
            ha='center', va='center', fontsize=8.5, color='#bf360c', zorder=3)
    ax.text(1.15, 7.7, 'Oil decline\ndQ/dt ≤ 0',
            ha='center', va='center', fontsize=8.5, color='#bf360c', zorder=3)
    ax.text(1.15, 6.5, 'λ warm-up\n0 → 0.10\nepoch 0–150',
            ha='center', va='center', fontsize=7.5, color='#555555', zorder=3)
    ax.annotate('', xy=(CX - W/2 - 0.04, 8.8),
                xytext=(0.2 + 1.9, 8.5),
                arrowprops=dict(arrowstyle='->', color='#e65100', lw=1.6),
                zorder=5)

    ax.text(CX, 14.7, 'Figure 5 — The architecture of the fully-connected\n'
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
