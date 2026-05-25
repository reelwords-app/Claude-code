"""
Publication-quality architecture figures — PINN polymer flood paper.
Matches SPE-218863-MS (Meng et al. 2024) style.

Figures:
  fig9_voronoi.png   — Voronoi-diagram model (Pelican Lake HP-6 geometry)
  fig10_pinn_structure.png — PINN structure (ref Fig 4)
  fig11_network_arch.png   — Network architecture (ref Fig 5)
  fig12_equations.png      — All model equations reference sheet
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import warnings, os

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family': 'DejaVu Sans'})
OUT_DIR = '/home/user/Claude-code/pinn_cmg_results/'
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [PLOT] {name}')


# =============================================================================
# Figure 9 — Voronoi-Diagram Model  (single panel, HP-6 geometry)
# =============================================================================
def make_fig9():
    """
    Pelican Lake HP-6: 5 horizontal wells P1–I1–P2–I2–P3
    Well length  4593.176 ft = 1400 m  (Y-axis)
    Well spacing  574.147 ft =  175 m  (X-axis)
    Source: Ugembe et al. manuscript, Section 2.2
    """
    SP   = 574.147 * 0.3048    # 175.0 m  spacing
    WL   = 4593.176 * 0.3048   # 1400.0 m length
    BUFY = 1148.294 * 0.3048   # 350.1 m  buffer in Y
    BUFX = SP * 0.9            # buffer in X

    Y0_w, Y1_w = BUFY, BUFY + WL
    FX0, FX1   = -BUFX,  4*SP + BUFX
    FY0, FY1   =  0.0,   BUFY + WL + BUFY
    ymid = (Y0_w + Y1_w) / 2

    WX   = [0.0, SP, 2*SP, 3*SP, 4*SP]
    WTYP = ['P', 'I', 'P', 'I', 'P']
    WNAM = ['P1', 'I1', 'P2', 'I2', 'P3']
    C_I, C_P = '#c62828', '#1565c0'

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_linewidth(1.6); sp.set_edgecolor('#444')

    # Voronoi bands (vertical strips — correct for parallel horizontal wells)
    bounds = [FX0, SP/2, 3*SP/2, 5*SP/2, 7*SP/2, FX1]
    fc_map = {'I': '#fce4ec', 'P': '#e3f2fd'}
    hat    = {'I': '///',    'P': ''}
    for i, (wt, wn) in enumerate(zip(WTYP, WNAM)):
        xL, xR = bounds[i], bounds[i+1]
        ax.add_patch(mpatches.Rectangle(
            (xL, FY0), xR-xL, FY1-FY0,
            facecolor=fc_map[wt], edgecolor='none',
            hatch=hat[wt], alpha=0.80, zorder=1))
        ax.text((xL+xR)/2, FY0+35, wn, ha='center', va='bottom',
                fontsize=9, color='#555', style='italic', zorder=2)

    # Voronoi boundary dashes
    for xb in bounds[1:-1]:
        ax.axvline(xb, color='#555', lw=1.2, ls='--', zorder=3, alpha=0.7)

    # Inter-region connections with T_ij labels
    for i in range(len(WX)-1):
        ax.annotate('', xy=(WX[i+1], ymid), xytext=(WX[i], ymid),
                    arrowprops=dict(arrowstyle='->', color='#c62828',
                                   lw=2.0, mutation_scale=18), zorder=5)
        ax.text((WX[i]+WX[i+1])/2, ymid+35, r'$T_{i,j}$',
                ha='center', fontsize=10, color='#c62828',
                fontweight='bold', zorder=6)

    # Wells
    for wx, wt, wn in zip(WX, WTYP, WNAM):
        col = C_I if wt=='I' else C_P
        mk  = 'D' if wt=='I' else 'o'
        ax.plot([wx,wx], [Y0_w,Y1_w], color=col, lw=5.5,
                solid_capstyle='round', zorder=6)
        ax.scatter(wx, ymid, s=170, marker=mk, color=col,
                   edgecolors='white', linewidths=1.8, zorder=8)
        ax.text(wx, Y1_w+30, wn, ha='center', va='bottom',
                fontsize=12, color=col, fontweight='bold', zorder=9)

    # Scale bar
    sb_y = FY0+55
    ax.plot([FX0+20, FX0+20+SP], [sb_y, sb_y], color='#333', lw=2.5, zorder=7)
    ax.text(FX0+20+SP/2, sb_y+20, '175 m',
            ha='center', fontsize=9, color='#333', zorder=8)

    # Well-length bracket
    ax.annotate('', xy=(FX1-12, Y1_w), xytext=(FX1-12, Y0_w),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.4))
    ax.text(FX1-8, ymid, '1 400 m',
            ha='left', va='center', fontsize=9, color='#555')

    ax.set_xlim(FX0-15, FX1+80)
    ax.set_ylim(FY0-15, FY1+70)
    ax.set_xlabel('X (m)', fontsize=12, labelpad=6)
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=6)
    ax.tick_params(labelsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Voronoi-Diagram Model — Pelican Lake HP-6 Pilot\n'
                 r'P1 – I1 – P2 – I2 – P3  $|$  175 m spacing  $|$  1 400 m well length',
                 fontsize=12, fontweight='bold', pad=12)

    handles = [
        Line2D([0],[0], color=C_I, lw=5, solid_capstyle='round', label='Injector (I1, I2)'),
        Line2D([0],[0], color=C_P, lw=5, solid_capstyle='round', label='Producer (P1, P2, P3)'),
        mpatches.Patch(facecolor='#fce4ec', edgecolor='#aaa', hatch='///', label='Injector region'),
        mpatches.Patch(facecolor='#e3f2fd', edgecolor='#aaa', label='Producer region'),
        Line2D([0],[0], color='#c62828', lw=2, label=r'Connection $T_{i,j}$'),
        Line2D([0],[0], color='#555', lw=1.2, ls='--', label='Voronoi boundary'),
    ]
    ax.legend(handles=handles, ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.08), fontsize=10,
              frameon=True, edgecolor='#ccc', framealpha=0.97,
              columnspacing=1.5, handlelength=2.2, borderpad=0.9)

    fig.tight_layout()
    save(fig, 'fig9_voronoi.png')


# =============================================================================
# Figure 10 — PINN Structure  (ref Fig 4)
# =============================================================================
def make_fig10():
    fig = plt.figure(figsize=(20, 9))
    ax  = fig.add_axes([0.01, 0.05, 0.98, 0.88])
    ax.set_xlim(0, 20); ax.set_ylim(0, 10)
    ax.axis('off'); ax.set_facecolor('white')

    R = 0.38

    def circ(x, y, r=R, fc='white', ec='#333', lw=2.0, z=4):
        ax.add_patch(plt.Circle((x,y), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))

    def conn(xs, ys_from, ys_to):
        for yf in ys_from:
            for yt in ys_to:
                ax.plot([xs[0]+R, xs[1]-R], [yf, yt], color='#d0d0d0', lw=0.5, zorder=1)

    def box(x, y, w, h, fc, ec, lw=2.2, z=3):
        ax.add_patch(FancyBboxPatch((x,y), w, h,
                     boxstyle='round,pad=0.14', facecolor=fc, edgecolor=ec, lw=lw, zorder=z))

    def arr(x1,y1,x2,y2, col='#555', lw=1.6, rad=0.0):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                   mutation_scale=18,
                                   connectionstyle=f'arc3,rad={rad}'), zorder=6)

    XI=1.2; XH=[3.5,5.5,7.5]; XO=9.8; XP=11.6; XL=14.85
    BPW=2.5; BLW=4.7
    inp_ys=[8.0,6.5,5.0,3.5,2.0]
    hid_ys=[8.2,7.0,5.5,4.0,2.8,1.6]
    out_ys=[7.2,5.0,2.8]

    # Layer headers
    for x,lbl in [(XI,'Input layer'),(XH[1],'Hidden layers'),(XO,'Output layer')]:
        ax.text(x, 9.55, lbl, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color='#111')

    # Input nodes
    inp_lbl = [r'$Q_1$', r'$\cdots$', r'$Q_{nI}$', r'$P_1^{wf}$', r'$t$']
    for lbl,y in zip(inp_lbl, inp_ys):
        circ(XI, y); ax.text(XI-R-0.07, y, lbl, ha='right', va='center', fontsize=10)

    # Hidden layers
    for xh in XH:
        for y in hid_ys: circ(xh, y)

    # Output nodes
    out_lbl = [r'$\hat{P}_1$', r'$\hat{S}_{wc}$', r'$\hat{Q}_{oil}$']
    for lbl,y in zip(out_lbl, out_ys):
        circ(XO, y); ax.text(XO+R+0.07, y, lbl, ha='left', va='center', fontsize=10)

    # Connections
    conn([XI,XH[0]], inp_ys, hid_ys)
    conn([XH[0],XH[1]], hid_ys, hid_ys)
    conn([XH[1],XH[2]], hid_ys, hid_ys)
    conn([XH[2],XO], hid_ys, out_ys)

    # Learnable Parameters (teal)
    LP_Y=2.7; LP_H=5.8
    box(XP, LP_Y, BPW, LP_H, '#e0f7fa', '#00838f')
    ax.text(XP+BPW/2, LP_Y+LP_H-0.28, 'Learnable Parameters',
            ha='center', va='top', fontsize=11, fontweight='bold', color='#006064')
    for k,p in enumerate([r'$T_{i,j}$',r'$V^0_{p,i}$',r'$WI_i$',
                           r"$k'_{rw}$",r"$k'_{ro}$",r'$S_{wc}$',r'$S_{or}$']):
        ax.text(XP+BPW/2, LP_Y+LP_H-0.85-k*0.62,
                p, ha='center', va='top', fontsize=10.5, color='#004d40', style='italic')

    # Intermediate labels
    mx = (XP+BPW+XL)/2
    for k,lbl in enumerate([r'$k_{rw}$',r'$k_{ro}$',r'$T_{w,i,j}$',r'$T_{o,i,j}$',r'$V_{p,i}$']):
        ax.text(mx, 7.7-k*0.72, lbl, ha='center', va='center',
                fontsize=10.5, color='#111', fontweight='bold')
    arr(XP+BPW+0.08, 5.55, XL-0.1, 7.2, col='#00838f')

    # Data Loss (blue)
    DL_Y=5.9; DL_H=3.4
    box(XL, DL_Y, BLW, DL_H, '#e8f4fd', '#1565c0')
    ax.text(XL+BLW/2, DL_Y+DL_H-0.28, 'Data Loss',
            ha='center', va='top', fontsize=11, fontweight='bold', color='#0d47a1')
    for k,ln in enumerate([
        r'$\hat{q}_{o} = WI\,\frac{k_{ro}}{\mu_o B_o}(P_j - P^{wf})$',
        r'$\hat{q}_{w} = WI\,\frac{k_{rw}}{\mu_w B_w}(P_j - P^{wf})$',
        r'$\mathcal{L}_D = \frac{1}{nP}\sum\![(q_o-\hat{q}_o)^2+(q_w-\hat{q}_w)^2]$',
    ]):
        ax.text(XL+BLW/2, DL_Y+DL_H-0.84-k*0.84,
                ln, ha='center', va='top', fontsize=8.8, color='#0d47a1')

    # Physics Loss (orange-red)
    PL_Y=1.0; PL_H=4.6
    box(XL, PL_Y, BLW, PL_H, '#fff8e1', '#e65100')
    ax.text(XL+BLW/2, PL_Y+PL_H-0.28, 'Physics Loss',
            ha='center', va='top', fontsize=11, fontweight='bold', color='#bf360c')
    for k,ln in enumerate([
        r'$\mathcal{L}_P = \sum T_{w,i,j}[\hat{P}_j-\hat{P}_i]+\hat{q}_{w,i}'
        r'-\frac{\partial}{\partial t}\!\left(\frac{V_{p,i}\hat{S}_{w,i}}{B_w}\right)$',
        r'$+\sum T_{o,i,j}[\hat{P}_j-\hat{P}_i]+\hat{q}_{o,i}'
        r'-\frac{\partial}{\partial t}\!\left(\frac{V_{p,i}(1-\hat{S}_{w,i})}{B_o}\right)$',
        r'$\mathcal{L} = \mathcal{L}_D + \lambda\,\mathcal{L}_P$',
        r'$\lambda(t)=\lambda_{\max}\!\cdot\!\min(1,\,t/t_{\rm warm})$',
    ]):
        ax.text(XL+BLW/2, PL_Y+PL_H-0.82-k*0.82,
                ln, ha='center', va='top', fontsize=8.8, color='#bf360c')

    # Arrows
    for oy in out_ys:
        arr(XO+R, oy, XP-0.05, LP_Y+LP_H/2, col='#00838f', lw=1.4)
    arr(XO+R, out_ys[0], XL-0.05, DL_Y+DL_H*0.75, col='#1565c0', lw=1.5, rad=-0.15)
    arr(XO+R, out_ys[1], XL-0.05, DL_Y+DL_H*0.40, col='#1565c0', lw=1.5, rad=0.05)
    arr(XO+R, out_ys[1], XL-0.05, PL_Y+PL_H*0.75, col='#e65100', lw=1.5, rad=0.15)
    arr(XO+R, out_ys[2], XL-0.05, PL_Y+PL_H*0.40, col='#e65100', lw=1.5, rad=-0.05)

    ax.text(10.0, 9.80,
            'Figure 4 — The structure of proposed PINN model',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#111')
    save(fig, 'fig10_pinn_structure.png')


# =============================================================================
# Figure 11 — Network architecture flowchart  (ref Fig 5)
# =============================================================================
def make_fig11():
    fig = plt.figure(figsize=(9, 14))
    ax  = fig.add_axes([0.04, 0.02, 0.92, 0.96])
    ax.set_xlim(0, 9); ax.set_ylim(0, 15)
    ax.axis('off')

    CX=3.8; W=4.2; DX=6.4; DW=2.2
    C_INP=('#c8e6c9','#2e7d32'); C_DNS=('#bbdefb','#0d47a1')
    C_ACT=('#e1f5fe','#0277bd'); C_SIG=('#e8eaf6','#3949ab')
    C_OUT=('#c8e6c9','#2e7d32')

    def block(yc,h,label,fc,ec,dim_in=None,dim_out=None):
        ax.add_patch(FancyBboxPatch((CX-W/2,yc-h/2),W,h,
                     boxstyle='round,pad=0.10',facecolor=fc,edgecolor=ec,lw=2.0,zorder=3))
        ax.text(CX,yc,label,ha='center',va='center',
                fontsize=11,fontweight='bold',color='#1a1a2e',zorder=4)
        if dim_in is not None:
            ax.add_patch(FancyBboxPatch((DX,yc-h/2+0.04),DW,h-0.08,
                         boxstyle='round,pad=0.06',facecolor='white',
                         edgecolor='#90a4ae',lw=1.0,zorder=3))
            ax.text(DX+DW/2,yc+0.13,f'Input dim:  {dim_in}',
                    ha='center',va='center',fontsize=8.5,color='#333',zorder=4)
            ax.text(DX+DW/2,yc-0.17,f'Output dim: {dim_out}',
                    ha='center',va='center',fontsize=8.5,color='#333',zorder=4)

    def darr(yf,yt):
        ax.annotate('',xy=(CX,yt+0.02),xytext=(CX,yf-0.02),
                    arrowprops=dict(arrowstyle='->',color='#333',lw=1.8,mutation_scale=18),zorder=5)

    layout = [
        (13.8,0.78,'Input:  t,  T_start,  Q_inj', C_INP,'3','3'),
        (12.5,0.78,'Dense Layer 1',                C_DNS,'3','64'),
        (11.2,0.65,'Tanh Activation',              C_ACT,'64','64'),
        (10.1,0.78,'Dense Layer 2',                C_DNS,'64','64'),
        ( 8.8,0.65,'Tanh Activation',              C_ACT,'64','64'),
        ( 7.7,0.78,'Dense Layer 3',                C_DNS,'64','64'),
        ( 6.4,0.65,'Tanh Activation',              C_ACT,'64','64'),
        ( 5.3,0.78,'Dense Output Layer',           C_DNS,'64','2'),
        ( 4.0,0.65,'Sigmoid Activation',           C_SIG,'2','2'),
        ( 2.8,0.78,'Output:  WC,  Q_oil_norm',     C_OUT,'2','2'),
    ]
    for (y,h,lbl,(fc,ec),di,do) in layout:
        block(y,h,lbl,fc,ec,dim_in=di,dim_out=do)
    for i in range(len(layout)-1):
        darr(layout[i][0]-layout[i][1]/2, layout[i+1][0]+layout[i+1][1]/2)

    ax.add_patch(FancyBboxPatch((0.2,5.5),1.9,5.0,boxstyle='round,pad=0.12',
                 facecolor='#fff8e1',edgecolor='#f9a825',lw=1.8,zorder=2))
    for y,txt in [(10.2,'Physics\nConstraints'),(9.0,'WC monotone\ndWC/dt ≥ 0'),
                  (7.7,'Oil decline\ndQ/dt ≤ 0'),(6.5,'λ warm-up\n0→0.10\nepoch 0–150')]:
        fs = 9.5 if y==10.2 else (7.5 if y==6.5 else 8.5)
        fw = 'bold' if y==10.2 else 'normal'
        col = '#e65100' if y==10.2 else ('#bf360c' if y in [9.0,7.7] else '#555')
        ax.text(1.15,y,txt,ha='center',va='center',fontsize=fs,fontweight=fw,color=col,zorder=3)
    ax.annotate('',xy=(CX-W/2-0.04,8.8),xytext=(0.2+1.9,8.5),
                arrowprops=dict(arrowstyle='->',color='#e65100',lw=1.6),zorder=5)

    ax.text(CX,14.7,'Figure 5 — Architecture of the fully-connected\nnetwork in the proposed PINN model',
            ha='center',va='center',fontsize=11.5,fontweight='bold',color='#111')
    save(fig, 'fig11_network_arch.png')


# =============================================================================
# Figure 12 — All model equations (reference sheet)
# =============================================================================
def make_fig12():
    """Clean equation reference card — all equations used in the PINN model."""

    sections = [
        ('Corey Relative Permeability', '#0d47a1', '#e3f2fd', [
            (r'$S = \dfrac{S_w - S_{wc}}{1 - S_{or} - S_{wc}}$',
             'Normalised water saturation'),
            (r'$k_{ro} = (1-S)^{2}$',
             'Oil relative permeability (Corey)'),
            (r'$k_{rw} = k^{\prime}_{rw}\,S^{2}$',
             'Water rel. perm. (endpoint $k^{\prime}_{rw}=0.2918$)'),
        ]),
        ('Polymer Viscosity — Hand Model', '#1b5e20', '#e8f5e9', [
            (r'$\mu_w(C_p) = \mu_{w0}\!\left(1 + 8\!\times\!10^{-4}C_p'
             r'+ 2\!\times\!10^{-7}C_p^{2}\right)$',
             'Water viscosity as function of polymer conc. $C_p$ [ppm]'),
            (r'$\mu_{w0}=1\text{ cp},\quad \mu_o=5000\text{ cp}$',
             'Reference viscosities (Pelican Lake heavy oil)'),
        ]),
        ('Buckley–Leverett Fractional Flow', '#4a148c', '#f3e5f5', [
            (r'$f_w = \dfrac{k_{rw}/\mu_w}{k_{rw}/\mu_w + k_{ro}/\mu_o}$',
             'Water fractional flow at reservoir conditions'),
            (r'$\mathrm{RF}(C_p) \propto \mu_w(C_p)^{0.35}$',
             'Recovery factor — Craig-Geffen-Morse sweep efficiency ($M\gg1$)'),
            (r'$\varphi(C_p) = \dfrac{\mu_w(C_p)^{0.35}}{\mu_w(C_{p,\mathrm{ref}})^{0.35}}$',
             'Concentration correction factor ($C_{p,\mathrm{ref}}=1000$ ppm)'),
        ]),
        ('Voronoi Physical Model', '#e65100', '#fff3e0', [
            (r'$\sum_{j=1}^{n}T_{w,i,j}[P_j-P_i]+q_{w,i}'
             r'=\dfrac{\partial}{\partial t}\!\left(\dfrac{V_{p,i}S_{w,i}}{B_w}\right)$',
             'Water material balance per Voronoi region $i$'),
            (r'$\sum_{j=1}^{n}T_{o,i,j}[P_j-P_i]+q_{o,i}'
             r'=\dfrac{\partial}{\partial t}\!\left(\dfrac{V_{p,i}(1-S_{w,i})}{B_o}\right)$',
             'Oil material balance per Voronoi region $i$'),
            (r'$T_{\alpha,i,j}=\dfrac{KA}{L}\cdot\dfrac{k_{r\alpha}}{\mu_\alpha B_\alpha}'
             r'=T_{i,j}\cdot\dfrac{k_{r\alpha}}{\mu_\alpha B_\alpha}$',
             'Phase transmissibility between regions $i$ and $j$'),
            (r'$q_{\alpha,i}=WI_i\cdot\dfrac{k_{r\alpha}}{\mu_\alpha B_\alpha}'
             r'(P_i - P_i^{wf})$',
             r'Well production rate (phase $\alpha\in\{o,w\}$)'),
        ]),
        ('Neural Network Loss Functions', '#bf360c', '#fff8e1', [
            (r'$\mathcal{L}_D = \dfrac{1}{nP}\sum_{i=1}^{nP}'
             r'\!\left[(q_{o,i}-\hat{q}_{o,i})^2+(q_{w,i}-\hat{q}_{w,i})^2\right]$',
             'Data loss — MSE between predicted and CMG STARS rates'),
            (r'$\mathcal{L}_P = \sum\mathrm{ReLU}(-\Delta WC)^2'
             r'+ \sum\mathrm{ReLU}(\Delta Q_{oil})^2$',
             'Physics loss — monotonicity: WC non-decreasing, oil non-increasing'),
            (r'$\mathcal{L} = \mathcal{L}_D + \lambda(t)\cdot\mathcal{L}_P$',
             'Total loss with curriculum weight $\lambda(t)$'),
            (r'$\lambda(t) = \lambda_{\max}\cdot\min\!\left(1,\,\dfrac{t}{t_{\mathrm{warm}}}\right)'
             r'\quad [\lambda_{\max}=0.10,\;t_{\mathrm{warm}}=150]$',
             'Curriculum warmup schedule (ramps over first 150 epochs)'),
        ]),
        ('Joint Optimisation Objective', '#4e342e', '#efebe9', [
            (r'$\mathrm{Cum}_{oil}(T_s, C_p) = \mathrm{Cum}_{NN}(T_s)'
             r'\;\times\;\varphi(C_p)$',
             'Two-stage surrogate: timing from PINN × BL concentration correction'),
            (r'$(T_s^*, C_p^*) = \arg\max_{T_s,\,C_p}\;\mathrm{Cum}_{oil}(T_s,C_p)$',
             r'Optimal polymer start day $T_s^*=682$ d, conc. $C_p^*=2000$ ppm'),
        ]),
    ]

    # Layout: one row per equation, grouped by section
    n_rows = sum(len(eqs) for _, _, _, eqs in sections) + len(sections)
    fig_h  = max(14, n_rows * 0.72 + 1.5)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis('off')
    ax.set_xlim(0, 15); ax.set_ylim(0, fig_h)

    y  = fig_h - 0.6
    pad_eq   = 0.15   # left indent for equation
    pad_desc = 7.8    # x position of description column

    ax.text(7.5, y, 'PINN Polymer Flood — Model Equations Reference',
            ha='center', va='top', fontsize=14, fontweight='bold', color='#111')
    y -= 0.65

    for sec_title, hdr_col, sec_fc, eqs in sections:
        # Section header bar
        ax.add_patch(FancyBboxPatch((0.1, y-0.32), 14.8, 0.46,
                     boxstyle='round,pad=0.06', facecolor=sec_fc,
                     edgecolor=hdr_col, lw=1.6, zorder=2))
        ax.text(0.35, y-0.06, sec_title, ha='left', va='center',
                fontsize=11, fontweight='bold', color=hdr_col, zorder=3)
        y -= 0.62

        for eq_str, desc_str in eqs:
            row_h = 0.82
            # Light row background
            ax.add_patch(mpatches.Rectangle((0.1, y-row_h+0.08), 14.8, row_h,
                         facecolor='#fafafa', edgecolor='#e0e0e0',
                         lw=0.8, zorder=1))
            # Equation (left)
            ax.text(pad_eq, y-row_h/2+0.08, eq_str,
                    ha='left', va='center', fontsize=11, color='#111',
                    zorder=3)
            # Description (right, smaller, gray)
            ax.text(pad_desc, y-row_h/2+0.08, desc_str,
                    ha='left', va='center', fontsize=9, color='#555',
                    style='italic', wrap=True, zorder=3)
            # Separator line
            ax.plot([0.1, 14.9], [y-row_h+0.08, y-row_h+0.08],
                    color='#e0e0e0', lw=0.6, zorder=2)
            y -= row_h

        y -= 0.18  # gap between sections

    # Column headers
    ax.text(0.35, fig_h-0.62-0.02, 'Equation', ha='left', va='center',
            fontsize=9, color='#888', style='italic')
    ax.text(pad_desc, fig_h-0.62-0.02, 'Description', ha='left', va='center',
            fontsize=9, color='#888', style='italic')

    fig.tight_layout(pad=0.5)
    save(fig, 'fig12_equations.png')


# =============================================================================
if __name__ == '__main__':
    print('[FIG9 ] Voronoi diagram ...')
    make_fig9()
    print('[FIG10] PINN structure ...')
    make_fig10()
    print('[FIG11] Network architecture ...')
    make_fig11()
    print('[FIG12] Equation reference sheet ...')
    make_fig12()
    print('\nDone.')
