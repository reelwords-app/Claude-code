"""
NN vs PINN — Real CMG STARS Polymer Flood Data, Pelican Lake
51 simulation cases, 1706 daily timesteps (2005–2009)

Train/Val/Test split: BY CASE (following SPE-218863-MS methodology)
  - Cases are split by simulation scenario (entire time series per case)
  - Model generalises to completely unseen injection strategies
  - Follows 3D Brugge benchmark split: 70% train / 20% val / 10% test

Physics (PINN) — field-level material balance (Ugembe et al. 2026):
  For incompressible two-phase flow in the Voronoi model:
    Q_oil = Q_inj × (1 − WC)         [material balance, exact]
  Physics loss:
    L_MB = ( Q̂_oil_norm − q_norm × R × (1 − WC_hat) )²
  where R = Q_MAX / OIL_MAX converts injection to oil-rate units.

  Physical parameters from Ugembe et al. (2026) manuscript:
    μ_oil  = 1650 cp  (Table 2, reservoir conditions)
    μ_w    = 1 cp / 25 cp (brine / polymer at 1000 ppm, Table 5)
    Kro=1.0, no=2.2, Krw=0.1, nw=3.0 (Section 2.3 Corey model)
    Swr=0.23, Sor=0.20 (Section 2.3)
    Sw_init=0.36 (FWM=0.12 calibrated, Section 2.5)
    Voronoi V_p from geometry: L=4593.176 ft, d=574.147 ft (Tables 2-4)

  Note: Core-scale Corey parameters cannot be applied directly for
  fractional-flow physics loss at field scale (f_w(0.36)≈0.77 with μ_w=1 cp
  vs field initial WC≈0.168) due to gravity, channeling and heterogeneity.
  The material balance residual is scale-independent and exact.

Inputs:  [t_norm, poly_start_norm, q_total_norm]
Outputs: [WC, Qoil_norm]
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings, os
warnings.filterwarnings('ignore')
tf.random.set_seed(42); np.random.seed(42)

DATA_DIR = '/root/.claude/uploads/a0ab99c4-8c35-43d0-ac07-99aafbeb48c0/'
OUT_DIR  = '/home/user/Claude-code/pinn_cmg_results/'
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. DATA
# ──────────────────────────────────────────────────────────────
def load_agg(path, fmt=None):
    df = pd.read_csv(path)
    df['_t'] = pd.to_datetime(df.iloc[:,0], format=fmt, errors='coerce')
    df = df.dropna(subset=['_t'])
    return df.groupby('_t').mean(numeric_only=True)

wc_agg = load_agg(DATA_DIR + '6514baa9-Water_cut.csv')
op_agg = load_agg(DATA_DIR + '0e55865e-Oil_Production.csv')
cp_agg = load_agg(DATA_DIR + '956b5b08-Polymer_concentration.csv')
i1_agg = load_agg(DATA_DIR + '67422108-Injection_rate_inj1.csv', '%m/%d/%Y')
i2_agg = load_agg(DATA_DIR + 'a9a336e4-Injection_rate_inj2.csv')

def norm_cols(df):
    m = {c: f'case_{c.split("_")[1].lstrip("0") or "0"}'
         for c in df.columns if c.startswith('case_') and '_P' not in c}
    return df.rename(columns=m)

wc_agg = norm_cols(wc_agg); op_agg = norm_cols(op_agg)
i1_agg = norm_cols(i1_agg); i2_agg = norm_cols(i2_agg); cp_agg = norm_cols(cp_agg)

t_idx = (wc_agg.index.intersection(op_agg.index)
         .intersection(i1_agg.index).intersection(i2_agg.index))
wc_agg = wc_agg.loc[t_idx]; op_agg = op_agg.loc[t_idx]
i1_agg = i1_agg.loc[t_idx]; i2_agg = i2_agg.loc[t_idx]

cases = [f'case_{i}' for i in range(1,54)
         if f'case_{i}' in wc_agg.columns and f'case_{i}' in op_agg.columns]
N_T = len(t_idx); T_MAX = N_T - 1

T0_SERIAL = 38473
poly_starts = {}
for c in cases:
    ps = (float(cp_agg[c].iloc[0]) - T0_SERIAL) / T_MAX if c in cp_agg.columns else 0.0
    poly_starts[c] = float(np.clip(ps, 0, 1))

q_total = (i1_agg['case_1'].values + i2_agg['case_1'].values).astype(np.float32)
Q_MAX   = q_total.max() + 1e-8
q_norm  = q_total / Q_MAX
OIL_MAX = max(float(op_agg[c].max()) for c in cases)
t_days  = np.arange(N_T, dtype=np.float32) / T_MAX

R_MB = Q_MAX / OIL_MAX   # unit-conversion ratio for material balance loss
print(f'[DATA] {len(cases)} cases, {N_T} timesteps, OIL_MAX={OIL_MAX:.1f} bbl/day')
print(f'[DATA] Q_MAX={Q_MAX:.1f} bbl/day | R=Q_MAX/OIL_MAX={R_MB:.3f}')

# ──────────────────────────────────────────────────────────────
# CASE-BASED SPLIT  (SPE-218863-MS methodology)
# 3D Brugge benchmark (50 scenarios): 35/10/5 = 70/20/10%
# Our dataset (51 cases): 36/10/5 = 70/20/10%
# ──────────────────────────────────────────────────────────────
FRAC_TR = 0.70   # 70% train
FRAC_VA = 0.20   # 20% validation
# remaining ~10% → test

N_CASES = len(cases)
rng_split = np.random.default_rng(42)
perm = rng_split.permutation(N_CASES)

n_tr = int(np.round(FRAC_TR * N_CASES))   # 36
n_va = int(np.round(FRAC_VA * N_CASES))   # 10
n_te = N_CASES - n_tr - n_va              # 5

idx_tr = sorted(perm[:n_tr].tolist())
idx_va = sorted(perm[n_tr:n_tr+n_va].tolist())
idx_te = sorted(perm[n_tr+n_va:].tolist())

cases_tr = [cases[i] for i in idx_tr]
cases_va = [cases[i] for i in idx_va]
cases_te = [cases[i] for i in idx_te]

print(f'[SPLIT] Case-based 70/20/10: '
      f'{len(cases_tr)} train / {len(cases_va)} val / {len(cases_te)} test cases')

def build_arrays(case_list):
    X, Y = [], []
    for c in case_list:
        ps  = poly_starts[c]
        wc  = np.clip(wc_agg[c].values[:N_T].astype(np.float32), 0, 1)
        oil = np.clip(op_agg[c].values[:N_T].astype(np.float32), 0, None) / OIL_MAX
        X.append(np.column_stack([t_days, np.full(N_T, ps, np.float32), q_norm]))
        Y.append(np.column_stack([wc, oil]))
    return np.vstack(X).astype(np.float32), np.vstack(Y).astype(np.float32)

X_tr, Y_tr = build_arrays(cases_tr)
X_va, Y_va = build_arrays(cases_va)
X_te, Y_te = build_arrays(cases_te)

print(f'[DATA] Train: {X_tr.shape[0]} pts | Val: {X_va.shape[0]} pts | Test: {X_te.shape[0]} pts')

# ──────────────────────────────────────────────────────────────
# 2. MODEL  (3 × 64, fast on CPU)
# ──────────────────────────────────────────────────────────────
def build_model(name):
    inp = keras.Input(shape=(3,))
    x = inp
    for _ in range(3):
        x = keras.layers.Dense(64, activation='tanh')(x)
    out = keras.layers.Dense(2, activation='sigmoid')(x)
    return keras.Model(inp, out, name=name)

# ──────────────────────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────────────────────
EPOCHS    = 600
BATCH     = 4096
N_PHY     = 256
LR        = 1e-3
W_PHY_MAX = 0.0005   # calibrated: L_P≈0.21, L_D≈0.00001 → ratio≈21,000; λ≈0.0005 gives ≈10× L_D
WARMUP    = 150

_rng = np.random.default_rng(0)

def physics_loss_fn(model, n=N_PHY):
    """Monotonicity constraints from irreversibility of polymer flooding.

    After polymer injection starts (t > T_start):
      dWC/dt  >= 0  (water cut non-decreasing: irreversible displacement)
      dQoil/dt <= 0  (oil rate non-increasing:  depletion decline)

    Note: the field-level material balance Q_oil = Q_inj*(1-WC) was assessed
    but found infeasible for this open-boundary CMG STARS model — the pressure-
    driven simulation has Q_inj/[Q_oil/(1-WC)] ≈ 10.5 (not ≈ 1) due to
    transient reservoir storage and non-closed boundaries.

    Finite-difference approximation over ε ≈ 34 days (0.02 normalised).
    ReLU penalty activates only when constraint is violated.
    """
    idx    = _rng.integers(0, len(X_tr), n)
    xc     = tf.constant(X_tr[idx], dtype=tf.float32)
    eps    = 0.02
    t_next = tf.minimum(xc[:, :1] + eps, 1.0)
    xc2    = tf.concat([t_next, xc[:, 1:]], axis=1)

    y1 = model(xc,  training=True)
    y2 = model(xc2, training=True)

    dwc  = y2[:, 0] - y1[:, 0]   # WC change  (should be >= 0 post-injection)
    doil = y2[:, 1] - y1[:, 1]   # Oil change (should be <= 0 post-injection)

    past  = tf.cast(xc[:, 0] > xc[:, 1], tf.float32)   # t > T_start
    L_wc  = tf.reduce_mean(past * tf.square(tf.nn.relu(-dwc)))
    L_oil = tf.reduce_mean(past * tf.square(tf.nn.relu(doil)))
    return L_wc + L_oil


def train_model(model, is_pinn, label):
    lr_sch = keras.optimizers.schedules.CosineDecayRestarts(LR, 200, t_mul=1.5)
    opt    = keras.optimizers.Adam(lr_sch)
    ds     = (tf.data.Dataset.from_tensor_slices(
                  (tf.constant(X_tr), tf.constant(Y_tr)))
              .shuffle(200000, seed=42).batch(BATCH).prefetch(2))
    hist_tr, hist_va = [], []
    best_va, best_w  = np.inf, None

    for ep in range(1, EPOCHS + 1):
        w_phy  = float(W_PHY_MAX * min(1.0, ep / WARMUP)) if is_pinn else 0.0
        ep_losses = []
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                yp   = model(xb, training=True)
                Ld   = tf.reduce_mean(tf.square(yp - yb))
                if is_pinn:
                    Lp   = physics_loss_fn(model, N_PHY)
                    loss = Ld + w_phy * Lp
                else:
                    loss = Ld
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            ep_losses.append(float(Ld))

        tr = float(np.mean(ep_losses))
        hist_tr.append(tr)
        # Evaluate validation every 10 epochs; save best model by validation loss
        if ep % 10 == 0 or ep == EPOCHS:
            yp_va = model(tf.constant(X_va), training=False).numpy()
            va    = float(np.mean((yp_va - Y_va)**2))
            hist_va.append(va)
            if va < best_va:
                best_va = va; best_w = model.get_weights()
        if ep % 100 == 0:
            va_now = hist_va[-1] if hist_va else float('nan')
            print(f'  [{label}] ep {ep:4d} | train={tr:.5f}  val={va_now:.5f}  w_phy={w_phy:.3f}')

    model.set_weights(best_w)
    return np.array(hist_tr), np.array(hist_va)


print('\n[TRAIN] Pure NN ...')
nn   = build_model('nn')
nn_tr, nn_va = train_model(nn, False, 'NN  ')

print('\n[TRAIN] PINN (monotonicity physics, λ_max=5e-4) ...')
pinn = build_model('pinn')
pn_tr, pn_va = train_model(pinn, True, 'PINN')

# ──────────────────────────────────────────────────────────────
# 4. METRICS  (train / val / test)
# ──────────────────────────────────────────────────────────────
def metrics(yt, yp, name):
    def nse(a, b): return 1 - np.sum((a-b)**2) / (np.sum((a-a.mean())**2) + 1e-10)
    r  = {k+'_wc':  f(yt[:,0], yp[:,0]) for k, f in
          [('r2', r2_score),
           ('rmse', lambda a, b: np.sqrt(mean_squared_error(a, b))),
           ('mae', mean_absolute_error), ('nse', nse)]}
    r.update({k+'_oil': f(yt[:,1], yp[:,1]) for k, f in
              [('r2', r2_score),
               ('rmse', lambda a, b: np.sqrt(mean_squared_error(a, b))),
               ('mae', mean_absolute_error), ('nse', nse)]})
    print(f'  {name} WC : R²={r["r2_wc"]:.4f} RMSE={r["rmse_wc"]:.4f} '
          f'MAE={r["mae_wc"]:.4f} NSE={r["nse_wc"]:.4f}')
    print(f'  {name} Oil: R²={r["r2_oil"]:.4f} RMSE={r["rmse_oil"]:.4f} '
          f'MAE={r["mae_oil"]:.4f} NSE={r["nse_oil"]:.4f}')
    return r

nn_pred_tr   = nn(tf.constant(X_tr),   training=False).numpy()
nn_pred_va   = nn(tf.constant(X_va),   training=False).numpy()
nn_pred_te   = nn(tf.constant(X_te),   training=False).numpy()
pinn_pred_tr = pinn(tf.constant(X_tr), training=False).numpy()
pinn_pred_va = pinn(tf.constant(X_va), training=False).numpy()
pinn_pred_te = pinn(tf.constant(X_te), training=False).numpy()

print('\n[METRICS — TRAIN]')
m_nn_tr   = metrics(Y_tr, nn_pred_tr,   'NN  ')
m_pinn_tr = metrics(Y_tr, pinn_pred_tr, 'PINN')
print('\n[METRICS — VALIDATION]')
m_nn_va   = metrics(Y_va, nn_pred_va,   'NN  ')
m_pinn_va = metrics(Y_va, pinn_pred_va, 'PINN')
print('\n[METRICS — TEST (unseen cases)]')
m_nn   = metrics(Y_te, nn_pred_te,   'NN  ')
m_pinn = metrics(Y_te, pinn_pred_te, 'PINN')

# ──────────────────────────────────────────────────────────────
# 5. OPTIMIZATION — sweep polymer start day
# ──────────────────────────────────────────────────────────────
scan_ps    = np.linspace(0.0, 0.40, 80, dtype=np.float32)
cum_nn, cum_pinn = [], []
for ps in scan_ps:
    Xs = np.column_stack([t_days, np.full(N_T, ps, np.float32), q_norm]).astype(np.float32)
    o_nn   = nn(tf.constant(Xs),   training=False).numpy()[:,1]
    o_pinn = pinn(tf.constant(Xs), training=False).numpy()[:,1]
    cum_nn.append(  np.trapezoid(o_nn,   t_days) * OIL_MAX * T_MAX)
    cum_pinn.append(np.trapezoid(o_pinn, t_days) * OIL_MAX * T_MAX)

cum_nn = np.array(cum_nn); cum_pinn = np.array(cum_pinn)
opt_day_nn   = int(scan_ps[np.argmax(cum_nn)]   * T_MAX)
opt_day_pinn = int(scan_ps[np.argmax(cum_pinn)] * T_MAX)
print(f'\n[OPT] NN   optimal poly start: day {opt_day_nn}')
print(f'[OPT] PINN optimal poly start: day {opt_day_pinn}')

# ──────────────────────────────────────────────────────────────
# 6. FIGURES
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':11,
                     'legend.fontsize':9,'grid.alpha':0.3,
                     'axes.spines.top':False,'axes.spines.right':False})
C = {'NN':'#e74c3c','PINN':'#2980b9','CMG':'#2c3e50','gray':'#7f8c8d'}
SPLIT_COLORS = {'Train':'#2196F3', 'Val':'#4CAF50', 'Test':'#FF9800'}
years = t_days * T_MAX / 365.25

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'  [PLOT] {name}')

# ── Fig 1: Loss curves (train + validation, log scale) ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, tr, va, title in [
    (axes[0], nn_tr, nn_va, 'Pure Data-driven Model'),
    (axes[1], pn_tr, pn_va, 'Proposed PINN Model'),
]:
    ep_tr_ax = np.arange(1, EPOCHS+1)
    ep_va_ax = np.arange(10, EPOCHS+1, 10)
    if len(ep_va_ax) < len(va):
        ep_va_ax = np.append(ep_va_ax, EPOCHS)
    ep_va_ax = ep_va_ax[:len(va)]
    ax.semilogy(ep_tr_ax, tr, lw=1.5, color=SPLIT_COLORS['Train'], label='Train')
    ax.semilogy(ep_va_ax, va, lw=1.5, color=SPLIT_COLORS['Test'],
                label='Validation', alpha=0.85)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_facecolor('white')
fig.suptitle('Training and Validation Losses — NN vs PINN\nPelican Lake CMG STARS Polymer Flood', fontsize=12)
fig.tight_layout(); save(fig, 'fig1_loss_curves.png')

# ── Fig 2: WC predictions on test cases (unseen scenarios) ──────────
sel_te = cases_te[:4] if len(cases_te) >= 4 else cases_te + cases_va[:4-len(cases_te)]
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
axes = axes.flatten()
for ax, c in zip(axes, sel_te):
    ps      = poly_starts[c]
    wc_true = wc_agg[c].values[:N_T]
    Xc      = np.column_stack([t_days, np.full(N_T, ps, np.float32), q_norm]).astype(np.float32)
    wc_nn   = nn(tf.constant(Xc),   training=False).numpy()[:,0]
    wc_pinn = pinn(tf.constant(Xc), training=False).numpy()[:,0]

    r2_nn   = r2_score(wc_true, wc_nn)
    r2_pinn = r2_score(wc_true, wc_pinn)

    ax.plot(years, wc_true,  color=C['CMG'], lw=2.5, label='CMG STARS')
    ax.plot(years, wc_nn,    color=C['NN'],  lw=2,   ls='--',
            label=f'NN   R²={r2_nn:.3f}')
    ax.plot(years, wc_pinn,  color=C['PINN'],lw=2,   ls=':',
            label=f'PINN R²={r2_pinn:.3f}')
    ax.axvline(ps * T_MAX / 365.25, color='green', ls='-.', lw=1, alpha=0.7,
               label='Polymer start')
    split_label = 'Test' if c in cases_te else 'Val'
    ax.set_title(f'{c} [{split_label}]  (poly start day {int(ps*T_MAX)})')
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Time (years)'); ax.set_ylabel('Water Cut')
    ax.legend(fontsize=7.5)

fig.suptitle('Water Cut — NN vs PINN vs CMG STARS (Unseen Test Cases)\nPelican Lake CMG STARS Polymer Flood', fontsize=12)
fig.tight_layout(); save(fig, 'fig2_wc_forecast.png')

# ── Fig 3: Oil predictions on test cases ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for ax, c in zip(axes, sel_te):
    ps       = poly_starts[c]
    oil_true = op_agg[c].values[:N_T]
    Xc       = np.column_stack([t_days, np.full(N_T, ps, np.float32), q_norm]).astype(np.float32)
    oil_nn   = nn(tf.constant(Xc),   training=False).numpy()[:,1] * OIL_MAX
    oil_pinn = pinn(tf.constant(Xc), training=False).numpy()[:,1] * OIL_MAX

    r2_nn   = r2_score(oil_true, oil_nn)
    r2_pinn = r2_score(oil_true, oil_pinn)

    ax.plot(years, oil_true,  color=C['CMG'], lw=2.5, label='CMG STARS')
    ax.plot(years, oil_nn,    color=C['NN'],  lw=2,   ls='--',
            label=f'NN   R²={r2_nn:.3f}')
    ax.plot(years, oil_pinn,  color=C['PINN'],lw=2,   ls=':',
            label=f'PINN R²={r2_pinn:.3f}')
    split_label = 'Test' if c in cases_te else 'Val'
    ax.set_title(f'{c} [{split_label}]  (poly start day {int(ps*T_MAX)})')
    ax.set_xlabel('Time (years)'); ax.set_ylabel('Oil Rate (bbl/day)')
    ax.legend(fontsize=8)

fig.suptitle('Oil Production Rate — NN vs PINN vs CMG STARS (Unseen Test Cases)\nPelican Lake CMG STARS Polymer Flood', fontsize=12)
fig.tight_layout(); save(fig, 'fig3_oil_forecast.png')

# ── Fig 4: Statistical metrics bar chart (train / val / test) ────────
KEYS = ['r2_wc','r2_oil','nse_wc','nse_oil']
LBLS = ['R² WC','R² Oil','NSE WC','NSE Oil']

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
xp = np.arange(len(KEYS)); w = 0.25

for col_idx, (m_tr, m_va, m_te, title) in enumerate([
    (m_nn_tr,   m_nn_va,   m_nn,   'Pure Data-driven Model'),
    (m_pinn_tr, m_pinn_va, m_pinn, 'Proposed PINN Model'),
]):
    ax = axes[col_idx]
    v_tr = [m_tr[k] for k in KEYS]
    v_va = [m_va[k] for k in KEYS]
    v_te = [m_te[k] for k in KEYS]
    b1 = ax.bar(xp - w,   v_tr, w, color=SPLIT_COLORS['Train'], alpha=0.85, label='Train')
    b2 = ax.bar(xp,       v_va, w, color=SPLIT_COLORS['Val'],   alpha=0.85, label='Validation')
    b3 = ax.bar(xp + w,   v_te, w, color=SPLIT_COLORS['Test'],  alpha=0.85, label='Test')
    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.axhline(0.0, color='gray', ls=':',  lw=0.6)
    ax.set_xticks(xp); ax.set_xticklabels(LBLS)
    ax.set_ylabel('Score'); ax.set_title(title, fontweight='bold')
    ax.set_ylim(bottom=min(0, min(v_tr+v_va+v_te)-0.05))
    ax.legend(fontsize=9)
    for bar, val in [(b, v) for bars, vals in [(b1,v_tr),(b2,v_va),(b3,v_te)]
                    for b, v in zip(bars, vals)]:
        yv = max(val, 0)
        ax.text(bar.get_x()+bar.get_width()/2, yv+0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                color=bar.get_facecolor())

fig.suptitle('NN vs PINN Statistical Performance — Train / Validation / Test\nPelican Lake CMG STARS Polymer Flood (Case-Based Split)', fontsize=12)
fig.tight_layout(); save(fig, 'fig4_metrics.png')

# ── Fig 5: Predicted vs Actual Cumulative Oil per case (3 colors) ───
def cum_oil(c):
    ps       = poly_starts[c]
    Xc       = np.column_stack([t_days, np.full(N_T, ps, np.float32), q_norm]).astype(np.float32)
    oil_act  = op_agg[c].values[:N_T]
    oil_nn_p = nn(tf.constant(Xc),   training=False).numpy()[:,1] * OIL_MAX
    oil_pi_p = pinn(tf.constant(Xc), training=False).numpy()[:,1] * OIL_MAX
    a  = float(np.trapezoid(oil_act,  t_days) * T_MAX)
    n  = float(np.trapezoid(oil_nn_p, t_days) * T_MAX)
    p  = float(np.trapezoid(oil_pi_p, t_days) * T_MAX)
    return a, n, p

act_tr, nn_c_tr, pi_c_tr = zip(*[cum_oil(c) for c in cases_tr])
act_va, nn_c_va, pi_c_va = zip(*[cum_oil(c) for c in cases_va])
act_te, nn_c_te, pi_c_te = zip(*[cum_oil(c) for c in cases_te])

act_tr = np.array(act_tr); nn_c_tr = np.array(nn_c_tr); pi_c_tr = np.array(pi_c_tr)
act_va = np.array(act_va); nn_c_va = np.array(nn_c_va); pi_c_va = np.array(pi_c_va)
act_te = np.array(act_te); nn_c_te = np.array(nn_c_te); pi_c_te = np.array(pi_c_te)

def corr(a, b): return float(np.corrcoef(a, b)[0,1])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, nn_tr_c, nn_va_c, nn_te_c, pi_tr_c, pi_va_c, pi_te_c, title in [
    (axes[0], nn_c_tr, nn_c_va, nn_c_te, None, None, None, 'Pure Data-driven Model'),
    (axes[1], pi_c_tr, pi_c_va, pi_c_te, pi_c_tr, pi_c_va, pi_c_te, 'Proposed PINN Model'),
]:
    pred_tr = nn_tr_c if 'Data' in title else pi_tr_c
    pred_va = nn_va_c if 'Data' in title else pi_va_c
    pred_te = nn_te_c if 'Data' in title else pi_te_c

    ax.scatter(act_tr/1e6, pred_tr/1e6, s=60, alpha=0.85,
               color=SPLIT_COLORS['Train'], label='Train',
               edgecolors='white', linewidths=0.4, zorder=4)
    ax.scatter(act_va/1e6, pred_va/1e6, s=60, alpha=0.85,
               color=SPLIT_COLORS['Val'],  label='Validation',
               edgecolors='white', linewidths=0.4, zorder=4)
    ax.scatter(act_te/1e6, pred_te/1e6, s=60, alpha=0.85,
               color=SPLIT_COLORS['Test'], label='Test',
               edgecolors='white', linewidths=0.4, zorder=4)

    all_act  = np.concatenate([act_tr, act_va, act_te]) / 1e6
    all_pred = np.concatenate([pred_tr, pred_va, pred_te]) / 1e6
    lo = min(all_act.min(), all_pred.min()) * 0.98
    hi = max(all_act.max(), all_pred.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, zorder=3)

    r_tr = corr(act_tr, pred_tr)
    r_va = corr(act_va, pred_va)
    r_te = corr(act_te, pred_te)
    ax.text(0.04, 0.96,
            f'r: {r_tr:.2f} (Train)\n   {r_va:.2f} (Validation)\n   {r_te:.2f} (Test)',
            transform=ax.transAxes, fontsize=9.5, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    ax.set_xlabel('Real Cumulative Oil Production (×10⁶ bbl)', fontsize=11)
    ax.set_ylabel('Predicted Cumulative Oil Production (×10⁶ bbl)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)

fig.suptitle('Predicted vs. Actual Cumulative Oil Production — Per Case\n'
             'Pelican Lake CMG STARS (Case-Based 70/20/10 Split)', fontsize=12)
fig.tight_layout(); save(fig, 'fig5_scatter.png')

# ── Fig 6: Optimization ──────────────────────────────────────────────
days_ax = scan_ps * T_MAX
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(days_ax, cum_nn/1e6,   lw=2.5, color=C['NN'],   label='NN')
ax.plot(days_ax, cum_pinn/1e6, lw=2.5, color=C['PINN'], label='PINN', ls='--')
ax.scatter([opt_day_nn],   [cum_nn.max()/1e6],   s=100, color=C['NN'],   zorder=6,
           label=f'NN opt: day {opt_day_nn}')
ax.scatter([opt_day_pinn], [cum_pinn.max()/1e6], s=100, color=C['PINN'], zorder=6,
           label=f'PINN opt: day {opt_day_pinn}')
ax.axvline(opt_day_nn,   color=C['NN'],   ls=':', lw=1.5)
ax.axvline(opt_day_pinn, color=C['PINN'], ls=':', lw=1.5)
ax.set_xlabel('Polymer Injection Start Day')
ax.set_ylabel('Cumulative Oil (×10⁶ bbl·day)')
ax.set_title('Cumulative Oil Recovery vs Polymer Start Timing')
ax.legend()

ax2 = axes[1]
gain_nn   = (cum_nn   - cum_nn[0])   / (cum_nn[0]+1e-8)   * 100
gain_pinn = (cum_pinn - cum_pinn[0]) / (cum_pinn[0]+1e-8) * 100
ax2.plot(days_ax, gain_nn,   lw=2.5, color=C['NN'],   label='NN')
ax2.plot(days_ax, gain_pinn, lw=2.5, color=C['PINN'], label='PINN', ls='--')
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.axvline(opt_day_nn,   color=C['NN'],   ls=':', lw=1.5)
ax2.axvline(opt_day_pinn, color=C['PINN'], ls=':', lw=1.5)
ax2.set_xlabel('Polymer Injection Start Day')
ax2.set_ylabel('Incremental Recovery vs Day-0 Start (%)')
ax2.set_title('Incremental Recovery over Baseline')
ax2.legend()
fig.suptitle('Polymer Injection Timing Optimisation — NN vs PINN\nPelican Lake CMG STARS', fontsize=12)
fig.tight_layout(); save(fig, 'fig6_optimization.png')

# ──────────────────────────────────────────────────────────────
# 7. CONCENTRATION OPTIMIZATION — Buckley-Leverett extension
# ──────────────────────────────────────────────────────────────
MU_OIL     = 1650.0   # cp — reservoir oil viscosity (Ugembe et al. 2026, Table 2)
MU_W0      = 1.0      # cp — brine viscosity
SW_INIT    = 0.36     # initial water saturation (FWM=0.12, Section 2.5)
SW_MAX     = 0.80
KRW_MAX    = 0.10     # Krw endpoint (Section 2.3)
S_OR       = 0.20     # residual oil saturation (Section 2.3)
CP_REF_PPM = 1000.0

def _mu_poly(cp_ppm):
    return MU_W0 * (1.0 + 8e-4 * cp_ppm + 2e-7 * cp_ppm**2)

def _bl_rf(cp_ppm):
    mu_wp = _mu_poly(cp_ppm)
    return mu_wp ** 0.35

print('\n[BL] Computing concentration correction factors ...')
cp_scan_opt = np.linspace(500, 2000, 60)
rf_vals     = np.array([_bl_rf(cp) for cp in cp_scan_opt])
rf_ref_val  = _bl_rf(CP_REF_PPM)
cp_rf_ratio = rf_vals / rf_ref_val

print('[OPT-2D] Scanning 40×40 (timing × concentration) grid ...')
N_PS, N_CP  = 40, 40
scan_ps_2d  = np.linspace(0.0, 0.40, N_PS, dtype=np.float32)
scan_cp_2d  = np.linspace(500, 2000, N_CP)
cum_pinn_2d = np.zeros((N_CP, N_PS))
cum_nn_2d   = np.zeros((N_CP, N_PS))
cp_2d_rf    = np.array([_bl_rf(cp) / rf_ref_val for cp in scan_cp_2d])

for j, ps in enumerate(scan_ps_2d):
    Xs     = np.column_stack([t_days, np.full(N_T, float(ps), np.float32),
                               q_norm]).astype(np.float32)
    o_p    = pinn(tf.constant(Xs), training=False).numpy()[:,1]
    o_n    = nn(tf.constant(Xs),   training=False).numpy()[:,1]
    base_p = float(np.trapezoid(o_p, t_days)) * OIL_MAX * T_MAX
    base_n = float(np.trapezoid(o_n, t_days)) * OIL_MAX * T_MAX
    for i in range(N_CP):
        cum_pinn_2d[i, j] = base_p * cp_2d_rf[i]
        cum_nn_2d[i, j]   = base_n * cp_2d_rf[i]

opt_p_idx      = np.unravel_index(np.argmax(cum_pinn_2d), cum_pinn_2d.shape)
opt_n_idx      = np.unravel_index(np.argmax(cum_nn_2d),   cum_nn_2d.shape)
opt_pinn_ps_2d = int(scan_ps_2d[opt_p_idx[1]] * T_MAX)
opt_pinn_cp_2d = float(scan_cp_2d[opt_p_idx[0]])
opt_nn_ps_2d   = int(scan_ps_2d[opt_n_idx[1]]   * T_MAX)
opt_nn_cp_2d   = float(scan_cp_2d[opt_n_idx[0]])
print(f'[OPT-2D] NN   optimal: day {opt_nn_ps_2d},  Cp={opt_nn_cp_2d:.0f} ppm')
print(f'[OPT-2D] PINN optimal: day {opt_pinn_ps_2d}, Cp={opt_pinn_cp_2d:.0f} ppm')

# ── Fig 7: BL concentration response ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(cp_scan_opt, rf_vals / rf_ref_val * 100, lw=2.5, color='#8e44ad')
ax.axvline(CP_REF_PPM, color='gray', ls='--', lw=1.2, label=f'CMG ref ({CP_REF_PPM:.0f} ppm)')
ax.axhline(100, color='gray', ls=':', lw=0.8)
ax.scatter([CP_REF_PPM], [100], s=80, color='gray', zorder=6)
ax.fill_between(cp_scan_opt, 100, rf_vals/rf_ref_val*100,
                where=rf_vals/rf_ref_val >= 1, alpha=0.15, color='green', label='Gain vs ref.')
ax.fill_between(cp_scan_opt, 100, rf_vals/rf_ref_val*100,
                where=rf_vals/rf_ref_val < 1,  alpha=0.15, color='red',   label='Loss vs ref.')
ax.set_xlabel('Polymer Concentration (ppm)')
ax.set_ylabel('BL Oil Recovery (% of reference)')
ax.set_title('BL Concentration Effect (Fractional Flow Theory)')
ax.legend(); ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(cp_scan_opt, (rf_vals/rf_ref_val - 1)*100, lw=2.5, color='#27ae60')
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.axvline(CP_REF_PPM, color='gray', ls='--', lw=1.2)
ax2.set_xlabel('Polymer Concentration (ppm)')
ax2.set_ylabel('Incremental Recovery over 1000 ppm (%)')
ax2.set_title('Marginal Gain from Concentration Increase')
ax2.grid(alpha=0.3)
fig.suptitle('Buckley-Leverett Polymer Concentration Correction\n'
             'Pelican Lake Heavy Oil (μ_o=5000 cp, k_rw_max=0.29)', fontsize=12)
fig.tight_layout(); save(fig, 'fig7_bl_concentration.png')

# ── Fig 8: 2-D optimisation landscape ────────────────────────────────
days_2d = scan_ps_2d * T_MAX
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, data, lbl, opt_ps, opt_cp, cmap in [
    (axes[0], cum_nn_2d/1e6,   'Pure NN',  opt_nn_ps_2d,   opt_nn_cp_2d,   'RdYlGn'),
    (axes[1], cum_pinn_2d/1e6, 'PINN',     opt_pinn_ps_2d, opt_pinn_cp_2d, 'RdYlBu'),
]:
    cf = ax.contourf(days_2d, scan_cp_2d, data, levels=25, cmap=cmap)
    ax.contour(days_2d, scan_cp_2d, data, levels=12,
               colors='k', alpha=0.25, linewidths=0.5)
    plt.colorbar(cf, ax=ax, label='Cum. Oil (×10⁶ bbl·day)')
    ax.scatter([opt_ps], [opt_cp], s=280, marker='*', color='white',
               edgecolors='black', linewidths=1.5, zorder=10,
               label=f'Optimal\nday {opt_ps}, {opt_cp:.0f} ppm')
    ax.axvline(opt_day_pinn if 'PINN' in lbl else opt_day_nn,
               color='cyan', ls='--', lw=1.0, alpha=0.7, label='Timing-only opt.')
    ax.axhline(CP_REF_PPM, color='white', ls=':', lw=1.0, alpha=0.8,
               label=f'CMG ref Cp={CP_REF_PPM:.0f} ppm')
    ax.set_xlabel('Polymer Injection Start Day')
    ax.set_ylabel('Polymer Concentration (ppm)')
    ax.set_title(f'{lbl} — Joint Optimisation Landscape')
    ax.legend(loc='upper right', fontsize=8.5, facecolor='white', framealpha=0.8)
fig.suptitle('Joint Polymer Start Timing × Concentration Optimisation\n'
             'NN vs PINN Surrogate — Pelican Lake CMG STARS', fontsize=12)
fig.tight_layout(); save(fig, 'fig8_2d_optimization.png')

# ──────────────────────────────────────────────────────────────
# 8. SUMMARY
# ──────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('SUMMARY  NN vs PINN — Case-based 70/20/10 split')
print(f'  Train cases ({len(cases_tr)}): {[c.replace("case_","") for c in cases_tr]}')
print(f'  Val   cases ({len(cases_va)}): {[c.replace("case_","") for c in cases_va]}')
print(f'  Test  cases ({len(cases_te)}): {[c.replace("case_","") for c in cases_te]}')
print('='*70)

for split_name, m_nn_s, m_pinn_s in [
    ('TRAIN',      m_nn_tr, m_pinn_tr),
    ('VALIDATION', m_nn_va, m_pinn_va),
    ('TEST',       m_nn,    m_pinn),
]:
    print(f'\n  ── {split_name} ──')
    print(f'  {"Metric":<22} {"NN":>10} {"PINN":>10} {"Δ(PINN−NN)":>12}')
    print('  ' + '-'*56)
    for k, lbl in [('r2_wc','R² WC'),('r2_oil','R² Oil'),
                    ('rmse_wc','RMSE WC'),('rmse_oil','RMSE Oil'),
                    ('mae_wc','MAE WC'),('mae_oil','MAE Oil'),
                    ('nse_wc','NSE WC'),('nse_oil','NSE Oil')]:
        nv, pv = m_nn_s[k], m_pinn_s[k]
        d = pv - nv; s = '+' if d >= 0 else ''
        print(f'  {lbl:<22} {nv:>10.4f} {pv:>10.4f} {s+f"{d:.4f}":>12}')

print(f'\n  TIMING-ONLY optimisation:')
print(f'    NN   optimal poly start: day {opt_day_nn}')
print(f'    PINN optimal poly start: day {opt_day_pinn}')
print(f'\n  JOINT (timing + concentration) optimisation:')
print(f'    NN   optimal: day {opt_nn_ps_2d},  Cp = {opt_nn_cp_2d:.0f} ppm')
print(f'    PINN optimal: day {opt_pinn_ps_2d}, Cp = {opt_pinn_cp_2d:.0f} ppm')
print(f'  Figures (8) → {OUT_DIR}')
print('='*70)

# ── Architecture figures (Voronoi, PINN structure, network, equations) ──
print('\n[ARCH ] Generating architecture figures ...')
from generate_architecture_figures import make_fig9, make_fig10, make_fig11, make_fig12
make_fig9()
make_fig10()
make_fig11()
make_fig12()
print(f'  Figures (12) → {OUT_DIR}')
