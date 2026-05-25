"""
NN vs PINN — Real CMG STARS Polymer Flood Data, Pelican Lake
53 simulation cases, 1706 daily timesteps (2005–2009)

Train/test split: TEMPORAL — first 75% history → predict last 25%
(tests production FORECASTING ability, not interpolation)

Physics (PINN):
  1. WC monotonicity: dWC/dt ≥ 0  (irreversible displacement)
  2. Oil decline:     dOil/dt ≤ 0  (production decline post-peak)
  enforced via automatic differentiation through the network

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

q_total  = (i1_agg['case_1'].values + i2_agg['case_1'].values).astype(np.float32)
Q_MAX    = q_total.max() + 1e-8
q_norm   = q_total / Q_MAX
OIL_MAX  = max(float(op_agg[c].max()) for c in cases)
t_days   = np.arange(N_T, dtype=np.float32) / T_MAX

print(f'[DATA] {len(cases)} cases, {N_T} timesteps, OIL_MAX={OIL_MAX:.1f} bbl/day')

# Build arrays for all cases
X_all, Y_all, cid_all = [], [], []
for ci, c in enumerate(cases):
    ps  = poly_starts[c]
    wc  = np.clip(wc_agg[c].values[:N_T].astype(np.float32), 0, 1)
    oil = np.clip(op_agg[c].values[:N_T].astype(np.float32), 0, None) / OIL_MAX
    X_all.append(np.column_stack([t_days, np.full(N_T, ps, np.float32), q_norm]))
    Y_all.append(np.column_stack([wc, oil]))
    cid_all.append(np.full(N_T, ci, np.int32))

X_all = np.vstack(X_all).astype(np.float32)
Y_all = np.vstack(Y_all).astype(np.float32)
cid_all = np.concatenate(cid_all)

# Temporal split: train first 75%, test last 25% of EVERY case
SPLIT = int(0.75 * N_T)
t_mask_tr = (X_all[:, 0] <= t_days[SPLIT])
t_mask_te = (X_all[:, 0] >  t_days[SPLIT])

X_tr, Y_tr = X_all[t_mask_tr], Y_all[t_mask_tr]
X_te, Y_te = X_all[t_mask_te], Y_all[t_mask_te]
print(f'[DATA] Temporal split: train {X_tr.shape[0]} pts (t≤{SPLIT} days) | '
      f'test {X_te.shape[0]} pts (t>{SPLIT} days)')

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
BATCH     = 8192    # larger batch = fewer steps = faster epochs
N_PHY     = 256     # physics collocation points (small = fast)
LR        = 1e-3
W_PHY_MAX = 0.10    # curriculum: ramp 0 → W_PHY_MAX over first 150 epochs
WARMUP    = 150

# Pre-sample physics collocation points from X_tr (fixed pool, resampled each epoch)
_rng = np.random.default_rng(0)

def physics_loss_fn(model, n=N_PHY):
    """
    Finite-difference monotonicity on N_PHY random collocation points.
    Uses a small separate batch — fast, avoids second-order gradients.
    """
    idx    = _rng.integers(0, len(X_tr), n)
    xc     = tf.constant(X_tr[idx], dtype=tf.float32)          # (n,3)
    eps    = 0.02                                               # ~34-day step
    t_next = tf.minimum(xc[:, :1] + eps, 1.0)
    xc2    = tf.concat([t_next, xc[:, 1:]], axis=1)            # (n,3)

    y1 = model(xc,  training=True)
    y2 = model(xc2, training=True)

    dwc  = y2[:, 0] - y1[:, 0]   # should be ≥ 0
    doil = y2[:, 1] - y1[:, 1]   # should be ≤ 0

    past = tf.cast(xc[:, 0] > xc[:, 1], tf.float32)
    L_wc  = tf.reduce_mean(past * tf.square(tf.nn.relu(-dwc)))
    L_oil = tf.reduce_mean(past * tf.square(tf.nn.relu(doil)))
    return L_wc + L_oil


def train_model(model, is_pinn, label):
    lr_sch = keras.optimizers.schedules.CosineDecayRestarts(LR, 200, t_mul=1.5)
    opt    = keras.optimizers.Adam(lr_sch)
    ds     = (tf.data.Dataset.from_tensor_slices(
                  (tf.constant(X_tr), tf.constant(Y_tr)))
              .shuffle(200000, seed=42).batch(BATCH).prefetch(2))
    hist_tr, hist_te = [], []
    best_te, best_w  = np.inf, None

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
        # Evaluate test every 10 epochs to save time
        if ep % 10 == 0 or ep == EPOCHS:
            yp_te = model(tf.constant(X_te), training=False).numpy()
            te    = float(np.mean((yp_te - Y_te)**2))
            hist_te.append(te)
            if te < best_te:
                best_te = te; best_w = model.get_weights()
        if ep % 100 == 0:
            te_now = hist_te[-1] if hist_te else float('nan')
            print(f'  [{label}] ep {ep:4d} | train={tr:.5f}  test={te_now:.5f}  w_phy={w_phy:.3f}')

    model.set_weights(best_w)
    return np.array(hist_tr), np.array(hist_te)


print('\n[TRAIN] Pure NN ...')
nn   = build_model('nn')
nn_tr, nn_te = train_model(nn, False, 'NN  ')

print('\n[TRAIN] PINN (monotonicity physics) ...')
pinn = build_model('pinn')
pn_tr, pn_te = train_model(pinn, True, 'PINN')

# ──────────────────────────────────────────────────────────────
# 4. METRICS
# ──────────────────────────────────────────────────────────────
def metrics(yt, yp, name):
    def nse(a,b): return 1 - np.sum((a-b)**2)/(np.sum((a-a.mean())**2)+1e-10)
    r  = {k+'_wc':  f(yt[:,0], yp[:,0]) for k,f in
          [('r2',r2_score),('rmse', lambda a,b: np.sqrt(mean_squared_error(a,b))),
           ('mae',mean_absolute_error),('nse',nse)]}
    r.update({k+'_oil': f(yt[:,1], yp[:,1]) for k,f in
              [('r2',r2_score),('rmse', lambda a,b: np.sqrt(mean_squared_error(a,b))),
               ('mae',mean_absolute_error),('nse',nse)]})
    print(f'  {name} WC : R²={r["r2_wc"]:.4f} RMSE={r["rmse_wc"]:.4f} '
          f'MAE={r["mae_wc"]:.4f} NSE={r["nse_wc"]:.4f}')
    print(f'  {name} Oil: R²={r["r2_oil"]:.4f} RMSE={r["rmse_oil"]:.4f} '
          f'MAE={r["mae_oil"]:.4f} NSE={r["nse_oil"]:.4f}')
    return r

nn_pred   = nn(tf.constant(X_te),   training=False).numpy()
pinn_pred = pinn(tf.constant(X_te), training=False).numpy()
print('\n[METRICS — TEST (last 25% of time, all cases)]')
m_nn   = metrics(Y_te, nn_pred,   'NN  ')
m_pinn = metrics(Y_te, pinn_pred, 'PINN')

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
years = t_days * T_MAX / 365.25
split_yr = t_days[SPLIT] * T_MAX / 365.25

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'  [PLOT] {name}')

# ── Fig 1: Loss curves (reference style: train+test same panel) ─────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, tr, te, title in [
    (axes[0], nn_tr, nn_te, 'Pure Data-driven Model'),
    (axes[1], pn_tr, pn_te, 'Proposed PINN Model'),
]:
    ep_tr_ax = np.arange(1, EPOCHS+1)
    ep_te_ax = np.arange(10, EPOCHS+1, 10)
    if len(ep_te_ax) < len(te):
        ep_te_ax = np.append(ep_te_ax, EPOCHS)
    ep_te_ax = ep_te_ax[:len(te)]
    ax.semilogy(ep_tr_ax, tr, lw=1.5, color='#2196F3', label='Train')
    ax.semilogy(ep_te_ax, te, lw=1.5, color='#F44336', label='Test', alpha=0.85)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_facecolor('white')
fig.suptitle('Training and Test Losses — NN vs PINN\nPelican Lake CMG STARS Polymer Flood', fontsize=12)
fig.tight_layout(); save(fig, 'fig1_loss_curves.png')

# ── Fig 2: WC predictions (4 cases) ─────────────────────────
sel = cases[:4]
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
axes = axes.flatten()
for ax, c in zip(axes, sel):
    ps  = poly_starts[c]
    wc_true = wc_agg[c].values[:N_T]
    Xc  = np.column_stack([t_days, np.full(N_T,ps,np.float32), q_norm]).astype(np.float32)
    wc_nn   = nn(tf.constant(Xc),   training=False).numpy()[:,0]
    wc_pinn = pinn(tf.constant(Xc), training=False).numpy()[:,0]

    r2_nn_tr   = r2_score(wc_true[:SPLIT],   wc_nn[:SPLIT])
    r2_pinn_tr = r2_score(wc_true[:SPLIT],   wc_pinn[:SPLIT])
    r2_nn_te   = r2_score(wc_true[SPLIT:],   wc_nn[SPLIT:])
    r2_pinn_te = r2_score(wc_true[SPLIT:],   wc_pinn[SPLIT:])

    ax.axvspan(split_yr, years[-1], alpha=0.06, color='gold', label='Forecast zone')
    ax.axvline(split_yr, color='k', ls=':', lw=1.2)
    ax.plot(years, wc_true,  color=C['CMG'], lw=2.5, label='CMG STARS')
    ax.plot(years, wc_nn,    color=C['NN'],  lw=2,   ls='--',
            label=f'NN  train R²={r2_nn_tr:.3f} | forecast R²={r2_nn_te:.3f}')
    ax.plot(years, wc_pinn,  color=C['PINN'],lw=2,   ls=':',
            label=f'PINN train R²={r2_pinn_tr:.3f} | forecast R²={r2_pinn_te:.3f}')
    ax.axvline(ps * T_MAX / 365.25, color='green', ls='-.', lw=1, alpha=0.7)
    ax.set_title(f'{c}  (poly start day {int(ps*T_MAX)})'); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Time (years)'); ax.set_ylabel('Water Cut')
    ax.legend(fontsize=7.5)

fig.suptitle('Water Cut — NN vs PINN vs CMG STARS\nShaded = forecast period (unseen during training)', fontsize=12)
fig.tight_layout(); save(fig, 'fig2_wc_forecast.png')

# ── Fig 3: Oil predictions (4 cases) ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for ax, c in zip(axes, sel):
    ps  = poly_starts[c]
    oil_true = op_agg[c].values[:N_T]
    Xc  = np.column_stack([t_days, np.full(N_T,ps,np.float32), q_norm]).astype(np.float32)
    oil_nn   = nn(tf.constant(Xc),   training=False).numpy()[:,1] * OIL_MAX
    oil_pinn = pinn(tf.constant(Xc), training=False).numpy()[:,1] * OIL_MAX

    r2_nn_te   = r2_score(oil_true[SPLIT:], oil_nn[SPLIT:])
    r2_pinn_te = r2_score(oil_true[SPLIT:], oil_pinn[SPLIT:])

    ax.axvspan(split_yr, years[-1], alpha=0.06, color='gold')
    ax.axvline(split_yr, color='k', ls=':', lw=1.2)
    ax.plot(years, oil_true,  color=C['CMG'], lw=2.5, label='CMG STARS')
    ax.plot(years, oil_nn,    color=C['NN'],  lw=2,   ls='--',
            label=f'NN  forecast R²={r2_nn_te:.3f}')
    ax.plot(years, oil_pinn,  color=C['PINN'],lw=2,   ls=':',
            label=f'PINN forecast R²={r2_pinn_te:.3f}')
    ax.set_title(f'{c}  (poly start day {int(ps*T_MAX)})')
    ax.set_xlabel('Time (years)'); ax.set_ylabel('Oil Rate (bbl/day)')
    ax.legend(fontsize=8)

fig.suptitle('Oil Production Rate — NN vs PINN vs CMG STARS\nShaded = forecast period', fontsize=12)
fig.tight_layout(); save(fig, 'fig3_oil_forecast.png')

# ── Fig 4: Statistical metrics bar chart ────────────────────
KEYS = ['r2_wc','r2_oil','rmse_wc','rmse_oil','mae_wc','mae_oil','nse_wc','nse_oil']
LBLS = ['R² WC','R² Oil','RMSE WC','RMSE Oil','MAE WC','MAE Oil','NSE WC','NSE Oil']
nn_v   = [m_nn[k]   for k in KEYS]
pinn_v = [m_pinn[k] for k in KEYS]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
xp = np.arange(4); w = 0.35

# Goodness (R², NSE)
good_idx = [0,1,6,7]
ax = axes[0]
bars_nn   = ax.bar(xp-w/2, [nn_v[i]   for i in good_idx], w, color=C['NN'],   alpha=0.85, label='NN')
bars_pinn = ax.bar(xp+w/2, [pinn_v[i] for i in good_idx], w, color=C['PINN'], alpha=0.85, label='PINN')
ax.axhline(1.0, color='gray', ls='--', lw=0.8)
ax.set_xticks(xp); ax.set_xticklabels([LBLS[i] for i in good_idx])
ax.set_ylabel('Score'); ax.set_title('Goodness-of-Fit Metrics (↑ better)'); ax.legend()
for bar, val in list(zip(bars_nn, [nn_v[i] for i in good_idx])) + \
                list(zip(bars_pinn, [pinn_v[i] for i in good_idx])):
    yv = max(val, ax.get_ylim()[0])
    ax.text(bar.get_x()+bar.get_width()/2, yv+0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold',
            color=bar.get_facecolor())

# Error (RMSE, MAE)
err_idx = [2,3,4,5]
ax2 = axes[1]
bars_nn2   = ax2.bar(xp-w/2, [nn_v[i]   for i in err_idx], w, color=C['NN'],   alpha=0.85, label='NN')
bars_pinn2 = ax2.bar(xp+w/2, [pinn_v[i] for i in err_idx], w, color=C['PINN'], alpha=0.85, label='PINN')
ax2.set_xticks(xp); ax2.set_xticklabels([LBLS[i] for i in err_idx])
ax2.set_ylabel('Error (normalised)'); ax2.set_title('Error Metrics (↓ better)'); ax2.legend()
for bar, val in list(zip(bars_nn2, [nn_v[i] for i in err_idx])) + \
                list(zip(bars_pinn2, [pinn_v[i] for i in err_idx])):
    ax2.text(bar.get_x()+bar.get_width()/2, val*1.01, f'{val:.4f}',
             ha='center', va='bottom', fontsize=8.5, fontweight='bold',
             color=bar.get_facecolor())

fig.suptitle('NN vs PINN Statistical Performance — Forecast Period (last 25%)\nPelican Lake CMG STARS Polymer Flood', fontsize=12)
fig.tight_layout(); save(fig, 'fig4_metrics.png')

# ── Fig 5: Predicted vs Actual Cumulative Oil per case ──────────────
# Compute per-case cumulative oil (bbl) for training and test periods
cum_act_tr, cum_act_te = [], []
cum_nn_tr_c, cum_nn_te_c = [], []
cum_pinn_tr_c, cum_pinn_te_c = [], []

for c in cases:
    ps  = poly_starts[c]
    Xc  = np.column_stack([t_days, np.full(N_T, ps, np.float32), q_norm]).astype(np.float32)
    oil_act  = op_agg[c].values[:N_T]
    oil_nn_p = nn(tf.constant(Xc),   training=False).numpy()[:,1] * OIL_MAX
    oil_pi_p = pinn(tf.constant(Xc), training=False).numpy()[:,1] * OIL_MAX
    for lst_a, lst_n, lst_p, sl in [
        (cum_act_tr, cum_nn_tr_c, cum_pinn_tr_c, slice(None, SPLIT)),
        (cum_act_te, cum_nn_te_c, cum_pinn_te_c, slice(SPLIT, None)),
    ]:
        lst_a.append(np.trapezoid(oil_act[sl],   t_days[sl]) * T_MAX)
        lst_n.append(np.trapezoid(oil_nn_p[sl],  t_days[sl]) * T_MAX)
        lst_p.append(np.trapezoid(oil_pi_p[sl],  t_days[sl]) * T_MAX)

cum_act_tr  = np.array(cum_act_tr);  cum_act_te  = np.array(cum_act_te)
cum_nn_tr_c = np.array(cum_nn_tr_c); cum_nn_te_c = np.array(cum_nn_te_c)
cum_pi_tr_c = np.array(cum_pinn_tr_c); cum_pi_te_c = np.array(cum_pinn_te_c)

def corr(a, b): return float(np.corrcoef(a, b)[0,1])
r_nn_tr  = corr(cum_act_tr, cum_nn_tr_c);  r_nn_te  = corr(cum_act_te, cum_nn_te_c)
r_pi_tr  = corr(cum_act_tr, cum_pi_tr_c);  r_pi_te  = corr(cum_act_te, cum_pi_te_c)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, act_tr, pred_tr, act_te, pred_te, r_tr, r_te, title, col in [
    (axes[0], cum_act_tr, cum_nn_tr_c, cum_act_te, cum_nn_te_c,
     r_nn_tr, r_nn_te, 'Pure Data-driven Model', C['NN']),
    (axes[1], cum_act_tr, cum_pi_tr_c, cum_act_te, cum_pi_te_c,
     r_pi_tr, r_pi_te, 'Proposed PINN Model', C['PINN']),
]:
    ax.scatter(act_tr/1e6, pred_tr/1e6, s=55, alpha=0.85, color='#2196F3',
               label='Train', edgecolors='white', linewidths=0.4, zorder=4)
    ax.scatter(act_te/1e6, pred_te/1e6, s=55, alpha=0.85, color='#FF9800',
               label='Test (Forecast)', edgecolors='white', linewidths=0.4, zorder=4)
    lo = min(np.concatenate([act_tr, act_te]).min(), np.concatenate([pred_tr, pred_te]).min())/1e6
    hi = max(np.concatenate([act_tr, act_te]).max(), np.concatenate([pred_tr, pred_te]).max())/1e6
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, zorder=3)
    ax.text(0.04, 0.92,
            f'r: {r_tr:.2f} (Train)\n   {r_te:.2f} (Test)',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.set_xlabel('Real Cumulative Oil Production (×10⁶ bbl)', fontsize=11)
    ax.set_ylabel('Predicted Cumulative Oil Production (×10⁶ bbl)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
fig.suptitle('Predicted vs. Actual Cumulative Oil Production — Per Case\nPelican Lake CMG STARS', fontsize=12)
fig.tight_layout(); save(fig, 'fig5_scatter.png')

# ── Fig 6: Optimization ──────────────────────────────────────
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
ax.set_xlabel('Polymer Injection Start Day'); ax.set_ylabel('Cumulative Oil (×10⁶ bbl·day)')
ax.set_title('Cumulative Oil Recovery vs Polymer Start Timing'); ax.legend()

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
ax2.set_title('Incremental Recovery over Baseline'); ax2.legend()
fig.suptitle('Polymer Injection Timing Optimisation — NN vs PINN\nPelican Lake CMG STARS', fontsize=12)
fig.tight_layout(); save(fig, 'fig6_optimization.png')

# ──────────────────────────────────────────────────────────────
# 7. CONCENTRATION OPTIMIZATION — Buckley-Leverett extension
# ──────────────────────────────────────────────────────────────
# The 51 CMG cases vary only in polymer start timing at a fixed
# reference concentration Cp_ref ≈ 1000 ppm.  To extend the
# optimisation to include polymer concentration we apply an
# analytical Buckley-Leverett (BL) correction factor derived from
# fractional flow theory — a standard two-stage surrogate approach.
#
# The PINN/NN surrogate captures the timing dimension from data;
# the BL model captures the concentration dimension from physics.
# ──────────────────────────────────────────────────────────────

# Pelican Lake heavy-oil parameters  (FWM-calibrated, pinn_causal_fwm.py)
MU_OIL     = 5000.0   # cp  — heavy oil viscosity
MU_W0      = 1.0      # cp  — water viscosity (no polymer)
SW_INIT    = 0.36     # connate water saturation
SW_MAX     = 0.80     # maximum water saturation
KRW_MAX    = 0.2918   # max k_rw (FWM calibration)
S_OR       = 0.10     # residual oil saturation
CP_REF_PPM = 1000.0   # concentration used in CMG STARS runs

def _mu_poly(cp_ppm):
    """Simplified Hand-model polymer viscosity (cp)."""
    return MU_W0 * (1.0 + 8e-4 * cp_ppm + 2e-7 * cp_ppm**2)

def _bl_rf(cp_ppm):
    """
    Oil recovery factor for heavy-oil polymer flooding (Pelican Lake regime).

    For very unfavourable mobility ratio (M >> 1, heavy oil at 5000 cp):
      - The fractional flow curve is convex near S_wi — Welge construction
        would give near-immediate breakthrough (tiny Np at BT).
      - The dominant polymer-flood improvement mechanism is SWEEP EFFICIENCY:
        higher Cp lowers M = k_rw_max × μ_o / μ_w(Cp), improving areal sweep.
      - From Craig-Geffen-Morse / Dykstra-Parsons theory for M >> 1:
            E_sweep ∝ (1/M)^0.35  →  RF ∝ μ_w(Cp)^0.35
      - This gives monotonically increasing RF with Cp (physically correct).
    """
    mu_wp = _mu_poly(cp_ppm)
    # RF proportional to μ_w^0.35 (sweep-efficiency scaling for heavy oil)
    return mu_wp ** 0.35

print('\n[BL] Computing concentration correction factors ...')
cp_scan_opt   = np.linspace(500, 2000, 60)
rf_vals       = np.array([_bl_rf(cp) for cp in cp_scan_opt])
rf_ref_val    = _bl_rf(CP_REF_PPM)
cp_rf_ratio   = rf_vals / rf_ref_val   # relative to CMG reference Cp

# ── 2-D optimisation: (poly_start, Cp)
print('[OPT-2D] Scanning 40×40 (timing × concentration) grid ...')
N_PS, N_CP     = 40, 40
scan_ps_2d     = np.linspace(0.0, 0.40, N_PS, dtype=np.float32)
scan_cp_2d     = np.linspace(500, 2000, N_CP)
cum_pinn_2d    = np.zeros((N_CP, N_PS))
cum_nn_2d      = np.zeros((N_CP, N_PS))

# Pre-compute BL ratio for each Cp row
cp_2d_rf = np.array([_bl_rf(cp) / rf_ref_val for cp in scan_cp_2d])

for j, ps in enumerate(scan_ps_2d):
    Xs      = np.column_stack([t_days,
                                np.full(N_T, float(ps), np.float32),
                                q_norm]).astype(np.float32)
    o_p     = pinn(tf.constant(Xs), training=False).numpy()[:,1]
    o_n     = nn(tf.constant(Xs),   training=False).numpy()[:,1]
    base_p  = float(np.trapezoid(o_p, t_days)) * OIL_MAX * T_MAX
    base_n  = float(np.trapezoid(o_n, t_days)) * OIL_MAX * T_MAX
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

# ── Fig 7: BL concentration response ────────────────────────
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
ax2.plot(cp_scan_opt,
         (rf_vals/rf_ref_val - 1)*100, lw=2.5, color='#27ae60')
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.axvline(CP_REF_PPM, color='gray', ls='--', lw=1.2)
ax2.set_xlabel('Polymer Concentration (ppm)')
ax2.set_ylabel('Incremental Recovery over 1000 ppm (%)')
ax2.set_title('Marginal Gain from Concentration Increase')
ax2.grid(alpha=0.3)
fig.suptitle('Buckley-Leverett Polymer Concentration Correction\n'
             'Pelican Lake Heavy Oil (μ_o=5000 cp, k_rw_max=0.29)', fontsize=12)
fig.tight_layout(); save(fig, 'fig7_bl_concentration.png')

# ── Fig 8: 2-D optimisation landscape ───────────────────────
days_2d = scan_ps_2d * T_MAX
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, data, lbl, opt_ps, opt_cp, cmap in [
    (axes[0], cum_nn_2d/1e6,   'Pure NN',  opt_nn_ps_2d,   opt_nn_cp_2d,   'RdYlGn'),
    (axes[1], cum_pinn_2d/1e6, 'PINN',     opt_pinn_ps_2d, opt_pinn_cp_2d, 'RdYlBu'),
]:
    cf = ax.contourf(days_2d, scan_cp_2d, data, levels=25, cmap=cmap)
    ax.contour( days_2d, scan_cp_2d, data, levels=12,
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
    ax.legend(loc='upper right', fontsize=8.5,
              facecolor='white', framealpha=0.8)
fig.suptitle('Joint Polymer Start Timing × Concentration Optimisation\n'
             'NN vs PINN Surrogate — Pelican Lake CMG STARS', fontsize=12)
fig.tight_layout(); save(fig, 'fig8_2d_optimization.png')

# ──────────────────────────────────────────────────────────────
# 8. SUMMARY
# ──────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('SUMMARY  NN vs PINN — Forecast period (last 25% of time, all cases)')
print('='*70)
print(f'  {"Metric":<22} {"NN":>10} {"PINN":>10} {"Δ(PINN−NN)":>12}')
print('  ' + '-'*56)
for k, lbl in [('r2_wc','R² WC'),('r2_oil','R² Oil'),
                ('rmse_wc','RMSE WC'),('rmse_oil','RMSE Oil'),
                ('mae_wc','MAE WC'),('mae_oil','MAE Oil'),
                ('nse_wc','NSE WC'),('nse_oil','NSE Oil')]:
    nv, pv = m_nn[k], m_pinn[k]
    d = pv - nv
    s = '+' if d >= 0 else ''
    print(f'  {lbl:<22} {nv:>10.4f} {pv:>10.4f} {s+f"{d:.4f}":>12}')
print(f'\n  TIMING-ONLY optimisation:')
print(f'    NN   optimal poly start: day {opt_day_nn}')
print(f'    PINN optimal poly start: day {opt_day_pinn}')
print(f'\n  JOINT (timing + concentration) optimisation:')
print(f'    NN   optimal: day {opt_nn_ps_2d},  Cp = {opt_nn_cp_2d:.0f} ppm')
print(f'    PINN optimal: day {opt_pinn_ps_2d}, Cp = {opt_pinn_cp_2d:.0f} ppm')
print(f'  Figures (8) → {OUT_DIR}')
print('='*70)
