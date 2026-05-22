"""
Physics-Informed Neural Network for Heavy Oil Polymer Flooding
Pelican Lake Field — SPE-166256 / SPE-179648
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get('DATA_DIR',   'data')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Physical constants (SPE-166256) ───────────────────────────────────────────
POROSITY        = 0.33
PERMEABILITY    = 3000.0     # mD
NET_PAY_M       = 2.4        # m
WELL_LEN_M      = 1400.0     # m
SPACING_M       = 175.0      # m

MU_OIL          = 1650.0     # cP
MU_WATER        = 1.0        # cP
MU_POLYMER_REF  = 22.0       # cP at 1000 ppm

SWC             = 0.224
SOR             = 0.206
KRW_MAX_INIT    = 0.216
KRO_MAX_INIT    = 1.0
NW_INIT         = 3.834
NO_INIT         = 1.885
RK              = 1.0

CP_MAX_PPM      = 2000.0
INJ_RATE_MAX    = 750.0      # STB/day
OPR_SCALE       = 500.0      # STB/day
TOTAL_DAYS      = 1705.0
N_TIMESTEPS     = 57

# Polymer viscosity power-law: mu = A * Cp^B  (calibrated to Table 6)
POLY_VISC_A     = 0.0002
POLY_VISC_B     = 1.661

# Loss weights
LAMBDA_R        = 0.5
LAMBDA_IC       = 0.5
LAMBDA_BC       = 0.5
LAMBDA_P        = 0.1

# Training hypers
EPOCHS          = 500
BATCH_SIZE      = 64
PATIENCE        = 50
LR              = 1e-3
LR_SW           = 1e-3
N_COLLOC        = 1000
N_IC            = 100
N_BC            = 100

# Splits
N_CASES         = 200
N_TRAIN         = 140
N_VAL           = 30
N_TEST          = 30


# ═════════════════════════════════════════════════════════════════════════════
# Custom Keras Layers
# ═════════════════════════════════════════════════════════════════════════════

class SwScaleLayer(layers.Layer):
    """Scale sigmoid output [0,1] to [SWC, 1-SOR]."""

    def __init__(self, swc=SWC, sor=SOR, **kw):
        super().__init__(**kw)
        self.swc = swc
        self.lo  = swc
        self.hi  = 1.0 - sor

    def call(self, s):
        return s * (self.hi - self.lo) + self.lo

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'swc': self.swc, 'sor': SOR})
        return cfg


class CMGViscosityLayer(layers.Layer):
    """Polymer solution viscosity from SPE-166256 Table 6 (power-law fit)."""

    def __init__(self, cp_max_ppm=CP_MAX_PPM, A=POLY_VISC_A, B=POLY_VISC_B,
                 mu_water=MU_WATER, **kw):
        super().__init__(**kw)
        self.cp_max_ppm = tf.constant(cp_max_ppm, dtype=tf.float32)
        self.A          = tf.constant(A,           dtype=tf.float32)
        self.B          = tf.constant(B,           dtype=tf.float32)
        self.mu_water   = tf.constant(mu_water,    dtype=tf.float32)

    def call(self, cp_norm):
        # cp_norm in [0, 1]; convert to ppm
        cp_ppm = cp_norm * self.cp_max_ppm
        # Power-law: mu = A * Cp^B
        mu_poly = self.A * tf.math.pow(cp_ppm + 1e-6, self.B)
        # Blend: at Cp=0 -> mu_water; at Cp>0 -> mu_poly
        mask    = tf.cast(cp_norm > 1e-4, tf.float32)
        mu_eff  = mask * mu_poly + (1.0 - mask) * self.mu_water
        return mu_eff

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'cp_max_ppm': float(self.cp_max_ppm.numpy()),
            'A': float(self.A.numpy()),
            'B': float(self.B.numpy()),
            'mu_water': float(self.mu_water.numpy()),
        })
        return cfg


class CoreyKrLayer(layers.Layer):
    """Corey relative permeability with TRAINABLE parameters."""

    def __init__(self, swc=SWC, sor=SOR, **kw):
        super().__init__(**kw)
        self.swc_const = tf.constant(swc,  dtype=tf.float32)
        self.sor_const = tf.constant(sor,  dtype=tf.float32)

    def build(self, input_shape):
        self.krw_max = self.add_weight(
            name='krw_max',
            shape=(), initializer=tf.constant_initializer(KRW_MAX_INIT),
            trainable=True)
        self.kro_max = self.add_weight(
            name='kro_max',
            shape=(), initializer=tf.constant_initializer(KRO_MAX_INIT),
            trainable=True)
        self.nw = self.add_weight(
            name='nw',
            shape=(), initializer=tf.constant_initializer(NW_INIT),
            trainable=True)
        self.no = self.add_weight(
            name='no',
            shape=(), initializer=tf.constant_initializer(NO_INIT),
            trainable=True)
        super().build(input_shape)

    def call(self, sw):
        # Clip trainable weights to physical bounds
        krw_max = tf.clip_by_value(self.krw_max, 0.01, 1.0)
        kro_max = tf.clip_by_value(self.kro_max, 0.1,  1.0)
        nw      = tf.clip_by_value(self.nw,      1.0,  8.0)
        no      = tf.clip_by_value(self.no,      1.0,  8.0)

        sw_c  = tf.clip_by_value(sw, self.swc_const, 1.0 - self.sor_const)
        swc   = self.swc_const
        sor   = self.sor_const

        sw_norm_w = (sw_c - swc)      / (1.0 - sor - swc + 1e-8)
        sw_norm_o = (1.0 - sor - sw_c) / (1.0 - sor - swc + 1e-8)

        sw_norm_w = tf.clip_by_value(sw_norm_w, 0.0, 1.0)
        sw_norm_o = tf.clip_by_value(sw_norm_o, 0.0, 1.0)

        krw = krw_max * tf.math.pow(sw_norm_w, nw)
        kro = kro_max * tf.math.pow(sw_norm_o, no)
        return tf.concat([krw, kro], axis=-1)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 2
        return tuple(shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'swc': float(self.swc_const.numpy()),
                    'sor': float(self.sor_const.numpy())})
        return cfg


class FractionalFlowLayer(layers.Layer):
    """Polymer-modified fractional flow fw = mob_w / (mob_w + mob_o)."""

    def __init__(self, mu_oil=MU_OIL, rk=RK, **kw):
        super().__init__(**kw)
        self.mu_oil = tf.constant(mu_oil, dtype=tf.float32)
        self.rk     = tf.constant(rk,     dtype=tf.float32)

    def call(self, inputs):
        krw, kro, mu_eff = inputs[0], inputs[1], inputs[2]
        mob_w = krw / (mu_eff * self.rk + 1e-8)
        mob_o = kro / (self.mu_oil + 1e-8)
        fw    = mob_w / (mob_w + mob_o + 1e-8)
        return fw

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'mu_oil': float(self.mu_oil.numpy()),
                    'rk': float(self.rk.numpy())})
        return cfg


# ═════════════════════════════════════════════════════════════════════════════
# Saturation Sub-model
# ═════════════════════════════════════════════════════════════════════════════

class SaturationSubmodel(keras.Model):
    """
    Predicts Sw(xD, tD, cp_norm, inj_norm).
    Must be a keras.Model subclass for physics-tape compatibility.
    """

    def __init__(self, swc=SWC, sor=SOR, **kw):
        super().__init__(**kw)
        self.bn0   = layers.BatchNormalization(name='sw_bn0')
        setattr(self, 'sw_d1', layers.Dense(128, activation='tanh',  name='sw_d1'))
        setattr(self, 'sw_d2', layers.Dense(256, activation='tanh',  name='sw_d2'))
        setattr(self, 'sw_d3', layers.Dense(256, activation='tanh',  name='sw_d3'))
        setattr(self, 'sw_d4', layers.Dense(128, activation='tanh',  name='sw_d4'))
        setattr(self, 'sw_d5', layers.Dense(64,  activation='tanh',  name='sw_d5'))
        self.sw_dr2 = layers.Dropout(0.05, name='sw_dr2')
        self.sw_dr3 = layers.Dropout(0.05, name='sw_dr3')
        self.sw_dr4 = layers.Dropout(0.05, name='sw_dr4')
        self.sw_out  = layers.Dense(1, activation='sigmoid', name='sw_out')
        self.sw_scale = SwScaleLayer(swc=swc, sor=sor, name='sw_scale')

    def call(self, inputs, training=False):
        x = self.bn0(inputs, training=training)
        x = self.sw_d1(x)
        x = self.sw_d2(x)
        x = self.sw_dr2(x, training=training)
        x = self.sw_d3(x)
        x = self.sw_dr3(x, training=training)
        x = self.sw_d4(x)
        x = self.sw_dr4(x, training=training)
        x = self.sw_d5(x)
        x = self.sw_out(x)
        x = self.sw_scale(x)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# Production PINN
# ═════════════════════════════════════════════════════════════════════════════

class PolymerPINN(keras.Model):
    """
    Full PINN: trunk + physics path + production head.
    Inputs : (t_hat, cp_norm, inj_norm, bhp_norm)
    Outputs: (OPR_field_norm, WC_field)
    """

    def __init__(self, sw_submodel: SaturationSubmodel, **kw):
        super().__init__(**kw)
        self.sw_submodel = sw_submodel

        # Physics layers
        self.visc_layer  = CMGViscosityLayer(name='visc_layer')
        self.kr_layer    = CoreyKrLayer(name='kr_layer')
        self.fw_layer    = FractionalFlowLayer(name='fw_layer')

        # Trunk
        self.trunk_bn  = layers.BatchNormalization(name='trunk_bn')
        setattr(self, 'trunk_d1', layers.Dense(128, activation='tanh', name='trunk_d1'))
        setattr(self, 'trunk_d2', layers.Dense(256, activation='tanh', name='trunk_d2'))
        setattr(self, 'trunk_d3', layers.Dense(256, activation='tanh', name='trunk_d3'))
        setattr(self, 'trunk_d4', layers.Dense(128, activation='tanh', name='trunk_d4'))
        setattr(self, 'trunk_d5', layers.Dense(64,  activation='tanh', name='trunk_d5'))
        self.trunk_dr2 = layers.Dropout(0.05, name='trunk_dr2')
        self.trunk_dr3 = layers.Dropout(0.05, name='trunk_dr3')
        self.trunk_dr4 = layers.Dropout(0.05, name='trunk_dr4')

        # Production head
        setattr(self, 'head_d1', layers.Dense(64,  activation='tanh',    name='head_d1'))
        setattr(self, 'head_d2', layers.Dense(32,  activation='tanh',    name='head_d2'))
        setattr(self, 'qo_out',  layers.Dense(1,   activation='softplus', name='qo_out'))
        setattr(self, 'alpha_d', layers.Dense(1,   activation='sigmoid',  name='alpha_d'))
        setattr(self, 'wc_raw',  layers.Dense(1,                          name='wc_raw'))

    def _trunk(self, x, training=False):
        x = self.trunk_bn(x, training=training)
        x = self.trunk_d1(x)
        x = self.trunk_d2(x)
        x = self.trunk_dr2(x, training=training)
        x = self.trunk_d3(x)
        x = self.trunk_dr3(x, training=training)
        x = self.trunk_d4(x)
        x = self.trunk_dr4(x, training=training)
        x = self.trunk_d5(x)
        return x

    def call(self, inputs, training=False):
        t_hat    = tf.expand_dims(inputs[:, 0], axis=-1)   # (B,1)
        cp_norm  = tf.expand_dims(inputs[:, 1], axis=-1)
        inj_norm = tf.expand_dims(inputs[:, 2], axis=-1)
        bhp_norm = tf.expand_dims(inputs[:, 3], axis=-1)

        # Physics path
        mu_eff  = self.visc_layer(cp_norm)                 # (B,1)

        sw_inp  = tf.concat([
            tf.fill(tf.shape(t_hat), 0.5),  # xD = 0.5 (midpoint)
            t_hat, cp_norm, inj_norm
        ], axis=-1)                                         # (B,4)
        sw_hat  = self.sw_submodel(sw_inp, training=training)  # (B,1)

        kr_out  = self.kr_layer(sw_hat)                    # (B,2)
        krw     = tf.expand_dims(kr_out[:, 0], axis=-1)
        kro     = tf.expand_dims(kr_out[:, 1], axis=-1)
        fw      = self.fw_layer([krw, kro, mu_eff])        # (B,1)

        # Trunk
        trunk_out = self._trunk(inputs, training=training) # (B,64)

        # Head
        combined = tf.concat([trunk_out, sw_hat, fw, mu_eff], axis=-1)
        h = self.head_d1(combined)
        h = self.head_d2(h)

        qo     = self.qo_out(h)                            # (B,1) > 0
        alpha  = self.alpha_d(h)                           # (B,1) in [0,1]
        wc_nn  = self.wc_raw(h)                            # (B,1) unconstrained
        wc     = tf.sigmoid(alpha * wc_nn + (1.0 - alpha) * fw)

        return tf.concat([qo, wc], axis=-1)                # (B,2)


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════════════════════

def detect_layout(df):
    """Return 'A' if columns are like case_001_P1, else 'B' for case_001."""
    cols = [c for c in df.columns if c != 'time']
    return 'A' if '_P' in cols[0] else 'B'


def load_data(data_dir=DATA_DIR):
    """
    Load all 5 CSVs and return processed arrays.

    Returns
    -------
    X_tr, y_tr, X_va, y_va, X_te, y_te  — float32 numpy arrays
    case_ids_test                         — list of case indices (0-based)
    inj_norm_series                       — shape (57,) shared injection profile
    """
    oil_path  = os.path.join(data_dir, 'Oil_Production.csv')
    wc_path   = os.path.join(data_dir, 'Water_cut.csv')
    cp_path   = os.path.join(data_dir, 'Polymer_concentration.csv')
    inj1_path = os.path.join(data_dir, 'Injection_rate_inj1.csv')
    inj2_path = os.path.join(data_dir, 'Injection_rate_inj2.csv')

    # ── Check data exists ────────────────────────────────────────────────────
    missing = [p for p in [oil_path, wc_path, cp_path, inj1_path, inj2_path]
               if not os.path.exists(p)]
    if missing:
        print(f"[WARN] Data files not found: {missing}")
        print("[INFO] Generating synthetic data for demonstration …")
        return _generate_synthetic(data_dir)

    oil_df  = pd.read_csv(oil_path)
    wc_df   = pd.read_csv(wc_path)
    cp_df   = pd.read_csv(cp_path)
    inj1_df = pd.read_csv(inj1_path)
    inj2_df = pd.read_csv(inj2_path)

    # ── Time axis ────────────────────────────────────────────────────────────
    if 'time' in oil_df.columns:
        t_vals = oil_df['time'].values.astype(np.float32)
        t_hat  = (t_vals - t_vals[0]) / (t_vals[-1] - t_vals[0] + 1e-8)
    else:
        t_hat  = np.linspace(0.0, 1.0, N_TIMESTEPS, dtype=np.float32)

    # ── Injection rate (identical across cases) ───────────────────────────────
    inj1_cols = [c for c in inj1_df.columns if c != 'time']
    inj2_cols = [c for c in inj2_df.columns if c != 'time']
    q1_series = inj1_df[inj1_cols[0]].values.astype(np.float32)
    q2_series = inj2_df[inj2_cols[0]].values.astype(np.float32)
    total_inj  = q1_series + q2_series
    inj_norm   = (total_inj / INJ_RATE_MAX).astype(np.float32)

    # ── Column helpers ────────────────────────────────────────────────────────
    def get_case_cols_A(df, case_id):
        tag = f'case_{case_id:03d}'
        return [c for c in df.columns if c.startswith(tag + '_P')]

    def get_case_col_B(df, case_id):
        tag = f'case_{case_id:03d}'
        return [c for c in df.columns if c == tag]

    n_cases = N_CASES
    t_steps = len(t_hat)

    X_list, y_list = [], []

    for ci in range(1, n_cases + 1):
        # ── Oil production ────────────────────────────────────────────────
        oil_cols = get_case_cols_A(oil_df, ci)
        if not oil_cols:
            # Try layout B fallback
            oil_cols = get_case_col_B(oil_df, ci)
            opr_arr = oil_df[oil_cols].values.astype(np.float32).sum(axis=1)
        else:
            opr_arr = oil_df[oil_cols].values.astype(np.float32).sum(axis=1)

        # ── Water cut ─────────────────────────────────────────────────────
        wc_cols = get_case_cols_A(wc_df, ci)
        if not wc_cols:
            wc_cols = get_case_col_B(wc_df, ci)
            wc_arr = wc_df[wc_cols].values.astype(np.float32).mean(axis=1)
        else:
            wc_arr = wc_df[wc_cols].values.astype(np.float32).mean(axis=1)

        # ── Polymer concentration ─────────────────────────────────────────
        cp_cols = get_case_col_B(cp_df, ci)
        cp_series = cp_df[cp_cols].values.astype(np.float32).flatten()
        cp_val    = float(cp_series[0])  # constant per case [0,1]

        # ── Build feature matrix for this case ───────────────────────────
        bhp_fixed = np.full(t_steps, 0.5, dtype=np.float32)
        cp_col    = np.full(t_steps, cp_val, dtype=np.float32)

        X_case = np.column_stack([t_hat, cp_col, inj_norm, bhp_fixed])
        y_case = np.column_stack([opr_arr / OPR_SCALE,
                                  np.clip(wc_arr, 0.0, 1.0)])

        X_list.append(X_case)
        y_list.append(y_case)

    X_all = np.vstack(X_list).astype(np.float32)   # (200*57, 4)
    y_all = np.vstack(y_list).astype(np.float32)   # (200*57, 2)

    # ── Case-level split ──────────────────────────────────────────────────────
    rng        = np.random.default_rng(SEED)
    perm       = rng.permutation(n_cases)
    tr_ids     = sorted(perm[:N_TRAIN])
    va_ids     = sorted(perm[N_TRAIN:N_TRAIN + N_VAL])
    te_ids     = sorted(perm[N_TRAIN + N_VAL:])

    def idx_for_cases(case_ids):
        idxs = []
        for c in case_ids:
            start = c * t_steps
            idxs.extend(range(start, start + t_steps))
        return np.array(idxs)

    X_tr = X_all[idx_for_cases(tr_ids)]
    y_tr = y_all[idx_for_cases(tr_ids)]
    X_va = X_all[idx_for_cases(va_ids)]
    y_va = y_all[idx_for_cases(va_ids)]
    X_te = X_all[idx_for_cases(te_ids)]
    y_te = y_all[idx_for_cases(te_ids)]

    assert X_tr.shape[1] == 4, "Expected 4 inputs: t, cp, inj, bhp"
    assert y_tr.shape[1] == 2, "Expected 2 outputs: OPR_norm, WC"

    print(f"[DATA] Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, te_ids, inj_norm


def _generate_synthetic(data_dir):
    """Generate synthetic data when CSVs are not present."""
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    t_hat_arr  = np.linspace(0.0, 1.0, N_TIMESTEPS, dtype=np.float32)
    inj_profile = (0.5 + 0.4 * np.sin(np.pi * t_hat_arr)).astype(np.float32)

    X_list, y_list = [], []
    for ci in range(N_CASES):
        cp_val = ci / (N_CASES - 1)                   # [0, 1]
        cp_col = np.full(N_TIMESTEPS, cp_val, dtype=np.float32)
        bhp    = np.full(N_TIMESTEPS, 0.5, dtype=np.float32)

        X_case = np.column_stack([t_hat_arr, cp_col, inj_profile, bhp])

        # Synthetic physics: polymer increases OPR, decreases WC
        base_opr = 0.030 * (1.0 + cp_val) * inj_profile
        base_wc  = 0.92  - 0.06 * cp_val - 0.02 * t_hat_arr
        opr      = base_opr + rng.normal(0, 0.001, N_TIMESTEPS).astype(np.float32)
        wc       = np.clip(base_wc + rng.normal(0, 0.003, N_TIMESTEPS).astype(np.float32),
                           0.0, 1.0)

        X_list.append(X_case)
        y_list.append(np.column_stack([opr, wc]))

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.vstack(y_list).astype(np.float32)

    rng2   = np.random.default_rng(SEED)
    perm   = rng2.permutation(N_CASES)
    tr_ids = sorted(perm[:N_TRAIN])
    va_ids = sorted(perm[N_TRAIN:N_TRAIN + N_VAL])
    te_ids = sorted(perm[N_TRAIN + N_VAL:])

    def idx_for_cases(case_ids):
        idxs = []
        for c in case_ids:
            start = c * N_TIMESTEPS
            idxs.extend(range(start, start + N_TIMESTEPS))
        return np.array(idxs)

    X_tr = X_all[idx_for_cases(tr_ids)]
    y_tr = y_all[idx_for_cases(tr_ids)]
    X_va = X_all[idx_for_cases(va_ids)]
    y_va = y_all[idx_for_cases(va_ids)]
    X_te = X_all[idx_for_cases(te_ids)]
    y_te = y_all[idx_for_cases(te_ids)]

    print(f"[SYNTH] Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, te_ids, inj_profile


# ═════════════════════════════════════════════════════════════════════════════
# Physics Loss Functions
# ═════════════════════════════════════════════════════════════════════════════

def compute_bl_residual(sw_model, kr_layer, fw_layer, visc_layer,
                        xD, tD, cp_norm_colloc, inj_colloc):
    """
    Buckley-Leverett residual: ∂Sw/∂tD + (dfw/dSw) * ∂Sw/∂xD = 0
    xD, tD: (N_COLLOC, 1) tensors with gradient tracking
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([xD, tD])
        sw_inp = tf.concat([xD, tD, cp_norm_colloc, inj_colloc], axis=-1)
        sw     = sw_model(sw_inp, training=True)

        kr_out = kr_layer(sw)
        krw    = tf.expand_dims(kr_out[:, 0], axis=-1)
        kro    = tf.expand_dims(kr_out[:, 1], axis=-1)
        mu_eff = visc_layer(cp_norm_colloc)
        fw     = fw_layer([krw, kro, mu_eff])

    dSw_dtD = tape2.gradient(sw, tD)
    dfw_dSw = tape2.gradient(fw, sw)
    dSw_dxD = tape2.gradient(sw, xD)
    del tape2

    # Guard None gradients
    if dSw_dtD is None: dSw_dtD = tf.zeros_like(tD)
    if dfw_dSw is None: dfw_dSw = tf.zeros_like(sw)
    if dSw_dxD is None: dSw_dxD = tf.zeros_like(xD)

    residual = dSw_dtD + dfw_dSw * dSw_dxD
    return tf.reduce_mean(tf.square(residual))


def compute_ic_loss(sw_model, n_ic=N_IC):
    """IC: Sw(xD, tD=0) = SWC"""
    xD_ic = tf.random.uniform((n_ic, 1), 0.0, 1.0)
    tD_ic = tf.zeros((n_ic, 1))
    cp_ic = tf.random.uniform((n_ic, 1), 0.0, 1.0)
    inj_ic = tf.random.uniform((n_ic, 1), 0.0, 1.0)

    sw_inp = tf.concat([xD_ic, tD_ic, cp_ic, inj_ic], axis=-1)
    sw_ic  = sw_model(sw_inp, training=True)
    target = tf.fill(tf.shape(sw_ic), SWC)
    return tf.reduce_mean(tf.square(sw_ic - target))


def compute_bc_loss(sw_model, n_bc=N_BC):
    """BC: Sw(xD=0, tD) = 1 - SOR"""
    xD_bc  = tf.zeros((n_bc, 1))
    tD_bc  = tf.random.uniform((n_bc, 1), 0.0, 1.0)
    cp_bc  = tf.random.uniform((n_bc, 1), 0.0, 1.0)
    inj_bc = tf.random.uniform((n_bc, 1), 0.0, 1.0)

    sw_inp = tf.concat([xD_bc, tD_bc, cp_bc, inj_bc], axis=-1)
    sw_bc  = sw_model(sw_inp, training=True)
    target = tf.fill(tf.shape(sw_bc), 1.0 - SOR)
    return tf.reduce_mean(tf.square(sw_bc - target))


def compute_penalty(y_pred):
    """Positivity and boundedness penalties."""
    qo  = tf.expand_dims(y_pred[:, 0], axis=-1)
    wc  = tf.expand_dims(y_pred[:, 1], axis=-1)
    p1  = tf.reduce_mean(tf.square(tf.nn.relu(-qo)))
    p2  = tf.reduce_mean(tf.square(tf.nn.relu(-wc)))
    p3  = tf.reduce_mean(tf.square(tf.nn.relu(wc - 1.0)))
    return p1 + p2 + p3


# ═════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═════════════════════════════════════════════════════════════════════════════

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10
    return float(1.0 - ss_res / ss_tot)


def train(model: PolymerPINN, sw_model: SaturationSubmodel,
          X_tr, y_tr, X_va, y_va):
    model_opt = keras.optimizers.Adam(learning_rate=LR)
    sw_opt    = keras.optimizers.Adam(learning_rate=LR_SW)

    X_tr_tf = tf.constant(X_tr, dtype=tf.float32)
    y_tr_tf = tf.constant(y_tr, dtype=tf.float32)
    X_va_tf = tf.constant(X_va, dtype=tf.float32)
    y_va_tf = tf.constant(y_va, dtype=tf.float32)

    dataset    = tf.data.Dataset.from_tensor_slices((X_tr_tf, y_tr_tf))
    dataset    = dataset.shuffle(buffer_size=8000, seed=SEED).batch(BATCH_SIZE)

    # History
    hist = {k: [] for k in ['total', 'data', 'bl', 'ic', 'bc', 'val_total']}

    best_val   = np.inf
    wait       = 0
    best_w_path = os.path.join(OUTPUT_DIR, 'pinn_polymer_best.weights.h5')

    # Pre-create physics layers references
    kr_layer  = model.kr_layer
    fw_layer  = model.fw_layer
    visc_layer = model.visc_layer

    for epoch in range(1, EPOCHS + 1):

        # ── Step 1: Physics update (ONCE per epoch) ───────────────────────
        xD_c   = tf.Variable(tf.random.uniform((N_COLLOC, 1), 0.0, 1.0))
        tD_c   = tf.Variable(tf.random.uniform((N_COLLOC, 1), 0.0, 1.0))
        cp_c   = tf.Variable(tf.random.uniform((N_COLLOC, 1), 0.0, 1.0))
        inj_c  = tf.Variable(tf.random.uniform((N_COLLOC, 1), 0.0, 1.0))

        with tf.GradientTape() as phys_tape:
            L_BL = compute_bl_residual(
                sw_model, kr_layer, fw_layer, visc_layer,
                xD_c, tD_c, cp_c, inj_c)
            L_IC = compute_ic_loss(sw_model, N_IC)
            L_BC = compute_bc_loss(sw_model, N_BC)
            phys_loss = LAMBDA_R * L_BL + LAMBDA_IC * L_IC + LAMBDA_BC * L_BC

        phys_grads = phys_tape.gradient(phys_loss,
                                        sw_model.trainable_variables)
        sw_opt.apply_gradients(
            zip(phys_grads, sw_model.trainable_variables))

        # ── Step 2: Data update (per batch) ──────────────────────────────
        epoch_data_loss = 0.0
        n_batches = 0
        for X_b, y_b in dataset:
            with tf.GradientTape() as data_tape:
                y_pred   = model(X_b, training=True)
                mse_opr  = tf.reduce_mean(tf.square(y_pred[:, 0] - y_b[:, 0]))
                mse_wc   = tf.reduce_mean(tf.square(y_pred[:, 1] - y_b[:, 1]))
                L_data   = mse_opr + mse_wc
                L_pen    = compute_penalty(y_pred)
                total_data = L_data + LAMBDA_P * L_pen

            data_grads = data_tape.gradient(total_data,
                                            model.trainable_variables)
            model_opt.apply_gradients(
                zip(data_grads, model.trainable_variables))
            epoch_data_loss += float(total_data)
            n_batches += 1

        epoch_data_loss /= max(n_batches, 1)
        total_loss = epoch_data_loss + float(phys_loss)

        # ── Validation ────────────────────────────────────────────────────
        y_va_pred = model(X_va_tf, training=False).numpy()
        val_mse   = float(np.mean((y_va_pred - y_va) ** 2))

        # ── Record history ────────────────────────────────────────────────
        hist['total'].append(total_loss)
        hist['data'].append(epoch_data_loss)
        hist['bl'].append(float(L_BL))
        hist['ic'].append(float(L_IC))
        hist['bc'].append(float(L_BC))
        hist['val_total'].append(val_mse)

        # ── Early stopping ────────────────────────────────────────────────
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            wait = 0
            model.save_weights(best_w_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        # ── Progress print every 10 epochs ───────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | total={total_loss:.4f} "
                  f"data={epoch_data_loss:.4f} BL={float(L_BL):.4f} "
                  f"IC={float(L_IC):.4f} BC={float(L_BC):.4f} "
                  f"val={val_mse:.4f}")

    # Restore best weights
    if os.path.exists(best_w_path):
        model.load_weights(best_w_path)

    # Save sw_submodel
    sw_save_path = os.path.join(OUTPUT_DIR, 'pinn_polymer_sw_submodel.keras')
    try:
        sw_model.save(sw_save_path)
    except Exception:
        sw_model.save_weights(os.path.join(OUTPUT_DIR, 'sw_submodel.weights.h5'))

    # Save history
    hist_path = os.path.join(OUTPUT_DIR, 'pinn_polymer_history.json')
    with open(hist_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.items()}, f)

    return hist


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_te, y_te):
    y_pred = model(tf.constant(X_te, dtype=tf.float32), training=False).numpy()

    opr_r2 = r_squared(y_te[:, 0], y_pred[:, 0])
    wc_r2  = r_squared(y_te[:, 1], y_pred[:, 1])
    print(f"\n[EVAL] Test OPR R²={opr_r2:.4f}  WC R²={wc_r2:.4f}")

    metrics = {'opr_r2': opr_r2, 'wc_r2': wc_r2}
    with open(os.path.join(OUTPUT_DIR, 'pinn_polymer_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return y_pred, metrics


# ═════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═════════════════════════════════════════════════════════════════════════════

def plot_fractional_flow(model: PolymerPINN):
    """Fractional flow curves for 5 Cp levels."""
    sw_vals = np.linspace(SWC, 1.0 - SOR, 200, dtype=np.float32)
    sw_tf   = tf.constant(sw_vals[:, np.newaxis])

    cp_ppm_list = [0, 500, 1000, 1500, 2000]
    fig, ax = plt.subplots(figsize=(8, 5))

    for cp_ppm in cp_ppm_list:
        cp_n  = tf.constant([[cp_ppm / CP_MAX_PPM]] * 200, dtype=tf.float32)
        kr_out = model.kr_layer(sw_tf)
        krw    = tf.expand_dims(kr_out[:, 0], axis=-1)
        kro    = tf.expand_dims(kr_out[:, 1], axis=-1)
        fw     = model.fw_layer([krw, kro, model.visc_layer(cp_n)]).numpy()
        ax.plot(sw_vals, fw.flatten(), label=f'Cp={cp_ppm} ppm')

    ax.set_xlabel('Water Saturation Sw')
    ax.set_ylabel('Fractional Flow fw')
    ax.set_title('Fractional Flow Curves (SPE-166256 Corey params)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fractional_flow.png'), dpi=150)
    plt.close(fig)
    print("[PLOT] fractional_flow.png saved")


def plot_training_history(hist):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].semilogy(hist['total'], label='Total')
    axes[0, 0].semilogy(hist['data'],  label='Data')
    axes[0, 0].semilogy(hist['val_total'], label='Val')
    axes[0, 0].set_title('Total & Data Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(hist['bl'], label='BL residual', color='red')
    axes[0, 1].set_title('Buckley-Leverett Residual')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(hist['ic'], label='IC loss', color='green')
    axes[1, 0].semilogy(hist['bc'], label='BC loss', color='purple')
    axes[1, 0].set_title('IC & BC Losses')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(hist['val_total'], color='orange')
    axes[1, 1].set_title('Validation Loss')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('PINN Training History — Pelican Lake Polymer Flooding')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)
    plt.close(fig)
    print("[PLOT] training_history.png saved")


def plot_parity(y_te, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # OPR parity
    opr_true = y_te[:, 0] * OPR_SCALE
    opr_pred = y_pred[:, 0] * OPR_SCALE
    ax1.scatter(opr_true, opr_pred, alpha=0.3, s=5, c='steelblue')
    mn, mx = opr_true.min(), opr_true.max()
    ax1.plot([mn, mx], [mn, mx], 'r--', lw=1)
    r2 = r_squared(y_te[:, 0], y_pred[:, 0])
    ax1.set_xlabel('CMG STARS OPR (STB/day)')
    ax1.set_ylabel('PINN OPR (STB/day)')
    ax1.set_title(f'Field OPR — R²={r2:.3f}')
    ax1.grid(True, alpha=0.3)

    # WC parity
    ax2.scatter(y_te[:, 1], y_pred[:, 1], alpha=0.3, s=5, c='darkorange')
    mn, mx = y_te[:, 1].min(), y_te[:, 1].max()
    ax2.plot([mn, mx], [mn, mx], 'r--', lw=1)
    r2w = r_squared(y_te[:, 1], y_pred[:, 1])
    ax2.set_xlabel('CMG STARS WC')
    ax2.set_ylabel('PINN WC')
    ax2.set_title(f'Field Water Cut — R²={r2w:.3f}')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('PINN vs CMG STARS — Test Set Parity')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'parity_test.png'), dpi=150)
    plt.close(fig)
    print("[PLOT] parity_test.png saved")


def plot_timeseries(model, X_te, y_te, te_ids, n_cases=3):
    """Time-series comparison for up to 3 test cases."""
    fig, axes = plt.subplots(n_cases, 2, figsize=(14, 4 * n_cases))
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    case_ids_to_plot = te_ids[:n_cases]
    t_hat_arr = np.linspace(0.0, 1.0, N_TIMESTEPS)

    for row, _ in enumerate(case_ids_to_plot):
        start = row * N_TIMESTEPS
        end   = start + N_TIMESTEPS

        if end > len(X_te):
            break

        X_case = X_te[start:end]
        y_case = y_te[start:end]
        y_pr   = model(tf.constant(X_case, dtype=tf.float32),
                       training=False).numpy()

        ax_opr = axes[row, 0]
        ax_wc  = axes[row, 1]

        ax_opr.plot(t_hat_arr, y_case[:, 0] * OPR_SCALE,
                    'b-o', ms=3, label='CMG STARS')
        ax_opr.plot(t_hat_arr, y_pr[:, 0]  * OPR_SCALE,
                    'r--', lw=2, label='PINN')
        ax_opr.set_ylabel('OPR (STB/day)')
        ax_opr.set_title(f'Case {case_ids_to_plot[row]+1} — Oil Rate')
        ax_opr.legend(); ax_opr.grid(True, alpha=0.3)

        ax_wc.plot(t_hat_arr, y_case[:, 1], 'b-o', ms=3, label='CMG STARS')
        ax_wc.plot(t_hat_arr, y_pr[:, 1],   'r--', lw=2, label='PINN')
        ax_wc.set_ylabel('Water Cut')
        ax_wc.set_title(f'Case {case_ids_to_plot[row]+1} — Water Cut')
        ax_wc.legend(); ax_wc.grid(True, alpha=0.3)

    fig.suptitle('Time-Series: CMG STARS vs PINN (Test Cases)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'timeseries_test.png'), dpi=150)
    plt.close(fig)
    print("[PLOT] timeseries_test.png saved")


def plot_saturation_profiles(sw_model):
    """Sw(xD) at 5 time snapshots for Cp = 1000 ppm."""
    xD_arr = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    tD_snaps = [0.1, 0.3, 0.5, 0.7, 0.9]
    cp_norm_val = 0.5   # 1000 ppm / 2000 ppm

    fig, ax = plt.subplots(figsize=(8, 5))
    for tD_val in tD_snaps:
        inp = np.column_stack([
            xD_arr,
            np.full_like(xD_arr, tD_val),
            np.full_like(xD_arr, cp_norm_val),
            np.full_like(xD_arr, 0.7)   # representative inj_norm
        ]).astype(np.float32)
        sw_out = sw_model(tf.constant(inp), training=False).numpy().flatten()
        ax.plot(xD_arr, sw_out, label=f'tD={tD_val:.1f}')

    ax.axhline(SWC,        color='k', ls=':', lw=1, label=f'SWC={SWC}')
    ax.axhline(1.0 - SOR,  color='gray', ls=':', lw=1, label=f'1-SOR={1-SOR:.3f}')
    ax.set_xlabel('Dimensionless Distance xD')
    ax.set_ylabel('Water Saturation Sw')
    ax.set_title('Saturation Profiles — Cp = 1000 ppm')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'saturation_profiles.png'), dpi=150)
    plt.close(fig)
    print("[PLOT] saturation_profiles.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# Ablation Study
# ═════════════════════════════════════════════════════════════════════════════

class DataOnlyPINN(keras.Model):
    """Ablation: data-only MLP baseline (no physics)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bn0 = layers.BatchNormalization(name='abl_bn0')
        setattr(self, 'abl_d1', layers.Dense(128, activation='tanh', name='abl_d1'))
        setattr(self, 'abl_d2', layers.Dense(256, activation='tanh', name='abl_d2'))
        setattr(self, 'abl_d3', layers.Dense(256, activation='tanh', name='abl_d3'))
        setattr(self, 'abl_d4', layers.Dense(128, activation='tanh', name='abl_d4'))
        setattr(self, 'abl_d5', layers.Dense(64,  activation='tanh', name='abl_d5'))
        setattr(self, 'abl_out', layers.Dense(2, name='abl_out'))

    def call(self, inputs, training=False):
        x = self.bn0(inputs, training=training)
        x = self.abl_d1(x)
        x = self.abl_d2(x)
        x = self.abl_d3(x)
        x = self.abl_d4(x)
        x = self.abl_d5(x)
        return self.abl_out(x)


def train_data_only(X_tr, y_tr, X_va, y_va, epochs=200):
    """Quick data-only training for ablation."""
    model  = DataOnlyPINN()
    opt    = keras.optimizers.Adam(LR)
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X_tr, tf.float32), tf.constant(y_tr, tf.float32))
    ).shuffle(8000, seed=SEED).batch(BATCH_SIZE)

    best_val = np.inf
    best_w   = None

    for epoch in range(1, epochs + 1):
        for X_b, y_b in dataset:
            with tf.GradientTape() as tape:
                y_p = model(X_b, training=True)
                loss = tf.reduce_mean(tf.square(y_p - y_b))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

        y_va_p = model(tf.constant(X_va, tf.float32), training=False).numpy()
        val_mse = float(np.mean((y_va_p - y_va) ** 2))
        if val_mse < best_val:
            best_val = val_mse
            best_w   = model.get_weights()

    if best_w:
        model.set_weights(best_w)
    return model


def run_ablation(full_model, X_tr, y_tr, X_va, y_va, X_te, y_te):
    print("\n[ABLATION] Training data-only baseline …")
    data_model = train_data_only(X_tr, y_tr, X_va, y_va, epochs=200)

    results = {}
    for label, m in [('Full PINN', full_model), ('Data-Only', data_model)]:
        y_p  = m(tf.constant(X_te, tf.float32), training=False).numpy()
        opr2 = r_squared(y_te[:, 0], y_p[:, 0])
        wc2  = r_squared(y_te[:, 1], y_p[:, 1])
        results[label] = {'OPR_R2': opr2, 'WC_R2': wc2}
        print(f"  {label}: OPR R²={opr2:.4f}  WC R²={wc2:.4f}")

    # Bar chart
    labels  = list(results.keys())
    opr_r2s = [results[l]['OPR_R2'] for l in labels]
    wc_r2s  = [results[l]['WC_R2']  for l in labels]
    x       = np.arange(len(labels))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, opr_r2s, w, label='OPR R²', color='steelblue')
    ax.bar(x + w/2, wc_r2s,  w, label='WC R²',  color='darkorange')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('R²')
    ax.set_title('Ablation Study — PINN vs Baseline')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'ablation_comparison.png'), dpi=150)
    plt.close(fig)
    print("[PLOT] ablation_comparison.png saved")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Optimisation — maximise cumulative oil over Cp
# ═════════════════════════════════════════════════════════════════════════════

def _try_trapezoid(y, x=None):
    """Portable numerical integration."""
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        try:
            return np.trapz(y, x)
        except Exception:
            from scipy.integrate import trapezoid as sp_trap
            return sp_trap(y, x)


def predict_cumulative_oil(model, cp_ppm, inj_norm_series,
                           t_hat_arr=None, bhp=0.5):
    """
    Predict cumulative field oil for a given cp_ppm using the real inj schedule.
    """
    if t_hat_arr is None:
        t_hat_arr = np.linspace(0.0, 1.0, N_TIMESTEPS, dtype=np.float32)

    cp_n = np.full(N_TIMESTEPS, cp_ppm / CP_MAX_PPM, dtype=np.float32)
    bhp_ = np.full(N_TIMESTEPS, bhp, dtype=np.float32)

    X = np.column_stack([t_hat_arr, cp_n, inj_norm_series, bhp_]).astype(np.float32)
    y = model(tf.constant(X), training=False).numpy()
    opr = y[:, 0] * OPR_SCALE  # STB/day

    # Integrate over time (days)
    t_days = t_hat_arr * TOTAL_DAYS
    cum_oil = _try_trapezoid(opr, t_days)
    return float(cum_oil)


def optimise_polymer(model, inj_norm_series):
    """Four optimisation algorithms over Cp ∈ [0, 2000] ppm."""
    print("\n[OPT] Optimising polymer concentration …")

    t_hat_arr = np.linspace(0.0, 1.0, N_TIMESTEPS, dtype=np.float32)

    def objective_neg(cp_ppm_arr):
        cp = float(np.clip(cp_ppm_arr[0], 0.0, CP_MAX_PPM))
        return -predict_cumulative_oil(model, cp, inj_norm_series, t_hat_arr)

    results = {}

    # ── 1. Differential Evolution ─────────────────────────────────────────
    try:
        from scipy.optimize import differential_evolution
        de_res = differential_evolution(
            objective_neg, bounds=[(0.0, CP_MAX_PPM)],
            maxiter=50, tol=1e-4, seed=SEED, workers=1)
        results['DE'] = {'cp_ppm': float(de_res.x[0]),
                         'cum_oil': -float(de_res.fun)}
        print(f"  DE:  Cp={results['DE']['cp_ppm']:.1f} ppm  "
              f"CumOil={results['DE']['cum_oil']:.0f} STB")
    except Exception as e:
        print(f"  DE failed: {e}")

    # ── 2. Bayesian Optimisation (grid scan as fallback) ──────────────────
    try:
        from skopt import gp_minimize
        bo_res = gp_minimize(
            objective_neg, dimensions=[(0.0, CP_MAX_PPM)],
            n_calls=30, random_state=SEED)
        results['BO'] = {'cp_ppm': float(bo_res.x[0]),
                         'cum_oil': -float(bo_res.fun)}
        print(f"  BO:  Cp={results['BO']['cp_ppm']:.1f} ppm  "
              f"CumOil={results['BO']['cum_oil']:.0f} STB")
    except ImportError:
        # Grid scan fallback
        cp_grid  = np.linspace(0, CP_MAX_PPM, 41)
        oils_g   = [predict_cumulative_oil(model, c, inj_norm_series, t_hat_arr)
                    for c in cp_grid]
        best_idx = int(np.argmax(oils_g))
        results['BO_grid'] = {'cp_ppm': float(cp_grid[best_idx]),
                              'cum_oil': float(oils_g[best_idx])}
        print(f"  BO(grid): Cp={results['BO_grid']['cp_ppm']:.1f} ppm  "
              f"CumOil={results['BO_grid']['cum_oil']:.0f} STB")
    except Exception as e:
        print(f"  BO failed: {e}")

    # ── 3. PSO ────────────────────────────────────────────────────────────
    try:
        import pyswarms as ps
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        pso_opt = ps.single.GlobalBestPSO(
            n_particles=20, dimensions=1,
            options=options, bounds=([0.0], [CP_MAX_PPM]))
        cost, pos = pso_opt.optimize(
            lambda p: np.array([objective_neg([pi]) for pi in p]),
            iters=50, verbose=False)
        results['PSO'] = {'cp_ppm': float(pos[0]), 'cum_oil': -float(cost)}
        print(f"  PSO: Cp={results['PSO']['cp_ppm']:.1f} ppm  "
              f"CumOil={results['PSO']['cum_oil']:.0f} STB")
    except ImportError:
        print("  PSO skipped (pyswarms not installed)")
    except Exception as e:
        print(f"  PSO failed: {e}")

    # ── 4. Genetic Algorithm ──────────────────────────────────────────────
    try:
        from scipy.optimize import differential_evolution
        ga_res = differential_evolution(
            objective_neg, bounds=[(0.0, CP_MAX_PPM)],
            strategy='best1bin', maxiter=80, popsize=15,
            mutation=(0.5, 1.5), recombination=0.7, seed=SEED + 1)
        results['GA'] = {'cp_ppm': float(ga_res.x[0]),
                         'cum_oil': -float(ga_res.fun)}
        print(f"  GA:  Cp={results['GA']['cp_ppm']:.1f} ppm  "
              f"CumOil={results['GA']['cum_oil']:.0f} STB")
    except Exception as e:
        print(f"  GA failed: {e}")

    # ── Sensitivity curve ─────────────────────────────────────────────────
    cp_range = np.linspace(0, CP_MAX_PPM, 41)
    oils_sens = [predict_cumulative_oil(model, c, inj_norm_series, t_hat_arr)
                 for c in cp_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cp_range, oils_sens, 'b-o', ms=4)
    ax.set_xlabel('Polymer Concentration Cp (ppm)')
    ax.set_ylabel('Cumulative Oil (STB)')
    ax.set_title('Polymer Concentration Optimisation — Pelican Lake')
    ax.grid(True, alpha=0.3)

    for label, res in results.items():
        if 'cp_ppm' in res:
            ax.axvline(res['cp_ppm'], ls='--', alpha=0.7, label=label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'optimisation_sensitivity.png'), dpi=150)
    plt.close(fig)
    print("[PLOT] optimisation_sensitivity.png saved")

    with open(os.path.join(OUTPUT_DIR, 'optimisation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PINN — Pelican Lake Heavy Oil Polymer Flooding")
    print("=" * 60)
    print(f"TensorFlow {tf.__version__}  |  Keras {keras.__version__}")
    print(f"Output dir: {os.path.abspath(OUTPUT_DIR)}")

    # ── Load data ─────────────────────────────────────────────────────────
    X_tr, y_tr, X_va, y_va, X_te, y_te, te_ids, inj_norm = load_data(DATA_DIR)

    # ── Build models ──────────────────────────────────────────────────────
    sw_model = SaturationSubmodel(name='sw_submodel')
    model    = PolymerPINN(sw_submodel=sw_model, name='polymer_pinn')

    # Warm-up build: force weight creation
    dummy = tf.zeros((2, 4), dtype=tf.float32)
    _ = model(dummy, training=False)
    print(f"[MODEL] PolymerPINN built — "
          f"{model.count_params():,} parameters")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n[TRAIN] Starting …")
    hist = train(model, sw_model, X_tr, y_tr, X_va, y_va)

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred, metrics = evaluate(model, X_te, y_te)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[PLOTS] Generating …")
    plot_fractional_flow(model)
    plot_training_history(hist)
    plot_parity(y_te, y_pred)
    plot_timeseries(model, X_te, y_te, te_ids, n_cases=min(3, len(te_ids)))
    plot_saturation_profiles(sw_model)

    # ── Ablation ──────────────────────────────────────────────────────────
    ablation = run_ablation(model, X_tr, y_tr, X_va, y_va, X_te, y_te)

    # ── Optimisation ──────────────────────────────────────────────────────
    opt_results = optimise_polymer(model, inj_norm)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  OPR R²  = {metrics['opr_r2']:.4f}")
    print(f"  WC  R²  = {metrics['wc_r2']:.4f}")
    print(f"  Learned Corey params:")
    kr = model.kr_layer
    print(f"    krw_max = {float(kr.krw_max):.4f}  (init {KRW_MAX_INIT})")
    print(f"    kro_max = {float(kr.kro_max):.4f}  (init {KRO_MAX_INIT})")
    print(f"    nw      = {float(kr.nw):.4f}  (init {NW_INIT})")
    print(f"    no      = {float(kr.no):.4f}  (init {NO_INIT})")
    print("=" * 60)
    print("Done.")


if __name__ == '__main__':
    main()
