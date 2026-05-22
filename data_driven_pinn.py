"""
Data-Driven Neural Network for Heavy Oil Polymer Flooding
Pelican Lake Field — SPE-166256 / SPE-179648

Architecture:
  Model 1 — PointwiseResNet: Deep Residual MLP (point-wise, per-timestep)
  Model 2 — BiLSTMModel:     Bidirectional LSTM (sequence-level, per-case)
  Ensemble: learned weighted combination with MC-Dropout uncertainty
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

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get('DATA_DIR',   'data')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs_dd')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Physical constants (SPE-166256) ──────────────────────────────────────────
POROSITY     = 0.33
PERMEABILITY = 3000.0    # mD
NET_PAY_M    = 2.4       # m
WELL_LEN_M   = 1400.0    # m
SPACING_M    = 175.0     # m

MU_OIL       = 1650.0    # cP
MU_WATER     = 1.0       # cP
SWC          = 0.224
SOR          = 0.206
CP_MAX_PPM   = 2000.0
INJ_RATE_MAX = 750.0     # STB/day
OPR_SCALE    = 500.0     # STB/day
TOTAL_DAYS   = 1705.0
N_TIMESTEPS  = 57
N_CASES      = 200
N_TRAIN      = 140
N_VAL        = 30
N_TEST       = 30

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS       = 500
BATCH_SIZE   = 64
PATIENCE     = 50
LR           = 1e-3
LR_LSTM      = 5e-4
N_MC         = 50        # MC Dropout inference samples


# ═════════════════════════════════════════════════════════════════════════════
# Custom Keras Layers
# ═════════════════════════════════════════════════════════════════════════════

class ResidualBlock(layers.Layer):
    """
    Dense residual block: Dense(units,tanh) -> Dense(units,tanh) + skip + LN.
    Projects skip if input width ≠ units.
    """

    def __init__(self, units, dropout_rate=0.05, **kw):
        super().__init__(**kw)
        self.units        = units
        self.dropout_rate = dropout_rate
        self._proj        = None

    def build(self, input_shape):
        in_u = int(input_shape[-1])
        setattr(self, '_d1',   layers.Dense(self.units, activation='tanh',
                                            name=self.name + '_d1'))
        setattr(self, '_d2',   layers.Dense(self.units, activation='tanh',
                                            name=self.name + '_d2'))
        setattr(self, '_drop', layers.Dropout(self.dropout_rate,
                                              name=self.name + '_dr'))
        setattr(self, '_ln',   layers.LayerNormalization(
                                              name=self.name + '_ln'))
        if in_u != self.units:
            setattr(self, '_proj', layers.Dense(self.units, use_bias=False,
                                                name=self.name + '_proj'))
        super().build(input_shape)

    def call(self, x, training=False):
        skip = self._proj(x) if self._proj is not None else x
        h    = self._d1(x)
        h    = self._d2(h)
        h    = self._drop(h, training=training)
        return self._ln(h + skip)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units, 'dropout_rate': self.dropout_rate})
        return cfg


class FeatureAttentionLayer(layers.Layer):
    """Learned per-feature scalar attention (softmax-normalised)."""

    def build(self, input_shape):
        n = int(input_shape[-1])
        self.attn_w = self.add_weight(
            name='attn_w', shape=(n,),
            initializer='ones', trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x * tf.nn.softmax(self.attn_w)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


class SqueezeExciteBlock(layers.Layer):
    """Channel squeeze-and-excitation (SE) recalibration."""

    def __init__(self, units, ratio=4, **kw):
        super().__init__(**kw)
        self.units = units
        self.ratio = ratio

    def build(self, input_shape):
        mid = max(1, self.units // self.ratio)
        setattr(self, '_se_sq', layers.Dense(mid, activation='relu',
                                             name=self.name + '_sq'))
        setattr(self, '_se_ex', layers.Dense(self.units, activation='sigmoid',
                                             name=self.name + '_ex'))
        super().build(input_shape)

    def call(self, x):
        s = self._se_sq(x)
        s = self._se_ex(s)
        return x * s

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units, 'ratio': self.ratio})
        return cfg


# ═════════════════════════════════════════════════════════════════════════════
# Model 1 — PointwiseResNet
# ═════════════════════════════════════════════════════════════════════════════

class PointwiseResNet(keras.Model):
    """
    Deep Residual Network for point-wise (per-timestep) production prediction.
    Input : (t_hat, cp_norm, inj_norm, bhp_norm) — shape (N, 4)
    Output: (OPR_field_norm, WC_field)            — shape (N, 2)
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bn_in = layers.BatchNormalization(name='rn_bn_in')
        self.attn  = FeatureAttentionLayer(name='rn_attn')

        setattr(self, 'rn_stem', layers.Dense(128, activation='tanh',
                                              name='rn_stem'))
        setattr(self, 'rn_rb1',  ResidualBlock(128, name='rn_rb1'))
        setattr(self, 'rn_rb2',  ResidualBlock(256, name='rn_rb2'))
        setattr(self, 'rn_se2',  SqueezeExciteBlock(256, name='rn_se2'))
        setattr(self, 'rn_rb3',  ResidualBlock(256, name='rn_rb3'))
        setattr(self, 'rn_rb4',  ResidualBlock(128, name='rn_rb4'))
        setattr(self, 'rn_se4',  SqueezeExciteBlock(128, name='rn_se4'))
        setattr(self, 'rn_rb5',  ResidualBlock(64,  name='rn_rb5'))
        setattr(self, 'rn_h1',   layers.Dense(32, activation='tanh',
                                              name='rn_h1'))
        setattr(self, 'rn_h2',   layers.Dense(16, activation='tanh',
                                              name='rn_h2'))
        setattr(self, 'rn_qo',   layers.Dense(1, activation='softplus',
                                              name='rn_qo'))
        setattr(self, 'rn_wc',   layers.Dense(1, activation='sigmoid',
                                              name='rn_wc'))

    def call(self, inputs, training=False):
        x = self.bn_in(inputs, training=training)
        x = self.attn(x)
        x = self.rn_stem(x)
        x = self.rn_rb1(x, training=training)
        x = self.rn_rb2(x, training=training)
        x = self.rn_se2(x)
        x = self.rn_rb3(x, training=training)
        x = self.rn_rb4(x, training=training)
        x = self.rn_se4(x)
        x = self.rn_rb5(x, training=training)
        x = self.rn_h1(x)
        x = self.rn_h2(x)
        qo = self.rn_qo(x)
        wc = self.rn_wc(x)
        return tf.concat([qo, wc], axis=-1)


# ═════════════════════════════════════════════════════════════════════════════
# Model 2 — BiLSTMModel
# ═════════════════════════════════════════════════════════════════════════════

class BiLSTMModel(keras.Model):
    """
    Bidirectional LSTM for sequence-level prediction.
    Input : (n_cases, N_TIMESTEPS, 4)
    Output: (n_cases, N_TIMESTEPS, 2)
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bn_in = layers.BatchNormalization(name='bi_bn')

        setattr(self, 'bi_lstm1', layers.Bidirectional(
            layers.LSTM(128, return_sequences=True,
                        dropout=0.05, recurrent_dropout=0.0),
            name='bi_lstm1'))
        setattr(self, 'bi_lstm2', layers.Bidirectional(
            layers.LSTM(64, return_sequences=True,
                        dropout=0.05, recurrent_dropout=0.0),
            name='bi_lstm2'))
        setattr(self, 'bi_lstm3', layers.Bidirectional(
            layers.LSTM(32, return_sequences=True,
                        dropout=0.05, recurrent_dropout=0.0),
            name='bi_lstm3'))
        setattr(self, 'bi_td1', layers.TimeDistributed(
            layers.Dense(32, activation='tanh'), name='bi_td1'))
        setattr(self, 'bi_td2', layers.TimeDistributed(
            layers.Dense(16, activation='tanh'), name='bi_td2'))
        setattr(self, 'bi_qo',  layers.TimeDistributed(
            layers.Dense(1, activation='softplus'), name='bi_qo'))
        setattr(self, 'bi_wc',  layers.TimeDistributed(
            layers.Dense(1, activation='sigmoid'),  name='bi_wc'))

    def call(self, inputs, training=False):
        # inputs: (batch, 57, 4)
        x = self.bn_in(inputs, training=training)
        x = self.bi_lstm1(x, training=training)
        x = self.bi_lstm2(x, training=training)
        x = self.bi_lstm3(x, training=training)
        x = self.bi_td1(x)
        x = self.bi_td2(x)
        qo = self.bi_qo(x)   # (batch, 57, 1)
        wc = self.bi_wc(x)   # (batch, 57, 1)
        return tf.concat([qo, wc], axis=-1)   # (batch, 57, 2)


# ═════════════════════════════════════════════════════════════════════════════
# Ensemble Model — learned weighting
# ═════════════════════════════════════════════════════════════════════════════

class EnsembleWeightLayer(layers.Layer):
    """Learnable scalar weight α ∈ (0,1) combining two prediction streams."""

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha', shape=(), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        pred_a, pred_b = inputs
        a = tf.sigmoid(self.alpha)
        return a * pred_a + (1.0 - a) * pred_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return super().get_config()


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading (self-contained — mirrors pinn_polymer_flooding.py)
# ═════════════════════════════════════════════════════════════════════════════

def load_data(data_dir=DATA_DIR):
    oil_path  = os.path.join(data_dir, 'Oil_Production.csv')
    wc_path   = os.path.join(data_dir, 'Water_cut.csv')
    cp_path   = os.path.join(data_dir, 'Polymer_concentration.csv')
    inj1_path = os.path.join(data_dir, 'Injection_rate_inj1.csv')
    inj2_path = os.path.join(data_dir, 'Injection_rate_inj2.csv')

    missing = [p for p in [oil_path, wc_path, cp_path, inj1_path, inj2_path]
               if not os.path.exists(p)]
    if missing:
        print(f"[WARN] Missing: {missing}. Generating synthetic data.")
        return _generate_synthetic()

    oil_df  = pd.read_csv(oil_path)
    wc_df   = pd.read_csv(wc_path)
    cp_df   = pd.read_csv(cp_path)
    inj1_df = pd.read_csv(inj1_path)
    inj2_df = pd.read_csv(inj2_path)

    if 'time' in oil_df.columns:
        t_v   = oil_df['time'].values.astype(np.float32)
        t_hat = (t_v - t_v[0]) / (t_v[-1] - t_v[0] + 1e-8)
    else:
        t_hat = np.linspace(0.0, 1.0, N_TIMESTEPS, dtype=np.float32)

    inj1_cols = [c for c in inj1_df.columns if c != 'time']
    inj2_cols = [c for c in inj2_df.columns if c != 'time']
    inj_norm = ((inj1_df[inj1_cols[0]].values +
                 inj2_df[inj2_cols[0]].values) / INJ_RATE_MAX).astype(np.float32)

    def cols_A(df, ci):
        tag = f'case_{ci:03d}'
        return [c for c in df.columns if c.startswith(tag + '_P')]

    def cols_B(df, ci):
        tag = f'case_{ci:03d}'
        return [c for c in df.columns if c == tag]

    X_list, y_list = [], []
    for ci in range(1, N_CASES + 1):
        ocols = cols_A(oil_df, ci) or cols_B(oil_df, ci)
        wcols = cols_A(wc_df,  ci) or cols_B(wc_df,  ci)
        ccols = cols_B(cp_df,  ci)

        opr = oil_df[ocols].values.astype(np.float32).sum(axis=1)
        wc  = wc_df[wcols].values.astype(np.float32).mean(axis=1)
        cp  = float(cp_df[ccols].values.astype(np.float32)[0, 0])

        X_list.append(np.column_stack([
            t_hat,
            np.full(N_TIMESTEPS, cp,  dtype=np.float32),
            inj_norm,
            np.full(N_TIMESTEPS, 0.5, dtype=np.float32),
        ]))
        y_list.append(np.column_stack([opr / OPR_SCALE,
                                       np.clip(wc, 0.0, 1.0)]))

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.vstack(y_list).astype(np.float32)
    return _split(X_all, y_all, inj_norm, t_hat)


def _split(X_all, y_all, inj_norm, t_hat):
    rng   = np.random.default_rng(SEED)
    perm  = rng.permutation(N_CASES)
    tr_ids = sorted(perm[:N_TRAIN])
    va_ids = sorted(perm[N_TRAIN:N_TRAIN + N_VAL])
    te_ids = sorted(perm[N_TRAIN + N_VAL:])

    def idx(ids):
        return np.concatenate([np.arange(c * N_TIMESTEPS,
                                         c * N_TIMESTEPS + N_TIMESTEPS)
                                for c in ids])

    X_tr, y_tr = X_all[idx(tr_ids)], y_all[idx(tr_ids)]
    X_va, y_va = X_all[idx(va_ids)], y_all[idx(va_ids)]
    X_te, y_te = X_all[idx(te_ids)], y_all[idx(te_ids)]

    assert X_tr.shape[1] == 4, "Expected 4 inputs"
    assert y_tr.shape[1] == 2, "Expected 2 outputs"
    print(f"[DATA] Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, te_ids, inj_norm, t_hat


def _generate_synthetic():
    rng = np.random.default_rng(SEED)
    t_hat    = np.linspace(0.0, 1.0, N_TIMESTEPS, dtype=np.float32)
    inj_prof = (0.5 + 0.4 * np.sin(np.pi * t_hat)).astype(np.float32)

    X_list, y_list = [], []
    for ci in range(N_CASES):
        cp = ci / (N_CASES - 1)
        X_list.append(np.column_stack([
            t_hat,
            np.full(N_TIMESTEPS, cp,  dtype=np.float32),
            inj_prof,
            np.full(N_TIMESTEPS, 0.5, dtype=np.float32),
        ]))
        opr = np.clip(0.030 * (1 + cp) * inj_prof +
                      rng.normal(0, 0.001, N_TIMESTEPS).astype(np.float32), 0, None)
        wc  = np.clip(0.92 - 0.06 * cp - 0.02 * t_hat +
                      rng.normal(0, 0.003, N_TIMESTEPS).astype(np.float32), 0, 1)
        y_list.append(np.column_stack([opr, wc]))

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.vstack(y_list).astype(np.float32)
    return _split(X_all, y_all, inj_prof, t_hat)


def to_sequences(X_flat, y_flat=None, n_cases=None, t_steps=N_TIMESTEPS):
    """Reshape flat (N*T, F) -> (N, T, F) for LSTM."""
    if n_cases is None:
        n_cases = X_flat.shape[0] // t_steps
    X_seq = X_flat[:n_cases * t_steps].reshape(n_cases, t_steps, -1)
    if y_flat is not None:
        y_seq = y_flat[:n_cases * t_steps].reshape(n_cases, t_steps, -1)
        return X_seq, y_seq
    return X_seq


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10
    return float(1.0 - ss_res / ss_tot)


def compute_penalty(y_pred):
    qo  = y_pred[:, :1]
    wc  = y_pred[:, 1:]
    return (tf.reduce_mean(tf.square(tf.nn.relu(-qo))) +
            tf.reduce_mean(tf.square(tf.nn.relu(-wc))) +
            tf.reduce_mean(tf.square(tf.nn.relu(wc - 1.0))))


def train_resnet(model, X_tr, y_tr, X_va, y_va, tag='resnet'):
    """Train PointwiseResNet with Adam + early stopping."""
    opt     = keras.optimizers.Adam(LR)
    dataset = (tf.data.Dataset
               .from_tensor_slices((tf.constant(X_tr, tf.float32),
                                    tf.constant(y_tr, tf.float32)))
               .shuffle(10000, seed=SEED)
               .batch(BATCH_SIZE))

    X_va_tf = tf.constant(X_va, tf.float32)
    best_val = np.inf
    wait     = 0
    hist     = {'loss': [], 'val_loss': []}
    best_w   = None

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0
        for X_b, y_b in dataset:
            with tf.GradientTape() as tape:
                y_p  = model(X_b, training=True)
                loss = (tf.reduce_mean(tf.square(y_p - y_b)) +
                        0.1 * compute_penalty(y_p))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += float(loss)
            n_batches  += 1

        epoch_loss /= max(n_batches, 1)
        y_va_p  = model(X_va_tf, training=False).numpy()
        val_mse = float(np.mean((y_va_p - y_va) ** 2))

        hist['loss'].append(epoch_loss)
        hist['val_loss'].append(val_mse)

        if val_mse < best_val - 1e-6:
            best_val = val_mse
            wait     = 0
            best_w   = model.get_weights()
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[{tag}] Early stop at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{tag}] Epoch {epoch:4d} | loss={epoch_loss:.4f} "
                  f"val={val_mse:.4f}")

    if best_w is not None:
        model.set_weights(best_w)
    return hist


def train_lstm(model, X_tr, y_tr, X_va, y_va, n_tr, n_va):
    """Train BiLSTMModel on reshaped sequence data."""
    X_tr_s, y_tr_s = to_sequences(X_tr, y_tr, n_cases=n_tr)
    X_va_s, y_va_s = to_sequences(X_va, y_va, n_cases=n_va)

    opt     = keras.optimizers.Adam(LR_LSTM)
    dataset = (tf.data.Dataset
               .from_tensor_slices((tf.constant(X_tr_s, tf.float32),
                                    tf.constant(y_tr_s, tf.float32)))
               .shuffle(500, seed=SEED)
               .batch(8))

    X_va_tf = tf.constant(X_va_s, tf.float32)
    best_val = np.inf
    wait     = 0
    hist     = {'loss': [], 'val_loss': []}
    best_w   = None

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches  = 0
        for X_b, y_b in dataset:
            with tf.GradientTape() as tape:
                y_p  = model(X_b, training=True)
                loss = tf.reduce_mean(tf.square(y_p - y_b))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += float(loss)
            n_batches  += 1

        epoch_loss /= max(n_batches, 1)
        y_va_p  = model(X_va_tf, training=False).numpy()
        val_mse = float(np.mean((y_va_p - y_va_s) ** 2))

        hist['loss'].append(epoch_loss)
        hist['val_loss'].append(val_mse)

        if val_mse < best_val - 1e-6:
            best_val = val_mse
            wait     = 0
            best_w   = model.get_weights()
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[LSTM] Early stop at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"[LSTM] Epoch {epoch:4d} | loss={epoch_loss:.4f} "
                  f"val={val_mse:.4f}")

    if best_w is not None:
        model.set_weights(best_w)
    return hist


# ═════════════════════════════════════════════════════════════════════════════
# Inference Helpers
# ═════════════════════════════════════════════════════════════════════════════

def predict_resnet(model, X_flat, training=False):
    """Flat (N, 4) -> (N, 2)."""
    return model(tf.constant(X_flat, tf.float32), training=training).numpy()


def predict_lstm(model, X_flat, n_cases):
    """Flat (N*57, 4) -> (N, 2) via sequence reshape."""
    X_seq = to_sequences(X_flat, n_cases=n_cases)
    y_seq = model(tf.constant(X_seq, tf.float32), training=False).numpy()
    return y_seq.reshape(-1, 2)


def ensemble_predict(resnet, lstm, X_flat, n_cases, alpha=0.6):
    """Weighted ensemble: alpha * resnet + (1-alpha) * lstm."""
    p_rn = predict_resnet(resnet, X_flat)
    p_ls = predict_lstm(lstm, X_flat, n_cases)
    return alpha * p_rn + (1.0 - alpha) * p_ls


def mc_predict(model, X_flat, n_samples=N_MC):
    """Monte Carlo Dropout uncertainty: mean and std over N_MC passes."""
    preds = np.stack([
        model(tf.constant(X_flat, tf.float32), training=True).numpy()
        for _ in range(n_samples)
    ], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def find_optimal_alpha(resnet, lstm, X_va, y_va, n_va):
    """Grid-search ensemble weight α on validation set."""
    best_alpha, best_mse = 0.5, np.inf
    for a in np.linspace(0.0, 1.0, 21):
        p = ensemble_predict(resnet, lstm, X_va, n_va, alpha=a)
        mse = float(np.mean((p - y_va) ** 2))
        if mse < best_mse:
            best_mse   = mse
            best_alpha = a
    print(f"[ENSEMBLE] Optimal α={best_alpha:.2f}  val_MSE={best_mse:.5f}")
    return best_alpha


# ═════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_history(hist_rn, hist_ls):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, hist, label in zip(axes,
                                [hist_rn, hist_ls],
                                ['PointwiseResNet', 'BiLSTM']):
        ax.semilogy(hist['loss'],     label='Train')
        ax.semilogy(hist['val_loss'], label='Val', ls='--')
        ax.set_title(f'{label} — Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('Data-Driven Model Training History')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dd_training_history.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_parity_with_uncertainty(y_te, y_pred_mean, y_pred_std):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    opr_true = y_te[:, 0] * OPR_SCALE
    opr_mean = y_pred_mean[:, 0] * OPR_SCALE
    opr_std  = y_pred_std[:, 0]  * OPR_SCALE

    ax1.errorbar(opr_true, opr_mean, yerr=2 * opr_std,
                 fmt='o', alpha=0.3, ms=3, color='steelblue',
                 ecolor='lightblue', elinewidth=0.5, capsize=0)
    mn, mx = opr_true.min(), opr_true.max()
    ax1.plot([mn, mx], [mn, mx], 'r--', lw=1)
    r2 = r_squared(y_te[:, 0], opr_mean / OPR_SCALE)
    ax1.set_xlabel('CMG STARS OPR (STB/day)')
    ax1.set_ylabel('DD-NN OPR ± 2σ (STB/day)')
    ax1.set_title(f'Field OPR — R²={r2:.3f}')
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(y_te[:, 1], y_pred_mean[:, 1], yerr=2 * y_pred_std[:, 1],
                 fmt='o', alpha=0.3, ms=3, color='darkorange',
                 ecolor='moccasin', elinewidth=0.5, capsize=0)
    mn, mx = y_te[:, 1].min(), y_te[:, 1].max()
    ax2.plot([mn, mx], [mn, mx], 'r--', lw=1)
    r2w = r_squared(y_te[:, 1], y_pred_mean[:, 1])
    ax2.set_xlabel('CMG STARS WC')
    ax2.set_ylabel('DD-NN WC ± 2σ')
    ax2.set_title(f'Field Water Cut — R²={r2w:.3f}')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Data-Driven Ensemble — Test Set Parity (with MC-Dropout ±2σ)')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dd_parity_test.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")
    return r_squared(y_te[:, 0], opr_mean / OPR_SCALE), r2w


def plot_timeseries_with_bands(resnet, X_te, y_te, te_ids,
                               n_cases=3, n_mc=N_MC):
    """Time-series with MC-Dropout uncertainty bands."""
    fig, axes = plt.subplots(n_cases, 2, figsize=(14, 4 * n_cases))
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    t_hat_arr = np.linspace(0.0, 1.0, N_TIMESTEPS)

    for row in range(min(n_cases, len(te_ids))):
        s   = row * N_TIMESTEPS
        e   = s   + N_TIMESTEPS
        if e > len(X_te):
            break
        X_c = X_te[s:e]
        y_c = y_te[s:e]

        mean, std = mc_predict(resnet, X_c, n_samples=n_mc)

        for col, (target_idx, label_true, label_pred, scale) in enumerate([
            (0, 'CMG STARS OPR', 'DD-NN OPR', OPR_SCALE),
            (1, 'CMG STARS WC',  'DD-NN WC',   1.0),
        ]):
            ax  = axes[row, col]
            yt  = y_c[:, target_idx] * scale
            yp  = mean[:, target_idx] * scale
            ys  = std[:, target_idx]  * scale

            ax.plot(t_hat_arr, yt, 'b-o', ms=3, lw=1.5, label=label_true)
            ax.plot(t_hat_arr, yp, 'r--', lw=2,          label=label_pred)
            ax.fill_between(t_hat_arr, yp - 2*ys, yp + 2*ys,
                            alpha=0.2, color='red', label='±2σ')
            ax.set_ylabel('OPR (STB/day)' if col == 0 else 'Water Cut')
            ax.set_title(f'Case {te_ids[row]+1}')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle('Data-Driven NN — Time-Series with MC-Dropout Uncertainty')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dd_timeseries_test.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_feature_importance(model: PointwiseResNet):
    """Visualise learned FeatureAttentionLayer weights."""
    attn_w   = model.attn.attn_w.numpy()
    norm_w   = np.exp(attn_w) / np.exp(attn_w).sum()
    feat_names = ['t_hat', 'cp_norm', 'inj_norm', 'bhp_norm']

    fig, ax = plt.subplots(figsize=(6, 4))
    colors  = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']
    bars    = ax.bar(feat_names, norm_w, color=colors)
    for bar, val in zip(bars, norm_w):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=10)
    ax.set_ylabel('Attention Weight (softmax)')
    ax.set_title('Learned Feature Importance — FeatureAttentionLayer')
    ax.set_ylim(0, norm_w.max() * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dd_feature_importance.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_cp_sensitivity(resnet, inj_norm, t_hat_arr):
    """OPR and WC response to Cp across the time horizon."""
    cp_levels = [0, 500, 1000, 1500, 2000]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for cp_ppm in cp_levels:
        cp_n  = np.full(N_TIMESTEPS, cp_ppm / CP_MAX_PPM, dtype=np.float32)
        bhp_  = np.full(N_TIMESTEPS, 0.5, dtype=np.float32)
        X_    = np.column_stack([t_hat_arr, cp_n, inj_norm, bhp_])
        y_    = resnet(tf.constant(X_, tf.float32), training=False).numpy()
        ax1.plot(t_hat_arr, y_[:, 0] * OPR_SCALE, label=f'{cp_ppm} ppm')
        ax2.plot(t_hat_arr, y_[:, 1],             label=f'{cp_ppm} ppm')

    ax1.set_xlabel('Normalised Time')
    ax1.set_ylabel('OPR (STB/day)')
    ax1.set_title('Field OPR vs Cp')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Normalised Time')
    ax2.set_ylabel('Water Cut')
    ax2.set_title('Field WC vs Cp')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle('Data-Driven Sensitivity to Polymer Concentration')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dd_cp_sensitivity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_model_comparison(metrics_rn, metrics_ls, metrics_ens):
    """Bar chart comparing ResNet, LSTM, and Ensemble."""
    labels  = ['ResNet', 'BiLSTM', 'Ensemble']
    opr_r2s = [metrics_rn['opr_r2'], metrics_ls['opr_r2'], metrics_ens['opr_r2']]
    wc_r2s  = [metrics_rn['wc_r2'],  metrics_ls['wc_r2'],  metrics_ens['wc_r2']]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, opr_r2s, w, label='OPR R²', color='steelblue')
    ax.bar(x + w/2, wc_r2s,  w, label='WC R²',  color='darkorange')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel('R²')
    ax.set_title('Data-Driven Model Comparison (Test Set)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for i, (o, wc) in enumerate(zip(opr_r2s, wc_r2s)):
        ax.text(i - w/2, o + 0.01, f'{o:.3f}', ha='center', fontsize=9)
        ax.text(i + w/2, wc + 0.01, f'{wc:.3f}', ha='center', fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dd_model_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Optimisation
# ═════════════════════════════════════════════════════════════════════════════

def _try_trapezoid(y, x=None):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        try:
            return np.trapz(y, x)
        except Exception:
            from scipy.integrate import trapezoid as sp_trap
            return sp_trap(y, x)


def predict_cumulative_oil(model, cp_ppm, inj_norm, t_hat_arr, bhp=0.5):
    cp_n  = np.full(N_TIMESTEPS, cp_ppm / CP_MAX_PPM, dtype=np.float32)
    bhp_  = np.full(N_TIMESTEPS, bhp, dtype=np.float32)
    X     = np.column_stack([t_hat_arr, cp_n, inj_norm, bhp_]).astype(np.float32)
    y     = model(tf.constant(X), training=False).numpy()
    opr   = y[:, 0] * OPR_SCALE
    t_days = t_hat_arr * TOTAL_DAYS
    return float(_try_trapezoid(opr, t_days))


def optimise_polymer(model, inj_norm, t_hat_arr):
    print("\n[OPT] Optimising polymer concentration (data-driven model) …")

    def obj_neg(cp_arr):
        cp = float(np.clip(cp_arr[0], 0.0, CP_MAX_PPM))
        return -predict_cumulative_oil(model, cp, inj_norm, t_hat_arr)

    results = {}

    # 1. Differential Evolution
    try:
        from scipy.optimize import differential_evolution
        res = differential_evolution(obj_neg, bounds=[(0.0, CP_MAX_PPM)],
                                     maxiter=60, tol=1e-4, seed=SEED)
        results['DE'] = {'cp_ppm': float(res.x[0]), 'cum_oil': -float(res.fun)}
        print(f"  DE:  Cp={results['DE']['cp_ppm']:.1f} ppm  "
              f"CumOil={results['DE']['cum_oil']:.0f} STB")
    except Exception as e:
        print(f"  DE failed: {e}")

    # 2. Bayesian Optimisation / grid fallback
    try:
        from skopt import gp_minimize
        res = gp_minimize(obj_neg, dimensions=[(0.0, CP_MAX_PPM)],
                          n_calls=30, random_state=SEED)
        results['BO'] = {'cp_ppm': float(res.x[0]), 'cum_oil': -float(res.fun)}
        print(f"  BO:  Cp={results['BO']['cp_ppm']:.1f} ppm  "
              f"CumOil={results['BO']['cum_oil']:.0f} STB")
    except ImportError:
        cp_grid = np.linspace(0, CP_MAX_PPM, 41)
        oils    = [predict_cumulative_oil(model, c, inj_norm, t_hat_arr)
                   for c in cp_grid]
        bi      = int(np.argmax(oils))
        results['BO_grid'] = {'cp_ppm': float(cp_grid[bi]),
                              'cum_oil': float(oils[bi])}
        print(f"  BO(grid): Cp={results['BO_grid']['cp_ppm']:.1f} ppm")
    except Exception as e:
        print(f"  BO failed: {e}")

    # 3. PSO
    try:
        import pyswarms as ps
        opt = ps.single.GlobalBestPSO(n_particles=20, dimensions=1,
                                       options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
                                       bounds=([0.0], [CP_MAX_PPM]))
        cost, pos = opt.optimize(
            lambda p: np.array([obj_neg([pi]) for pi in p]),
            iters=60, verbose=False)
        results['PSO'] = {'cp_ppm': float(pos[0]), 'cum_oil': -float(cost)}
        print(f"  PSO: Cp={results['PSO']['cp_ppm']:.1f} ppm")
    except ImportError:
        print("  PSO skipped (pyswarms not installed)")
    except Exception as e:
        print(f"  PSO failed: {e}")

    # 4. GA (via DE with different strategy)
    try:
        from scipy.optimize import differential_evolution
        res = differential_evolution(obj_neg, bounds=[(0.0, CP_MAX_PPM)],
                                     strategy='best1bin', maxiter=80,
                                     popsize=15, mutation=(0.5, 1.5),
                                     recombination=0.7, seed=SEED + 1)
        results['GA'] = {'cp_ppm': float(res.x[0]), 'cum_oil': -float(res.fun)}
        print(f"  GA:  Cp={results['GA']['cp_ppm']:.1f} ppm  "
              f"CumOil={results['GA']['cum_oil']:.0f} STB")
    except Exception as e:
        print(f"  GA failed: {e}")

    # Sensitivity curve
    cp_range = np.linspace(0, CP_MAX_PPM, 41)
    oils_s   = [predict_cumulative_oil(model, c, inj_norm, t_hat_arr)
                for c in cp_range]
    fig, ax  = plt.subplots(figsize=(8, 5))
    ax.plot(cp_range, oils_s, 'b-o', ms=4)
    ax.set_xlabel('Polymer Concentration (ppm)')
    ax.set_ylabel('Cumulative Oil (STB)')
    ax.set_title('DD-NN Polymer Optimisation — Pelican Lake')
    ax.grid(True, alpha=0.3)
    for lbl, r in results.items():
        if 'cp_ppm' in r:
            ax.axvline(r['cp_ppm'], ls='--', alpha=0.7, label=lbl)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dd_optimisation_sensitivity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")

    with open(os.path.join(OUTPUT_DIR, 'dd_optimisation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Data-Driven NN — Pelican Lake Heavy Oil Polymer Flooding")
    print("=" * 60)
    print(f"TensorFlow {tf.__version__}  |  Keras {keras.__version__}")

    # ── Data ──────────────────────────────────────────────────────────────
    (X_tr, y_tr, X_va, y_va, X_te, y_te,
     te_ids, inj_norm, t_hat_arr) = load_data(DATA_DIR)

    # ── Build & train ResNet ──────────────────────────────────────────────
    print("\n[RESNET] Building …")
    resnet = PointwiseResNet(name='resnet')
    _ = resnet(tf.zeros((2, 4), tf.float32))
    print(f"  Parameters: {resnet.count_params():,}")

    print("[RESNET] Training …")
    hist_rn = train_resnet(resnet, X_tr, y_tr, X_va, y_va, tag='ResNet')

    # ── Build & train BiLSTM ──────────────────────────────────────────────
    print("\n[LSTM] Building …")
    bilstm = BiLSTMModel(name='bilstm')
    _ = bilstm(tf.zeros((2, N_TIMESTEPS, 4), tf.float32))
    print(f"  Parameters: {bilstm.count_params():,}")

    print("[LSTM] Training …")
    hist_ls = train_lstm(bilstm, X_tr, y_tr, X_va, y_va,
                         n_tr=N_TRAIN, n_va=N_VAL)

    # ── Save weights ──────────────────────────────────────────────────────
    resnet.save_weights(os.path.join(OUTPUT_DIR, 'dd_resnet.weights.h5'))
    bilstm.save_weights(os.path.join(OUTPUT_DIR, 'dd_bilstm.weights.h5'))

    # ── Ensemble weight optimisation ──────────────────────────────────────
    alpha = find_optimal_alpha(resnet, bilstm, X_va, y_va, n_va=N_VAL)

    # ── Evaluate individual models ────────────────────────────────────────
    print("\n[EVAL] Individual models …")
    p_rn_te = predict_resnet(resnet, X_te)
    p_ls_te = predict_lstm(bilstm, X_te, n_cases=N_TEST)
    p_en_te = alpha * p_rn_te + (1.0 - alpha) * p_ls_te

    metrics_rn  = {'opr_r2': r_squared(y_te[:, 0], p_rn_te[:, 0]),
                   'wc_r2':  r_squared(y_te[:, 1], p_rn_te[:, 1])}
    metrics_ls  = {'opr_r2': r_squared(y_te[:, 0], p_ls_te[:, 0]),
                   'wc_r2':  r_squared(y_te[:, 1], p_ls_te[:, 1])}
    metrics_ens = {'opr_r2': r_squared(y_te[:, 0], p_en_te[:, 0]),
                   'wc_r2':  r_squared(y_te[:, 1], p_en_te[:, 1])}

    print(f"  ResNet  — OPR R²={metrics_rn['opr_r2']:.4f}  "
          f"WC R²={metrics_rn['wc_r2']:.4f}")
    print(f"  BiLSTM  — OPR R²={metrics_ls['opr_r2']:.4f}  "
          f"WC R²={metrics_ls['wc_r2']:.4f}")
    print(f"  Ensemble— OPR R²={metrics_ens['opr_r2']:.4f}  "
          f"WC R²={metrics_ens['wc_r2']:.4f}")

    all_metrics = {
        'resnet':   metrics_rn,
        'bilstm':   metrics_ls,
        'ensemble': metrics_ens,
        'alpha':    float(alpha),
    }
    with open(os.path.join(OUTPUT_DIR, 'dd_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # ── MC Dropout uncertainty on test set ────────────────────────────────
    print("[EVAL] MC Dropout uncertainty …")
    mean_mc, std_mc = mc_predict(resnet, X_te, n_samples=N_MC)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[PLOTS] Generating …")
    plot_training_history(hist_rn, hist_ls)
    plot_parity_with_uncertainty(y_te, mean_mc, std_mc)
    plot_timeseries_with_bands(resnet, X_te, y_te, te_ids,
                               n_cases=min(3, len(te_ids)))
    plot_feature_importance(resnet)
    plot_cp_sensitivity(resnet, inj_norm, t_hat_arr)
    plot_model_comparison(metrics_rn, metrics_ls, metrics_ens)

    # ── Optimisation ──────────────────────────────────────────────────────
    opt_results = optimise_polymer(resnet, inj_norm, t_hat_arr)

    # ── Save history ──────────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, 'dd_history.json'), 'w') as f:
        json.dump({'resnet': {k: [float(v) for v in vs]
                              for k, vs in hist_rn.items()},
                   'bilstm': {k: [float(v) for v in vs]
                              for k, vs in hist_ls.items()}}, f)

    print("\n" + "=" * 60)
    print("SUMMARY — Data-Driven Ensemble")
    print(f"  Ensemble OPR R² = {metrics_ens['opr_r2']:.4f}")
    print(f"  Ensemble WC  R² = {metrics_ens['wc_r2']:.4f}")
    print(f"  Optimal α       = {alpha:.2f}")
    print("=" * 60)
    print("Done.")


if __name__ == '__main__':
    main()
