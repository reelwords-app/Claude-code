"""
Physics-Informed Neural Network for Pelican Lake Heavy-Oil Polymer Flooding
Continuation of: "Physics-Informed Machine Learning for Heavy Oil Polymer Flooding
                  with Mobile Water Fraction"

All parameters from the manuscript. No CSV files.
CMG STARS reference curves reconstructed from manuscript Tables 6 & 7.
Dual-tape training: physics (BL PDE) + data (CMG reference per well).
Three-producer architecture: P1, P2, P3.
Trainable Corey relative permeability parameters.
"""

import os
import json
import warnings
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model, optimizers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "pinn_polymer_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 1. MANUSCRIPT PARAMETERS  (Tables 2, 4; Sections 2.3–2.5)
# ===========================================================================

class Params:
    # Reservoir — Table 2
    depth_ft      = 1475.0
    T_F           = 63.0
    P_init_psi    = 380.0
    mu_oil_cp     = 1650.0
    Bo            = 5.65
    GOR           = 28.07          # SCF/STB
    Pb_psi        = 304.58
    rock_comp     = 2.3e-4         # psi⁻¹

    # Corey relative permeability — Section 2.3
    Swr     = 0.23
    Sro     = 0.20
    Krw_max = 0.10                 # trainable initial value
    Kro_max = 1.00                 # trainable initial value
    nw      = 3.0                  # trainable initial value
    no      = 2.2                  # trainable initial value

    # Geological layers — Table 4 (Good Pay layer dominates)
    phi  = 0.312
    K_md = 3000.0

    # Mobile Water Fraction — Section 2.5
    FWM       = 0.12
    Swi_core  = 0.30
    Swinitial = 0.36               # = 0.30 + 0.12*(1-0.30-0.20)
    Sor       = 0.20

    # Polymer — Section 2.4
    Cp_base_ppm = 1000.0
    mu_poly_cp  = 25.0             # at 1000 ppm
    salinity    = 8222.0           # ppm
    adsorption  = 10.0             # µg/g
    RRF         = 2.0
    IPV         = 0.10

    # Well geometry
    n_producers     = 3
    n_injectors     = 2
    well_length_ft  = 4593.176
    well_spacing_ft = 574.147
    BHP_prod_psi    = 120.0
    BHP_inj_psi     = 550.0

    # Simulation period — 57 monthly steps (May 2005 – Dec 2009)
    t_end      = 56.0              # months
    n_timesteps= 57

    # Injection polymer schedule (ppm per phase)
    inj_schedule = [
        ( 0,  5,  600.0),
        ( 5, 12,  500.0),
        (12, 30,  800.0),
        (30, 57, 1000.0),
    ]

    # CMG STARS targets — Table 6 (with FWM model)
    cmg_r2    = {"P1": 0.9987, "P2": 0.9960, "P3": 0.9906}
    cmg_nrmse = {"P1": 0.0119, "P2": 0.0216, "P3": 0.0317}
    cmg_cum   = {"P1": 0.0014, "P2": 0.0292, "P3": 0.0366}

    # CMG water-cut endpoints — Table 7
    wc_initial = {"P1": 0.168, "P2": 0.168, "P3": 0.168}
    wc_final   = {"P1": 0.606, "P2": 0.598, "P3": 0.605}

    # Training
    epochs_physics    = 500
    epochs_production = 800
    lr_physics  = 5e-4
    lr_prod     = 1e-3
    n_colloc    = 2000
    batch_size  = 57               # one full well time-series per batch step
    patience    = 80

    # Loss weights
    w_bl   = 1.0
    w_pt   = 0.5
    w_ic   = 10.0
    w_bc   = 10.0
    w_data = 1.0
    w_pen  = 0.5

    # Optimisation
    Cp_min_ppm = 500.0
    Cp_max_ppm = 2000.0

P = Params()


# ===========================================================================
# 2. CMG REFERENCE DATA  (reconstructed from manuscript Tables 6 & 7)
# ===========================================================================

def _cp_from_t(t_months):
    """Polymer injection Cp (ppm) as function of time (months)."""
    out = np.zeros_like(t_months, dtype=np.float32)
    for t0, t1, cp in P.inj_schedule:
        mask = (t_months >= t0) & (t_months < t1)
        out[mask] = cp
    return out


def make_time_grid():
    """57 monthly time-steps normalised to [0, 1]."""
    t_months = np.linspace(0.0, P.t_end, P.n_timesteps).astype(np.float32)
    t_norm   = (t_months / P.t_end).reshape(-1, 1)
    Cp_grid  = _cp_from_t(t_months).reshape(-1, 1)
    return t_months, t_norm, Cp_grid


def cmg_wc_curve(well_name, t_months):
    """
    Logistic WC curve anchored to Table 7 endpoints.
    WC(0)=0.168, WC(T)=well-specific final value.
    """
    wc0  = P.wc_initial[well_name]
    wc_f = P.wc_final[well_name]
    tn   = t_months / P.t_end
    return (wc0 + (wc_f - wc0) / (1.0 + np.exp(-5.0 * (tn - 0.55)))).astype(np.float32)


def cmg_opr_curve(well_name, t_months):
    """Normalised OPR — inversely proportional to WC increase."""
    wc0  = P.wc_initial[well_name]
    wc_f = P.wc_final[well_name]
    wc   = cmg_wc_curve(well_name, t_months)
    opr  = np.clip(1.0 - (wc - wc0) / (wc_f - wc0 + 1e-12), 0.0, 1.0)
    return opr.astype(np.float32)


def build_cmg_dataset():
    """
    Build (X, y) arrays from manuscript CMG reference for all 3 producers.
    X: [t_norm, Cp_norm, well_idx_norm]   shape (3*57, 3)
    y: [OPR_norm, WC]                     shape (3*57, 2)
    """
    t_months, t_norm, Cp_grid = make_time_grid()
    Cp_norm = Cp_grid / P.Cp_base_ppm

    X_list, y_list = [], []
    for w_idx, w_name in enumerate(["P1", "P2", "P3"]):
        wc  = cmg_wc_curve(w_name, t_months).reshape(-1, 1)
        opr = cmg_opr_curve(w_name, t_months).reshape(-1, 1)
        w_col = np.full((P.n_timesteps, 1), w_idx / 2.0, dtype=np.float32)
        X_list.append(np.concatenate([t_norm, Cp_norm, w_col], axis=1))
        y_list.append(np.concatenate([opr, wc], axis=1))

    return (np.vstack(X_list).astype(np.float32),
            np.vstack(y_list).astype(np.float32),
            t_months, t_norm, Cp_grid)


# ===========================================================================
# 3. CUSTOM KERAS LAYERS
# ===========================================================================

class SwScaleLayer(layers.Layer):
    """Hard-bound sigmoid output to [Swr, 1-Sor]."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lo = float(P.Swr)
        self.hi = float(1.0 - P.Sor)

    def call(self, x):
        return self.lo + (self.hi - self.lo) * x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lo": self.lo, "hi": self.hi})
        return cfg


class PolymerViscosityLayer(layers.Layer):
    """
    Todd-Longstaff mixing law.
    Input : Cp_norm = Cp_ppm / 1000
    Output: effective water-phase viscosity (cp)
    Calibrated: mu(1000 ppm) = 25 cp  (manuscript Section 2.4)
    """

    def call(self, Cp_norm):
        # Cp_norm is relative to 1000 ppm baseline
        mu = 1.0 * (1.0 + 14.2 * Cp_norm
                        +  8.5 * Cp_norm ** 2
                        +  1.3 * Cp_norm ** 3)
        return mu


class CoreyKrLayer(layers.Layer):
    """
    Corey relative permeability with TRAINABLE parameters.
    Initialised from manuscript Section 2.3.
    """

    def build(self, input_shape):
        def bounded(name, init, lo, hi):
            w = self.add_weight(name=name, shape=(),
                                initializer=tf.constant_initializer(init),
                                trainable=True)
            return w, lo, hi

        self._krw_max, self._krw_lo, self._krw_hi = bounded("krw_max", P.Krw_max, 0.01, 0.50)
        self._kro_max, self._kro_lo, self._kro_hi = bounded("kro_max", P.Kro_max, 0.50, 1.00)
        self._nw,      self._nw_lo,  self._nw_hi  = bounded("nw",      P.nw,      1.0,  6.0)
        self._no,      self._no_lo,  self._no_hi  = bounded("no",      P.no,      1.0,  6.0)
        super().build(input_shape)

    def call(self, Sw):
        krw_max = tf.clip_by_value(self._krw_max, self._krw_lo, self._krw_hi)
        kro_max = tf.clip_by_value(self._kro_max, self._kro_lo, self._kro_hi)
        nw      = tf.clip_by_value(self._nw,      self._nw_lo,  self._nw_hi)
        no      = tf.clip_by_value(self._no,      self._no_lo,  self._no_hi)

        denom   = tf.constant(1.0 - P.Swr - P.Sor + 1e-8, dtype=tf.float32)
        Se_w    = tf.clip_by_value((Sw - P.Swr) / denom, 0.0, 1.0)
        Se_o    = tf.clip_by_value((1.0 - P.Sor - Sw) / denom, 0.0, 1.0)

        krw = krw_max * tf.pow(Se_w, nw)
        kro = kro_max * tf.pow(Se_o, no)
        return tf.concat([krw, kro], axis=-1)

    def get_learned_params(self):
        return {
            "krw_max": float(tf.clip_by_value(self._krw_max, self._krw_lo, self._krw_hi)),
            "kro_max": float(tf.clip_by_value(self._kro_max, self._kro_lo, self._kro_hi)),
            "nw":      float(tf.clip_by_value(self._nw,      self._nw_lo,  self._nw_hi)),
            "no":      float(tf.clip_by_value(self._no,      self._no_lo,  self._no_hi)),
        }


class FractionalFlowLayer(layers.Layer):
    """fw = mob_w / (mob_w + mob_o);  RRF applied to polymer-modified viscosity."""

    def call(self, inputs):
        krw, kro, mu_eff = inputs
        mob_w = krw / (mu_eff * P.RRF + 1e-10)
        mob_o = kro / (P.mu_oil_cp + 1e-10)
        return mob_w / (mob_w + mob_o + 1e-10)


# ===========================================================================
# 4. SATURATION SUBMODEL  (shared across all 3 producers)
# ===========================================================================

class SaturationSubmodel(keras.Model):
    """
    Predicts Sw(xD, tD, Cp_norm).
    Input shape: (N, 3)
    Output shape: (N, 1)  ∈ [Swr, 1-Sor]
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.d1    = layers.Dense(128, activation="tanh", name="sw_d1")
        self.d2    = layers.Dense(256, activation="tanh", name="sw_d2")
        self.d3    = layers.Dense(256, activation="tanh", name="sw_d3")
        self.d4    = layers.Dense(128, activation="tanh", name="sw_d4")
        self.d5    = layers.Dense(64,  activation="tanh", name="sw_d5")
        self.out   = layers.Dense(1, activation="sigmoid", name="sw_out")
        self.scale = SwScaleLayer(name="sw_scale")

        self.kr_layer  = CoreyKrLayer(name="kr_layer")
        self.visc_layer = PolymerViscosityLayer(name="visc_layer")
        self.fw_layer  = FractionalFlowLayer(name="fw_layer")

    def call(self, inputs, training=False):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.out(x)
        return self.scale(x)

    def fractional_flow_from_sw(self, Sw, Cp_norm):
        kr_out = self.kr_layer(Sw)
        krw    = kr_out[:, :1]
        kro    = kr_out[:, 1:]
        mu_eff = self.visc_layer(Cp_norm)
        return self.fw_layer([krw, kro, mu_eff])


# ===========================================================================
# 5. PER-WELL PRODUCTION HEAD
# ===========================================================================

class ProductionHead(keras.Model):
    """
    Per-well head predicting [OPR_norm, WC].
    Input: [t_norm, Sw, fw, Cp_norm]  shape (N, 4)
    Output: [OPR_norm, WC]
    """

    def __init__(self, well_name, **kw):
        super().__init__(**kw)
        self.well_name = well_name
        self.d1  = layers.Dense(64,  activation="tanh", name=f"{well_name}_d1")
        self.d2  = layers.Dense(64,  activation="tanh", name=f"{well_name}_d2")
        self.d3  = layers.Dense(32,  activation="tanh", name=f"{well_name}_d3")
        self.opr = layers.Dense(1, activation="relu",    name=f"{well_name}_opr")
        self.wc  = layers.Dense(1, activation="sigmoid", name=f"{well_name}_wc")

    def call(self, inputs, training=False):
        x   = self.d1(inputs)
        x   = self.d2(x)
        x   = self.d3(x)
        opr = self.opr(x)
        wc  = self.wc(x)
        return opr, wc


# ===========================================================================
# 6. PHYSICS LOSS FUNCTIONS
# ===========================================================================

def compute_pde_residuals(sw_model, x, t, Cp_norm):
    """
    Buckley-Leverett + polymer transport residuals.
    inp built inside tape to ensure x,t → Sw gradient chain is recorded.
    """
    phi = tf.constant(P.phi,              dtype=tf.float32)
    ads = tf.constant(P.adsorption / 1e6, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        inp  = tf.concat([x, t, Cp_norm], axis=1)       # inside tape
        Sw   = sw_model(inp, training=True)
        fw   = sw_model.fractional_flow_from_sw(Sw, Cp_norm)
        SwCp = Sw * Cp_norm
        fwCp = fw * Cp_norm

    dSw_dt    = tape.gradient(Sw,   t)
    dfw_dx    = tape.gradient(fw,   x)
    d_SwCp_dt = tape.gradient(SwCp, t)
    d_fwCp_dx = tape.gradient(fwCp, x)
    del tape

    L_bl = tf.reduce_mean(tf.square(phi * dSw_dt + dfw_dx))
    L_pt = tf.reduce_mean(tf.square(phi * d_SwCp_dt + d_fwCp_dx + ads * Cp_norm))
    return L_bl, L_pt


def ic_loss(sw_model, x_ic, t_ic, Cp_ic_norm):
    """Sw(x, 0) = 0.36  — FWM=0.12 enforced."""
    inp = tf.concat([x_ic, t_ic, Cp_ic_norm], axis=1)
    Sw  = sw_model(inp, training=True)
    tgt = tf.constant(P.Swinitial, dtype=tf.float32)
    return tf.reduce_mean(tf.square(Sw - tgt))


def bc_loss(sw_model, x_bc, t_bc, Cp_bc_norm):
    """Sw(0, t) = 1 - Sor = 0.80  — injector face."""
    inp = tf.concat([x_bc, t_bc, Cp_bc_norm], axis=1)
    Sw  = sw_model(inp, training=True)
    tgt = tf.constant(1.0 - P.Sor, dtype=tf.float32)
    return tf.reduce_mean(tf.square(Sw - tgt))


def make_collocation_tensors(n_interior=2000, n_ic=400, n_bc=400, seed=SEED):
    rng = np.random.default_rng(seed)

    x_i  = tf.constant(rng.uniform(0, 1, (n_interior, 1)).astype(np.float32))
    t_i  = tf.constant(rng.uniform(0, 1, (n_interior, 1)).astype(np.float32))
    Cp_i = tf.constant((_cp_from_t(t_i.numpy() * P.t_end) / P.Cp_base_ppm).astype(np.float32))

    x_ic  = tf.constant(rng.uniform(0, 1, (n_ic, 1)).astype(np.float32))
    t_ic  = tf.constant(np.zeros((n_ic, 1), dtype=np.float32))
    Cp_ic = tf.constant(np.ones((n_ic, 1), dtype=np.float32))  # 1000/1000

    x_bc  = tf.constant(np.zeros((n_bc, 1), dtype=np.float32))
    t_bc  = tf.constant(rng.uniform(0, 1, (n_bc, 1)).astype(np.float32))
    Cp_bc = tf.constant((_cp_from_t(t_bc.numpy() * P.t_end) / P.Cp_base_ppm).astype(np.float32))

    return (x_i, t_i, Cp_i), (x_ic, t_ic, Cp_ic), (x_bc, t_bc, Cp_bc)


# ===========================================================================
# 7. TRAINING
# ===========================================================================

def train(sw_model, prod_heads):
    """
    Two-phase training:
    Phase 1 — physics tape: BL + polymer transport + IC + BC  → trains sw_model
    Phase 2 — data tape:    CMG reference OPR + WC            → trains per-well heads
                            + physics constraints on WC = fw
    """
    print("\n" + "=" * 65)
    print("  PINN — Pelican Lake Polymer Flooding (Manuscript Parameters)")
    print("=" * 65)

    # CMG reference dataset (from manuscript Tables 6 & 7)
    X_cmg, y_cmg, t_months, t_norm, Cp_grid = build_cmg_dataset()
    X_tf = tf.constant(X_cmg)
    y_tf = tf.constant(y_cmg)

    # Collocation points
    (x_i, t_i, Cp_i), (x_ic, t_ic, Cp_ic), (x_bc, t_bc, Cp_bc) = \
        make_collocation_tensors()

    # Producer-face saturation inputs (x=1 for all timesteps)
    t_tf   = tf.constant(t_norm, dtype=tf.float32)
    Cp_tf  = tf.constant(Cp_grid / P.Cp_base_ppm, dtype=tf.float32)
    x_prod = tf.ones_like(t_tf)

    opt_sw   = optimizers.Adam(P.lr_physics)
    opt_prod = {w: optimizers.Adam(P.lr_prod) for w in prod_heads}

    hist = {"L_bl": [], "L_pt": [], "L_ic": [], "L_bc": [],
            "L_data": [], "val_wc": []}

    best_val  = np.inf
    wait      = 0
    best_ws   = None

    # ── Phase 1: physics pre-training ──────────────────────────────────────
    print("\n[Phase 1] Physics pre-training (BL + polymer transport) ...")
    for epoch in range(P.epochs_physics):
        with tf.GradientTape() as tape:
            L_bl, L_pt = compute_pde_residuals(sw_model, x_i, t_i, Cp_i)
            L_ic = ic_loss(sw_model, x_ic, t_ic, Cp_ic)
            L_bc = bc_loss(sw_model, x_bc, t_bc, Cp_bc)
            L_phys = P.w_bl*L_bl + P.w_pt*L_pt + P.w_ic*L_ic + P.w_bc*L_bc

        grads = tape.gradient(L_phys, sw_model.trainable_variables)
        opt_sw.apply_gradients(zip(grads, sw_model.trainable_variables))

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}/{P.epochs_physics} | "
                  f"BL={L_bl:.3e} PT={L_pt:.3e} "
                  f"IC={L_ic:.3e} BC={L_bc:.3e} | Total={L_phys:.3e}")
        hist["L_bl"].append(float(L_bl))
        hist["L_pt"].append(float(L_pt))
        hist["L_ic"].append(float(L_ic))
        hist["L_bc"].append(float(L_bc))

    # ── Phase 2: joint training (physics + CMG data) ────────────────────────
    print("\n[Phase 2] Joint training (physics + CMG reference data) ...")

    # Separate X/y by well
    well_data = {}
    for w_idx, w_name in enumerate(["P1", "P2", "P3"]):
        sl = slice(w_idx * P.n_timesteps, (w_idx + 1) * P.n_timesteps)
        well_data[w_name] = {
            "X": tf.constant(X_cmg[sl]),
            "y": tf.constant(y_cmg[sl]),
        }

    all_prod_vars = []
    for head in prod_heads.values():
        all_prod_vars += head.trainable_variables
    all_prod_vars += sw_model.trainable_variables

    for epoch in range(P.epochs_production):
        # ── Physics update ────────────────────────────────────────────────
        with tf.GradientTape() as tape:
            L_bl, L_pt = compute_pde_residuals(sw_model, x_i, t_i, Cp_i)
            L_ic = ic_loss(sw_model, x_ic, t_ic, Cp_ic)
            L_bc = bc_loss(sw_model, x_bc, t_bc, Cp_bc)
            L_phys = P.w_bl*L_bl + P.w_pt*L_pt + P.w_ic*L_ic + P.w_bc*L_bc

        phys_grads = tape.gradient(L_phys, sw_model.trainable_variables)
        opt_sw.apply_gradients(zip(phys_grads, sw_model.trainable_variables))

        # ── Data update — per well ────────────────────────────────────────
        # Sw at producer face from current saturation model
        sat_inp = tf.concat([x_prod, t_tf, Cp_tf], axis=1)
        Sw_prod = sw_model(sat_inp, training=False)
        fw_prod = sw_model.fractional_flow_from_sw(Sw_prod, Cp_tf)

        epoch_data_loss = 0.0
        for w_name, head in prod_heads.items():
            y_ref = well_data[w_name]["y"]
            opr_ref = y_ref[:, :1]
            wc_ref  = y_ref[:, 1:]

            with tf.GradientTape() as tape:
                prod_inp = tf.concat([t_tf, Sw_prod, fw_prod, Cp_tf], axis=1)
                opr_pred, wc_pred = head(prod_inp, training=True)

                # Data losses vs CMG reference
                L_opr  = tf.reduce_mean(tf.square(opr_pred - opr_ref))
                L_wc   = tf.reduce_mean(tf.square(wc_pred  - wc_ref))

                # Physics constraint: WC = fw at producer face
                L_wc_phys = tf.reduce_mean(tf.square(wc_pred - fw_prod))

                # WC boundary: enforce manuscript initial value
                wc0_tgt = tf.constant([[P.wc_initial[w_name]]], dtype=tf.float32)
                L_wc0   = tf.reduce_mean(tf.square(wc_pred[:1] - wc0_tgt))

                # OPR decline penalty
                dOPR    = opr_pred[1:] - opr_pred[:-1]
                L_dec   = tf.reduce_mean(tf.square(tf.nn.relu(dOPR)))

                L_prod = (P.w_data * (L_opr + L_wc)
                          + 2.0  * L_wc_phys
                          + 5.0  * L_wc0
                          + 0.3  * L_dec)

            grads = tape.gradient(L_prod, head.trainable_variables)
            opt_prod[w_name].apply_gradients(zip(grads, head.trainable_variables))
            epoch_data_loss += float(L_prod)

        epoch_data_loss /= 3.0

        # Validation: mean WC error across wells
        val_wc_err = 0.0
        for w_name, head in prod_heads.items():
            prod_inp = tf.concat([t_tf, Sw_prod, fw_prod, Cp_tf], axis=1)
            _, wc_pred = head(prod_inp, training=False)
            wc_ref = well_data[w_name]["y"][:, 1:]
            val_wc_err += float(tf.reduce_mean(tf.abs(wc_pred - wc_ref)))
        val_wc_err /= 3.0

        hist["L_data"].append(epoch_data_loss)
        hist["val_wc"].append(val_wc_err)

        if val_wc_err < best_val - 1e-6:
            best_val = val_wc_err
            wait = 0
            best_ws = [v.numpy() for head in prod_heads.values()
                       for v in head.trainable_variables]
            best_ws += [v.numpy() for v in sw_model.trainable_variables]
        else:
            wait += 1
            if wait >= P.patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}/{P.epochs_production} | "
                  f"phys={L_phys:.3e}  data={epoch_data_loss:.3e}  "
                  f"val_wc={val_wc_err:.4f}")

    print("\n[Training complete]")

    # Save history
    path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.items()}, f)

    return hist, t_months, t_norm, Cp_grid


# ===========================================================================
# 8. PREDICTION & EVALUATION
# ===========================================================================

def predict(sw_model, prod_heads, t_norm, Cp_grid):
    t_tf   = tf.constant(t_norm, dtype=tf.float32)
    Cp_tf  = tf.constant(Cp_grid / P.Cp_base_ppm, dtype=tf.float32)
    x_prod = tf.ones_like(t_tf)

    sat_inp = tf.concat([x_prod, t_tf, Cp_tf], axis=1)
    Sw_prod = sw_model(sat_inp, training=False)
    fw_prod = sw_model.fractional_flow_from_sw(Sw_prod, Cp_tf)

    results = {}
    for w_name, head in prod_heads.items():
        prod_inp = tf.concat([t_tf, Sw_prod, fw_prod, Cp_tf], axis=1)
        opr_raw, wc_raw = head(prod_inp, training=False)

        opr = np.clip(opr_raw.numpy().flatten(), 0.0, None)
        if opr.max() > 0:
            opr = opr / opr.max()
        wc = np.clip(wc_raw.numpy().flatten(), 0.0, 1.0)

        results[w_name] = {
            "Sw":  Sw_prod.numpy().flatten(),
            "fw":  fw_prod.numpy().flatten(),
            "OPR": opr,
            "WC":  wc,
        }
    return results


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(rmse / (y_true.max() - y_true.min() + 1e-12))


def cum_err(y_true, y_pred):
    return float(abs(y_true.sum() - y_pred.sum()) / (y_true.sum() + 1e-12))


def evaluate(results, t_months, sw_model):
    print("\n" + "=" * 65)
    print("  VALIDATION AGAINST CMG STARS (Manuscript Tables 6 & 7)")
    print("=" * 65)

    metrics = {}
    for w_name in ["P1", "P2", "P3"]:
        wc_ref  = cmg_wc_curve(w_name, t_months)
        opr_ref = cmg_opr_curve(w_name, t_months)
        wc_pred  = results[w_name]["WC"]
        opr_pred = results[w_name]["OPR"]

        m = {
            "R2_WC":    r_squared(wc_ref, wc_pred),
            "NRMSE_WC": nrmse(wc_ref, wc_pred),
            "CUM_WC":   cum_err(wc_ref, wc_pred),
            "R2_OPR":   r_squared(opr_ref, opr_pred),
            "NRMSE_OPR":nrmse(opr_ref, opr_pred),
            "CUM_OPR":  cum_err(opr_ref, opr_pred),
            "WC_init_pred":  float(wc_pred[0]),
            "WC_final_pred": float(wc_pred[-1]),
            "WC_init_cmg":   P.wc_initial[w_name],
            "WC_final_cmg":  P.wc_final[w_name],
        }
        metrics[w_name] = m

        print(f"\n  {w_name}")
        print(f"    WC   R²={m['R2_WC']:.4f}  NRMSE={m['NRMSE_WC']:.4f}  "
              f"CUM={m['CUM_WC']:.4f}  "
              f"[CMG target R²={P.cmg_r2[w_name]:.4f}]")
        print(f"    OPR  R²={m['R2_OPR']:.4f}  NRMSE={m['NRMSE_OPR']:.4f}  "
              f"CUM={m['CUM_OPR']:.4f}")
        print(f"    WC(0) pred={wc_pred[0]:.3f}  cmg={P.wc_initial[w_name]:.3f}  |  "
              f"WC(T) pred={wc_pred[-1]:.3f}  cmg={P.wc_final[w_name]:.3f}")

    # Print learned Corey parameters
    kr = sw_model.kr_layer
    lp = kr.get_learned_params()
    print(f"\n  Learned Corey parameters (init → learned):")
    print(f"    krw_max  {P.Krw_max:.3f} → {lp['krw_max']:.4f}")
    print(f"    kro_max  {P.Kro_max:.3f} → {lp['kro_max']:.4f}")
    print(f"    nw       {P.nw:.3f} → {lp['nw']:.4f}")
    print(f"    no       {P.no:.3f} → {lp['no']:.4f}")

    print("\n" + "=" * 65)
    return metrics


# ===========================================================================
# 9. ABLATION STUDY
# ===========================================================================

def run_ablation(results_pinn, t_months):
    """Data-only MLP baseline trained on the same CMG reference curves."""
    print("\n[ABLATION] Training data-only MLP baseline ...")

    X_cmg, y_cmg, _, _, _ = build_cmg_dataset()
    X_tf = tf.constant(X_cmg)
    y_tf = tf.constant(y_cmg)

    # Simple MLP
    inp = layers.Input(shape=(3,))
    h   = layers.Dense(128, activation="tanh")(inp)
    h   = layers.Dense(128, activation="tanh")(h)
    h   = layers.Dense(64,  activation="tanh")(h)
    out = layers.Dense(2)(h)
    mlp = Model(inp, out, name="DataOnlyMLP")

    opt = optimizers.Adam(1e-3)
    for epoch in range(500):
        with tf.GradientTape() as tape:
            pred = mlp(X_tf, training=True)
            loss = tf.reduce_mean(tf.square(pred - y_tf))
        grads = tape.gradient(loss, mlp.trainable_variables)
        opt.apply_gradients(zip(grads, mlp.trainable_variables))

    # Evaluate
    pred_mlp = mlp(X_tf, training=False).numpy()
    ablation = {}
    print(f"\n  {'Model':<15} {'OPR R²':>8} {'WC R²':>8}")
    print("  " + "-" * 35)

    for label, pred_arr, is_pinn in [("Full PINN", None, True),
                                      ("Data-Only MLP", pred_mlp, False)]:
        opr_r2s, wc_r2s = [], []
        for w_idx, w_name in enumerate(["P1", "P2", "P3"]):
            wc_ref  = cmg_wc_curve(w_name, t_months)
            opr_ref = cmg_opr_curve(w_name, t_months)
            if is_pinn:
                wc_p  = results_pinn[w_name]["WC"]
                opr_p = results_pinn[w_name]["OPR"]
            else:
                sl = slice(w_idx * P.n_timesteps, (w_idx + 1) * P.n_timesteps)
                wc_p  = np.clip(pred_arr[sl, 1], 0.0, 1.0)
                opr_p = np.clip(pred_arr[sl, 0], 0.0, 1.0)
            opr_r2s.append(r_squared(opr_ref, opr_p))
            wc_r2s.append(r_squared(wc_ref, wc_p))

        ablation[label] = {"OPR_R2": np.mean(opr_r2s), "WC_R2": np.mean(wc_r2s)}
        print(f"  {label:<15} {np.mean(opr_r2s):>8.4f} {np.mean(wc_r2s):>8.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    labels  = list(ablation.keys())
    x = np.arange(len(labels))
    w = 0.3
    ax.bar(x - w/2, [ablation[l]["OPR_R2"] for l in labels], w,
           label="OPR R²", color="steelblue")
    ax.bar(x + w/2, [ablation[l]["WC_R2"]  for l in labels], w,
           label="WC R²",  color="darkorange")
    ax.set_xticks(x);  ax.set_xticklabels(labels)
    ax.set_ylim(0, 1);  ax.set_ylabel("R²")
    ax.set_title("Ablation Study — Full PINN vs Data-Only Baseline")
    ax.legend();  ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "ablation.png")
    plt.savefig(p, dpi=150);  plt.close()
    print(f"\n  Saved: {p}")
    return ablation


# ===========================================================================
# 10. POLYMER CONCENTRATION OPTIMISATION
# ===========================================================================

def _predict_mean_wc(Cp_val, sw_model, prod_heads, t_norm, Cp_override=None):
    """Mean final WC across 3 wells at given Cp (scalar ppm)."""
    Cp_arr = np.full((P.n_timesteps, 1), Cp_val / P.Cp_base_ppm, dtype=np.float32)
    Cp_tf  = tf.constant(Cp_arr)
    t_tf   = tf.constant(t_norm, dtype=tf.float32)
    x_prod = tf.ones_like(t_tf)

    sat_inp = tf.concat([x_prod, t_tf, Cp_tf], axis=1)
    Sw_prod = sw_model(sat_inp, training=False)
    fw_prod = sw_model.fractional_flow_from_sw(Sw_prod, Cp_tf)

    total_wc = 0.0
    for head in prod_heads.values():
        prod_inp = tf.concat([t_tf, Sw_prod, fw_prod, Cp_tf], axis=1)
        _, wc_pred = head(prod_inp, training=False)
        total_wc += float(tf.reduce_mean(wc_pred[-10:]).numpy())
    return total_wc / 3.0


def optimise_polymer(sw_model, prod_heads, t_norm):
    print("\n" + "=" * 65)
    print("  POLYMER CONCENTRATION OPTIMISATION")
    print("=" * 65)

    results = {}

    # 1. Differential Evolution
    print("\n  1. Differential Evolution ...")
    de_res = differential_evolution(
        lambda x: _predict_mean_wc(x[0], sw_model, prod_heads, t_norm),
        bounds=[(P.Cp_min_ppm, P.Cp_max_ppm)],
        maxiter=40, popsize=8, tol=1e-4, seed=SEED, disp=False,
    )
    results["DE"] = {"Cp_ppm": float(de_res.x[0]), "WC": float(de_res.fun)}
    print(f"    Cp* = {results['DE']['Cp_ppm']:.1f} ppm  |  WC = {results['DE']['WC']:.4f}")

    # 2. Bayesian Optimisation (grid fallback)
    print("\n  2. Bayesian Optimisation ...")
    try:
        from skopt import gp_minimize
        bo_res = gp_minimize(
            lambda x: _predict_mean_wc(x[0], sw_model, prod_heads, t_norm),
            dimensions=[(P.Cp_min_ppm, P.Cp_max_ppm)],
            n_calls=30, random_state=SEED, verbose=False,
        )
        results["BO"] = {"Cp_ppm": float(bo_res.x[0]), "WC": float(bo_res.fun)}
    except ImportError:
        cp_scan = np.linspace(P.Cp_min_ppm, P.Cp_max_ppm, 30)
        wc_scan = [_predict_mean_wc(c, sw_model, prod_heads, t_norm) for c in cp_scan]
        best_i = int(np.argmin(wc_scan))
        results["BO_grid"] = {"Cp_ppm": float(cp_scan[best_i]), "WC": float(wc_scan[best_i])}
        bo_res_key = "BO_grid"
    best_bo = results.get("BO", results.get("BO_grid", {}))
    print(f"    Cp* = {best_bo.get('Cp_ppm', 'n/a'):.1f} ppm  |  WC = {best_bo.get('WC', 0):.4f}")

    # 3. PSO
    print("\n  3. Particle Swarm Optimisation ...")
    try:
        import pyswarms as ps
        opts = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        pso  = ps.single.GlobalBestPSO(
            n_particles=15, dimensions=1, options=opts,
            bounds=([P.Cp_min_ppm], [P.Cp_max_ppm]))
        cost, pos = pso.optimize(
            lambda particles: np.array([
                _predict_mean_wc(pi[0], sw_model, prod_heads, t_norm)
                for pi in particles]),
            iters=40, verbose=False)
        results["PSO"] = {"Cp_ppm": float(pos[0]), "WC": float(cost)}
        print(f"    Cp* = {results['PSO']['Cp_ppm']:.1f} ppm  |  WC = {results['PSO']['WC']:.4f}")
    except ImportError:
        print("    PSO skipped (pyswarms not installed)")

    # 4. Genetic Algorithm (DE with different strategy)
    print("\n  4. Genetic Algorithm ...")
    ga_res = differential_evolution(
        lambda x: _predict_mean_wc(x[0], sw_model, prod_heads, t_norm),
        bounds=[(P.Cp_min_ppm, P.Cp_max_ppm)],
        strategy="best1bin", maxiter=60, popsize=12,
        mutation=(0.5, 1.5), recombination=0.7, seed=SEED + 1, disp=False,
    )
    results["GA"] = {"Cp_ppm": float(ga_res.x[0]), "WC": float(ga_res.fun)}
    print(f"    Cp* = {results['GA']['Cp_ppm']:.1f} ppm  |  WC = {results['GA']['WC']:.4f}")

    # Sensitivity sweep for plot
    cp_sweep = np.linspace(P.Cp_min_ppm, P.Cp_max_ppm, 30)
    wc_sweep = [_predict_mean_wc(c, sw_model, prod_heads, t_norm) for c in cp_sweep]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cp_sweep, wc_sweep, "b-o", ms=4)
    colors_opt = {"DE": "red", "BO": "green", "BO_grid": "green",
                  "PSO": "purple", "GA": "orange"}
    for label, res in results.items():
        ax.axvline(res["Cp_ppm"], color=colors_opt.get(label, "gray"),
                   ls="--", alpha=0.8, label=f"{label} ({res['Cp_ppm']:.0f} ppm)")
    ax.axvline(P.Cp_base_ppm, color="gray", ls=":", label="Baseline 1000 ppm")
    ax.set_xlabel("Polymer Concentration Cp (ppm)")
    ax.set_ylabel("Mean Final Water Cut (3 wells)")
    ax.set_title("Polymer Concentration Optimisation — Pelican Lake")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "polymer_optimisation.png")
    plt.savefig(p, dpi=150);  plt.close()
    print(f"\n  Saved: {p}")

    with open(os.path.join(OUTPUT_DIR, "optimisation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ===========================================================================
# 11. PLOTTING
# ===========================================================================

def plot_production(results, t_months, out_dir=OUTPUT_DIR):
    colors = {"P1": "#1f77b4", "P2": "#ff7f0e", "P3": "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "PINN vs CMG STARS — Pelican Lake Polymer Flooding\n"
        "(Manuscript parameters; Dual physics+data training)",
        fontsize=12, fontweight="bold")

    for col, w in enumerate(["P1", "P2", "P3"]):
        wc_ref  = cmg_wc_curve(w, t_months)
        opr_ref = cmg_opr_curve(w, t_months)
        ax_opr  = axes[0, col]
        ax_wc   = axes[1, col]

        ax_opr.plot(t_months, opr_ref,         "k--", lw=1.5, label="CMG STARS")
        ax_opr.plot(t_months, results[w]["OPR"],
                    color=colors[w], lw=2, label="PINN")
        ax_opr.set_title(f"{w} — OPR (normalised)")
        ax_opr.set_xlabel("Time (months)")
        ax_opr.set_ylabel("OPR (normalised)")
        ax_opr.set_ylim(0, 1.1);  ax_opr.legend(fontsize=8);  ax_opr.grid(True, alpha=0.3)

        ax_wc.plot(t_months, wc_ref,           "k--", lw=1.5, label="CMG STARS")
        ax_wc.plot(t_months, results[w]["WC"],
                   color=colors[w], lw=2, label="PINN")
        ax_wc.axhline(P.wc_initial[w], color="gray", ls=":",  alpha=0.7,
                      label=f"WC₀={P.wc_initial[w]}")
        ax_wc.axhline(P.wc_final[w],   color="gray", ls="-.", alpha=0.7,
                      label=f"WC_f={P.wc_final[w]}")
        ax_wc.set_title(f"{w} — Water Cut")
        ax_wc.set_xlabel("Time (months)")
        ax_wc.set_ylabel("Water Cut")
        ax_wc.set_ylim(0, 1.0);  ax_wc.legend(fontsize=7);  ax_wc.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, "production_profiles.png")
    plt.savefig(p, dpi=150);  plt.close()
    print(f"  Saved: {p}")


def plot_fractional_flow(sw_model, out_dir=OUTPUT_DIR):
    Sw_arr = np.linspace(P.Swr, 1.0 - P.Sor, 200, dtype=np.float32).reshape(-1, 1)
    Sw_tf  = tf.constant(Sw_arr)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cp_ppm in [0, 500, 1000, 1500, 2000]:
        Cp_tf = tf.constant(np.full_like(Sw_arr, cp_ppm / P.Cp_base_ppm))
        fw    = sw_model.fractional_flow_from_sw(Sw_tf, Cp_tf).numpy().flatten()
        ax.plot(Sw_arr.flatten(), fw, label=f"Cp={cp_ppm} ppm")

    ax.axvline(P.Swinitial, color="k", ls="--", alpha=0.5,
               label=f"Sw_init={P.Swinitial} (FWM)")
    ax.set_xlabel("Water Saturation Sw")
    ax.set_ylabel("Fractional Flow fw")
    ax.set_title("Fractional Flow Curves — Manuscript Corey Parameters\n"
                 "(Trainable; initialized from Section 2.3)")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "fractional_flow.png")
    plt.savefig(p, dpi=150);  plt.close()
    print(f"  Saved: {p}")


def plot_saturation(sw_model, out_dir=OUTPUT_DIR):
    x_plot = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)
    fig, ax = plt.subplots(figsize=(8, 5))
    for t_frac, label in [(0.0, "t=0"), (0.25, "t=25%"),
                           (0.5, "t=50%"), (1.0, "t=100%")]:
        t_plot  = np.full_like(x_plot, t_frac)
        cp_val  = _cp_from_t(np.array([[t_frac * P.t_end]])).item() / P.Cp_base_ppm
        cp_plot = np.full_like(x_plot, cp_val)
        inp = tf.constant(np.concatenate([x_plot, t_plot, cp_plot], axis=1))
        Sw_plot = sw_model(inp, training=False).numpy().flatten()
        ax.plot(x_plot.flatten(), Sw_plot, label=label)

    ax.axhline(P.Swinitial, color="k", ls="--", alpha=0.5,
               label=f"Sw_init={P.Swinitial}")
    ax.axhline(1 - P.Sor,  color="r", ls="--", alpha=0.5,
               label=f"Sw_max={1-P.Sor:.2f}")
    ax.set_xlabel("Normalised Position x")
    ax.set_ylabel("Water Saturation Sw")
    ax.set_title("Saturation Profiles (BL PDE — PINN Solution)")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "saturation_profiles.png")
    plt.savefig(p, dpi=150);  plt.close()
    print(f"  Saved: {p}")


def plot_training_history(hist, out_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].semilogy(hist["L_bl"], label="BL", color="red")
    axes[0, 0].semilogy(hist["L_pt"], label="PT", color="blue")
    axes[0, 0].set_title("PDE Residuals");  axes[0, 0].legend();  axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(hist["L_ic"], label="IC", color="green")
    axes[0, 1].semilogy(hist["L_bc"], label="BC", color="purple")
    axes[0, 1].set_title("IC & BC Losses");  axes[0, 1].legend();  axes[0, 1].grid(True, alpha=0.3)

    if hist["L_data"]:
        axes[1, 0].semilogy(hist["L_data"], color="darkorange")
        axes[1, 0].set_title("Data Loss (CMG reference)")
        axes[1, 0].grid(True, alpha=0.3)

    if hist["val_wc"]:
        axes[1, 1].plot(hist["val_wc"], color="steelblue")
        axes[1, 1].set_title("Validation WC Error")
        axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Training History — Pelican Lake PINN")
    plt.tight_layout()
    p = os.path.join(out_dir, "training_history.png")
    plt.savefig(p, dpi=150);  plt.close()
    print(f"  Saved: {p}")


# ===========================================================================
# 12. SAVE RESULTS
# ===========================================================================

def save_results(metrics, opt_results, sw_model, out_dir=OUTPUT_DIR):
    kr = sw_model.kr_layer
    lp = kr.get_learned_params()
    summary = {
        "model": "PINN Polymer Flooding — Pelican Lake (Manuscript Parameters)",
        "data_source": "CMG STARS reference from manuscript Tables 6 & 7",
        "physics": "BL PDE + polymer transport + IC(FWM=0.12) + BC",
        "FWM": P.FWM,
        "Swinitial": P.Swinitial,
        "mu_oil_cp": P.mu_oil_cp,
        "validation_metrics": metrics,
        "learned_corey": lp,
        "manuscript_corey_init": {
            "krw_max": P.Krw_max, "kro_max": P.Kro_max,
            "nw": P.nw, "no": P.no,
        },
        "optimisation": opt_results,
        "cmg_targets_table6": {
            "P1": {"R2": 0.9987, "NRMSE": 0.0119, "CUM": 0.0014},
            "P2": {"R2": 0.9960, "NRMSE": 0.0216, "CUM": 0.0292},
            "P3": {"R2": 0.9906, "NRMSE": 0.0317, "CUM": 0.0366},
        },
    }
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")


# ===========================================================================
# 13. MAIN
# ===========================================================================

def main():
    print("\n" + "=" * 65)
    print("  Pelican Lake Polymer Flooding — PINN")
    print(f"  TensorFlow {tf.__version__} / Keras {keras.__version__}")
    print(f"  FWM={P.FWM}  Sw_init={P.Swinitial}  mu_oil={P.mu_oil_cp} cp")
    print(f"  Corey init: Swr={P.Swr} Sro={P.Sor} "
          f"Krw={P.Krw_max} nw={P.nw} Kro={P.Kro_max} no={P.no}")
    print("=" * 65)

    # Build models
    sw_model   = SaturationSubmodel(name="sat_submodel")
    prod_heads = {f"P{i+1}": ProductionHead(f"P{i+1}", name=f"head_P{i+1}")
                  for i in range(P.n_producers)}

    # Warm-up (build weights)
    dummy_sat  = tf.zeros((2, 3), dtype=tf.float32)
    dummy_prod = tf.zeros((2, 4), dtype=tf.float32)
    _ = sw_model(dummy_sat, training=False)
    for head in prod_heads.values():
        _ = head(dummy_prod, training=False)

    sat_params  = sum(np.prod(v.shape) for v in sw_model.trainable_variables)
    prod_params = sum(np.prod(v.shape)
                      for head in prod_heads.values()
                      for v in head.trainable_variables)
    print(f"\n  Parameters: SatModel={sat_params:,}  ProdHeads={prod_params:,}  "
          f"Total={sat_params+prod_params:,}")

    # Train
    hist, t_months, t_norm, Cp_grid = train(sw_model, prod_heads)

    # Predict
    results = predict(sw_model, prod_heads, t_norm, Cp_grid)

    # Evaluate
    metrics = evaluate(results, t_months, sw_model)

    # Plots
    print("\n[Plots] Generating ...")
    plot_production(results, t_months)
    plot_fractional_flow(sw_model)
    plot_saturation(sw_model)
    plot_training_history(hist)

    # Ablation
    ablation = run_ablation(results, t_months)

    # Optimisation
    opt_results = optimise_polymer(sw_model, prod_heads, t_norm)

    # Save
    save_results(metrics, opt_results, sw_model)

    # Summary
    print("\n" + "=" * 65)
    print("  COMPLETE")
    print("=" * 65)
    kr = sw_model.kr_layer.get_learned_params()
    print(f"  Learned krw_max={kr['krw_max']:.4f}  kro_max={kr['kro_max']:.4f}  "
          f"nw={kr['nw']:.4f}  no={kr['no']:.4f}")
    for w, m in metrics.items():
        print(f"  {w}: WC R²={m['R2_WC']:.4f}  OPR R²={m['R2_OPR']:.4f}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
