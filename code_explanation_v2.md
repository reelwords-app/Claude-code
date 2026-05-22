# Understanding the PINN Code — A Complete Beginner's Guide
### Pelican Lake Heavy Oil Polymer Flooding — `pinn_polymer_flooding.py`

---

> **Who this document is for:** You have never written a line of Python before. You do not know what a neural network is. You may not even know what a reservoir is. That is perfectly fine. By the end of this document you will understand every single line in this code, what it does, why it is there, and how it connects to the physics of your oil field.
>
> We will use simple analogies throughout. When a concept is technical, we will explain it in plain English first, then show you the code.

---

## Table of Contents

1. [The Big Picture — What This Code Does](#1-the-big-picture)
2. [Imports — Borrowing Tools](#2-imports)
3. [Constants — The Physical World in Numbers](#3-constants)
4. [Custom Layers — Teaching the Machine About Physics](#4-custom-layers)
5. [SaturationSubmodel — The Water Saturation Brain](#5-saturationsubmodel)
6. [PolymerPINN — The Main Prediction Engine](#6-polymerpinn)
7. [Loading Data — Reading Your CMG Files](#7-loading-data)
8. [Physics Loss Functions — Enforcing the Laws of Nature](#8-physics-loss-functions)
9. [The Training Loop — Learning From Both Data and Physics](#9-training-loop)
10. [Evaluation — Grading the Model](#10-evaluation)
11. [Visualisation — Drawing the Results](#11-visualisation)
12. [Ablation Study — Proving Physics Helps](#12-ablation-study)
13. [Optimisation — Finding the Best Polymer Dose](#13-optimisation)
14. [Main Function — The Master Conductor](#14-main-function)
15. [Glossary of Terms](#15-glossary)

---

## 1. The Big Picture

### What is this code trying to do?

Imagine you run an oil field called **Pelican Lake** in Canada. You are injecting a special chemical called **polymer** into the ground to push heavy oil towards your production wells. The oil is very thick — like cold honey — and polymer helps push it more efficiently.

You have a **reservoir simulator** called CMG STARS. Think of it like a very detailed video game that simulates the underground. You ran this game 200 times with different settings and saved the results as CSV files. Each run is called a **case**.

Now the problem: CMG takes a long time to run. What if you need to test 10,000 different scenarios to find the best polymer dose? You would need years.

**The solution:** Train a neural network to be a "student" of CMG. Show it the 200 cases. Let it learn the patterns. Then it can answer new questions in milliseconds instead of hours.

But here is the catch — a regular neural network is a pure pattern-matcher. It might learn nonsense that violates the laws of physics. That is dangerous for engineering decisions.

**The PINN (Physics-Informed Neural Network) solution:** We do two things at once:
1. Train on the 200 CMG cases (data)
2. Simultaneously force the network to obey the physical equations that govern fluid flow underground (physics)

This is like teaching a student with two teachers — one shows them real exam answers, the other makes sure they understand WHY the answers are correct, not just memorise them.

### The physical equations involved

The two main equations that govern what happens underground are:

**Buckley-Leverett equation** (how water saturation moves):
```
φ × ∂Sw/∂t  +  ∂fw/∂x  =  0
```
Read this as: "The rate of change of water saturation with time, plus the rate of change of fractional flow with distance, equals zero." This is a conservation law — water does not appear from nothing.

**Polymer transport equation** (how polymer moves):
```
φ × ∂(Sw × Cp)/∂t  +  ∂(fw × Cp)/∂x  +  adsorption  =  0
```
Read this as: "The change in polymer stored in the water, plus the change in polymer flowing, plus polymer sticking to rock, equals zero."

These equations are what the code enforces mathematically. Now let us read the code.

---

## 2. Imports

```python
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
```

**What is an `import`?**

Python does not know how to do everything by itself. Just like a carpenter does not make their own screwdrivers — they buy tools. `import` is how Python gets tools from a toolbox.

| Import | What it does | Analogy |
|--------|-------------|---------|
| `os` | Talks to the computer's operating system — creates folders, reads paths | A filing cabinet manager |
| `json` | Reads and writes JSON files (structured text data) | A translator for a special text format |
| `warnings` | Controls warning messages | A volume knob for warnings |
| `numpy` (np) | Fast math with arrays — vectors, matrices, statistics | A scientific calculator on steroids |
| `pandas` (pd) | Reads spreadsheets (CSV files), organises data in tables | Excel in Python |
| `matplotlib` | Draws graphs and plots | The drawing program |
| `tensorflow` (tf) | The main machine-learning engine — builds and trains neural networks | The engine of a car |
| `keras` | A friendlier interface to TensorFlow — makes building networks easier | The steering wheel of that same car |
| `layers` | The building blocks of neural networks — Dense, Dropout, BatchNorm | LEGO bricks |

```python
matplotlib.use('Agg')
```
This tells matplotlib: "Do not try to open a window on screen. Save all plots to files instead." This is necessary when running on a server with no screen.

```python
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
```
These two lines silence unnecessary messages. The training already prints enough information — we do not need TensorFlow printing technical warnings every second.

---

## 3. Constants

```python
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

**What is a random seed?**

Computers generate "random" numbers using a mathematical formula. If you start the formula at the same point every time, you get the same sequence of "random" numbers. This is called **reproducibility** — if you run this code twice, you get the same result. The number 42 is traditional (from "The Hitchhiker's Guide to the Galaxy") but any number works.

---

```python
DATA_DIR   = os.environ.get('DATA_DIR',   'data')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

**Paths — where to find things.**

- `DATA_DIR` = the folder where your 200 CSV files live. Default: a folder named `data/`.
- `OUTPUT_DIR` = the folder where plots and results will be saved. Default: `outputs/`.
- `os.makedirs(..., exist_ok=True)` = "Create the outputs folder if it does not exist. If it already exists, do not complain."

---

### Physical Constants — The Numbers From Your Manuscript

These are the heart of the update. Every number comes from the Pelican Lake paper.

```python
POROSITY = 0.312
```
**Porosity** (φ) is the fraction of the rock that is empty space (pores) where fluid can sit. 0.312 means 31.2% of the rock volume is pore space. From Table 4 of the manuscript, Good Pay layer.

```python
PERMEABILITY = 3000.0   # millidarcies
```
**Permeability** measures how easily fluid flows through rock. 3000 md is very high — this is a good reservoir. Like the difference between water flowing through sand vs. solid concrete.

```python
NET_PAY_M  = 2.4    # metres
WELL_LEN_M = 1400.0 # metres
```
The thickness of the oil-bearing layer (2.4 m) and the length of the horizontal well (1.4 km). Used for volume calculations.

```python
MU_OIL     = 1650.0  # centipoise (cP)
MU_WATER   =    1.0  # cP
MU_POLYMER_REF = 25.0  # cP at 1000 ppm
```
**Viscosity** (μ) measures how thick a fluid is. 
- Water viscosity = 1 cP (the reference).
- Pelican Lake oil is 1650 times thicker than water. That is why it is called **heavy oil** — it barely flows.
- Polymer solution at 1000 ppm concentration: 25 cP. Still 25× thicker than water. Polymer makes the injected water thicker so it sweeps the reservoir more evenly instead of fingering through.

```python
SWC  = 0.23   # connate water saturation
SOR  = 0.20   # residual oil saturation
```
**SWC** (S_wc) = the minimum water saturation — water is always present in pores even in the driest zone (23% of pore space is always water). You cannot drive it below this.

**SOR** (S_or) = the residual oil saturation — the oil that is permanently trapped in pores no matter how much water you inject (20% of pore space is always oil). This is the oil left behind that you can never recover with waterflooding.

So the range of water saturation the model ever predicts is: **SWC to (1 − SOR) = 0.23 to 0.80**.

```python
KRW_MAX_INIT = 0.10   # maximum relative permeability to water
KRO_MAX_INIT = 1.00   # maximum relative permeability to oil
NW_INIT      = 3.0    # Corey water exponent
NO_INIT      = 2.2    # Corey oil exponent
```
These are the **Corey relative permeability** parameters. Relative permeability (kr) describes how well each fluid flows at a given saturation — it is not constant, it depends on how much of each fluid is present.

The Corey model says:
```
krw = KRW_MAX × ((Sw - Swc) / (1 - Swc - Sor))^nw
kro = KRO_MAX × ((1 - Sor - Sw) / (1 - Swc - Sor))^no
```
- When Sw = SWC (minimum water): krw = 0, kro = KRO_MAX = 1.00. All oil flows, no water.
- When Sw = 1−SOR (maximum water): krw = KRW_MAX = 0.10, kro = 0. All water flows, no oil.
- The exponents (nw=3, no=2.2) control how curved the relationship is.

**KRW_MAX = 0.10 is the key manuscript correction.** It was 0.216 in the original code. At maximum water saturation, water only flows 10% as well as oil at its best. This is typical for heavy oil — oil has much better relative permeability.

**These parameters are TRAINABLE** — the neural network starts with these values but will adjust them during training to better fit the data. It is like giving an expert their initial guess and letting them refine it.

```python
FWM      = 0.12   # Mobile Water Fraction
SWI_CORE = 0.30   # irreducible Sw from core measurements
SWINITIAL = 0.36  # = 0.30 + 0.12*(1 - 0.30 - 0.20)
```
**This is the KEY innovation of the Pelican Lake paper.**

Standard models assume the reservoir starts at SWC (connate water only = immobile). But Pelican Lake has a **mobile water fraction (FWM)** — there is already some moveable water present before injection starts.

The initial water saturation is:
```
Swinitial = Swi_core + FWM × (1 - Swi_core - Sor)
          = 0.30     + 0.12 × (1 - 0.30 - 0.20)
          = 0.30     + 0.12 × 0.50
          = 0.30     + 0.06
          = 0.36
```
This 0.36 (not 0.23) is used as the **initial condition** of the PINN. It changes the predicted water cut at the start of production.

```python
RK = 2.0   # Residual Resistance Factor (RRF)
```
**RRF (Residual Resistance Factor)** = after polymer flows through rock, it reduces the rock's permeability to water permanently (polymer adsorbs onto rock surfaces, narrowing pore throats). RRF = 2.0 means the permeability to water is cut in half after polymer treatment. This divides mob_w by 2, which makes polymer even more effective at diverting flow to unswept zones.

```python
CP_MAX_PPM  = 2000.0  # maximum Cp in dataset (ppm)
SALINITY_PPM = 8222.0
ADSORPTION  = 10.0    # µg/g rock
IPV         = 0.10    # Inaccessible Pore Volume
```
- **CP_MAX_PPM**: The highest polymer concentration tested = 2000 ppm. All concentrations are normalised by dividing by this number (so cp_norm ranges 0 to 1).
- **SALINITY**: Formation water salt content — affects polymer rheology.
- **ADSORPTION**: Polymer that sticks permanently to rock (10 micrograms per gram of rock).
- **IPV**: 10% of pore volume cannot be accessed by polymer molecules (they are too big for very small pores).

```python
INJ_RATE_MAX  = 750.0    # STB/day
OPR_SCALE     = 500.0    # STB/day
TOTAL_DAYS    = 1705.0   # days (May 2005 to Dec 2009)
N_TIMESTEPS   = 57       # monthly time steps
```
Scaling constants used to normalise inputs and outputs to the [0, 1] range. Neural networks train better on normalised numbers.

```python
CMG_R2    = {'P1': 0.9987, 'P2': 0.9960, 'P3': 0.9906}
CMG_NRMSE = {'P1': 0.0119, 'P2': 0.0216, 'P3': 0.0317}
CMG_CUM   = {'P1': 0.0014, 'P2': 0.0292, 'P3': 0.0366}
WC_INITIAL = {'P1': 0.168, 'P2': 0.168, 'P3': 0.168}
WC_FINAL   = {'P1': 0.606, 'P2': 0.598, 'P3': 0.605}
```
These are the **benchmark targets** from the manuscript (Tables 6 & 7). P1, P2, P3 are the three production wells.
- **R²** (R-squared) = how well the model matches the data. 1.0 = perfect.
- **NRMSE** = Normalised Root Mean Squared Error. 0.0119 means 1.19% average error.
- **WC_INITIAL/WC_FINAL** = water cut (fraction of produced fluid that is water) at start and end of production.

These numbers are targets to compare against after training. They tell us: the manuscript achieved R² = 0.9987 — can our PINN match that?

```python
LAMBDA_R   = 0.5   # weight on Buckley-Leverett residual
LAMBDA_IC  = 0.5   # weight on initial condition
LAMBDA_BC  = 0.5   # weight on boundary condition
LAMBDA_P   = 0.1   # weight on penalty terms
```
**Loss weights** — the training tries to minimise a total "error score" called the loss. This loss is a sum of several parts. The lambdas control how important each part is. LAMBDA_R = 0.5 means the physics residual contributes with weight 0.5 compared to the data loss (which has weight 1.0).

```python
EPOCHS     = 500   # maximum training passes through the data
BATCH_SIZE = 64    # data points processed per gradient step
PATIENCE   = 50    # stop if no improvement after 50 epochs
LR         = 1e-3  # learning rate for the main model
LR_SW      = 1e-3  # learning rate for the saturation submodel
N_COLLOC   = 1000  # number of physics collocation points per epoch
N_IC       = 100   # points for initial condition enforcement
N_BC       = 100   # points for boundary condition enforcement
```
**What is an epoch?** One full pass through all training data. With 500 epochs, we show the model the data up to 500 times.

**What is a batch?** Instead of feeding all data at once, we feed 64 rows at a time. This is faster and the noise actually helps the model generalise.

**What is a learning rate?** How big a step the model takes when updating its weights. 1e-3 = 0.001. Too large: overshoots, never converges. Too small: takes forever. This value is tried-and-true for Adam optimizer.

**What are collocation points?** Random points in (x, time, Cp) space where we check whether the physics equations are satisfied. We do not need data at these points — we just evaluate the PDE residual.

```python
N_CASES = 200
N_TRAIN = 140
N_VAL   = 30
N_TEST  = 30
```
200 CMG cases split: 140 for training, 30 for validation (tuning), 30 for final testing. 140+30+30 = 200. ✓

---

## 4. Custom Layers

### What is a Keras Layer?

A **layer** in a neural network is one processing step. Data flows in → computation happens → result flows out. Keras has many built-in layers, but we can build custom ones that encode specific physics.

Think of a layer like a station on an assembly line. Each station transforms the product in some way.

---

### 4.1 SwScaleLayer — Clamping Water Saturation

```python
class SwScaleLayer(layers.Layer):
    """Scale sigmoid output [0,1] to [SWC, 1-SOR]."""

    def __init__(self, swc=SWC, sor=SOR, **kw):
        super().__init__(**kw)
        self.swc = swc
        self.lo  = swc         # lower bound = 0.23
        self.hi  = 1.0 - sor   # upper bound = 0.80

    def call(self, s):
        return s * (self.hi - self.lo) + self.lo
```

**The problem:** The neural network outputs a number between 0 and 1 (using `sigmoid` activation). But physically, water saturation cannot go below SWC (0.23) or above 1−SOR (0.80). The raw network output needs to be stretched to the correct physical range.

**The formula:**
```
Sw = s × (0.80 − 0.23) + 0.23
   = s × 0.57 + 0.23
```
- If s = 0 (minimum sigmoid output): Sw = 0.23 = SWC ✓
- If s = 1 (maximum sigmoid output): Sw = 0.80 = 1−SOR ✓
- If s = 0.5 (middle): Sw = 0.515 ✓

This is a **linear scaling** — simple, no new parameters, just rescales the range. It guarantees the physics constraint is hard-coded.

```python
    def compute_output_shape(self, input_shape):
        return input_shape
```
This tells Keras: "My output has the same shape as my input. I just change the values, not the dimensions."

```python
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'swc': self.swc, 'sor': SOR})
        return cfg
```
This method allows the layer to be saved to disk and reloaded. Keras needs to know the parameters used to create the layer.

---

### 4.2 CMGViscosityLayer — Todd-Longstaff Polymer Viscosity

```python
class CMGViscosityLayer(layers.Layer):
    """
    Polymer solution viscosity — Todd-Longstaff model (Section 2.4).
    Calibrated: mu(1000 ppm) = 25 cp.
    """

    def __init__(self, mu_water=MU_WATER, **kw):
        super().__init__(**kw)
        self.mu_water = tf.constant(mu_water, dtype=tf.float32)
```

**What does this layer do?** Given a normalised polymer concentration (cp_norm = cp_ppm / 2000), it returns the effective viscosity of the polymer solution.

```python
    def call(self, cp_norm):
        cn = 2.0 * cp_norm
        mu_poly = self.mu_water * (1.0
                                   + 14.2 * cn
                                   +  8.5 * cn ** 2
                                   +  1.3 * cn ** 3)
```

**Why multiply by 2.0?** 
- `cp_norm = cp_ppm / CP_MAX_PPM = cp_ppm / 2000`
- The Todd-Longstaff model uses `cn = cp_ppm / 1000` (concentration in kg/m³ or normalised to 1000 ppm reference)
- So `cn = cp_ppm / 1000 = 2 × (cp_ppm / 2000) = 2 × cp_norm`

**The viscosity formula:**
```
μ_polymer = μ_water × (1 + 14.2×cn + 8.5×cn² + 1.3×cn³)
```
Let us verify this gives 25 cP at 1000 ppm:
- At 1000 ppm: cn = 1000/1000 = 1
- μ = 1 × (1 + 14.2×1 + 8.5×1² + 1.3×1³)
- μ = 1 + 14.2 + 8.5 + 1.3 = 25.0 cP ✓

At 0 ppm (no polymer): cn = 0, μ = 1 × 1 = 1 cP = pure water ✓

```python
        mask   = tf.cast(cp_norm > 1e-4, tf.float32)
        mu_eff = mask * mu_poly + (1.0 - mask) * self.mu_water
        return mu_eff
```

**The mask trick:** When cp_norm is essentially zero (less than 0.0001), we want exactly mu_water (not a slightly-off computed value with floating-point noise). 

- `cp_norm > 1e-4` → True (1) where polymer is present, False (0) where it is not
- `tf.cast(..., tf.float32)` → Convert True/False to 1.0/0.0
- `mu_eff = mask × mu_poly + (1-mask) × mu_water`
  - Where polymer exists: mu_eff = 1 × mu_poly + 0 × mu_water = mu_poly
  - Where no polymer: mu_eff = 0 × mu_poly + 1 × mu_water = mu_water

This avoids division by zero and numerical instability near cp = 0.

---

### 4.3 CoreyKrLayer — Relative Permeability (Trainable!)

```python
class CoreyKrLayer(layers.Layer):
    """Corey relative permeability — TRAINABLE parameters."""

    def __init__(self, swc=SWC, sor=SOR, **kw):
        super().__init__(**kw)
        self.swc_const = tf.constant(swc, dtype=tf.float32)
        self.sor_const = tf.constant(sor, dtype=tf.float32)
```

SWC and SOR are stored as **constants** (not trainable) — they are fixed from core analysis. The Corey shape parameters (krw_max, kro_max, nw, no) will be learnable.

```python
    def build(self, input_shape):
        self.krw_max = self.add_weight(
            name='krw_max', shape=(),
            initializer=tf.constant_initializer(KRW_MAX_INIT),  # start at 0.10
            trainable=True)
        self.kro_max = self.add_weight(
            name='kro_max', shape=(),
            initializer=tf.constant_initializer(KRO_MAX_INIT),  # start at 1.00
            trainable=True)
        self.nw = self.add_weight(
            name='nw', shape=(),
            initializer=tf.constant_initializer(NW_INIT),       # start at 3.0
            trainable=True)
        self.no = self.add_weight(
            name='no', shape=(),
            initializer=tf.constant_initializer(NO_INIT),       # start at 2.2
            trainable=True)
        super().build(input_shape)
```

**What does `build` mean?** Keras calls `build()` the first time data flows through a layer. It is where you create trainable weights (parameters that the optimizer will adjust during training).

**`self.add_weight(..., trainable=True)`** = "Create a number that the neural network can change during training, starting at the initialiser value."

So after training, krw_max might become 0.092 instead of 0.10 — it found a slightly better fit to the data.

```python
    def call(self, sw):
        krw_max = tf.clip_by_value(self.krw_max, 0.01, 1.0)
        kro_max = tf.clip_by_value(self.kro_max, 0.1,  1.0)
        nw      = tf.clip_by_value(self.nw,      1.0,  8.0)
        no      = tf.clip_by_value(self.no,      1.0,  8.0)
```

**`tf.clip_by_value(x, min, max)`** = "If x < min, use min. If x > max, use max. Otherwise use x."

Without clipping, the optimizer might push krw_max to -0.5 (negative permeability is physically impossible). Clipping enforces physical bounds.

```python
        sw_c  = tf.clip_by_value(sw, self.swc_const, 1.0 - self.sor_const)
        denom = 1.0 - self.sor_const - self.swc_const + 1e-8
```

- `sw_c` = water saturation clamped to the physical range [SWC, 1-SOR]
- `denom` = the normalisation denominator = 1 - 0.20 - 0.23 = 0.57. The `+ 1e-8` prevents division by zero.

```python
        sw_norm_w = tf.clip_by_value((sw_c - self.swc_const) / denom, 0.0, 1.0)
        sw_norm_o = tf.clip_by_value((1.0 - self.sor_const - sw_c) / denom, 0.0, 1.0)
```

These are the **normalised saturations** for the Corey formula:
- `sw_norm_w` = (Sw − Swc) / (1 − Swc − Sor) → how "full" of mobile water the pore is
- `sw_norm_o` = (1 − Sor − Sw) / (1 − Swc − Sor) → how "full" of mobile oil the pore is

Both range from 0 to 1.

```python
        krw = krw_max * tf.math.pow(sw_norm_w, nw)
        kro = kro_max * tf.math.pow(sw_norm_o, no)
        return tf.concat([krw, kro], axis=-1)
```

**The Corey power law:**
- `krw = 0.10 × (sw_norm_w)^3.0` — water relative permeability
- `kro = 1.00 × (sw_norm_o)^2.2` — oil relative permeability

`tf.concat([krw, kro], axis=-1)` = stack them side by side → output shape: (N, 2) where column 0 = krw, column 1 = kro.

**Example:** At Sw = 0.50:
- sw_norm_w = (0.50 − 0.23) / 0.57 = 0.47
- sw_norm_o = (1 − 0.20 − 0.50) / 0.57 = 0.53
- krw = 0.10 × 0.47³ = 0.10 × 0.104 = 0.0104
- kro = 1.00 × 0.53^2.2 = 0.258

Oil flows much better than water at this saturation — consistent with heavy oil behaviour.

---

### 4.4 FractionalFlowLayer — How Much of the Flow is Water?

```python
class FractionalFlowLayer(layers.Layer):
    """
    fw = mob_w / (mob_w + mob_o)
    RK = RRF = 2.0 (Residual Resistance Factor, Section 2.4)
    """

    def __init__(self, mu_oil=MU_OIL, rk=RK, **kw):
        super().__init__(**kw)
        self.mu_oil = tf.constant(mu_oil, dtype=tf.float32)   # 1650 cP
        self.rk     = tf.constant(rk,     dtype=tf.float32)   # 2.0
```

```python
    def call(self, inputs):
        krw, kro, mu_eff = inputs[0], inputs[1], inputs[2]
        mob_w = krw / (mu_eff * self.rk + 1e-8)
        mob_o = kro / (self.mu_oil      + 1e-8)
        fw    = mob_w / (mob_w + mob_o  + 1e-8)
        return fw
```

**Mobility** = how easily a fluid moves = kr / μ (relative permeability divided by viscosity).

- `mob_w` = krw / (μ_polymer × RRF) = water mobility
  - Note: RRF = 2.0 doubles the effective resistance, halving water mobility
  - This is the Residual Resistance Factor from polymer adsorption
- `mob_o` = kro / μ_oil = oil mobility

**Fractional flow** (fw) = the fraction of total flow that is water:
```
fw = mob_w / (mob_w + mob_o)
```

**Example with no polymer:**
- μ_eff = 1 cP, RRF = 2.0, at Sw = 0.50: mob_w = 0.0104 / (1 × 2) = 0.0052
- mob_o = 0.258 / 1650 = 0.000156
- fw = 0.0052 / (0.0052 + 0.000156) = 0.97

So 97% of the produced fluid is water! This reflects the severe water channelling problem in heavy oil reservoirs — the oil is so viscous that water races through it.

**With 1000 ppm polymer:**
- μ_eff = 25 cP: mob_w = 0.0104 / (25 × 2) = 0.000208
- mob_o = 0.000156 (unchanged, polymer does not affect oil)
- fw = 0.000208 / (0.000208 + 0.000156) = 0.57

Polymer dropped water cut from 97% to 57% at this saturation. This is the physics of polymer flooding.

The `+ 1e-8` terms prevent division by zero in edge cases.

---

## 5. SaturationSubmodel

```python
class SaturationSubmodel(keras.Model):
    """
    Predicts Sw(xD, tD, cp_norm, inj_norm).
    Input shape: (N, 4)   Output shape: (N, 1)  ∈ [SWC, 1-SOR]
    """
```

This is a **sub-neural-network** dedicated to one job: predict the water saturation at any location and time, given the conditions.

**Inputs (4 numbers):**
1. `xD` — dimensionless position (0 = injector, 1 = producer)
2. `tD` — dimensionless time (0 = start, 1 = end of field life)
3. `cp_norm` — normalised polymer concentration (0 = no polymer, 1 = 2000 ppm)
4. `inj_norm` — normalised injection rate

**Output (1 number):**
- `Sw` — water saturation ∈ [0.23, 0.80]

```python
    def __init__(self, swc=SWC, sor=SOR, **kw):
        super().__init__(**kw)
        self.bn0    = layers.BatchNormalization(name='sw_bn0')
        setattr(self, 'sw_d1', layers.Dense(128, activation='tanh',  name='sw_d1'))
        setattr(self, 'sw_d2', layers.Dense(256, activation='tanh',  name='sw_d2'))
        setattr(self, 'sw_d3', layers.Dense(256, activation='tanh',  name='sw_d3'))
        setattr(self, 'sw_d4', layers.Dense(128, activation='tanh',  name='sw_d4'))
        setattr(self, 'sw_d5', layers.Dense(64,  activation='tanh',  name='sw_d5'))
        self.sw_dr2  = layers.Dropout(0.05, name='sw_dr2')
        self.sw_dr3  = layers.Dropout(0.05, name='sw_dr3')
        self.sw_dr4  = layers.Dropout(0.05, name='sw_dr4')
        self.sw_out  = layers.Dense(1, activation='sigmoid', name='sw_out')
        self.sw_scale = SwScaleLayer(swc=swc, sor=sor, name='sw_scale')
```

**Why `setattr`?** Keras tracks layers by looking at the model's attributes. Normally you write `self.sw_d1 = layers.Dense(...)` — that works. But with a loop or complex naming, Keras might miss some layers. `setattr(self, 'sw_d1', layers.Dense(...))` is exactly equivalent — it sets `self.sw_d1` — but is explicit enough that Keras always detects it. It is a safety pattern.

**The architecture:**
```
Input (4) → BatchNorm → Dense(128,tanh) → Dense(256,tanh) → Dropout(5%)
         → Dense(256,tanh) → Dropout(5%) → Dense(128,tanh) → Dropout(5%)
         → Dense(64,tanh) → Dense(1,sigmoid) → SwScaleLayer → Sw ∈ [0.23,0.80]
```

**What are these layers?**

**`BatchNormalization`** — normalises the input data so all features have similar scale. Think of it as saying "make all the numbers comparable" before feeding them to the network. Without this, a feature ranging [0, 2000] might dominate over one ranging [0, 1].

**`Dense(128, activation='tanh')`** — a fully connected layer with 128 neurons and `tanh` activation.
- A neuron computes: `output = tanh(w₁×input₁ + w₂×input₂ + ... + bias)`
- `tanh` squashes the output to (−1, +1). Good for physics problems because it has a smooth, continuous derivative (needed for the PDE residual computation).
- 128 neurons = 128 such computations in parallel.

**`Dropout(0.05)`** — randomly turns off 5% of neurons during each training step. This is **regularisation** — it prevents the model from memorising the training data without generalising. Think of it like: if you always study with all neurons, you memorise the notes. Dropout forces you to learn the concept, not the specific words.

**`Dense(1, activation='sigmoid')`** — the final neuron outputs a single number in (0, 1).

**`SwScaleLayer`** — rescales from (0,1) to [SWC, 1−SOR] = [0.23, 0.80].

```python
    def call(self, inputs, training=False):
        x = self.bn0(inputs, training=training)
        x = self.sw_d1(x)
        x = self.sw_d2(x)
        x = self.sw_dr2(x, training=training)
        ...
        x = self.sw_scale(x)
        return x
```

`call()` is the forward pass — data flows through layers in order. The `training=training` argument matters for BatchNorm and Dropout — they behave differently during training vs. inference:
- During training: BatchNorm uses the batch statistics; Dropout randomly drops neurons.
- During inference: BatchNorm uses stored running statistics; Dropout is disabled.

---

## 6. PolymerPINN

This is the **main model** — the complete PINN that predicts oil production rate and water cut.

```python
class PolymerPINN(keras.Model):
    """
    Full PINN: trunk + physics path + production head.
    Inputs : (t_hat, cp_norm, inj_norm, bhp_norm)  shape (N, 4)
    Outputs: (OPR_field_norm, WC_field)             shape (N, 2)
    """
```

**Inputs (4 numbers per time step):**
1. `t_hat` — normalised time (0 to 1)
2. `cp_norm` — normalised polymer concentration
3. `inj_norm` — normalised injection rate
4. `bhp_norm` — normalised bottom-hole pressure

**Outputs (2 numbers per time step):**
1. `OPR_norm` — normalised oil production rate
2. `WC` — water cut (fraction)

```python
    def __init__(self, sw_submodel: SaturationSubmodel, **kw):
        super().__init__(**kw)
        self.sw_submodel  = sw_submodel     # the saturation brain
        self.visc_layer   = CMGViscosityLayer(name='visc_layer')   # viscosity
        self.kr_layer     = CoreyKrLayer(name='kr_layer')           # permeability
        self.fw_layer     = FractionalFlowLayer(name='fw_layer')    # fractional flow
```

The PolymerPINN **owns** the SaturationSubmodel and the three physics layers. This embedding is what makes it a PINN — the physics layers are not external functions, they are part of the model.

```python
        # Trunk network
        self.trunk_bn  = layers.BatchNormalization(name='trunk_bn')
        setattr(self, 'trunk_d1', layers.Dense(128, activation='tanh', ...))
        ...
        setattr(self, 'trunk_d5', layers.Dense(64, activation='tanh',  ...))
```

The **trunk** is another deep neural network. It processes the raw inputs (time, Cp, injection rate, BHP) into a rich feature representation (64 numbers). Think of it as a "context-understanding" network.

```python
        # Production head
        setattr(self, 'head_d1', layers.Dense(64,  activation='tanh',    ...))
        setattr(self, 'head_d2', layers.Dense(32,  activation='tanh',    ...))
        setattr(self, 'qo_out',  layers.Dense(1,   activation='softplus', ...))
        setattr(self, 'alpha_d', layers.Dense(1,   activation='sigmoid',  ...))
        setattr(self, 'wc_raw',  layers.Dense(1,                          ...))
```

The **head** combines physics outputs with trunk features to produce final predictions.

- `qo_out` uses `softplus` activation: `softplus(x) = log(1 + e^x)`. This is always positive (oil production rate cannot be negative). Like `relu` but smooth.
- `alpha_d` outputs a blending weight α ∈ (0, 1)
- `wc_raw` outputs an unconstrained number

### The forward pass:

```python
    def call(self, inputs, training=False):
        t_hat    = tf.expand_dims(inputs[:, 0], axis=-1)
        cp_norm  = tf.expand_dims(inputs[:, 1], axis=-1)
        inj_norm = tf.expand_dims(inputs[:, 2], axis=-1)
        bhp_norm = tf.expand_dims(inputs[:, 3], axis=-1)
```

`inputs` arrives as shape (N, 4) — N rows, 4 columns. We split it into 4 separate columns, each shape (N, 1). `tf.expand_dims(..., axis=-1)` adds the trailing dimension (changes shape from (N,) to (N,1)).

```python
        # Physics path
        mu_eff  = self.visc_layer(cp_norm)
```
Compute effective viscosity from polymer concentration.

```python
        sw_inp  = tf.concat([
            tf.fill(tf.shape(t_hat), 0.5),   # xD = 0.5 (midpoint)
            t_hat, cp_norm, inj_norm
        ], axis=-1)
        sw_hat  = self.sw_submodel(sw_inp, training=training)
```

Call the saturation submodel at xD = 0.5 (midpoint of the reservoir). `tf.fill(shape, value)` creates a tensor filled with the same value — here, 0.5 for every row.

We concatenate [xD=0.5, tD, Cp, inj] into a (N,4) input for the saturation model. It outputs Sw ∈ [0.23, 0.80].

Why xD = 0.5? The PolymerPINN operates at the field scale (time series of production), not spatially resolved. We use the midpoint saturation as a representative average.

```python
        kr_out  = self.kr_layer(sw_hat)
        krw     = tf.expand_dims(kr_out[:, 0], axis=-1)
        kro     = tf.expand_dims(kr_out[:, 1], axis=-1)
        fw      = self.fw_layer([krw, kro, mu_eff])
```

Compute kr (Corey) and fractional flow from the physics layers. Now `fw` is the physically-computed water cut fraction.

```python
        # Trunk
        trunk_out = self._trunk(inputs, training=training)
```

Run the raw inputs through the trunk network.

```python
        # Head
        combined = tf.concat([trunk_out, sw_hat, fw, mu_eff], axis=-1)
        h = self.head_d1(combined)
        h = self.head_d2(h)

        qo    = self.qo_out(h)
        alpha = self.alpha_d(h)
        wc_nn = self.wc_raw(h)
        wc    = tf.sigmoid(alpha * wc_nn + (1.0 - alpha) * fw)
```

**The key design decision:** Water cut is a **blend** of the physics prediction (fw) and the neural network prediction (wc_nn):
```
wc = sigmoid(α × wc_nn + (1−α) × fw)
```
- `alpha` is learned during training (0 to 1)
- If α = 0: pure physics fractional flow
- If α = 1: pure neural network
- If α = 0.5: 50/50 blend

This design lets the model be close to physics while having flexibility to correct for field-scale heterogeneity that the simple 1D Buckley-Leverett equation cannot capture.

```python
        return tf.concat([qo, wc], axis=-1)
```

Stack oil rate and water cut into shape (N, 2) — the final output.

---

## 7. Loading Data

```python
def load_data(data_dir=DATA_DIR):
```

This function loads your 200 CMG simulation cases from CSV files.

```python
    oil_path  = os.path.join(data_dir, 'Oil_Production.csv')
    wc_path   = os.path.join(data_dir, 'Water_cut.csv')
    cp_path   = os.path.join(data_dir, 'Polymer_concentration.csv')
    inj1_path = os.path.join(data_dir, 'Injection_rate_inj1.csv')
    inj2_path = os.path.join(data_dir, 'Injection_rate_inj2.csv')
```

Five CSV files — each has 57 rows (monthly time steps) and 200 columns (cases).

```python
    missing = [p for p in [oil_path, wc_path, cp_path, inj1_path, inj2_path]
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Required CSV files not found: {missing}\n"
            f"Place your 200 CMG cases in: {os.path.abspath(data_dir)}/")
```

**Fail loudly** — if any CSV file is missing, immediately raise an error with a clear message telling you where to put the files. The code NEVER creates synthetic data as a fallback. You must provide the real CMG cases.

```python
    oil_df  = pd.read_csv(oil_path)
    wc_df   = pd.read_csv(wc_path)
    cp_df   = pd.read_csv(cp_path)
    inj1_df = pd.read_csv(inj1_path)
    inj2_df = pd.read_csv(inj2_path)
```

`pd.read_csv()` reads a CSV file into a **DataFrame** — think of it as a table in Python. Rows = time steps, columns = different variables or cases.

```python
    if 'time' in oil_df.columns:
        t_vals = oil_df['time'].values.astype(np.float32)
        t_hat  = (t_vals - t_vals[0]) / (t_vals[-1] - t_vals[0] + 1e-8)
    else:
        t_hat  = np.linspace(0.0, 1.0, N_TIMESTEPS, dtype=np.float32)
```

If the CSV has a 'time' column, use it. Normalise it to [0, 1]:
```
t_hat = (t - t_min) / (t_max - t_min)
```
Otherwise create 57 evenly spaced values from 0 to 1.

```python
    q1_series = inj1_df[inj1_cols[0]].values.astype(np.float32)
    q2_series = inj2_df[inj2_cols[0]].values.astype(np.float32)
    total_inj  = q1_series + q2_series
    inj_norm   = (total_inj / INJ_RATE_MAX).astype(np.float32)
```

Two injection wells (inj1 and inj2) — their rates are summed and normalised by 750 STB/day.

```python
    for ci in range(1, N_CASES + 1):   # ci = 1, 2, 3, ..., 200
```

Loop through all 200 cases.

```python
        oil_cols = get_case_cols_A(oil_df, ci)
        if not oil_cols:
            oil_cols = get_case_col_B(oil_df, ci)
        opr_arr = oil_df[oil_cols].values.astype(np.float32).sum(axis=1)
```

**Two CSV layouts supported:**
- **Layout A**: columns named `case_001_P1`, `case_001_P2`, `case_001_P3` (per-well data) → sum all 3 producers
- **Layout B**: single column named `case_001` (already totalled)

`sum(axis=1)` = sum along columns (adding P1+P2+P3 for each time step).

```python
        wc_arr = wc_df[wc_cols].values.astype(np.float32).mean(axis=1)
```

Water cut is **averaged** across the wells (not summed — it is a fraction, not a rate).

```python
        cp_val = float(cp_series[0])   # normalised [0, 1]
```

Polymer concentration is constant for each case (each CMG case uses one Cp value throughout). Take the first value — they are all the same.

```python
        bhp_fixed = np.full(t_steps, 0.5, dtype=np.float32)
        cp_col    = np.full(t_steps, cp_val, dtype=np.float32)

        X_case = np.column_stack([t_hat, cp_col, inj_norm, bhp_fixed])
        y_case = np.column_stack([opr_arr / OPR_SCALE,
                                  np.clip(wc_arr, 0.0, 1.0)])
```

For each case, build a matrix:
- **X_case**: shape (57, 4) — time, cp, inj, BHP for each of the 57 months
- **y_case**: shape (57, 2) — oil rate (normalised), water cut (clipped to [0,1])

`np.clip(wc_arr, 0.0, 1.0)` = ensure water cut is between 0% and 100%.

```python
    X_all = np.vstack(X_list).astype(np.float32)   # (200*57, 4) = (11400, 4)
    y_all = np.vstack(y_list).astype(np.float32)   # (200*57, 2) = (11400, 2)
```

Stack all 200 cases vertically: 200 × 57 = **11,400 rows** total.

```python
    rng    = np.random.default_rng(SEED)
    perm   = rng.permutation(N_CASES)
    tr_ids = sorted(perm[:N_TRAIN])         # first 140 (shuffled)
    va_ids = sorted(perm[N_TRAIN:N_TRAIN+N_VAL])  # next 30
    te_ids = sorted(perm[N_TRAIN+N_VAL:])  # last 30
```

**Case-level random split** — shuffle the 200 case indices and assign:
- 140 cases to training
- 30 to validation
- 30 to test

This is important: we split by **case**, not by row. If we split by row, the model might train on time step 45 of case 7 and test on time step 46 of case 7 — that would be cheating (they are the same experiment). Case-level split ensures test cases are completely unseen.

```python
    def idx_for_cases(case_ids):
        idxs = []
        for c in case_ids:
            start = c * t_steps       # first row of this case
            idxs.extend(range(start, start + t_steps))   # all 57 rows
        return np.array(idxs)
```

For a given list of case IDs, find the row indices in X_all that belong to those cases. Case c occupies rows `[c × 57, c × 57 + 57)`.

```python
    assert X_tr.shape[1] == 4, 'Expected 4 inputs: t, cp, inj, bhp'
    assert y_tr.shape[1] == 2, 'Expected 2 outputs: OPR_norm, WC'
```

**Assertions** are sanity checks. If the shapes are wrong, Python raises an error immediately with a clear message. Better to fail early than to train for 10 hours on wrong data.

---

## 8. Physics Loss Functions

This is where the PINN differs from a regular neural network. We have three physics constraints to enforce.

### 8.1 Buckley-Leverett Residual

```python
def compute_bl_residual(sw_model, kr_layer, fw_layer, visc_layer,
                        xD, tD, cp_norm_colloc, inj_colloc):
    """
    Buckley-Leverett residual: ∂Sw/∂tD + (dfw/dSw) * ∂Sw/∂xD = 0
    """
```

The Buckley-Leverett PDE says (in dimensionless form):
```
∂Sw/∂tD + (dfw/dSw) × ∂Sw/∂xD = 0
```

This means: the rate of change of water saturation in time plus the wave speed (dfw/dSw, called the fractional flow derivative) times the spatial gradient must equal zero. This is conservation of mass.

To check this, we need **derivatives** of a neural network's output with respect to its inputs. This is done using **automatic differentiation**.

```python
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([xD, tD])
```

**`tf.GradientTape`** is TensorFlow's automatic differentiation tool. Imagine it as a recording device:
- "Start recording every computation that involves xD and tD."
- `tape2.watch([xD, tD])` = explicitly tell the tape to track these variables.
- `persistent=True` = keep the recording after one gradient computation (we need multiple gradients).

```python
        sw_inp = tf.concat([xD, tD, cp_norm_colloc, inj_colloc], axis=-1)
        sw     = sw_model(sw_inp, training=True)
```

Build the input and run the saturation model. **Critically:** the input is built INSIDE the `with tape2:` block. This ensures the tape records: xD → sw_inp → sw. Without this, the tape cannot trace the chain from xD to sw and would return None for the gradient.

```python
        kr_out = kr_layer(sw)
        krw    = tf.expand_dims(kr_out[:, 0], axis=-1)
        kro    = tf.expand_dims(kr_out[:, 1], axis=-1)
        mu_eff = visc_layer(cp_norm_colloc)
        fw     = fw_layer([krw, kro, mu_eff])
```

Also computed inside the tape: sw → kr → fw. The tape records the entire chain.

```python
    dSw_dtD = tape2.gradient(sw, tD)
    dfw_dSw = tape2.gradient(fw, sw)
    dSw_dxD = tape2.gradient(sw, xD)
    del tape2
```

Now ask the tape: "What is the derivative of sw with respect to tD?" etc.

- `dSw_dtD` = ∂Sw/∂tD (how fast saturation changes in time)
- `dfw_dSw` = dfw/dSw (the fractional flow derivative — the Buckley-Leverett wave speed)
- `dSw_dxD` = ∂Sw/∂xD (how saturation varies in space)

`del tape2` = free the memory. The `persistent` tape does not automatically delete itself.

```python
    if dSw_dtD is None: dSw_dtD = tf.zeros_like(tD)
    if dfw_dSw is None: dfw_dSw = tf.zeros_like(sw)
    if dSw_dxD is None: dSw_dxD = tf.zeros_like(xD)
```

Safety: if TensorFlow cannot compute a gradient (rare edge case — e.g., constant output), replace None with zeros so the code does not crash.

```python
    residual = dSw_dtD + dfw_dSw * dSw_dxD
    return tf.reduce_mean(tf.square(residual))
```

**The PDE residual:** If the Buckley-Leverett equation is satisfied exactly:
```
∂Sw/∂tD + dfw/dSw × ∂Sw/∂xD = 0
```
So the residual = 0. We square it (always positive) and take the mean over all collocation points. The training will drive this toward zero.

---

### 8.2 Initial Condition Loss

```python
def compute_ic_loss(sw_model, n_ic=N_IC):
    """
    IC: Sw(xD, tD=0) = SWINITIAL = 0.36  (FWM=0.12 incorporated).
    """
    xD_ic  = tf.random.uniform((n_ic, 1), 0.0, 1.0)
    tD_ic  = tf.zeros((n_ic, 1))    # tD = 0 (start of time)
    cp_ic  = tf.random.uniform((n_ic, 1), 0.0, 1.0)
    inj_ic = tf.random.uniform((n_ic, 1), 0.0, 1.0)
```

At time zero (tD = 0), the reservoir has not been flooded yet. The initial water saturation everywhere must be **SWINITIAL = 0.36** (from FWM model). 

We sample 100 random points in space (xD) and concentration (Cp) — all at tD = 0 — and check that the model predicts 0.36 at all of them.

```python
    sw_inp = tf.concat([xD_ic, tD_ic, cp_ic, inj_ic], axis=-1)
    sw_ic  = sw_model(sw_inp, training=True)
    target = tf.fill(tf.shape(sw_ic), SWINITIAL)   # 0.36 for every point
    return tf.reduce_mean(tf.square(sw_ic - target))
```

The loss = mean squared error between predicted Sw and the target 0.36. Training will push the model to always predict 0.36 at t=0.

---

### 8.3 Boundary Condition Loss

```python
def compute_bc_loss(sw_model, n_bc=N_BC):
    """BC: Sw(xD=0, tD) = 1 - SOR = 0.80  (injector face)."""
    xD_bc  = tf.zeros((n_bc, 1))   # xD = 0 (injector face)
    tD_bc  = tf.random.uniform((n_bc, 1), 0.0, 1.0)
```

At the injection face (xD = 0), we are injecting polymer water. The rock at the injector is fully swept — water saturation there should be at maximum = 1 − SOR = 0.80.

```python
    target = tf.fill(tf.shape(sw_bc), 1.0 - SOR)   # 0.80
    return tf.reduce_mean(tf.square(sw_bc - target))
```

Same idea as IC loss — mean squared error from the target.

---

### 8.4 Penalty Function

```python
def compute_penalty(y_pred):
    """Positivity and boundedness penalties."""
    qo = tf.expand_dims(y_pred[:, 0], axis=-1)
    wc = tf.expand_dims(y_pred[:, 1], axis=-1)
    p1 = tf.reduce_mean(tf.square(tf.nn.relu(-qo)))     # penalise qo < 0
    p2 = tf.reduce_mean(tf.square(tf.nn.relu(-wc)))     # penalise wc < 0
    p3 = tf.reduce_mean(tf.square(tf.nn.relu(wc - 1.0)))  # penalise wc > 1
    return p1 + p2 + p3
```

**`tf.nn.relu(x)`** = max(0, x). It returns the value if positive, zero otherwise.

- `relu(-qo)` is positive only when qo is negative (impossible physically)
- `relu(-wc)` is positive only when wc is negative (water cut < 0% is impossible)
- `relu(wc - 1.0)` is positive only when wc > 1 (water cut > 100% is impossible)

Squaring and averaging these creates a soft penalty — the model is not hard-blocked from these values, but there is a cost. Combined with `softplus` activation on qo, this reinforces physical bounds.

---

## 9. The Training Loop

```python
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10
    return float(1.0 - ss_res / ss_tot)
```

**R² (R-squared)** = the fraction of variance in the data explained by the model.
- R² = 1.0: perfect prediction
- R² = 0.0: model is no better than predicting the mean
- R² < 0: model is worse than predicting the mean (very bad)

Formula: `R² = 1 − SS_res / SS_tot`
- `SS_res` = sum of squared errors (residuals)
- `SS_tot` = total variance of the true data

---

```python
def train(model: PolymerPINN, sw_model: SaturationSubmodel,
          X_tr, y_tr, X_va, y_va):
```

The training function takes the two models and the train/validation data.

```python
    model_opt = keras.optimizers.Adam(learning_rate=LR)
    sw_opt    = keras.optimizers.Adam(learning_rate=LR_SW)
```

**Two separate optimizers** — one for the main PolymerPINN, one for the SaturationSubmodel. This is because the saturation model is trained with physics loss while the main model is trained with data loss. Keeping them separate gives finer control.

**Adam (Adaptive Moment Estimation)** is the most popular neural network optimizer. It automatically adjusts the step size for each parameter based on its history. Think of it as a smart learning rate that goes faster on highways (smooth loss surfaces) and slower on winding roads (noisy gradients).

```python
    dataset = tf.data.Dataset.from_tensor_slices((X_tr_tf, y_tr_tf))
    dataset = dataset.shuffle(buffer_size=8000, seed=SEED).batch(BATCH_SIZE)
```

**`tf.data.Dataset`** is TensorFlow's efficient data pipeline.
- `from_tensor_slices` = treat each row as one sample
- `.shuffle(buffer_size=8000)` = randomly reorder rows before each epoch (prevents the model from learning the order)
- `.batch(64)` = group rows into mini-batches of 64

Each time through the dataset, we get batches of 64 rows.

```python
    for epoch in range(1, EPOCHS + 1):
```

The outer loop: repeat training up to 500 times (epochs).

```python
        # ── Step 1: Physics update ─────────────────────────────────────────
        xD_c  = tf.Variable(tf.random.uniform((N_COLLOC, 1), 0.0, 1.0))
        tD_c  = tf.Variable(tf.random.uniform((N_COLLOC, 1), 0.0, 1.0))
        cp_c  = tf.Variable(tf.random.uniform((N_COLLOC, 1), 0.0, 1.0))
        inj_c = tf.Variable(tf.random.uniform((N_COLLOC, 1), 0.0, 1.0))
```

**Collocation points** — 1000 random locations in the (xD, tD, Cp, inj) space. These are NOT from your CSV data. They are randomly generated to check PDE satisfaction across the entire space.

Why `tf.Variable` instead of `tf.constant`? The GradientTape can differentiate with respect to `tf.Variable` by default. For `tf.Tensor` (constant), we need `tape.watch()` explicitly (done inside `compute_bl_residual`).

```python
        with tf.GradientTape() as phys_tape:
            L_BL = compute_bl_residual(...)
            L_IC = compute_ic_loss(sw_model, N_IC)
            L_BC = compute_bc_loss(sw_model, N_BC)
            phys_loss = LAMBDA_R * L_BL + LAMBDA_IC * L_IC + LAMBDA_BC * L_BC
```

Compute all three physics losses and combine them (weighted sum).

```python
        phys_grads = phys_tape.gradient(phys_loss, sw_model.trainable_variables)
        sw_opt.apply_gradients(zip(phys_grads, sw_model.trainable_variables))
```

**The gradient step:**
1. `tape.gradient(loss, variables)` = compute ∂loss/∂weight for every weight in sw_model
2. `apply_gradients(zip(grads, vars))` = update each weight: `w ← w − lr × ∂loss/∂w`

This is **gradient descent** — move the weights in the direction that reduces the physics loss.

**`zip(phys_grads, sw_model.trainable_variables)`** — pairs each gradient with its corresponding weight. The optimizer then subtracts α×gradient from each weight.

```python
        # ── Step 2: Data update ────────────────────────────────────────────
        for X_b, y_b in dataset:
            with tf.GradientTape() as data_tape:
                y_pred  = model(X_b, training=True)
                mse_opr = tf.reduce_mean(tf.square(y_pred[:, 0] - y_b[:, 0]))
                mse_wc  = tf.reduce_mean(tf.square(y_pred[:, 1] - y_b[:, 1]))
                L_data  = mse_opr + mse_wc
                L_pen   = compute_penalty(y_pred)
                total_data = L_data + LAMBDA_P * L_pen
```

For each mini-batch from your CMG data:
- Predict OPR and WC
- Compute MSE for each
- Add penalty for physical bound violations
- Record total data loss

```python
            data_grads = data_tape.gradient(total_data, model.trainable_variables)
            model_opt.apply_gradients(zip(data_grads, model.trainable_variables))
```

Update ALL weights of the PolymerPINN model to reduce the data loss.

Note: the data update touches ALL of model's weights (including the sw_submodel embedded inside), while the physics update only touches sw_model's weights. This two-step training couples the two models.

```python
        # ── Validation ─────────────────────────────────────────────────────
        y_va_pred = model(X_va_tf, training=False).numpy()
        val_mse   = float(np.mean((y_va_pred - y_va) ** 2))
```

After each epoch, evaluate on the **validation set** (data the model has never been trained on). `training=False` disables Dropout.

```python
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            wait = 0
            model.save_weights(best_w_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break
```

**Early stopping** — if the validation loss does not improve by at least 1e-6 for 50 consecutive epochs, stop training. This prevents **overfitting** (when the model memorises the training data instead of learning general patterns).

The best weights are saved to disk every time we find a new best. If training stops early, we load the best weights — not the final ones (which may have started overfitting).

```python
    if os.path.exists(best_w_path):
        model.load_weights(best_w_path)
```

Restore the best weights after training.

```python
    hist_path = os.path.join(OUTPUT_DIR, 'pinn_polymer_history.json')
    with open(hist_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.items()}, f)
```

Save the training history (all loss values per epoch) as a JSON file. `[float(v) for v in vals]` converts numpy floats to Python floats — JSON cannot serialise numpy types directly.

---

## 10. Evaluation

```python
def evaluate(model, X_te, y_te):
    y_pred = model(tf.constant(X_te, dtype=tf.float32), training=False).numpy()

    opr_r2 = r_squared(y_te[:, 0], y_pred[:, 0])
    wc_r2  = r_squared(y_te[:, 1], y_pred[:, 1])
    print(f'\n[EVAL] Test OPR R²={opr_r2:.4f}  WC R²={wc_r2:.4f}')
```

Run the model on the 30 test cases (never seen during training) and compute R² for oil rate and water cut.

```python
    print('\n[CMG STARS Targets — Table 6 (manuscript)]')
    for w in ['P1', 'P2', 'P3']:
        print(f'  {w}: R²={CMG_R2[w]}  NRMSE={CMG_NRMSE[w]:.4f}  ...')
```

Print the manuscript targets alongside your results for comparison. If your R² is near 0.9987, the PINN is matching CMG STARS quality.

---

## 11. Visualisation

Four plots are generated:

### `plot_fractional_flow` — The Foundation Curve

Shows fw vs. Sw for different polymer concentrations (0, 500, 1000, 1500, 2000 ppm). This is the physical foundation of displacement efficiency. You can see:
- Without polymer: fractional flow rises steeply (poor sweep)
- With 2000 ppm polymer: the curve shifts — more oil-favourable
- Vertical dashed line at Swinitial = 0.36 shows the starting condition (FWM model)

### `plot_training_history` — How Well Did Training Go?

Four subplots showing how each loss component evolved over epochs:
- Total loss and data loss should decrease smoothly
- BL residual should decrease (physics being satisfied better)
- IC and BC losses should decrease (initial/boundary conditions respected)
- Validation loss should parallel training loss (no overfitting if they track together)

### `plot_parity` — Predicted vs. Actual

Two scatter plots (OPR and WC):
- X-axis: CMG STARS values (ground truth)
- Y-axis: PINN predictions
- Red dashed line: perfect prediction (y = x)
- Points clustered on the red line = good model

### `plot_timeseries` — Time Evolution for Specific Cases

For 3 test cases, shows the monthly time series of OPR and WC:
- Blue circles: CMG STARS (truth)
- Red dashed: PINN prediction
- Good overlap means the model captures the dynamics

### `plot_saturation_profiles` — Where is the Water?

Shows the saturation profile Sw(xD) at different times (t = 0.1, 0.3, 0.5, 0.7, 0.9 of field life) for 1000 ppm polymer:
- At tD = 0.1: saturation front just starting to advance
- At tD = 0.9: most of the reservoir has been swept
- You can see the **Buckley-Leverett shock front** — the sharp jump in saturation moving from injector (xD=0) to producer (xD=1)

---

## 12. Ablation Study

```python
class DataOnlyPINN(keras.Model):
    """Ablation: data-only MLP baseline (no physics)."""
```

**What is an ablation study?** In science, you remove one component to see how much it contributes. Here: train a model with NO physics equations (just learns from data) and compare against the full PINN.

The `DataOnlyPINN` is a simple neural network:
- Same architecture (5 Dense layers, BatchNorm)
- No physics layers (no Corey, no fractional flow, no BL residual)
- No SaturationSubmodel
- Just: inputs → 5 Dense layers → 2 outputs

```python
def run_ablation(full_model, X_tr, y_tr, X_va, y_va, X_te, y_te):
    data_model = train_data_only(X_tr, y_tr, X_va, y_va, epochs=200)
    
    for label, m in [('Full PINN', full_model), ('Data-Only', data_model)]:
        y_p  = m(tf.constant(X_te, tf.float32), training=False).numpy()
        opr2 = r_squared(y_te[:, 0], y_p[:, 0])
        wc2  = r_squared(y_te[:, 1], y_p[:, 1])
```

Compare R² for both models on the same test set. The result bar chart shows whether adding physics improves accuracy. Typically: PINN > data-only, especially for **extrapolation** (conditions outside the training range).

---

## 13. Optimisation — Finding the Best Polymer Dose

The trained PINN can now answer "what if?" questions in milliseconds. The optimisation finds the polymer concentration that maximises cumulative oil recovery.

```python
def predict_cumulative_oil(model, cp_ppm, inj_norm_series, ...):
    X    = np.column_stack([t_hat_arr, cp_n, inj_norm_series, bhp_])
    y    = model(tf.constant(X), training=False).numpy()
    opr  = y[:, 0] * OPR_SCALE   # convert to STB/day
    t_days = t_hat_arr * TOTAL_DAYS   # convert to days
    return float(_try_trapezoid(opr, t_days))   # integrate: area under curve
```

**Numerical integration** (`trapezoid`) of the OPR time series gives cumulative oil in STB (Stock Tank Barrels). The trapezoidal rule approximates the area under the curve by connecting adjacent points with straight lines.

```python
def optimise_polymer(model, inj_norm_series):
    def objective_neg(cp_ppm_arr):
        cp = float(np.clip(cp_ppm_arr[0], 0.0, CP_MAX_PPM))
        return -predict_cumulative_oil(...)   # negative because we MINIMISE
```

Optimisers **minimise** (find the lowest point). We want to **maximise** cumulative oil. The trick: maximise f(x) ≡ minimise −f(x). So we return negative cumulative oil.

**Four algorithms are tried:**

1. **Differential Evolution (DE)** — simulates natural selection. Maintains a population of candidate Cp values. Each generation, candidates combine (crossover) and compete (selection). Robust, finds global optimum.

2. **Bayesian Optimisation (BO)** — builds a probability model of the objective function and uses it to choose the most promising point to evaluate next. Very efficient when evaluations are expensive (though here they are fast).

3. **Particle Swarm Optimisation (PSO)** — simulates a swarm of birds. Each "particle" (candidate Cp) has position and velocity. Particles share information about the best location found. They converge to the optimum.

4. **Genetic Algorithm (GA)** — another evolutionary approach. Uses mutation (random changes) and selection. Similar to DE but with different crossover strategy.

```python
    cp_range  = np.linspace(0, CP_MAX_PPM, 41)
    oils_sens = [predict_cumulative_oil(model, c, ...) for c in cp_range]
```

Also compute a **sensitivity curve** — cumulative oil for 41 evenly-spaced Cp values from 0 to 2000 ppm. This plots the entire landscape, showing the optimal region. Typically there is a peak around 800-1200 ppm (too little = poor sweep, too much = expensive with diminishing returns).

---

## 14. Main Function

```python
def main():
    print('=' * 65)
    print('PINN — Pelican Lake Heavy Oil Polymer Flooding')
    ...
```

The `main()` function is the master conductor — it calls every other function in order:

1. `load_data()` → load your 200 CMG cases
2. Build `SaturationSubmodel` and `PolymerPINN`
3. Run a **dummy forward pass** to initialise all weights:
   ```python
   dummy = tf.zeros((2, 4), dtype=tf.float32)
   _ = model(dummy, training=False)
   ```
   Keras builds layer weights lazily (the first time data flows through). We pass 2 dummy rows to trigger this.
4. `train()` → train for up to 500 epochs
5. `evaluate()` → compute R² on test set
6. Generate all 5 plots
7. `run_ablation()` → compare PINN vs data-only
8. `optimise_polymer()` → find best Cp
9. Print final summary showing trained Corey parameters

```python
if __name__ == '__main__':
    main()
```

**What does `if __name__ == '__main__'` mean?**

When Python runs a file, it sets `__name__` to `'__main__'` if that file was called directly (e.g. `python pinn_polymer_flooding.py`). If the file was imported by another file, `__name__` is the file's name instead.

This guard means: only run `main()` if this file was executed directly — not if it was imported as a library. It is a best practice for every Python script.

---

## 15. Glossary

| Term | Plain English |
|------|--------------|
| **Neural network** | A mathematical system with many adjustable numbers (weights) that learns to map inputs to outputs from examples |
| **PINN** | A neural network trained simultaneously on data AND physical equations |
| **Layer** | One processing step in a neural network |
| **Dense layer** | A layer where every input connects to every output |
| **Activation function** | A mathematical function applied after each layer (tanh, sigmoid, relu, softplus) |
| **Weight / parameter** | An adjustable number inside the neural network |
| **Training** | The process of adjusting weights to reduce the loss |
| **Loss / loss function** | A number measuring how wrong the model is |
| **Gradient** | The derivative of the loss — tells you which direction to move weights |
| **Gradient descent** | Update weights by subtracting a fraction of the gradient |
| **Adam optimizer** | A smart gradient descent method that adapts the step size |
| **Learning rate** | How big a step the optimizer takes (too big = unstable; too small = slow) |
| **Epoch** | One full pass through the training data |
| **Batch** | A subset of data processed together before one weight update |
| **Dropout** | Randomly disabling neurons during training to prevent overfitting |
| **Batch normalisation** | Rescaling layer inputs to have mean=0, std=1 for stable training |
| **Overfitting** | Model memorises training data and performs poorly on new data |
| **Early stopping** | Stop training when validation performance stops improving |
| **R²** | Fraction of variance explained (1.0 = perfect) |
| **MSE** | Mean Squared Error — average of squared prediction errors |
| **NRMSE** | Normalised Root Mean Squared Error (fraction of data range) |
| **Collocation points** | Random space/time points where PDE satisfaction is enforced |
| **Automatic differentiation** | Exact computation of derivatives through a computer program |
| **GradientTape** | TensorFlow's automatic differentiation tool |
| **Buckley-Leverett** | The PDE governing 1D two-phase flow (Darcy + mass conservation) |
| **Fractional flow** | Fraction of total fluid flow that is water at a given saturation |
| **Relative permeability** | How well a fluid flows at a given saturation (0 to 1 scale) |
| **Corey model** | Power-law formula for relative permeability curves |
| **Viscosity** | Thickness of a fluid (water=1, heavy oil=1650, polymer=25 cP) |
| **Porosity** | Fraction of rock volume that is pore space |
| **Permeability** | How easily fluid flows through rock |
| **Water saturation (Sw)** | Fraction of pore volume occupied by water |
| **SWC / Swc** | Irreducible water saturation (minimum Sw) |
| **SOR / Sor** | Residual oil saturation (minimum remaining oil fraction) |
| **FWM** | Mobile Water Fraction — initially mobile water present |
| **SWINITIAL** | Initial Sw incorporating FWM (=0.36 for Pelican Lake) |
| **RRF / RK** | Residual Resistance Factor — permeability reduction from polymer |
| **CMG STARS** | Commercial reservoir simulator used as ground truth |
| **Ablation study** | Experiment removing one component to measure its contribution |
| **Differential Evolution** | Global optimisation algorithm mimicking biological evolution |
| **Bayesian Optimisation** | Smart optimisation building a surrogate model of the objective |
| **PSO** | Particle Swarm Optimisation — swarm-intelligence algorithm |
| **Trapezoid rule** | Numerical integration: area under a curve by summing trapezoids |
| **Cumulative oil** | Total oil produced over the field life (STB) |
| **STB** | Stock Tank Barrel — volume of oil at surface conditions |

---

*Document covers all 1169 lines of `pinn_polymer_flooding.py`. For questions about specific sections, refer to the line numbers in the code.*
