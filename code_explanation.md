# Understanding the Pelican Lake PINN Code
## A Complete Beginner's Guide — Explained Like You're 5

---

## Before We Start: The Big Picture in Simple Words

Imagine you have an oil field underground. Oil is trapped in rock, like juice soaked into a sponge. To get it out, engineers pump water (with a special slippery ingredient called **polymer**) into the ground through some wells. The water pushes the oil toward other wells where it comes out.

The problem: **nobody can see underground.** We have to guess what's happening — how much oil is coming out, how much water is mixed in — using math and computers.

That's what this code does. It builds a **smart computer brain** (a neural network) that:
1. Knows the **physics** of how oil and water flow underground (like rules of a game)
2. Learns from what a big industrial computer simulator (CMG STARS) already calculated
3. Can then **predict** what will happen without running the expensive simulator

This type of brain is called a **Physics-Informed Neural Network (PINN)**.

---

## Part 0: What is a Neural Network? (For Complete Beginners)

Think of a neural network like a **series of filters**. You put a number in one end, it passes through many layers of simple math operations, and a useful number comes out the other end.

```
Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
  0.3  →  math   →   math   →   math   →   0.72
```

Each "layer" has:
- **Neurons**: little calculators
- **Weights**: knobs that get tuned during training (like adjusting a radio dial)
- **Activation function**: decides if a neuron "fires" or not

**Training** = showing the network many examples and adjusting the knobs until it gives correct answers.

In our PINN, the network learns to predict **water saturation** (how wet the rock is) and then uses that to predict **oil production** and **water cut** (fraction of liquid that is water).

---

## Part 1: The File Header (Lines 1–30)

```python
"""
Physics-Informed Neural Network for Pelican Lake Heavy-Oil Polymer Flooding
...
"""
```

The text inside `"""triple quotes"""` is a **docstring** — a note to humans explaining what the file does. Python ignores it when running the code.

---

```python
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
```

**`import`** means "go get this toolbox and bring it here."

| Import | What it is | Analogy |
|---|---|---|
| `os` | Operating system tools | File cabinet — find/create folders |
| `json` | Save/load structured data | Save your game |
| `warnings` | Control warning messages | Mute annoying alerts |
| `numpy as np` | Math on arrays of numbers | A super calculator |
| `tensorflow as tf` | Build neural networks | The engine |
| `keras` | Friendly layer on top of TensorFlow | Easier steering wheel |
| `matplotlib` | Draw graphs | Pencil and paper |
| `scipy.optimize` | Find the best value of something | Search for treasure |

`matplotlib.use("Agg")` — this tells matplotlib **not** to open a window on screen (useful on servers). It saves graphs to files instead.

---

```python
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
```

These two lines **silence** unimportant messages so the output is clean. Think of it as turning off the TV while you work.

---

```python
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

**Random seed** — computers can't truly be random; they follow a recipe. By setting the seed to 42 (any number works), we guarantee the same "random" numbers every run. This makes experiments **reproducible** — you get the same result every time, like using the same recipe to bake a cake.

---

```python
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "pinn_polymer_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

`OUTPUT_DIR` is where we save graphs and results. `os.makedirs` creates that folder if it doesn't exist. `exist_ok=True` means "don't crash if the folder is already there."

---

## Part 2: The Manuscript Parameters — `class Params` (Lines 40–130)

```python
class Params:
    depth_ft  = 1475.0
    T_F       = 63.0
    ...
```

A **class** is a container — a box that holds related information together.

`Params` holds **every physical number** about the Pelican Lake oil field, taken directly from the research manuscript. Using a class keeps everything organized in one place. If you need to change a value, you change it once here and it updates everywhere.

Let's go through each group:

### Reservoir Properties (what the underground rock/fluid is like)

```python
depth_ft   = 1475.0   # How deep underground the reservoir is (feet)
T_F        = 63.0     # Temperature (Fahrenheit) — affects oil thickness
P_init_psi = 380.0    # Starting pressure in the reservoir (psi = pounds per square inch)
mu_oil_cp  = 1650.0   # Oil viscosity (how thick/sticky the oil is, in centipoise)
Bo         = 5.65     # Oil formation volume factor — oil expands underground
```

**Viscosity** is how thick a fluid is. Water = 1 cp. Honey ≈ 10,000 cp. This oil at 1650 cp is like very thick syrup — that's why we need polymer to push it.

```python
GOR    = 28.07    # Gas-Oil Ratio: how much gas dissolves in the oil
Pb_psi = 304.58   # Bubble point: below this pressure, gas bubbles out of oil
rock_comp = 2.3e-4  # How much the rock squeezes when pressure changes
```

`2.3e-4` = 0.00023. The `e-4` means "move the decimal 4 places left." Scientific notation for tiny numbers.

### Corey Relative Permeability (Section 2.3 of manuscript)

```python
Swr     = 0.23    # Residual water saturation
Sro     = 0.20    # Residual oil saturation
Krw_max = 0.10    # Maximum water relative permeability
Kro_max = 1.00    # Maximum oil relative permeability
nw      = 3.0     # Corey water exponent
no      = 2.2     # Corey oil exponent
```

**Saturation** = fraction of pore space filled by a fluid (0 to 1).
- `Swr = 0.23` means at minimum, 23% of the pore space always has water stuck in it (can't remove it)
- `Sro = 0.20` means at minimum, 20% always has oil stuck (can't push all oil out)

**Relative permeability** (kr) = how easily a fluid flows through rock when another fluid is also present. Like trying to walk through a crowd — harder when more people are there.

`Krw_max = 0.10` means water's maximum flow ability is only 10% of the rock's full capacity. This is low because the oil is thick and blocks water flow paths.

`nw = 3.0` and `no = 2.2` are **shape parameters** for the Corey equations (explained in Part 4).

### Mobile Water Fraction (the KEY innovation from the manuscript)

```python
FWM       = 0.12    # Mobile Water Fraction
Swi_core  = 0.30    # Water saturation measured in the lab on a core sample
Swinitial = 0.36    # Actual initial water saturation in the field
Sor       = 0.20    # Same as Sro above
```

This is the **most important discovery** in the manuscript.

In the lab, scientists measure water saturation = 0.30. But in the real field, it's 0.36. Why the difference?

`FWM = 0.12` means 12% of the pore space between "stuck" and "free" water is actually **mobile** (moving) water that lab tests miss.

The formula:
```
Swinitial = Swi_core + FWM × (1 - Swi_core - Sor)
           = 0.30 + 0.12 × (1 - 0.30 - 0.20)
           = 0.30 + 0.12 × 0.50
           = 0.36
```

Without FWM, the model predicts almost **zero** initial water cut. With FWM, it predicts 16.8% — matching the real field data perfectly.

### Polymer Properties

```python
Cp_base_ppm = 1000.0   # Polymer concentration injected (parts per million)
mu_poly_cp  = 25.0     # Polymer solution viscosity at 1000 ppm
salinity    = 8222.0   # Salt content of water (ppm) — affects polymer performance
adsorption  = 10.0     # How much polymer sticks to the rock (µg per gram of rock)
RRF         = 2.0      # Residual Resistance Factor
IPV         = 0.10     # Inaccessible Pore Volume
```

**Polymer** (HPAM = partially hydrolysed polyacrylamide) is added to water to make it thicker, so it pushes the thick oil better. Think of adding cornstarch to water.

`mu_poly = 25 cp` at 1000 ppm means the polymer solution is 25× thicker than plain water.

`RRF = 2.0` means polymer permanently reduces rock permeability by half — some polymer sticks to rock walls and clogs pores slightly. This slows water but also the oil, net effect helps recovery.

`IPV = 0.10` means 10% of pore space is too tiny for polymer molecules to enter. Polymer can't push oil out of those tiny pores.

### Well Geometry

```python
n_producers     = 3          # Number of production wells (P1, P2, P3)
n_injectors     = 2          # Number of injection wells
well_length_ft  = 4593.176   # How long each horizontal well is
well_spacing_ft = 574.147    # Distance between wells
BHP_prod_psi    = 120.0      # Bottom-hole pressure at producers (low = oil flows in)
BHP_inj_psi     = 550.0      # Bottom-hole pressure at injectors (high = pushes fluid)
```

Pelican Lake uses **horizontal wells** — drilled sideways through the reservoir, not straight down. This gives much more contact with the oil.

### Time

```python
t_end       = 56.0   # Total simulation time in months (May 2005 to Dec 2009)
n_timesteps = 57     # 57 monthly snapshots
```

The simulation covers 4.5 years of production history.

### CMG STARS Validation Targets (Tables 6 & 7)

```python
cmg_r2    = {"P1": 0.9987, "P2": 0.9960, "P3": 0.9906}
cmg_nrmse = {"P1": 0.0119, "P2": 0.0216, "P3": 0.0317}
cmg_cum   = {"P1": 0.0014, "P2": 0.0292, "P3": 0.0366}

wc_initial = {"P1": 0.168, "P2": 0.168, "P3": 0.168}
wc_final   = {"P1": 0.606, "P2": 0.598, "P3": 0.605}
```

`{"P1": 0.9987, ...}` is a **Python dictionary** — like a phone book where the name ("P1") maps to a number (0.9987).

**R²** (R-squared) = how well predictions match reality. 1.0 = perfect. 0.9987 means 99.87% accuracy — CMG STARS matches field data extremely well.

**NRMSE** = Normalised Root Mean Squared Error. Smaller is better. 0.0119 = 1.19% average error.

**Water cut (WC)** = fraction of liquid coming out that is water.
- `wc_initial = 0.168` → at the start, 16.8% of liquid is water
- `wc_final P1 = 0.606` → at the end, 60.6% is water (polymer flooding has progressed)

---

## Part 3: CMG Reference Data — How We Create "Ground Truth" Without CSVs

Since we have no CSV files, we reconstruct CMG's output from the numbers reported in the paper.

### `_cp_from_t(t_months)` — Polymer injection schedule

```python
def _cp_from_t(t_months):
    out = np.zeros_like(t_months, dtype=np.float32)
    for t0, t1, cp in P.inj_schedule:
        mask = (t_months >= t0) & (t_months < t1)
        out[mask] = cp
    return out
```

`inj_schedule = [(0,5,600), (5,12,500), (12,30,800), (30,57,1000)]`

This reads: "from month 0 to 5, inject 600 ppm; from month 5 to 12, inject 500 ppm; ..."

`np.zeros_like(t_months)` — make an array of zeros the same shape as `t_months`.

`mask = (t_months >= t0) & (t_months < t1)` — a True/False array marking which time values fall in this period.

`out[mask] = cp` — fill those positions with the polymer concentration. Like coloring specific cells in a spreadsheet.

**Result**: Given any array of month values, you get back the corresponding polymer concentrations.

### `cmg_wc_curve(well_name, t_months)` — Reconstruct water cut from manuscript

```python
def cmg_wc_curve(well_name, t_months):
    wc0  = P.wc_initial[well_name]   # 0.168 for all wells
    wc_f = P.wc_final[well_name]     # e.g., 0.606 for P1
    tn   = t_months / P.t_end        # normalise time to [0,1]
    return wc0 + (wc_f - wc0) / (1.0 + np.exp(-5.0 * (tn - 0.55)))
```

This is a **logistic (S-shaped) curve** — the classic shape of water cut in polymer flooding:
- Starts low (water hasn't broken through yet)
- Rises steeply in the middle (water breaks through)
- Flattens at the top (mostly water at the end)

The formula `1 / (1 + exp(-k*(t-t0)))` is the **sigmoid function**:
- `k = 5.0` controls how steep the rise is
- `t0 = 0.55` means the steepest point is at 55% through simulation time
- The result is scaled from `wc0` to `wc_f`

This gives us a smooth, physically realistic water cut curve from just the two endpoints reported in Table 7.

### `build_cmg_dataset()` — Assemble training data from manuscript

```python
def build_cmg_dataset():
    t_months, t_norm, Cp_grid = make_time_grid()
    Cp_norm = Cp_grid / P.Cp_base_ppm

    X_list, y_list = [], []
    for w_idx, w_name in enumerate(["P1", "P2", "P3"]):
        wc  = cmg_wc_curve(w_name, t_months).reshape(-1, 1)
        opr = cmg_opr_curve(w_name, t_months).reshape(-1, 1)
        w_col = np.full((P.n_timesteps, 1), w_idx / 2.0, dtype=np.float32)
        X_list.append(np.concatenate([t_norm, Cp_norm, w_col], axis=1))
        y_list.append(np.concatenate([opr, wc], axis=1))
```

`enumerate(["P1","P2","P3"])` gives pairs: `(0,"P1"), (1,"P2"), (2,"P3")`.

`w_col = w_idx / 2.0` — encodes which well we're looking at as a number: P1=0.0, P2=0.5, P3=1.0. The network uses this to tell wells apart.

`np.concatenate([t_norm, Cp_norm, w_col], axis=1)` — glues three columns together into one matrix:

```
[time,  Cp_norm,  well_id]
[0.00,  0.60,     0.0    ]   ← P1, month 0
[0.018, 0.60,     0.0    ]   ← P1, month 1
...
[0.00,  0.60,     0.5    ]   ← P2, month 0
...
```

`np.vstack(X_list)` — stacks the three wells' data vertically into one big matrix of shape `(171, 3)` = 3 wells × 57 months.

---

## Part 4: Custom Keras Layers — Building Physics Into the Network

**Why custom layers?** Standard neural network layers just do matrix multiplications. We need layers that implement **physical equations** so the network respects the laws of physics, not just statistics.

### `SwScaleLayer` — Force output into physical range

```python
class SwScaleLayer(layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.lo = float(P.Swr)        # 0.23
        self.hi = float(1.0 - P.Sor) # 0.80

    def call(self, x):
        return self.lo + (self.hi - self.lo) * x
```

A **sigmoid** function outputs values between 0 and 1. But water saturation must be between 0.23 and 0.80 (physical limits). This layer **stretches** the sigmoid output to that range:

```
Input x ∈ [0, 1]  →  Output ∈ [0.23, 0.80]
Output = 0.23 + (0.80 - 0.23) × x
       = 0.23 + 0.57 × x
```

When x=0 → output=0.23 (minimum saturation = residual water)
When x=1 → output=0.80 (maximum saturation = 1 - residual oil)

The `call(self, x)` method is what runs when you "call" the layer with data. `super().__init__(**kw)` calls the parent class constructor — always needed for Keras layers.

### `PolymerViscosityLayer` — Todd-Longstaff viscosity model

```python
class PolymerViscosityLayer(layers.Layer):
    def call(self, Cp_norm):
        mu = 1.0 * (1.0 + 14.2 * Cp_norm
                        +  8.5 * Cp_norm ** 2
                        +  1.3 * Cp_norm ** 3)
        return mu
```

`Cp_norm = Cp_ppm / 1000.0` — normalised polymer concentration.

This polynomial formula calculates how thick the polymer solution is. We chose coefficients (14.2, 8.5, 1.3) so that:
- At Cp=0 (no polymer): mu = 1.0 cp (plain water)
- At Cp=1000 ppm (Cp_norm=1): mu = 1+14.2+8.5+1.3 = **25 cp** ✓ (matches manuscript)

`**` means "to the power of": `Cp_norm ** 2` = Cp_norm squared.

### `CoreyKrLayer` — Trainable relative permeability (the physics heart)

This is the most important physics layer. It calculates how easily water and oil flow through rock at a given water saturation.

```python
class CoreyKrLayer(layers.Layer):
    def build(self, input_shape):
        self._krw_max = self.add_weight(
            name="krw_max", shape=(),
            initializer=tf.constant_initializer(P.Krw_max),
            trainable=True)
        ...
```

`build()` is called once to create the layer's **trainable weights** — the knobs that get adjusted during training. `shape=()` means a single scalar number (not an array).

`initializer=tf.constant_initializer(P.Krw_max)` — start this knob at the manuscript value (0.10). Training will adjust it to better fit the data.

`trainable=True` — yes, this number should change during training.

```python
    def call(self, Sw):
        krw_max = tf.clip_by_value(self._krw_max, 0.01, 0.50)
        ...
        denom  = 1.0 - P.Swr - P.Sor + 1e-8
        Se_w   = tf.clip_by_value((Sw - P.Swr) / denom, 0.0, 1.0)
        Se_o   = tf.clip_by_value((1.0 - P.Sor - Sw) / denom, 0.0, 1.0)

        krw = krw_max * tf.pow(Se_w, nw)
        kro = kro_max * tf.pow(Se_o, no)
        return tf.concat([krw, kro], axis=-1)
```

`tf.clip_by_value(x, min, max)` — keep values within bounds. Prevents physically impossible values like negative permeability.

`Se_w` = **effective water saturation** = normalised between 0 (all residual) and 1 (maximum mobile water).

`1e-8` = 0.00000001 — a tiny number added to denominators to prevent **division by zero** errors.

**Corey equations:**
```
krw = Krw_max × Se_w^nw    (water relative permeability)
kro = Kro_max × Se_o^no    (oil relative permeability)
```

These power-law curves look like this:
- When Sw is low (little water): krw≈0 (water barely flows), kro≈max (oil flows freely)
- When Sw is high (lots of water): krw≈max (water flows freely), kro≈0 (oil can't flow)

`tf.concat([krw, kro], axis=-1)` — combine the two values into one output tensor with 2 columns.

### `FractionalFlowLayer` — Calculate what fraction is water

```python
class FractionalFlowLayer(layers.Layer):
    def call(self, inputs):
        krw, kro, mu_eff = inputs   # unpack three inputs
        mob_w = krw / (mu_eff * P.RRF + 1e-10)
        mob_o = kro / (P.mu_oil_cp  + 1e-10)
        return mob_w / (mob_w + mob_o + 1e-10)
```

**Mobility** = how easily a fluid moves = kr / viscosity.
- High kr + low viscosity = high mobility (flows easily)
- Low kr + high viscosity = low mobility (flows with difficulty)

Water mobility: `mob_w = krw / (mu_eff × RRF)` — divided by effective viscosity × resistance factor.

Oil mobility: `mob_o = kro / mu_oil` — oil at 1650 cp is very thick, so mobility is low.

**Fractional flow** = `mob_w / (mob_w + mob_o)` — what fraction of total flow is water.

If mob_w = 0.5 and mob_o = 0.5 → fw = 0.5 (half water, half oil)
If mob_w = 2.0 and mob_o = 0.5 → fw = 0.8 (80% water — bad!)
If mob_w = 0.1 and mob_o = 0.5 → fw = 0.17 (17% water — good, mostly oil!)

This is why polymer helps: it increases mu_eff (makes water thicker), reducing mob_w, which reduces fw, which means more oil flows relative to water.

---

## Part 5: The Saturation Submodel

```python
class SaturationSubmodel(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.d1    = layers.Dense(128, activation="tanh", name="sw_d1")
        self.d2    = layers.Dense(256, activation="tanh", name="sw_d2")
        self.d3    = layers.Dense(256, activation="tanh", name="sw_d3")
        self.d4    = layers.Dense(128, activation="tanh", name="sw_d4")
        self.d5    = layers.Dense(64,  activation="tanh", name="sw_d5")
        self.out   = layers.Dense(1, activation="sigmoid", name="sw_out")
        self.scale = SwScaleLayer(name="sw_scale")
```

`keras.Model` is a **base class** — it gives our class all the machinery of a neural network (training, saving, etc.) for free.

`layers.Dense(128, activation="tanh")` — a **fully connected layer**:
- 128 neurons
- Each neuron computes: `output = tanh(weights × input + bias)`
- `tanh` activation function squishes output to range (-1, +1)

**Why tanh?** PINNs work best with smooth activation functions that have smooth derivatives (needed for physics equations).

```python
        self.kr_layer   = CoreyKrLayer(name="kr_layer")
        self.visc_layer = PolymerViscosityLayer(name="visc_layer")
        self.fw_layer   = FractionalFlowLayer(name="fw_layer")
```

These physics layers live inside the saturation model. So the model knows about kr, viscosity, and fractional flow.

```python
    def call(self, inputs, training=False):
        x = self.d1(inputs)   # 3 inputs → 128 outputs
        x = self.d2(x)        # 128 → 256
        x = self.d3(x)        # 256 → 256
        x = self.d4(x)        # 256 → 128
        x = self.d5(x)        # 128 → 64
        x = self.out(x)       # 64 → 1  (sigmoid, range [0,1])
        return self.scale(x)  # stretch to [0.23, 0.80]
```

Data flows through the layers like water through a pipe, getting transformed at each step. The shape starts at (N, 3), grows to (N, 256) in the middle, then narrows back to (N, 1) at the output. This hourglass shape is called an **encoder-decoder** pattern.

`training=False` parameter: some layers (like Dropout) behave differently during training vs. inference. Passing `training=True` during training enables this.

```python
    def fractional_flow_from_sw(self, Sw, Cp_norm):
        kr_out = self.kr_layer(Sw)
        krw    = kr_out[:, :1]    # first column
        kro    = kr_out[:, 1:]    # second column
        mu_eff = self.visc_layer(Cp_norm)
        return self.fw_layer([krw, kro, mu_eff])
```

`kr_out[:, :1]` — Python slice: all rows (`:`) of column 0 only (`:1`). Gets the krw column.
`kr_out[:, 1:]` — all rows, column 1 onwards. Gets the kro column.

This helper method takes Sw and Cp, and uses the physics layers to compute fractional flow — the fraction of fluid that is water.

---

## Part 6: The Production Head (One Per Well)

```python
class ProductionHead(keras.Model):
    def __init__(self, well_name, **kw):
        super().__init__(**kw)
        self.well_name = well_name
        self.d1  = layers.Dense(64, activation="tanh")
        self.d2  = layers.Dense(64, activation="tanh")
        self.d3  = layers.Dense(32, activation="tanh")
        self.opr = layers.Dense(1, activation="relu",    name=f"{well_name}_opr")
        self.wc  = layers.Dense(1, activation="sigmoid", name=f"{well_name}_wc")
```

We have **three** of these — one for P1, one for P2, one for P3. Each learns the specific production behaviour of its well.

**Why separate heads?** Each well has slightly different geology and location. Sharing the saturation model (which describes general physics) but using separate production heads allows the model to be:
- Physically consistent (shared physics)
- Well-specific (separate production learning)

`activation="relu"` for OPR — "ReLU" (Rectified Linear Unit) = `max(0, x)`. Forces OPR to be non-negative (can't produce negative oil).

`activation="sigmoid"` for WC — forces WC between 0 and 1 (it's a fraction: can't be negative or more than 100%).

```python
    def call(self, inputs, training=False):
        x   = self.d1(inputs)
        x   = self.d2(x)
        x   = self.d3(x)
        opr = self.opr(x)
        wc  = self.wc(x)
        return opr, wc      # returns TWO separate outputs
```

One network, two outputs: oil production rate AND water cut. Python can return multiple values with a comma.

---

## Part 7: The Physics Loss Functions — The Heart of PINN

This is where the **magic** of Physics-Informed Neural Networks happens. Instead of just matching data, we also force the network to obey physical laws.

### `compute_pde_residuals` — Buckley-Leverett + Polymer Transport PDEs

```python
def compute_pde_residuals(sw_model, x, t, Cp_norm):
    phi = tf.constant(P.phi,              dtype=tf.float32)
    ads = tf.constant(P.adsorption / 1e6, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        inp  = tf.concat([x, t, Cp_norm], axis=1)
        Sw   = sw_model(inp, training=True)
        fw   = sw_model.fractional_flow_from_sw(Sw, Cp_norm)
        SwCp = Sw * Cp_norm
        fwCp = fw * Cp_norm
```

**`tf.GradientTape`** is TensorFlow's automatic differentiation tool. Think of it as a **recording device** that watches every math operation you do, so it can later calculate derivatives (slopes) automatically.

`tape.watch(x)` and `tape.watch(t)` — tell the tape "pay special attention to x and t; we'll want their derivatives later."

**CRITICAL:** `inp = tf.concat([x, t, Cp_norm], axis=1)` is built **inside** the tape. This is essential — the tape must see the full chain of computation from x and t to the final output. If you build `inp` outside the tape, the tape loses track of the chain and returns `None` for derivatives. This was the bug we fixed earlier!

```python
    dSw_dt    = tape.gradient(Sw,   t)    # ∂Sw/∂t
    dfw_dx    = tape.gradient(fw,   x)    # ∂fw/∂x
    d_SwCp_dt = tape.gradient(SwCp, t)    # ∂(Sw·Cp)/∂t
    d_fwCp_dx = tape.gradient(fwCp, x)   # ∂(fw·Cp)/∂x
    del tape
```

`tape.gradient(y, x)` = "what is the derivative of y with respect to x?" = how much does y change when x changes a tiny bit?

The **Buckley-Leverett PDE** (conservation of mass for two-phase flow):

```
φ × ∂Sw/∂t + ∂fw/∂x = 0
```

In words: "The rate of change of water in the rock (∂Sw/∂t) plus the spatial change in water flow (∂fw/∂x) must be zero — water is neither created nor destroyed."

This is the **conservation of mass** — one of the most fundamental laws in physics.

The **polymer transport PDE**:

```
φ × ∂(Sw·Cp)/∂t + ∂(fw·Cp)/∂x + α·Cp = 0
```

Same idea but tracking the polymer — plus a loss term `α·Cp` for polymer adsorption (polymer sticking to the rock and disappearing).

```python
    L_bl = tf.reduce_mean(tf.square(phi * dSw_dt + dfw_dx))
    L_pt = tf.reduce_mean(tf.square(phi * d_SwCp_dt + d_fwCp_dx + ads * Cp_norm))
```

If the physics equation is perfectly satisfied: `phi * dSw_dt + dfw_dx = 0`.
The loss is this expression **squared** — so if it equals zero, the loss is zero. If it's not zero, the loss is positive, and training pushes the network to reduce it.

`tf.reduce_mean(tf.square(...))` = mean squared error (MSE) — square each value, then take the average.

### `ic_loss` — Initial Condition: Sw at time zero

```python
def ic_loss(sw_model, x_ic, t_ic, Cp_ic_norm):
    inp = tf.concat([x_ic, t_ic, Cp_ic_norm], axis=1)
    Sw  = sw_model(inp, training=True)
    tgt = tf.constant(P.Swinitial, dtype=tf.float32)  # 0.36
    return tf.reduce_mean(tf.square(Sw - tgt))
```

At time = 0, everywhere in the reservoir, water saturation = 0.36 (incorporating FWM = 0.12).

This forces the network to start from the correct physical initial state.

### `bc_loss` — Boundary Condition: Sw at the injector

```python
def bc_loss(sw_model, x_bc, t_bc, Cp_bc_norm):
    inp = tf.concat([x_bc, t_bc, Cp_bc_norm], axis=1)
    Sw  = sw_model(inp, training=True)
    tgt = tf.constant(1.0 - P.Sor, dtype=tf.float32)  # 0.80
    return tf.reduce_mean(tf.square(Sw - tgt))
```

At x = 0 (the injector face), water is being pumped in continuously, so Sw = 1 - Sor = 0.80 — the maximum physically possible saturation.

### `make_collocation_tensors` — Random points for enforcing physics

```python
def make_collocation_tensors(n_interior=2000, ...):
    rng = np.random.default_rng(seed)

    x_i = tf.constant(rng.uniform(0, 1, (n_interior, 1)).astype(np.float32))
    t_i = tf.constant(rng.uniform(0, 1, (n_interior, 1)).astype(np.float32))
```

We can't check the PDE at every point in space and time — there are infinitely many. Instead, we randomly sample 2000 points (called **collocation points**) and require the PDE to hold at those points.

`rng.uniform(0, 1, (2000, 1))` — 2000 random numbers between 0 and 1, in a column of shape (2000, 1).

`tf.constant(...)` — converts numpy array to a TensorFlow constant (faster for GPU computation).

---

## Part 8: The Training Loop — Where the Learning Happens

```python
def train(sw_model, prod_heads):
```

This function orchestrates the entire training process. It's like a **coach** directing practice sessions.

### Setting up optimizers

```python
opt_sw   = optimizers.Adam(P.lr_physics)    # lr = 5e-4 = 0.0005
opt_prod = {w: optimizers.Adam(P.lr_prod)   # lr = 1e-3 = 0.001
            for w in prod_heads}
```

**Adam optimizer** — the most popular neural network trainer. It adjusts the learning rate automatically for each weight. Think of it as a smart GPS that adjusts its route as conditions change.

`lr` = **learning rate** = how big a step to take when adjusting weights.
- Too large: overshoots, never converges (like jumping too far)
- Too small: trains forever (like shuffling tiny steps)
- 0.0005 and 0.001 are good starting values for PINNs

`{w: optimizers.Adam(...) for w in prod_heads}` — Python **dictionary comprehension**: create one optimizer per well.

### Phase 1: Physics Pre-training

```python
for epoch in range(P.epochs_physics):
    with tf.GradientTape() as tape:
        L_bl, L_pt = compute_pde_residuals(sw_model, x_i, t_i, Cp_i)
        L_ic = ic_loss(sw_model, x_ic, t_ic, Cp_ic)
        L_bc = bc_loss(sw_model, x_bc, t_bc, Cp_bc)
        L_phys = P.w_bl*L_bl + P.w_pt*L_pt + P.w_ic*L_ic + P.w_bc*L_bc

    grads = tape.gradient(L_phys, sw_model.trainable_variables)
    opt_sw.apply_gradients(zip(grads, sw_model.trainable_variables))
```

One **epoch** = one full pass through the training data.

`with tf.GradientTape() as tape:` — starts the outer recording. This tape watches the model's weights (automatically) and records all math operations.

`L_phys = w_bl*L_bl + w_pt*L_pt + w_ic*L_ic + w_bc*L_bc`

**Weighted total loss** — combine all physics constraints into one number. Weights decide which constraints are most important:
- `w_ic = 10.0` and `w_bc = 10.0` — initial and boundary conditions are 10× more important than PDE residuals
- This forces the network to learn the correct starting conditions first

`tape.gradient(L_phys, sw_model.trainable_variables)` — ask the tape: "how much would each weight need to change to reduce L_phys?" This returns **gradients** — the mathematical answer to that question.

**Gradient** = slope of the loss with respect to each weight. If gradient is positive, reducing the weight reduces the loss. If negative, increasing it reduces the loss.

`opt_sw.apply_gradients(zip(grads, sw_model.trainable_variables))` — apply the computed gradients: move each weight a small step in the direction that reduces loss.

`zip(grads, sw_model.trainable_variables)` — pairs each gradient with the corresponding weight, like zipping two lists together: `[(grad1, weight1), (grad2, weight2), ...]`.

### Phase 2: Joint Training (Physics + CMG Data)

```python
for epoch in range(P.epochs_production):
    # ── Physics update ──
    with tf.GradientTape() as tape:
        L_bl, L_pt = compute_pde_residuals(...)
        L_phys = ...
    phys_grads = tape.gradient(L_phys, sw_model.trainable_variables)
    opt_sw.apply_gradients(zip(phys_grads, sw_model.trainable_variables))
```

First, update the saturation model with physics (same as Phase 1).

```python
    # ── Saturation at producer face ──
    sat_inp = tf.concat([x_prod, t_tf, Cp_tf], axis=1)
    Sw_prod = sw_model(sat_inp, training=False)
    fw_prod = sw_model.fractional_flow_from_sw(Sw_prod, Cp_tf)
```

Get the saturation at `x=1` (the producer face — where oil comes out). Use `training=False` because we don't want this pass to update the model — we're just reading values to pass to the production heads.

```python
    for w_name, head in prod_heads.items():
        y_ref   = well_data[w_name]["y"]
        opr_ref = y_ref[:, :1]
        wc_ref  = y_ref[:, 1:]

        with tf.GradientTape() as tape:
            prod_inp = tf.concat([t_tf, Sw_prod, fw_prod, Cp_tf], axis=1)
            opr_pred, wc_pred = head(prod_inp, training=True)

            L_opr = tf.reduce_mean(tf.square(opr_pred - opr_ref))
            L_wc  = tf.reduce_mean(tf.square(wc_pred  - wc_ref))
```

For each well, compute how much the prediction differs from the CMG reference.

`y_ref[:, :1]` — all rows, first column only → OPR reference values
`y_ref[:, 1:]` — all rows, second column → WC reference values

```python
            # Physics constraint: WC = fw at producer face
            L_wc_phys = tf.reduce_mean(tf.square(wc_pred - fw_prod))

            # Manuscript initial WC constraint
            wc0_tgt = tf.constant([[P.wc_initial[w_name]]])
            L_wc0   = tf.reduce_mean(tf.square(wc_pred[:1] - wc0_tgt))

            # OPR should not increase over time (decline constraint)
            dOPR  = opr_pred[1:] - opr_pred[:-1]
            L_dec = tf.reduce_mean(tf.square(tf.nn.relu(dOPR)))
```

Three extra physics constraints for the production head:

1. **`L_wc_phys`**: WC should approximately equal the fractional flow `fw` from physics. This couples the data-driven head to the physics model.

2. **`L_wc0`**: First timestep WC must equal 0.168 (from manuscript Table 7). `wc_pred[:1]` = first row only.

3. **`L_dec`**: Oil production should decline over time (it always does in a producing field). `dOPR = opr_pred[1:] - opr_pred[:-1]` = differences between consecutive time steps. `tf.nn.relu(dOPR)` = only the *positive* differences (increases are penalised; decreases are fine).

```python
            L_prod = (P.w_data * (L_opr + L_wc)
                      + 2.0 * L_wc_phys
                      + 5.0 * L_wc0
                      + 0.3 * L_dec)

        grads = tape.gradient(L_prod, head.trainable_variables)
        opt_prod[w_name].apply_gradients(zip(grads, head.trainable_variables))
```

**Total production loss** = data fit + physics constraints (weighted). Gradient descent optimises only the **production head's** weights, not the saturation model.

### Early Stopping

```python
if val_wc_err < best_val - 1e-6:
    best_val = val_wc_err
    wait = 0
    best_ws = [v.numpy() for ...]
else:
    wait += 1
    if wait >= P.patience:
        print(f"\n  Early stopping at epoch {epoch + 1}")
        break
```

**Early stopping** = stop training when performance stops improving. This prevents **overfitting** (memorising training data instead of learning the underlying patterns).

If validation error improves by more than `1e-6`: reset the patience counter, save the best weights.
If not: increment the counter. After `patience=80` epochs without improvement, stop.

---

## Part 9: Prediction — Using the Trained Model

```python
def predict(sw_model, prod_heads, t_norm, Cp_grid):
    t_tf   = tf.constant(t_norm, dtype=tf.float32)
    Cp_tf  = tf.constant(Cp_grid / P.Cp_base_ppm, dtype=tf.float32)
    x_prod = tf.ones_like(t_tf)
```

`tf.ones_like(t_tf)` — array of 1.0s the same shape as t_tf. This sets x=1 (producer face position) for all time steps.

```python
    sat_inp = tf.concat([x_prod, t_tf, Cp_tf], axis=1)
    Sw_prod = sw_model(sat_inp, training=False)
    fw_prod = sw_model.fractional_flow_from_sw(Sw_prod, Cp_tf)
```

Run the saturation model at x=1 (producer face) to get Sw at each time step. Then compute the fractional flow.

```python
    for w_name, head in prod_heads.items():
        prod_inp = tf.concat([t_tf, Sw_prod, fw_prod, Cp_tf], axis=1)
        opr_raw, wc_raw = head(prod_inp, training=False)

        opr = np.clip(opr_raw.numpy().flatten(), 0.0, None)
        if opr.max() > 0:
            opr = opr / opr.max()   # normalise to [0, 1]
        wc = np.clip(wc_raw.numpy().flatten(), 0.0, 1.0)
```

`tensor.numpy()` — convert TensorFlow tensor to numpy array (regular Python math after this).
`tensor.flatten()` — convert 2D (N,1) to 1D (N,) — removes the extra dimension.
`np.clip(arr, 0, None)` — ensure no negative values. `None` means "no upper limit."

---

## Part 10: Evaluation — How Good Is It?

```python
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)
```

**R² (R-squared)** = "how much better is my model than just guessing the average?"

- `ss_res` = Sum of Squared Residuals = total prediction error
- `ss_tot` = Total Sum of Squares = error if you just guessed the mean every time
- R² = 1 - (my_error / baseline_error)
- R² = 1.0 → perfect; R² = 0.0 → no better than guessing the mean; R² < 0 → worse than mean

`+ 1e-12` prevents division by zero if all true values are identical.

```python
def nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(rmse / (y_true.max() - y_true.min() + 1e-12))
```

**NRMSE** = Root Mean Squared Error, divided by the range. Expresses error as a fraction of the data's spread. 0.02 = 2% error.

---

## Part 11: Polymer Optimisation — Finding the Best Injection Rate

```python
def _predict_mean_wc(Cp_val, sw_model, prod_heads, t_norm, ...):
    Cp_arr = np.full((P.n_timesteps, 1), Cp_val / P.Cp_base_ppm, ...)
    ...
    # Run the model with this Cp
    # Return: average final WC across 3 wells
    return total_wc / 3.0
```

This function runs the full model for any polymer concentration `Cp_val` and returns the average final water cut. Lower WC = better oil recovery.

### Differential Evolution

```python
de_res = differential_evolution(
    lambda x: _predict_mean_wc(x[0], ...),
    bounds=[(500.0, 2000.0)],
    maxiter=40, popsize=8, ...
)
```

**Differential Evolution (DE)** is an optimization algorithm inspired by evolution:

1. Start with 8 "individuals" — each is a candidate Cp value between 500 and 2000 ppm
2. For each individual: combine with two others randomly to create a "mutant"
3. If mutant has lower WC (better), replace the original
4. Repeat 40 generations
5. The survivor is the best Cp

`lambda x: _predict_mean_wc(x[0], ...)` — an anonymous function. `x` is a list, `x[0]` is the Cp value.

`bounds=[(500.0, 2000.0)]` — search only between 500 and 2000 ppm.

### Bayesian Optimisation

```python
from skopt import gp_minimize
bo_res = gp_minimize(objective, dimensions=[(500, 2000)], n_calls=30, ...)
```

**Bayesian Optimisation** uses a probabilistic model (Gaussian Process) to intelligently decide which Cp to try next. Instead of random search, it models uncertainty and tries points where it thinks improvement is most likely. Efficient with few function evaluations.

### PSO (Particle Swarm Optimisation)

```python
pso = ps.single.GlobalBestPSO(n_particles=15, ...)
cost, pos = pso.optimize(objective, iters=40)
```

**PSO** — inspired by how birds flock or fish school. 15 "particles" (candidate Cp values) move through the search space:
- Each particle remembers its personal best location
- Each particle is attracted toward the global best found by any particle
- Particles gradually converge on the optimum

### Genetic Algorithm

```python
ga_res = differential_evolution(
    objective, bounds=[...],
    strategy="best1bin", mutation=(0.5, 1.5), recombination=0.7, ...
)
```

**Genetic Algorithm (GA)** — inspired by biological evolution:
- Population of solutions → select the best → mutate → crossover → new generation
- `strategy="best1bin"` → specific mutation strategy
- `mutation=(0.5, 1.5)` → how much to mutate
- `recombination=0.7` → 70% chance of accepting mutated offspring

Using four different optimisers and comparing results gives us **confidence** that the answer is correct — if all four agree on ~1200-1400 ppm, we're likely right.

---

## Part 12: Plotting — Making Graphs

```python
def plot_production(results, t_months, out_dir=OUTPUT_DIR):
    colors = {"P1": "#1f77b4", "P2": "#ff7f0e", "P3": "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
```

`plt.subplots(2, 3, figsize=(16, 9))` — create a figure with a 2×3 grid of subplots. 2 rows (OPR, WC) × 3 columns (P1, P2, P3) = 6 panels total.

`figsize=(16, 9)` — 16 inches wide × 9 inches tall (widescreen).

```python
    for col, w in enumerate(["P1", "P2", "P3"]):
        ax_opr = axes[0, col]   # top row, column col
        ax_wc  = axes[1, col]   # bottom row, column col

        ax_opr.plot(t_months, opr_ref, "k--", lw=1.5, label="CMG STARS")
        ax_opr.plot(t_months, results[w]["OPR"], color=colors[w], lw=2, label="PINN")
```

`"k--"` — `k` = black, `--` = dashed line.
`lw=1.5` — line width.
`label="CMG STARS"` — text for the legend.

```python
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
```

`tight_layout()` — automatically adjust spacing so labels don't overlap.
`savefig(path, dpi=150)` — save to file at 150 dots per inch (good quality).
`plt.close()` — free memory. Always close figures you've saved!

---

## Part 13: The `main()` Function — Putting It All Together

```python
def main():
    # 1. Build models
    sw_model   = SaturationSubmodel(name="sat_submodel")
    prod_heads = {f"P{i+1}": ProductionHead(f"P{i+1}", ...)
                  for i in range(P.n_producers)}
```

`f"P{i+1}"` — an **f-string** (formatted string). The `{}` injects the value of the expression inside. When `i=0`, this gives `"P1"`.

```python
    # 2. Warm-up: build weights by passing dummy data
    dummy_sat  = tf.zeros((2, 3), dtype=tf.float32)
    _ = sw_model(dummy_sat, training=False)
```

Neural networks in Keras don't actually create their weights until they see real data (they need to know the input shape). We pass `tf.zeros((2, 3))` — a tiny 2-row, 3-column array of zeros — just to trigger weight creation.

`_` = the output is discarded (we only care about the side effect of creating weights).

```python
    # 3. Train
    hist, t_months, t_norm, Cp_grid = train(sw_model, prod_heads)

    # 4. Predict
    results = predict(sw_model, prod_heads, t_norm, Cp_grid)

    # 5. Evaluate
    metrics = evaluate(results, t_months, sw_model)

    # 6-8. Plot, ablate, optimise, save
    ...
```

Everything runs in order. Python is procedural — it executes top to bottom.

```python
if __name__ == "__main__":
    main()
```

`__name__ == "__main__"` is True only when you run this file directly (`python pinn_polymer_flooding.py`). If another file imports this one, `__name__` would be the module name, and `main()` wouldn't run automatically. This is the standard Python pattern for runnable scripts.

---

## Summary: The Complete Flow

```
MANUSCRIPT PARAMETERS
        │
        ▼
CMG REFERENCE DATA         COLLOCATION POINTS
(WC/OPR curves from        (random x, t, Cp
 Tables 6 & 7)              for physics)
        │                         │
        ▼                         ▼
┌──────────────────────────────────────────┐
│              TRAINING LOOP               │
│                                          │
│  Phase 1: Physics only (500 epochs)      │
│  ┌─────────────────────────────────┐     │
│  │  SaturationSubmodel learns:     │     │
│  │  - BL PDE (water conservation)  │     │
│  │  - Polymer transport            │     │
│  │  - IC: Sw(x,0) = 0.36 (FWM)    │     │
│  │  - BC: Sw(0,t) = 0.80          │     │
│  └─────────────────────────────────┘     │
│                                          │
│  Phase 2: Physics + Data (800 epochs)    │
│  ┌─────────────────────────────────┐     │
│  │  SaturationSubmodel continues   │     │
│  │  + ProductionHead P1 learns:    │     │
│  │    - Match CMG WC curve         │     │
│  │    - Match CMG OPR curve        │     │
│  │    - WC = fw (physics link)     │     │
│  │    - WC(0) = 0.168              │     │
│  │  + ProductionHead P2 (same)     │     │
│  │  + ProductionHead P3 (same)     │     │
│  └─────────────────────────────────┘     │
└──────────────────────────────────────────┘
        │
        ▼
   PREDICTIONS
   OPR(t), WC(t) for P1, P2, P3
        │
        ├── EVALUATION vs CMG STARS
        │   (R², NRMSE, CUM_ERR per well)
        │
        ├── ABLATION STUDY
        │   (PINN vs plain MLP — how much does physics help?)
        │
        └── OPTIMISATION
            (DE + BO + PSO + GA find best Cp ∈ [500,2000] ppm)
```

---

## Key Concepts to Remember

| Concept | Simple Explanation |
|---|---|
| **Neural Network** | A brain made of math — learns patterns from examples |
| **PINN** | Neural network that also obeys physics equations |
| **Saturation Sw** | How wet the rock is (0=dry, 1=fully wet) |
| **Fractional flow fw** | Fraction of liquid that is water (0=all oil, 1=all water) |
| **Corey kr** | How easily each fluid flows at a given wetness |
| **Buckley-Leverett PDE** | Physics equation for how water moves through rock |
| **GradientTape** | TensorFlow's recorder for automatic differentiation |
| **Loss function** | Score of how wrong the model is (we want to minimise it) |
| **Gradient descent** | Method of adjusting weights to reduce loss |
| **Epoch** | One full training pass through the data |
| **FWM** | Mobile Water Fraction — extra water missed by lab tests |
| **Collocation points** | Random points where we check if physics is satisfied |
| **Early stopping** | Stop training when performance stops improving |

---

## How to Apply This to a Different Field

If you wanted to do this for a completely different oil field, you would:

1. **Replace `class Params`** with your field's numbers from your own paper/report
2. **Replace `wc_initial` and `wc_final`** with your CMG or field-measured endpoints
3. **Adjust `n_producers`** if you have a different number of wells
4. **Check `inj_schedule`** — update the polymer injection timing for your field
5. **Keep all the code structure identical** — only the numbers in `Params` change

The physics (BL PDE, Corey kr, fractional flow) is universal — it applies to any oil reservoir. Only the specific values differ from field to field.

This is the power of PINN: **one framework, endlessly reusable.**
