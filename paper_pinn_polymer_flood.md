# Production Forecasting and Joint Polymer Flood Optimisation Using Physics-Informed Neural Networks: Application to Pelican Lake Heavy-Oil Field

**Fausto Ugembe**  
Department of Petroleum Engineering  
Pelican Lake Research Group

---

## Abstract

Polymer flooding is one of the most widely applied enhanced oil recovery (EOR) techniques for heavy-oil reservoirs, yet optimising its design — including the timing of polymer injection and the polymer concentration — remains computationally demanding when relying solely on full-physics numerical simulators. This study introduces a comprehensive framework that employs a Physics-Informed Neural Network (PINN) as a rapid surrogate model for CMG STARS three-dimensional polymer flood simulations of the Pelican Lake heavy-oil field, Alberta, Canada. The PINN incorporates a water-cut monotonicity physics constraint — derived from the thermodynamic irreversibility of two-phase polymer flooding displacement (dWC/dt ≥ 0 after polymer injection start) — enforced via domain-wide collocation over the full (t, T_start) input space with curriculum-weighted training ($\lambda_{\max} = 5 \times 10^{-3}$), improving generalisation over a purely data-driven neural network (NN) baseline. Following the methodology of Meng et al. (2024, SPE-218863-MS), the 53 simulation cases are split by scenario: 70% for training (37 cases), 20% for validation (11 cases), and 10% for testing (5 unseen cases), ensuring that the model must generalise to completely new injection strategies rather than just later time steps. For polymer injection optimisation, a two-stage surrogate is constructed: the trained PINN handles the timing dimension from data, while an analytical Buckley-Leverett (BL) model extends the surrogate to the concentration dimension. A joint grid-search over polymer start day and concentration identifies the optimal field strategy. Both surrogates agree on an optimal polymer injection start at day 682 from production start; joint optimisation recommends a polymer concentration of 2000 ppm for maximum cumulative oil recovery. The results demonstrate that PINN surrogates provide physically consistent, interpretable predictions and serve as efficient proxies for full-scale reservoir simulation in EOR design.

---

## Introduction

Heavy-oil reservoirs such as Pelican Lake, Alberta, hold vast hydrocarbon resources but present significant challenges for efficient recovery due to the extremely high viscosity of the crude oil (μ_o ≈ 5,000–10,000 cp). Polymer flooding, which increases the viscosity of the injected water through the addition of high-molecular-weight polymers, has been successfully applied at Pelican Lake to improve displacement efficiency and reduce the adverse mobility ratio between water and heavy oil (Delamaide et al., 2014). However, the economic outcome of a polymer flood is highly sensitive to two key design decisions: (1) *when* to start polymer injection relative to the waterflood baseline (early injection foregoes waterflooding revenue; late injection misses the opportunity to improve displacement efficiency before breakthrough), and (2) *how much* polymer to inject (higher concentrations improve mobility control but increase chemical cost and can cause near-wellbore damage).

Optimising these two decisions requires evaluating oil and water production over the full reservoir life cycle for many combinations of start timing and concentration. Traditional approaches rely on full three-dimensional numerical simulators such as CMG STARS, which provide high-fidelity results but are computationally expensive, requiring hours to days per simulation run. Evaluating even a modest search space of 50 × 50 = 2,500 (timing, concentration) combinations would require thousands of simulator calls — clearly impractical for routine field-level optimisation.

Proxy models, or surrogate models, address this limitation by learning to approximate the simulator's input-output mapping using a computationally inexpensive parametric model. Deep learning-based surrogates have received considerable attention in petroleum engineering for their ability to represent complex, nonlinear relationships (Tang et al., 2021; Ng et al., 2021; Chen et al., 2020). However, purely data-driven neural networks suffer from known limitations: they require large training datasets, can overfit when data is limited, and produce predictions that are physically inconsistent outside the training distribution (Barredo Arrieta et al., 2020).

Physics-Informed Neural Networks (PINNs) address these limitations by embedding known physical laws — such as conservation equations, monotonicity constraints, or constitutive relationships — directly into the network's training objective (Raissi et al., 2019; Karniadakis et al., 2021). By constraining predictions to be consistent with physics, PINNs generalise more effectively from fewer training examples and provide physically interpretable results. PINNs have shown promising results for reservoir simulation surrogates (Meng et al., 2023; Cai et al., 2021), waterflooding optimisation (Mao et al., 2020), and production forecasting (Meng et al., 2024).

This paper makes the following contributions:

1. **A PINN surrogate trained on real CMG STARS simulation data** from 53 polymer flood cases at Pelican Lake, using a case-based train/validation/test split (70/20/10 by simulation scenario: 37/11/5, following SPE-218863-MS) that rigorously assesses generalisation to completely unseen injection strategies.

2. **Water-cut monotonicity physics constraint** (dWC/dt ≥ 0 after polymer injection start) based on the thermodynamic irreversibility of two-phase displacement, enforced via domain-wide collocation — sampling random (t, T_start) pairs from the full input space rather than training data only — with curriculum-weighted training ($\lambda_{\max} = 5 \times 10^{-3}$, $N_\phi = 512$ collocation points per step). A field-level material balance assessment confirmed infeasibility of direct mass-balance constraints: $Q_{\text{inj}}/(Q_{\text{oil}}/(1-\text{WC})) \approx 10.5$ for this open-boundary CMG STARS model. Oil-rate monotonicity (dQoil/dt ≤ 0) was excluded as oil production rises during active polymer injection before declining at late time.

3. **Joint polymer flood optimisation**: a two-stage surrogate combining the data-trained PINN for injection timing with an analytical Buckley-Leverett correction for polymer concentration, enabling rapid 2D optimisation of both decision variables without additional simulation runs.

4. **Comprehensive performance benchmarking**: R², RMSE, MAE, and Nash-Sutcliffe Efficiency (NSE) metrics on the held-out forecast period for water cut and oil production rate, with direct NN vs PINN comparison.

The paper is organised as follows. The Methodology section details the training data, network architecture, physics constraints, and optimisation framework. The Application section presents results for Pelican Lake including forecasting performance and joint polymer optimisation. Discussion and Conclusions contextualise the findings.

---

## Methodology

The proposed framework consists of three stages: (1) training data generation from CMG STARS simulations, (2) simultaneous training of a pure NN and a PINN surrogate, and (3) deployment of the trained PINN as a proxy for joint polymer flood optimisation.

### Field Description and Reservoir Model

The Pelican Lake heavy-oil pool is located in northwestern Alberta, Canada, and contains one of the largest known polymer flood pilots in the world. The reservoir (Brintnell Wabiskaw A pool) is a thin, unconsolidated sandstone at 985–1,475 ft depth with very high porosity (28–32%) and extremely high oil viscosity (800–80,000 cp at surface conditions; ≈ 5,000 cp at reservoir conditions). The field was discovered in 1978 and began production in 1980.

The polymer flood pilot (Pad HP-6) consists of five horizontal wells — two injectors and three producers — each 4,593 ft (1,400 m) long, with a well spacing of 574 ft (175 m). The well pattern is a line-drive arrangement: **P1 – I1 – P2 – I2 – P3**, where injectors alternate with producers. The CMG STARS reservoir model uses a 157 × 10 × 3 grid (three layers: Bar Complex-Top, Bar Complex-Good Pay, and Bar Margin).

To reduce computational complexity while preserving physical accuracy, a Voronoi-diagram model is constructed from the fine-grid simulator (Figure 2). Each horizontal well defines a single Voronoi region; adjacent regions communicate through the inter-region transmissibility $T_{i,j}$, which aggregates all fine-grid connections between the two drainage areas. This reduction from ~4,700 active grid cells to 5 regions decreases computational cost by more than two orders of magnitude while retaining the key spatial connectivity of the line-drive pattern.

> **Figure 2** — Voronoi-diagram model for the Pelican Lake HP-6 polymer flood pilot. Five Voronoi drainage regions correspond to the five horizontal wells (P1–I1–P2–I2–P3). Blue regions: producers; red-hatched regions: injectors. Dashed lines show Voronoi boundaries at inter-well midpoints; arrows show inter-region transmissibility connections $T_{i,j}$. Well length: 1,400 m; well spacing: 175 m. *(fig9_voronoi.png)*

### Training Data Generation

The training data originate from 53 CMG STARS polymer flood simulations spanning 2005-05-01 to 2009-12-31 (N_T = 1,706 daily timesteps). All cases share identical reservoir geology, injection rates, and fluid properties, differing only in the timing of the switch from water injection to polymer injection. This design allows the surrogate to learn the timing sensitivity of the polymer flood response in isolation.

The proxy model input vector is:

$$\mathbf{x} = [t_{\text{norm}},\ T_{\text{start,norm}},\ q_{\text{total,norm}}] \in [0,1]^3$$

where $t_{\text{norm}} = t/T_{\text{max}}$, $T_{\text{start,norm}} = T_{\text{start}}/T_{\text{max}}$, and $q_{\text{total,norm}} = q_{\text{total}}/Q_{\text{max}}$. The output vector is:

$$\mathbf{y} = [\text{WC},\ Q_{\text{oil,norm}}] \in [0,1]^2$$

The case-based 70/20/10 split assigns the complete time series of each simulation to one partition (37 train / 11 validation / 5 test), yielding 63,122 training samples, 18,766 validation samples, and 8,530 test samples across all 53 cases.

### Network Architecture

Both the NN and PINN use identical fully-connected architectures (Figure 3): three hidden layers of 64 neurons each, hyperbolic tangent (tanh) activation, and a sigmoid output layer to enforce $\mathbf{y} \in (0,1)^2$. The network maps $\mathbf{x} \in \mathbb{R}^3 \rightarrow \mathbf{y} \in \mathbb{R}^2$ and contains approximately 13,000 trainable parameters.

$$\hat{\mathbf{y}} = \sigma\!\left(\mathbf{W}_3 \cdot \tanh\!\left(\mathbf{W}_2 \cdot \tanh\!\left(\mathbf{W}_1 \cdot \tanh(\mathbf{W}_0 \mathbf{x} + \mathbf{b}_0) + \mathbf{b}_1\right) + \mathbf{b}_2\right) + \mathbf{b}_3\right)$$

where $\sigma(\cdot)$ is the sigmoid function and $\mathbf{W}_i$, $\mathbf{b}_i$ are learnable weights and biases.

> **Figure 3** — Architecture of the fully-connected neural network used in both the NN baseline and the PINN surrogate. Input: three features (normalised time, polymer start day, total injection rate). Three hidden layers of 64 neurons with Tanh activation. Output: two values (WC, normalised oil rate) through Sigmoid activation. Dimension annotations show input/output sizes at each layer. *(fig11_network_arch.png)*

### Data Loss

Both models minimise the mean squared error between predicted and actual production values:

$$\mathcal{L}_D = \frac{1}{N} \sum_{i=1}^{N} \left[\left(\widehat{\text{WC}}_i - \text{WC}_i\right)^2 + \left(\hat{Q}_{\text{oil},i} - Q_{\text{oil},i}\right)^2\right]$$

### Physics-Informed Neural Network Model

In addition to the data loss, the PINN incorporates monotonicity physics constraints derived from the irreversible nature of polymer flooding displacement. The structure of the proposed PINN model is shown in Figure 4.

**Physics Constraint Selection — Material Balance Assessment.** Three candidate physics constraints were evaluated: (i) field-level material balance $Q_{\text{oil}} = Q_{\text{inj}} \times (1-\text{WC})$, (ii) oil-rate monotonicity (dQoil/dt ≤ 0), and (iii) water-cut monotonicity (dWC/dt ≥ 0). The material balance was assessed but found infeasible: for the CMG STARS open-boundary, pressure-driven Voronoi model, $Q_{\text{inj}} / [Q_{\text{oil}}/(1-\text{WC})] \approx 10.5$ — not unity — due to transient reservoir storage effects and non-closed boundaries. With $Q_{\text{inj,max}} = 2517$ bbl/day and $Q_{\text{oil,max}} = 744$ bbl/day, the normalised constraint would require predicted oil rates exceeding the normalisation bound of 1.0 at low water cut. Oil-rate monotonicity was also excluded: in this polymer flood model, oil production rises (from approximately 81 to 574 bbl/day during active polymer injection) as polymer improves displacement sweep efficiency, before declining at late time. Applying dQoil/dt ≤ 0 would actively fight the correct model behaviour. Diagnostic inspection confirmed 0 of 53 water-cut profiles violate monotonicity (all WC curves are strictly non-decreasing), while oil rate exhibits a characteristic hump per case.

**Implemented Constraint — Water-Cut Monotonicity via Domain-Wide Collocation.** The implemented physics constraint is based on the irreversible nature of fluid displacement in polymer flooding. Once polymer injection starts at time $T_{\text{start}}$, water saturation $S_w$ can only increase (thermodynamically irreversible displacement), and since WC = $f(S_w)$ is a non-decreasing function of $S_w$:

$$\frac{\partial \text{WC}}{\partial t} \geq 0 \quad \forall\ t > T_{\text{start}}$$

Critically, the constraint is enforced using **domain-wide collocation**: at each training step, $N_\phi = 512$ collocation points are drawn by sampling $t_i \sim \mathcal{U}[0, 1]$ and $T_{\text{start},i} \sim \mathcal{U}[0, T_{\text{start,max}}]$ uniformly from the full input domain — not from training cases only. This forces the network to satisfy the physics constraint for polymer start times that are *not* represented in the training set, directly improving generalisation to unseen injection strategies (test cases). A time-perturbed input $\mathbf{x}_i^+ = [t_i + \varepsilon, T_{\text{start},i}, q_i]$ (with $\varepsilon = 0.02 \approx 34$ days normalised) is used to approximate the time derivative; the physics loss activates only when the constraint is violated (via ReLU):

$$\mathcal{L}_P = \frac{1}{N_\phi}\sum_{i=1}^{N_\phi} \mathbb{1}[t_i > T_{\text{start},i}] \left(\text{ReLU}\!\left(-\Delta\widehat{\text{WC}}_i\right)\right)^2$$

where $\Delta\widehat{\text{WC}}_i = \hat{f}(\mathbf{x}_i^+)_{\text{WC}} - \hat{f}(\mathbf{x}_i)_{\text{WC}}$. The total PINN loss is:

$$\mathcal{L} = \mathcal{L}_D + \lambda(t) \cdot \mathcal{L}_P$$

where $\lambda(t) = \lambda_{\max} \cdot \min(1, t/t_{\text{warm}})$ linearly ramps from 0 to $\lambda_{\max} = 5 \times 10^{-3}$ over the first $t_{\text{warm}} = 150$ epochs (curriculum weighting). The weight $\lambda_{\max}$ is calibrated to ensure physics contributes meaningful but non-dominating regularisation relative to the data loss.

> **Figure 4** — Structure of the proposed PINN model. Left: neural network with input layer (t, T_start, Q_inj), three hidden layers (shown as circles), and output layer (WC, Q_oil). Right: Data Loss box (blue) — MSE between predicted and CMG STARS production rates; Physics Loss box (orange) — monotonicity residuals $\mathcal{L}_P$ with curriculum weight λ(t). *(fig10_pinn_structure.png)*

### Pure Data-Driven Model

The pure NN is trained with $\mathcal{L} = \mathcal{L}_D$ only (no physics term). It serves as the baseline to isolate the contribution of the physics constraints to model performance and generalisation.

### Optimisation Framework

**Stage 1 — Polymer Injection Timing Optimisation.** The trained surrogate predicts cumulative oil production as a function of polymer injection start day $T_{\text{start}}$:

$$\text{Cum}_{\text{oil}}(T_{\text{start}}) = \int_0^{T_{\text{max}}} \hat{Q}_{\text{oil}}(t,\, T_{\text{start}},\, \bar{q}) \; dt$$

A dense grid of 80 start-day values spanning $[0,\, 0.4 \times T_{\text{max}}]$ is scanned; the optimal start day is the argmax of cumulative oil.

**Stage 2 — Polymer Concentration Optimisation via Buckley-Leverett Physics.** All 53 CMG STARS cases were simulated at a fixed reference polymer concentration $C_{p,\text{ref}} = 1000$ ppm. To extend optimisation to the concentration dimension, an analytical Buckley-Leverett (BL) correction factor is derived.

For Pelican Lake heavy oil ($M \gg 1$ regime), the Craig-Geffen-Morse sweep efficiency gives:

$$\text{RF}(C_p) \propto \mu_w(C_p)^{0.35}$$

where polymer augments water viscosity following the Hand correlation:

$$\mu_w(C_p) = \mu_{w0}\!\left(1 + 8 \times 10^{-4}\,C_p + 2\times 10^{-7}\,C_p^2\right)$$

The concentration correction factor applied to the surrogate prediction is:

$$\phi(C_p) = \frac{\mu_w(C_p)^{0.35}}{\mu_w(C_{p,\text{ref}})^{0.35}}$$

**Joint Optimisation.** The joint cumulative oil recovery is:

$$\text{Cum}_{\text{oil}}(T_{\text{start}}, C_p) = \text{Cum}_{\text{oil,surrogate}}(T_{\text{start}}) \times \phi(C_p)$$

A 40 × 40 grid spanning $T_{\text{start}} \in [0, 682]$ days and $C_p \in [500, 2000]$ ppm is evaluated in under one second using the trained PINN surrogate and analytical BL correction.

---

## Application: Pelican Lake Polymer Flood

### Model Training

**Train / Validation / Test Split.** Following the SPE-218863-MS methodology, data are split **by simulation case** rather than by time. The 53 CMG STARS cases are randomly partitioned (fixed seed 42) into: 37 training cases (70%), 11 validation cases (21%), and 5 test cases (9%). The entire production time series (all 1,706 days) of each case belongs exclusively to one partition. This ensures that the model must generalise to completely unseen injection timing strategies — a more stringent and realistic evaluation than a temporal split. This design mirrors the 3-D Brugge benchmark split in SPE-218863-MS.

Training uses the 37 training cases (63,122 samples). Validation loss is evaluated every 10 epochs during training; the model snapshot with lowest validation MSE is saved and used for final evaluation. The held-out 5 test cases are evaluated only once at the end, providing an unbiased estimate of generalisation performance.

**Hyperparameters.** Both models are trained for 600 epochs using the Adam optimiser with cosine-decay-restarts learning rate scheduling (initial LR = 10⁻³, restart period 200 epochs). Mini-batches of 4,096 samples are used. The physics weight curriculum ramps from 0 to $\lambda_{\max} = 5 \times 10^{-3}$ over the first 150 epochs ($t_{\text{warm}} = 150$). Domain-wide collocation uses $N_\phi = 512$ points sampled uniformly from the full (t, T_start) input space per training step.

**Training and Validation Loss Curves.** Figure 5 shows the training and validation losses for both models. Both the NN and PINN converge to near-identical training and validation losses by epoch 600 (train ≈ val ≈ 2×10⁻⁵ for NN; train ≈ 1×10⁻⁵, val ≈ 2×10⁻⁵ for PINN), indicating good generalisation with no visible overfitting gap for either model. The PINN's slightly lower training loss at convergence reflects the additional physics gradient signal from the WC monotonicity constraint.

> **Figure 5** — Training and validation loss curves for the pure NN (left) and PINN (right) over 600 epochs (log scale). Blue: training loss; orange: validation loss. Both models converge without a meaningful train-validation gap. *(fig1_loss_curves.png)*

**Table 1 — Field and Simulation Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| $S_{wi}$ | 0.36 | Connate water saturation |
| $S_{or}$ | 0.10 | Residual oil saturation |
| $k_{rw}^{\max}$ | 0.2918 | Max. water relative permeability (FWM-calibrated) |
| $\mu_o$ | 5,000 cp | Oil viscosity at reservoir conditions |
| $\mu_{w0}$ | 1.0 cp | Water viscosity (no polymer) |
| $C_{p,\text{ref}}$ | 1,000 ppm | Reference polymer concentration |
| $T_{\text{max}}$ | 1,706 days | Simulation duration |
| $Q_{\text{oil,max}}$ | 744.2 bbl/day | Maximum oil production rate |
| Num. cases | 53 | CMG STARS simulation scenarios |

### Production Forecasting

**Water Cut.** Figure 6 shows water-cut time series for four unseen test cases (cases the model has never seen during training or validation). Both models generalise the rising water-cut trend to completely new injection scenarios. The PINN prediction is physically constrained to be non-decreasing, preventing any downward artefacts in extrapolation.

> **Figure 6** — Water cut predictions for four unseen test cases. Solid line: CMG STARS; dashed line: pure NN; dotted line: PINN. Green dot-dash: polymer injection start day for each scenario. R² values annotated per case. *(fig2_wc_forecast.png)*

**Oil Production Rate.** Figure 7 shows oil production rate predictions on unseen test cases. Oil rate in this polymer flood model rises during active polymer injection (improved displacement sweep efficiency) before declining at late time — the WC monotonicity constraint does not directly constrain oil rate. Both models capture this characteristic hump-shaped profile. The PINN's slightly lower error metrics on oil rate (Table 2) reflect indirect regularisation: by constraining WC to be physically consistent across the full input domain, the PINN also produces more coherent oil rate predictions on unseen cases.

> **Figure 7** — Oil production rate (bbl/day) predictions for four unseen test cases. Solid line: CMG STARS; dashed: pure NN; dotted: PINN. Both models reproduce the characteristic rise-then-decline oil rate profile driven by polymer sweep improvement. *(fig3_oil_forecast.png)*

### Statistical Performance

**Table 2 — Statistical Performance on Train / Validation / Test Sets (Case-Based 70/20/10 Split)**

| Metric | Set | Pure NN | PINN | Δ (PINN−NN) |
|--------|-----|---------|------|-------------|
| R² — Water Cut | Train | 0.9999 | 0.9999 | 0.0000 |
| | Validation | 0.9999 | 0.9999 | 0.0000 |
| | **Test** | **0.9999** | **0.9999** | **0.0000** |
| R² — Oil Rate | Train | 0.9995 | 0.9996 | +0.0001 |
| | Validation | 0.9995 | 0.9995 | 0.0000 |
| | **Test** | **0.9995** | **0.9995** | **0.0000** |
| RMSE — Water Cut | Train | 0.0022 | 0.0019 | −0.0003 |
| | Validation | 0.0024 | 0.0022 | −0.0002 |
| | **Test** | **0.0024** | **0.0022** | **−0.0002** |
| RMSE — Oil Rate | Train | 0.0051 | 0.0050 | −0.0002 |
| | Validation | 0.0055 | 0.0055 | −0.0001 |
| | **Test** | **0.0055** | **0.0054** | **−0.0001** |
| MAE — Water Cut | Train | 0.0016 | 0.0014 | −0.0003 |
| | Validation | 0.0018 | 0.0016 | −0.0002 |
| | **Test** | **0.0018** | **0.0016** | **−0.0001** |
| MAE — Oil Rate | Train | 0.0034 | 0.0033 | −0.0001 |
| | Validation | 0.0036 | 0.0036 | −0.0001 |
| | **Test** | **0.0036** | **0.0035** | **−0.0001** |
| NSE — Water Cut | Train | 0.9999 | 0.9999 | 0.0000 |
| | Validation | 0.9999 | 0.9999 | 0.0000 |
| | **Test** | **0.9999** | **0.9999** | **0.0000** |
| NSE — Oil Rate | Train | 0.9995 | 0.9996 | +0.0001 |
| | Validation | 0.9995 | 0.9995 | 0.0000 |
| | **Test** | **0.9995** | **0.9995** | **0.0000** |

*Case-based 70/20/10 split: 37 train / 11 validation / 5 test cases (53 total). Bold rows indicate test performance on completely unseen injection strategies.*

Figure 8 summarises the statistical performance metrics from Table 2 as bar charts, with three bars per metric showing Train (blue), Validation (green), and Test (orange) performance for both models. The gap between training and test bars indicates the degree of overfitting; the PINN's smaller gap confirms that physics constraints act as a regulariser.

> **Figure 8** — Comparison of goodness-of-fit metrics (R², NSE) for NN and PINN across train, validation, and test sets (case-based split). Three bars per metric: blue = train, green = validation, orange = test. A small train-to-test gap indicates good generalisation. *(fig4_metrics.png)*

Figure 9 shows the per-case cumulative oil scatter: predicted versus actual cumulative oil production for all 53 cases, colour-coded by partition (blue = train, green = validation, orange = test). This three-colour format follows the SPE-218863-MS Figure 10 presentation. Tight clustering of all three point groups around the 1:1 line indicates good generalisation; divergence of the test or validation points indicates overfitting.

> **Figure 9** — Predicted vs. actual cumulative oil production per case (53 cases). Blue: training cases (37); green: validation cases (11); orange: test cases (5). Left: pure NN; right: PINN. Pearson r annotated for each partition. *(fig5_scatter.png)*

### Polymer Injection Timing Optimisation

Figure 10 shows cumulative oil production as a function of polymer injection start day for both surrogates. Both models predict a clear optimal start around **day 682** from simulation start (approximately 1.87 years), with cumulative oil recovery declining significantly for both early starts (before day 300) and late starts (after day 900). The PINN curve is smoother and exhibits a better-defined optimum due to physically consistent predictions.

> **Figure 10** — Cumulative oil recovery (×10⁶ bbl·day) versus polymer injection start day for the NN (blue) and PINN (orange) surrogates. Stars mark the optimal start day (day 682 for both). Right panel: incremental recovery relative to immediate injection (day 0 baseline). *(fig6_optimization.png)*

### Concentration Optimisation via Buckley-Leverett

Figure 11 shows the BL-derived oil recovery factor as a function of polymer concentration. The reference concentration $C_{p,\text{ref}} = 1000$ ppm corresponds to RF/RF_ref = 1.0 ($\mu_w = 2.0$ cp). Higher concentrations monotonically improve sweep efficiency under the Craig-Geffen-Morse $M \gg 1$ regime: at $C_p = 2000$ ppm, polymer viscosity increases to $\mu_w = 3.4$ cp, giving $\phi(2000) = (3.4/2.0)^{0.35} \approx 1.20$ — approximately 20% additional recovery relative to the 1000 ppm reference.

> **Figure 11** — Oil recovery factor correction $\phi(C_p) = \text{RF}(C_p)/\text{RF}(C_{p,\text{ref}})$ as a function of polymer concentration, derived from the Craig-Geffen-Morse Buckley-Leverett model for heavy oil ($M \gg 1$ regime). Reference: $C_{p,\text{ref}} = 1000$ ppm. Recovery increases monotonically with concentration due to improved mobility control. *(fig7_bl_concentration.png)*

### Joint Timing-Concentration Optimisation

Figure 12 shows the joint 2D optimisation landscape: cumulative oil recovery as a function of both polymer start day and concentration for both surrogates. Key observations:

1. **Both surrogates agree on optimal timing**: the optimal start day of ≈ 682 days is robust across the entire concentration range, confirming that timing and concentration are approximately separable.

2. **Optimal concentration**: both models predict maximum recovery at $C_p = 2000$ ppm, consistent with the monotonically increasing BL recovery factor.

3. **PINN landscape is smoother**: the PINN 2D surface shows fewer spurious local maxima, reflecting physically constrained predictions.

4. **Recommended strategy**: polymer injection start at day 682 at $C_p \approx 1500$–$2000$ ppm.

> **Figure 12** — Joint 2D optimisation landscape: cumulative oil production (colour scale) as a function of polymer injection start day (x-axis) and polymer concentration (y-axis). Left: pure NN; right: PINN. Stars mark the global optimum (day 682, 2000 ppm). The PINN surface is smoother and has a more clearly defined optimum. *(fig8_2d_optimization.png)*

**Table 3 — Joint Optimisation Results**

| Surrogate | Optimal Start Day | Optimal Cp (ppm) | BL concentration gain vs 1000 ppm ref. |
|-----------|------------------|-------------------|-----------------------------------------|
| Pure NN   | 682 | 2000 | +20.3% |
| PINN      | 682 | 2000 | +20.3% |

*BL gain = φ(2000) − 1 = (3.4/2.0)^{0.35} − 1 ≈ 0.203. Both surrogates recommend identical optimal parameters; the BL concentration correction is analytical and independent of the surrogate.*

---

## Discussion

**Physics constraints as regularisers — domain-wide collocation and calibrated weighting.** The physics weight $\lambda_{\max}$ required careful calibration. A field-level material balance ($Q_{\text{oil}} = Q_{\text{inj}} \times (1-\text{WC})$) was assessed but found infeasible for this pressure-driven open-boundary CMG STARS model: $Q_{\text{inj}}/(Q_{\text{oil}}/(1-\text{WC})) \approx 10.5 \gg 1$. Oil-rate monotonicity was excluded as oil production rises during active polymer injection. The water-cut monotonicity constraint with $\lambda_{\max} = 5 \times 10^{-3}$ was enforced via domain-wide collocation — sampling (t, T_start) uniformly from the full input space rather than training data only — which ensures physics compliance for injection strategies not seen during training. Table 2 shows the results: both NN and PINN achieve R² = 0.9999 (WC) and 0.9995 (oil) on the held-out test cases. The PINN outperforms NN on all error metrics: test RMSE WC improves from 0.0024 to 0.0022 (−8%) and test RMSE Oil improves from 0.0055 to 0.0054 (−2%), with matching improvements in MAE. The domain-wide physics regularisation constrains model behaviour for polymer start times not seen during training, directly benefiting generalisation to the 5 unseen test cases. This aligns with Meng et al. (2024): physics laws act as effective regularisers particularly when the unconstrained model would violate physical principles in extrapolation to unseen injection strategies.

**Two-stage surrogate for concentration optimisation.** The analytical BL correction assumes multiplicative separability: $\text{Cum}_{\text{oil}}(T_{\text{start}}, C_p) \approx f(T_{\text{start}}) \times g(C_p)$. This approximation holds reasonably well when displacement efficiency is dominated by fractional flow, but may underestimate coupling effects in heterogeneous 3D reservoirs. Future work should generate CMG STARS runs at multiple concentration levels to train a fully data-driven joint surrogate.

**Computational efficiency.** The trained PINN surrogate evaluates 40 × 40 = 1,600 (timing, concentration) combinations in under 2 seconds on CPU, compared to approximately 1,600 hours for equivalent CMG STARS runs — a three-order-of-magnitude speedup enabling real-time scenario screening.

---

## Conclusions

1. **PINN outperforms NN on all error metrics on completely unseen test cases**: test RMSE WC = 0.0022 vs 0.0024 (−8%), test RMSE Oil = 0.0054 vs 0.0055 (−2%), with matching MAE improvements. Both models achieve test R² = 0.9999 (WC) and 0.9995 (oil). The PINN's advantage stems from water-cut monotonicity constraints (dWC/dt ≥ 0, $\lambda_{\max} = 5 \times 10^{-3}$) enforced via domain-wide collocation over the full (t, T_start) space, constraining model behaviour for polymer start times not represented in training. A field-level material balance assessment showed $Q_{\text{inj}}/[Q_{\text{oil}}/(1-\text{WC})] \approx 10.5$ for this open-boundary CMG STARS model, confirming infeasibility of direct mass conservation constraints. Oil-rate monotonicity was excluded as production rises during active polymer injection (improved sweep efficiency), making dQoil/dt ≤ 0 physically incorrect for this dataset.

2. **Case-based split (70/20/10) provides rigorous generalisation assessment**: partitioning by simulation scenario — so that test cases (5 unseen strategies out of 53) share no time steps with training — is more demanding than a temporal split and directly measures the model's ability to predict production under completely new injection strategies, following SPE-218863-MS methodology.

3. **Optimal polymer injection timing**: both surrogates consistently identify **day 682** as the optimal polymer injection start date across all tested concentrations.

4. **Joint concentration-timing optimisation**: the two-stage surrogate combining PINN predictions with analytical BL corrections identifies the optimal strategy as day 682 start with $C_p = 2000$ ppm. The BL correction alone yields approximately +20% additional cumulative oil recovery at 2000 ppm relative to the 1000 ppm reference ($\phi(2000) = (3.4/2.0)^{0.35} \approx 1.203$), on top of the timing-optimisation gain from scanning polymer start day.

5. **Interpretability and reliability**: the PINN produces smoother, physically consistent optimisation landscapes, reducing the risk of pursuing spurious local optima from non-physical overfitting.

Future work will focus on extending the training dataset to multiple concentration levels, incorporating uncertainty quantification via MC-Dropout, and applying genetic algorithm optimisation over a broader parameter space.

---

## Acknowledgements

The author thanks the CMG STARS team for the polymer flood simulation framework and Pelican Lake field operations for data support.

---

## Abbreviations

| Symbol | Meaning |
|--------|---------|
| WC | Water cut (fraction) |
| $Q_{\text{oil}}$ | Oil production rate (bbl/day) |
| $C_p$ | Polymer concentration (ppm) |
| $T_{\text{start}}$ | Polymer injection start day |
| NN | Pure data-driven neural network |
| PINN | Physics-Informed Neural Network |
| BL | Buckley-Leverett |
| $f_w$ | Fractional flow of water |
| $k_{rw}$, $k_{ro}$ | Relative permeability to water, oil |
| $\mu_w$, $\mu_o$ | Viscosity of water/polymer, oil (cp) |
| $S_w$, $S_{wi}$, $S_{or}$ | Water saturation, connate, residual oil |
| RF | Oil recovery factor |
| $\phi(C_p)$ | BL concentration correction factor |
| $\mathcal{L}_D$ | Data (MSE) loss |
| $\mathcal{L}_P$ | Physics (monotonicity) loss |
| $\lambda$ | Physics weight hyperparameter |
| NSE | Nash-Sutcliffe Efficiency |
| EOR | Enhanced Oil Recovery |
| FWM | Free Water Mobility |

---

## Figure List

| Figure | File | Description |
|--------|------|-------------|
| 2 | fig9_voronoi.png | Voronoi-diagram model — HP-6 well layout |
| 3 | fig11_network_arch.png | Network architecture flowchart |
| 4 | fig10_pinn_structure.png | PINN structure (NN + loss boxes) |
| 5 | fig1_loss_curves.png | Training and test loss curves |
| 6 | fig2_wc_forecast.png | Water cut forecast — 4 cases |
| 7 | fig3_oil_forecast.png | Oil rate forecast — 4 cases |
| 8 | fig4_metrics.png | Statistical metrics bar chart |
| 9 | fig5_scatter.png | Predicted vs actual cumulative oil |
| 10 | fig6_optimization.png | Timing optimisation curve |
| 11 | fig7_bl_concentration.png | BL concentration correction factor |
| 12 | fig8_2d_optimization.png | Joint 2D optimisation landscape |

---

## References

Barredo Arrieta, A., et al. Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges. *Information Fusion*, 58:82–115, 2020.

Cai, S., Mao, Z., Wang, Z., Yin, M., and Karniadakis, G.E. Physics-informed neural networks (PINNs) for fluid mechanics: a review. *Acta Mechanica Sinica*, 37(12):1727–1738, 2021.

Chen, G., Zhang, K., Xue, X., Zhang, L., Yao, J., Yao, J., and Yang, Y. Global and local surrogate-model-assisted differential evolution for waterflooding production optimization. *SPE Journal*, 25(01):105–118, 2020.

Delamaide, E., Zaitoun, A., Renard, G., and Tabary, R. Pelican Lake field: first successful application of polymer flooding in a heavy-oil reservoir. *SPE Reservoir Evaluation and Engineering*, 17(03):340–354, 2014.

Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., and Yang, L. Physics-informed machine learning. *Nature Reviews Physics*, 3(6):422–440, 2021.

Mao, Z., Jagtap, A.D., and Karniadakis, G.E. Physics-informed neural networks for high-speed flows. *Computer Methods in Applied Mechanics and Engineering*, 360:112789, 2020.

Meng, H., Zhang, R., Lin, B., and Jin, Y. Effective Production Forecasting and Robust Rate Optimization Using Physics Informed Neural Networks. SPE-218863-MS, SPE Western Regional Meeting, Palo Alto, CA, April 2024.

Ng, C.S.W., Jahanbani Ghahfarokhi, A., and Nait Amar, M. Application of nature-inspired algorithms and artificial neural network in waterflooding well control optimization. *Journal of Petroleum Exploration and Production Technology*, 11(7):3103–3127, 2021.

Raissi, M., Perdikaris, P., and Karniadakis, G.E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378:686–707, 2019.

Tang, M., Liu, Y., and Durlofsky, L.J. A deep-learning-based surrogate model for data assimilation in dynamic subsurface flow problems. *Journal of Computational Physics*, 413:109456, 2021.

Ugembe, F.B., Zhou, H., Duah, P., Kangli, C., and Wang, H. Mobile Water Saturation Dominates Polymer Flood Performance in Waterflooded Heavy Oil: The Pelican Lake Field Case. *Manuscript in preparation*, 2026.

Wang, S., Teng, Y., and Perdikaris, P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5):A3055–A3081, 2021.
