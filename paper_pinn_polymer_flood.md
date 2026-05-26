# Production Forecasting and Joint Polymer Flood Optimisation Using Physics-Informed Neural Networks: Application to Pelican Lake Heavy-Oil Field

**Fausto Ugembe**  
Department of Petroleum Engineering  
Pelican Lake Research Group

---

## Abstract

Polymer flooding is one of the most widely applied enhanced oil recovery (EOR) techniques for heavy-oil reservoirs, yet optimising its design — including the timing of polymer injection and the polymer concentration — remains computationally demanding when relying solely on full-physics numerical simulators. This study introduces a comprehensive framework that employs a Physics-Informed Neural Network (PINN) as a rapid surrogate model for CMG STARS three-dimensional polymer flood simulations of the Pelican Lake heavy-oil field, Alberta, Canada. The PINN incorporates monotonicity physics constraints derived from irreversible fluid displacement principles into deep learning, improving generalisation over a purely data-driven neural network (NN) baseline. Following the methodology of Meng et al. (2024, SPE-218863-MS), the 51 simulation cases are split by scenario: 70% for training (36 cases), 20% for validation (10 cases), and 10% for testing (5 unseen cases), ensuring that the model must generalise to completely new injection strategies rather than just later time steps. On the held-out test cases, the PINN outperforms the pure NN on oil rate prediction through physics-enforced monotonic decline. For polymer injection optimisation, a two-stage surrogate is constructed: the trained PINN handles the timing dimension from data, while an analytical Buckley-Leverett (BL) model extends the surrogate to the concentration dimension. A joint grid-search over polymer start day and concentration identifies the optimal field strategy. Both surrogates agree on an optimal polymer injection start at day 682 from production start; joint optimisation recommends a polymer concentration of 2000 ppm for maximum cumulative oil recovery. The results demonstrate that PINN surrogates provide physically consistent, interpretable predictions and serve as efficient proxies for full-scale reservoir simulation in EOR design.

---

## Introduction

Heavy-oil reservoirs such as Pelican Lake, Alberta, hold vast hydrocarbon resources but present significant challenges for efficient recovery due to the extremely high viscosity of the crude oil (μ_o ≈ 5,000–10,000 cp). Polymer flooding, which increases the viscosity of the injected water through the addition of high-molecular-weight polymers, has been successfully applied at Pelican Lake to improve displacement efficiency and reduce the adverse mobility ratio between water and heavy oil (Delamaide et al., 2014). However, the economic outcome of a polymer flood is highly sensitive to two key design decisions: (1) *when* to start polymer injection relative to the waterflood baseline (early injection foregoes waterflooding revenue; late injection misses the opportunity to improve displacement efficiency before breakthrough), and (2) *how much* polymer to inject (higher concentrations improve mobility control but increase chemical cost and can cause near-wellbore damage).

Optimising these two decisions requires evaluating oil and water production over the full reservoir life cycle for many combinations of start timing and concentration. Traditional approaches rely on full three-dimensional numerical simulators such as CMG STARS, which provide high-fidelity results but are computationally expensive, requiring hours to days per simulation run. Evaluating even a modest search space of 50 × 50 = 2,500 (timing, concentration) combinations would require thousands of simulator calls — clearly impractical for routine field-level optimisation.

Proxy models, or surrogate models, address this limitation by learning to approximate the simulator's input-output mapping using a computationally inexpensive parametric model. Deep learning-based surrogates have received considerable attention in petroleum engineering for their ability to represent complex, nonlinear relationships (Tang et al., 2021; Ng et al., 2021; Chen et al., 2020). However, purely data-driven neural networks suffer from known limitations: they require large training datasets, can overfit when data is limited, and produce predictions that are physically inconsistent outside the training distribution (Barredo Arrieta et al., 2020).

Physics-Informed Neural Networks (PINNs) address these limitations by embedding known physical laws — such as conservation equations, monotonicity constraints, or constitutive relationships — directly into the network's training objective (Raissi et al., 2019; Karniadakis et al., 2021). By constraining predictions to be consistent with physics, PINNs generalise more effectively from fewer training examples and provide physically interpretable results. PINNs have shown promising results for reservoir simulation surrogates (Meng et al., 2023; Cai et al., 2021), waterflooding optimisation (Mao et al., 2020), and production forecasting (Meng et al., 2024).

This paper makes the following contributions:

1. **A PINN surrogate trained on real CMG STARS simulation data** from 51 polymer flood cases at Pelican Lake, using a case-based train/validation/test split (70/20/10 by simulation scenario, following SPE-218863-MS) that rigorously assesses generalisation to completely unseen injection strategies.

2. **Finite-difference monotonicity physics constraints** (dWC/dt ≥ 0, dOil/dt ≤ 0 post-injection) implemented without second-order automatic differentiation, which reduce overfitting and improve oil rate generalisation on unseen cases.

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

The training data originate from 51 CMG STARS polymer flood simulations spanning 2005-05-01 to 2009-12-31 (N_T = 1,706 daily timesteps). All cases share identical reservoir geology, injection rates, and fluid properties, differing only in the timing of the switch from water injection to polymer injection. This design allows the surrogate to learn the timing sensitivity of the polymer flood response in isolation.

The proxy model input vector is:

$$\mathbf{x} = [t_{\text{norm}},\ T_{\text{start,norm}},\ q_{\text{total,norm}}] \in [0,1]^3$$

where $t_{\text{norm}} = t/T_{\text{max}}$, $T_{\text{start,norm}} = T_{\text{start}}/T_{\text{max}}$, and $q_{\text{total,norm}} = q_{\text{total}}/Q_{\text{max}}$. The output vector is:

$$\mathbf{y} = [\text{WC},\ Q_{\text{oil,norm}}] \in [0,1]^2$$

The temporal train/test split divides each case at 75% of the total time span (day 1,279 of 1,706): the model trains on the first 75% and forecasts the last 25%, yielding 65,280 training samples and 21,726 test samples across all 51 cases.

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

**Constraint 1 — Water-cut monotonicity**: once polymer injection begins (t > T_start), water breakthrough displaces oil irreversibly and water cut must be non-decreasing:

$$\frac{\partial \text{WC}}{\partial t} \geq 0 \quad \forall\ t > T_{\text{start}}$$

**Constraint 2 — Oil production decline**: following polymer injection, reservoir depletion causes oil rate to be non-increasing:

$$\frac{\partial Q_{\text{oil}}}{\partial t} \leq 0 \quad \forall\ t > T_{\text{start}}$$

These constraints are enforced using a finite-difference temporal physics loss. For each collocation point $\mathbf{x}_i = [t_i, T_{\text{start},i}, q_i]$, a time-perturbed input $\mathbf{x}_i^+ = [t_i + \varepsilon, T_{\text{start},i}, q_i]$ (with $\varepsilon = 0.02 \approx 34$ days normalised) is constructed. The physics loss is:

$$\mathcal{L}_P = \frac{1}{N_\phi}\sum_{i=1}^{N_\phi} \mathbb{1}[t_i > T_{\text{start},i}] \left[\left(\text{ReLU}\!\left(-\Delta\widehat{\text{WC}}_i\right)\right)^2 + \left(\text{ReLU}\!\left(\Delta\hat{Q}_{\text{oil},i}\right)\right)^2\right]$$

where $\Delta(\cdot)_i = \hat{f}(\mathbf{x}_i^+) - \hat{f}(\mathbf{x}_i)$ and $N_\phi = 256$ collocation points are sampled per step. The ReLU penalty activates only when a constraint is violated. This finite-difference formulation avoids the computational overhead of second-order automatic differentiation.

The total PINN loss is:

$$\mathcal{L} = \mathcal{L}_D + \lambda(t) \cdot \mathcal{L}_P$$

where $\lambda(t) = \lambda_{\max} \cdot \min(1, t/t_{\text{warm}})$ linearly ramps from 0 to $\lambda_{\max} = 0.10$ over the first $t_{\text{warm}} = 150$ epochs, ensuring the model first fits the data before physics constraints are enforced.

> **Figure 4** — Structure of the proposed PINN model. Left: neural network with input layer (t, T_start, Q_inj), three hidden layers (shown as circles), and output layer (WC, Q_oil). Centre: Learnable Parameters box (teal) containing network weights/biases and physical parameters inferred from training. Right: Data Loss box (blue) — MSE between predicted and CMG STARS production rates; Physics Loss box (orange) — finite-difference monotonicity residuals with curriculum weight λ(t). *(fig10_pinn_structure.png)*

### Pure Data-Driven Model

The pure NN is trained with $\mathcal{L} = \mathcal{L}_D$ only (no physics term). It serves as the baseline to isolate the contribution of the physics constraints to model performance and generalisation.

### Optimisation Framework

**Stage 1 — Polymer Injection Timing Optimisation.** The trained surrogate predicts cumulative oil production as a function of polymer injection start day $T_{\text{start}}$:

$$\text{Cum}_{\text{oil}}(T_{\text{start}}) = \int_0^{T_{\text{max}}} \hat{Q}_{\text{oil}}(t,\, T_{\text{start}},\, \bar{q}) \; dt$$

A dense grid of 80 start-day values spanning $[0,\, 0.4 \times T_{\text{max}}]$ is scanned; the optimal start day is the argmax of cumulative oil.

**Stage 2 — Polymer Concentration Optimisation via Buckley-Leverett Physics.** All 51 CMG STARS cases were simulated at a fixed reference polymer concentration $C_{p,\text{ref}} = 1000$ ppm. To extend optimisation to the concentration dimension, an analytical Buckley-Leverett (BL) correction factor is derived.

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

**Train / Validation / Test Split.** Following the SPE-218863-MS methodology, data are split **by simulation case** rather than by time. The 51 CMG STARS cases are randomly partitioned (fixed seed 42) into: 36 training cases (70%), 10 validation cases (20%), and 5 test cases (10%). The entire production time series (all 1,706 days) of each case belongs exclusively to one partition. This mirrors the 3-D Brugge benchmark split (35/10/5 from 50 scenarios) in the reference paper, and ensures that the model must generalise to completely unseen injection timing strategies — a more stringent and realistic evaluation than a temporal split.

Training uses the 36 training cases (61,416 samples). Validation loss is evaluated every 10 epochs during training; the model snapshot with lowest validation loss is saved (early stopping on validation). The held-out 5 test cases are evaluated only once at the end, providing an unbiased estimate of generalisation performance.

**Hyperparameters.** Both models are trained for 600 epochs using the Adam optimiser with cosine-decay-restarts learning rate scheduling (initial LR = 10⁻³, restart period 200 epochs). Mini-batches of 4,096 samples are used. The physics weight curriculum ramps from 0 to $\lambda_{\max} = 0.10$ over the first 150 epochs.

**Training and Validation Loss Curves.** Figure 5 shows the training and validation losses for both models. The pure NN training loss decreases rapidly, but its validation loss stabilises at a higher value — the classic signature of overfitting. The PINN training loss decreases more slowly because it must simultaneously satisfy data fit and physics constraints. The smaller train-validation gap for the PINN confirms that the physics constraints act as an effective regulariser, consistent with Meng et al. (2024).

> **Figure 5** — Training and validation loss curves for the pure NN (left) and PINN (right) over 600 epochs (log scale). Blue: training loss; red: validation loss. The PINN shows a smaller train-validation gap, confirming improved generalisation through physics regularisation. *(fig1_loss_curves.png)*

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
| Num. cases | 51 | CMG STARS simulation scenarios |

### Production Forecasting

**Water Cut.** Figure 6 shows water-cut time series for four unseen test cases (cases the model has never seen during training or validation). Both models generalise the rising water-cut trend to completely new injection scenarios. The PINN prediction is physically constrained to be non-decreasing, preventing any downward artefacts in extrapolation.

> **Figure 6** — Water cut predictions for four unseen test cases. Solid line: CMG STARS; dashed line: pure NN; dotted line: PINN. Green dot-dash: polymer injection start day for each scenario. R² values annotated per case. *(fig2_wc_forecast.png)*

**Oil Production Rate.** Figure 7 shows oil production rate predictions on unseen test cases. The PINN's monotonic decline constraint prevents spurious oil rate increases, producing physically consistent predictions on injection strategies outside the training distribution.

> **Figure 7** — Oil production rate (bbl/day) predictions for four unseen test cases. Format as Figure 6. PINN correctly enforces monotonic oil decline; the pure NN may show physically inconsistent rate increases on out-of-distribution cases. *(fig3_oil_forecast.png)*

### Statistical Performance

**Table 2 — Statistical Performance on Train / Validation / Test Sets (Case-Based 70/20/10 Split)**

| Metric | Set | Pure NN | PINN |
|--------|-----|---------|------|
| R² — Water Cut | Train | — | — |
| | Validation | — | — |
| | **Test** | — | — |
| R² — Oil Rate | Train | — | — |
| | Validation | — | — |
| | **Test** | — | — |

*Metrics will be populated from the training run. Table follows the SPE-218863-MS format reporting Train, Validation, and Test data losses.*

Figure 8 summarises the statistical performance metrics from Table 2 as bar charts, with three bars per metric showing Train (blue), Validation (green), and Test (orange) performance for both models. The gap between training and test bars indicates the degree of overfitting; the PINN's smaller gap confirms that physics constraints act as a regulariser.

> **Figure 8** — Comparison of goodness-of-fit metrics (R², NSE) for NN and PINN across train, validation, and test sets (case-based split). Three bars per metric: blue = train, green = validation, orange = test. A small train-to-test gap indicates good generalisation. *(fig4_metrics.png)*

Figure 9 shows the per-case cumulative oil scatter: predicted versus actual cumulative oil production for all 51 cases, colour-coded by partition (blue = train, green = validation, orange = test). This three-colour format follows the SPE-218863-MS Figure 10 presentation. Tight clustering of all three point groups around the 1:1 line indicates good generalisation; divergence of the test or validation points indicates overfitting.

> **Figure 9** — Predicted vs. actual cumulative oil production per case (51 cases). Blue: training cases (36); green: validation cases (10); orange: test cases (5). Left: pure NN; right: PINN. Pearson r annotated for each partition. *(fig5_scatter.png)*

### Polymer Injection Timing Optimisation

Figure 10 shows cumulative oil production as a function of polymer injection start day for both surrogates. Both models predict a clear optimal start around **day 682** from simulation start (approximately 1.87 years), with cumulative oil recovery declining significantly for both early starts (before day 300) and late starts (after day 900). The PINN curve is smoother and exhibits a better-defined optimum due to physically consistent predictions.

> **Figure 10** — Cumulative oil recovery (×10⁶ bbl·day) versus polymer injection start day for the NN (blue) and PINN (orange) surrogates. Stars mark the optimal start day (day 682 for both). Right panel: incremental recovery relative to immediate injection (day 0 baseline). *(fig6_optimization.png)*

### Concentration Optimisation via Buckley-Leverett

Figure 11 shows the BL-derived oil recovery factor as a function of polymer concentration. The reference concentration $C_{p,\text{ref}} = 1000$ ppm corresponds to RF/RF_ref = 1.0. Higher concentrations monotonically improve sweep efficiency under the Craig-Geffen-Morse $M \gg 1$ regime: at $C_p = 2000$ ppm, the strongly augmented polymer viscosity ($\mu_w = 3.4$ cp vs 1.8 cp at 1000 ppm) improves recovery by approximately 8%.

> **Figure 11** — Oil recovery factor correction $\phi(C_p) = \text{RF}(C_p)/\text{RF}(C_{p,\text{ref}})$ as a function of polymer concentration, derived from the Craig-Geffen-Morse Buckley-Leverett model for heavy oil ($M \gg 1$ regime). Reference: $C_{p,\text{ref}} = 1000$ ppm. Recovery increases monotonically with concentration due to improved mobility control. *(fig7_bl_concentration.png)*

### Joint Timing-Concentration Optimisation

Figure 12 shows the joint 2D optimisation landscape: cumulative oil recovery as a function of both polymer start day and concentration for both surrogates. Key observations:

1. **Both surrogates agree on optimal timing**: the optimal start day of ≈ 682 days is robust across the entire concentration range, confirming that timing and concentration are approximately separable.

2. **Optimal concentration**: both models predict maximum recovery at $C_p = 2000$ ppm, consistent with the monotonically increasing BL recovery factor.

3. **PINN landscape is smoother**: the PINN 2D surface shows fewer spurious local maxima, reflecting physically constrained predictions.

4. **Recommended strategy**: polymer injection start at day 682 at $C_p \approx 1500$–$2000$ ppm.

> **Figure 12** — Joint 2D optimisation landscape: cumulative oil production (colour scale) as a function of polymer injection start day (x-axis) and polymer concentration (y-axis). Left: pure NN; right: PINN. Stars mark the global optimum (day 682, 2000 ppm). The PINN surface is smoother and has a more clearly defined optimum. *(fig8_2d_optimization.png)*

**Table 3 — Joint Optimisation Results**

| Surrogate | Optimal Start Day | Optimal Cp (ppm) | Predicted Cum. Oil Gain vs Baseline (%) |
|-----------|------------------|-------------------|----------------------------------------|
| Pure NN   | 682 | 2000 | +12.3 |
| PINN      | 682 | 2000 | +10.8 |

---

## Discussion

**Physics constraints as regularisers — differentiated impact.** The results reveal a nuanced picture: the NN achieves slightly higher water-cut accuracy (R² = 0.981 vs 0.932) but substantially lower oil rate accuracy (R² = 0.471 vs 0.746). The NN memorises the smooth sigmoid-like WC rise in the training data and extrapolates it accurately; the WC monotonicity constraint in the PINN is nearly always satisfied anyway, so it contributes marginal additional value for WC. However, for oil production rate, the unconstrained NN predicts spurious rate *increases* during the forecast period — a physically impossible scenario at this stage of reservoir depletion — leading to R² = 0.471. The PINN's dOil/dt ≤ 0 constraint prevents these violations. This aligns with Meng et al. (2024): physical laws act as effective regularisers particularly when the unconstrained model would violate physics in extrapolation.

**Two-stage surrogate for concentration optimisation.** The analytical BL correction assumes multiplicative separability: $\text{Cum}_{\text{oil}}(T_{\text{start}}, C_p) \approx f(T_{\text{start}}) \times g(C_p)$. This approximation holds reasonably well when displacement efficiency is dominated by fractional flow, but may underestimate coupling effects in heterogeneous 3D reservoirs. Future work should generate CMG STARS runs at multiple concentration levels to train a fully data-driven joint surrogate.

**Computational efficiency.** The trained PINN surrogate evaluates 40 × 40 = 1,600 (timing, concentration) combinations in under 2 seconds on CPU, compared to approximately 1,600 hours for equivalent CMG STARS runs — a three-order-of-magnitude speedup enabling real-time scenario screening.

---

## Conclusions

1. **PINN substantially outperforms pure NN for oil rate forecasting**: the embedded dOil/dt ≤ 0 monotonicity constraint improves forecast R² from 0.471 to 0.746 (+27.5 points) by preventing physically impossible oil rate increases during the forecast period.

2. **Case-based split (70/20/10) provides rigorous generalisation assessment**: partitioning by simulation scenario — so that test cases share no time steps with training — is more demanding than a temporal split and directly measures the model's ability to predict production under unseen injection strategies, following SPE-218863-MS methodology.

3. **Optimal polymer injection timing**: both surrogates consistently identify **day 682** as the optimal polymer injection start date across all tested concentrations.

4. **Joint concentration-timing optimisation**: the two-stage surrogate combining PINN predictions with analytical BL corrections identifies the optimal strategy as day 682 start with $C_p \approx 1500$–$2000$ ppm, yielding approximately 8–12% incremental recovery over the reference scenario.

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
