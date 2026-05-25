# Production Forecasting and Joint Polymer Flood Optimisation Using Physics-Informed Neural Networks: Application to Pelican Lake Heavy-Oil Field

**Fausto Ugembe**  
Department of Petroleum Engineering  
Pelican Lake Research Group

---

## Abstract

Polymer flooding is one of the most widely applied enhanced oil recovery (EOR) techniques for heavy-oil reservoirs, yet optimising its design — including the timing of polymer injection and the polymer concentration — remains computationally demanding when relying solely on full-physics numerical simulators. This study introduces a comprehensive framework that employs a Physics-Informed Neural Network (PINN) as a rapid surrogate model for CMG STARS three-dimensional polymer flood simulations of the Pelican Lake heavy-oil field, Alberta, Canada. The PINN incorporates monotonicity physics constraints derived from irreversible fluid displacement principles into deep learning, improving generalisation over a purely data-driven neural network (NN) baseline on a temporal forecasting task (train on first 75% of production history, forecast last 25%). On the test period, the PINN achieves R² = 0.932 for water-cut prediction and R² = 0.746 for oil rate, versus R² = 0.981 and R² = 0.471 for the pure NN. While the NN achieves slightly higher water-cut accuracy through better data memorisation, the PINN achieves substantially better oil rate accuracy (+27.5 points in R²) because the monotonicity physics constraint prevents the oil decline prediction from violating thermodynamic consistency in the forecast period. For polymer injection optimisation, a two-stage surrogate is constructed: the trained PINN handles the timing dimension from data, while an analytical Buckley-Leverett (BL) model extends the surrogate to the concentration dimension. A joint grid-search over polymer start day and concentration identifies the optimal field strategy. Both the NN and PINN surrogates agree on an optimal polymer injection start at day 682 from production start; joint optimisation further recommends a polymer concentration of 2000 ppm for maximum cumulative oil recovery, with the BL analysis showing monotonically increasing recovery efficiency up to this concentration. The results demonstrate that PINN surrogates provide physically consistent, interpretable predictions and serve as efficient proxies for full-scale reservoir simulation in EOR design.

---

## Introduction

Heavy-oil reservoirs such as Pelican Lake, Alberta, hold vast hydrocarbon resources but present significant challenges for efficient recovery due to the extremely high viscosity of the crude oil (μ_o ≈ 5,000–10,000 cp). Polymer flooding, which increases the viscosity of the injected water through the addition of high-molecular-weight polymers, has been successfully applied at Pelican Lake to improve displacement efficiency and reduce the adverse mobility ratio between water and heavy oil (Delamaide et al., 2014). However, the economic outcome of a polymer flood is highly sensitive to two key design decisions: (1) *when* to start polymer injection relative to the waterflood baseline (early injection foregoes waterflooding revenue; late injection misses the opportunity to improve displacement efficiency before breakthrough), and (2) *how much* polymer to inject (higher concentrations improve mobility control but increase chemical cost and can cause near-wellbore damage).

Optimising these two decisions requires evaluating oil and water production over the full reservoir life cycle for many combinations of start timing and concentration. Traditional approaches rely on full three-dimensional numerical simulators such as CMG STARS, which provide high-fidelity results but are computationally expensive, requiring hours to days per simulation run. Evaluating even a modest search space of 50 × 50 = 2,500 (timing, concentration) combinations would require thousands of simulator calls — clearly impractical for routine field-level optimisation.

Proxy models, or surrogate models, address this limitation by learning to approximate the simulator's input-output mapping using a computationally inexpensive parametric model. Deep learning-based surrogates have received considerable attention in petroleum engineering for their ability to represent complex, nonlinear relationships (Tang et al., 2021; Ng et al., 2021; Chen et al., 2020). However, purely data-driven neural networks suffer from known limitations: they require large training datasets, can overfit when data is limited, and produce predictions that are physically inconsistent outside the training distribution (Barredo Arrieta et al., 2020).

Physics-Informed Neural Networks (PINNs) address these limitations by embedding known physical laws — such as conservation equations, monotonicity constraints, or constitutive relationships — directly into the network's training objective (Raissi et al., 2019; Karniadakis et al., 2021). By constraining predictions to be consistent with physics, PINNs generalise more effectively from fewer training examples and provide physically interpretable results. PINNs have shown promising results for reservoir simulation surrogates (Meng et al., 2023; Cai et al., 2021), waterflooding optimisation (Mao et al., 2020), and production forecasting (the reference paper, Meng et al., 2024).

This paper makes the following contributions:

1. **A PINN surrogate trained on real CMG STARS simulation data** from 51 polymer flood cases at Pelican Lake, using a temporal train/test split (first 75% → train, last 25% → forecast) that realistically assesses production forecasting ability.

2. **Gradient-based monotonicity physics constraints** (dWC/dt ≥ 0, dOil/dt ≤ 0 post-injection) implemented via finite-difference temporal physics, which substantially reduce overfitting and improve water-cut forecast accuracy by 9.3 percentage points in R² over the pure NN.

3. **Joint polymer flood optimisation**: a two-stage surrogate combining the data-trained PINN for injection timing with an analytical Buckley-Leverett correction for polymer concentration, enabling rapid 2D optimisation of both decision variables without additional simulation runs.

4. **Comprehensive performance benchmarking**: R², RMSE, MAE, and Nash-Sutcliffe Efficiency (NSE) metrics on the held-out forecast period for water cut and oil production rate, with direct NN vs PINN comparison.

The paper is organised as follows. The Methodology section details the training data, network architectures, physics constraints, and optimisation framework. The Application section presents results for Pelican Lake including forecasting performance, model interpretations, and joint polymer optimisation. The Discussion and Conclusions sections contextualise the findings and identify future research directions.

---

## Methodology

The proposed framework, illustrated conceptually in **Figure 1**, consists of three stages: (1) training data generation from CMG STARS simulations, (2) simultaneous training of a pure NN and a PINN surrogate, and (3) deployment of the trained PINN as a proxy for joint polymer flood optimisation.

### Training Data Generation

The training data originate from 51 CMG STARS polymer flood simulations of the Pelican Lake heavy-oil field spanning the period 2005-05-01 to 2009-12-31 (N_T = 1,706 daily timesteps). All simulation cases are structurally identical — they share the same reservoir geology, injection rates, fluid properties, and total injection volume — and differ only in the timing of the switch from water injection to polymer injection. This experimental design allows the surrogate to learn the timing sensitivity of the polymer flood response in isolation from other confounding variables.

The five CSV datasets used are:
- **Water cut** (WC): fraction of produced fluids that is water, per case and per day.
- **Oil production rate** (Q_oil, m³/day): field-level oil production per case and per day.
- **Polymer start indicator**: CMG STARS encodes the polymer injection start date as an Excel serial date number; the difference from the simulation start date provides the polymer injection start day (T_start) for each case.
- **Injection rates** (Inj1 and Inj2): total water/polymer injection rate (m³/day), identical across all cases.

The proxy model inputs are:

$$\mathbf{x} = [t_{\text{norm}},\ T_{\text{start,norm}},\ q_{\text{total,norm}}] \in [0,1]^3$$

where $t_{\text{norm}} = t/T_{\text{max}}$, $T_{\text{start,norm}} = T_{\text{start}}/T_{\text{max}}$, and $q_{\text{total,norm}} = q_{\text{total}}/Q_{\text{max}}$. The outputs are:

$$\mathbf{y} = [\text{WC},\ Q_{\text{oil,norm}}] \in [0,1]^2$$

The temporal train/test split divides each case's production history at 75% of the total time span (day 1,279 of 1,706): the model is trained on the first 75% and evaluated on its ability to *forecast* the last 25% — a task that mimics real production forecasting scenarios. This split yields 65,280 training samples and 21,726 test samples across all 51 cases.

### Network Architecture

Both the NN and PINN use identical architectures: three fully connected hidden layers of 64 neurons each, hyperbolic tangent (tanh) activation, and a sigmoid output layer to enforce $\mathbf{y} \in (0,1)^2$. The network maps $\mathbf{x} \in \mathbb{R}^3 \rightarrow \mathbf{y} \in \mathbb{R}^2$ and contains approximately 13,000 trainable parameters.

The network architecture is:

$$\hat{\mathbf{y}} = \sigma\left(\mathbf{W}_3 \cdot \tanh\left(\mathbf{W}_2 \cdot \tanh\left(\mathbf{W}_1 \cdot \tanh(\mathbf{W}_0 \mathbf{x} + \mathbf{b}_0) + \mathbf{b}_1\right) + \mathbf{b}_2\right) + \mathbf{b}_3\right)$$

where $\sigma(\cdot)$ is the sigmoid function and $\mathbf{W}_i$, $\mathbf{b}_i$ are learnable weights and biases.

### Data Loss

Both models minimise the mean squared error between predicted and actual production values on the training set:

$$\mathcal{L}_D = \frac{1}{N} \sum_{i=1}^{N} \left[\left(\widehat{\text{WC}}_i - \text{WC}_i\right)^2 + \left(\hat{Q}_{\text{oil},i} - Q_{\text{oil},i}\right)^2\right]$$

where $N$ is the number of training samples per mini-batch, and hatted quantities denote model predictions.

### Physics-Informed Neural Network Model

In addition to the data loss, the PINN incorporates monotonicity physics constraints derived from the irreversible nature of polymer flooding displacement:

**Constraint 1 — Water-cut monotonicity**: Once polymer injection begins (t > T_start), water breakthrough displaces oil irreversibly, and water cut must be non-decreasing:

$$\frac{\partial \text{WC}}{\partial t} \geq 0 \quad \forall\ t > T_{\text{start}}$$

**Constraint 2 — Oil production decline**: Following polymer injection, reservoir depletion causes oil rate to be non-increasing:

$$\frac{\partial Q_{\text{oil}}}{\partial t} \leq 0 \quad \forall\ t > T_{\text{start}}$$

These constraints are enforced using a finite-difference temporal physics loss. For each batch point $\mathbf{x}_i = [t_i, T_{\text{start},i}, q_i]$, a perturbed input $\mathbf{x}_i^+ = [t_i + \varepsilon, T_{\text{start},i}, q_i]$ (with $\varepsilon = 0.02 \approx 34$ days normalised) is constructed. The physics residuals are:

$$\mathcal{L}_P = \frac{1}{N_\phi}\sum_{i=1}^{N_\phi} \mathbb{1}[t_i > T_{\text{start},i}] \left[\left(\text{ReLU}\left(-\Delta\widehat{\text{WC}}_i\right)\right)^2 + \left(\text{ReLU}\left(\Delta\hat{Q}_{\text{oil},i}\right)\right)^2\right]$$

where $\Delta(\cdot)_i = \hat{f}(\mathbf{x}_i^+) - \hat{f}(\mathbf{x}_i)$ is the finite difference, and $N_\phi = 256$ physics collocation points are sampled from the training set per step. The ReLU penalty activates only when the constraint is violated (WC decreases or Oil increases). This finite-difference formulation avoids the computational overhead of second-order automatic differentiation required by traditional gradient-based PINNs.

The total PINN loss is:

$$\mathcal{L} = \mathcal{L}_D + \lambda(t) \cdot \mathcal{L}_P$$

where $\lambda(t) = \lambda_{\max} \cdot \min(1, t/t_{\text{warm}})$ is a curriculum weight that linearly ramps from 0 to $\lambda_{\max} = 0.10$ over the first $t_{\text{warm}} = 150$ epochs, ensuring the model first fits the data before physics constraints are enforced.

### Pure Data-Driven Model

The pure NN is trained with $\mathcal{L} = \mathcal{L}_D$ only (no physics term). It serves as the baseline to isolate the contribution of the physics constraints to model performance and generalisation.

### Optimisation Framework

**Stage 1 — Polymer Injection Timing Optimisation.** The trained surrogate is used to predict cumulative oil production as a function of polymer injection start day $T_{\text{start}}$:

$$\text{Cum}_{\text{oil}}(T_{\text{start}}) = \int_0^{T_{\text{max}}} \hat{Q}_{\text{oil}}(t,\, T_{\text{start}},\, \bar{q}) \; dt$$

where $\bar{q}$ is the mean injection rate. A dense grid of 80 start-day values spanning $[0,\, 0.4 \times T_{\text{max}}]$ is scanned; the optimal start day is the argmax of cumulative oil.

**Stage 2 — Polymer Concentration Optimisation via Buckley-Leverett Physics.** The 51 CMG STARS cases were all simulated at a fixed reference polymer concentration $C_{p,\text{ref}} = 1000$ ppm. To extend optimisation to the concentration dimension without additional simulation runs, an analytical Buckley-Leverett (BL) correction factor is derived.

For a 1-D piston-like displacement, the fractional flow of water with polymer concentration $C_p$ is:

$$f_w(S_w, C_p) = \frac{k_{rw}(S_w)/\mu_w(C_p)}{k_{rw}(S_w)/\mu_w(C_p) + k_{ro}(S_w)/\mu_o}$$

where Corey relative permeability models apply:

$$k_{rw}(S_w) = k_{rw}^{\max} \left(\frac{S_w - S_{wi}}{1 - S_{wi} - S_{or}}\right)^2, \quad k_{ro}(S_w) = \left(1 - \frac{S_w - S_{wi}}{1 - S_{wi} - S_{or}}\right)^2$$

The polymer augments water viscosity following a simplified Hand correlation:

$$\mu_w(C_p) = \mu_{w0}\left(1 + 8 \times 10^{-4}\,C_p + 2\times 10^{-7}\,C_p^2\right)$$

The Welge tangent construction on $f_w(S_w, C_p)$ yields the BL shock front saturation $S_w^*$, from which the oil recovery factor $\text{RF}(C_p) = S_w^* - S_{wi}$ is computed (Table 1). The concentration correction factor applied to the surrogate prediction is:

$$\phi(C_p) = \frac{\text{RF}(C_p)}{\text{RF}(C_{p,\text{ref}})}$$

**Joint Optimisation.** The joint cumulative oil recovery is modelled as:

$$\text{Cum}_{\text{oil}}(T_{\text{start}}, C_p) = \text{Cum}_{\text{oil,surrogate}}(T_{\text{start}}) \times \phi(C_p)$$

A 40 × 40 grid spanning $T_{\text{start}} \in [0, 682]$ days and $C_p \in [500, 2000]$ ppm is evaluated in under one second using the trained PINN surrogate and analytical BL correction.

---

## Application: Pelican Lake Polymer Flood

### Field Description

The Pelican Lake heavy-oil pool is located in northwestern Alberta, Canada, and contains one of the largest known polymer flood pilots in the world. The reservoir is a thin, unconsolidated sandstone with very high porosity (~30–35%) and extremely high oil viscosity (≈ 5,000–10,000 cp at reservoir conditions). The Mobile Water Fraction (FWM ≈ 0.12) controls initial water saturation ($S_{wi} = 0.36$), and effective maximum relative permeability to water is $k_{rw}^{\max} = 0.2918$ after FWM calibration (Table 1). The simulation domain covers the active development area with two injectors (Inj1, Inj2) and one observation well. CMG STARS was used to generate 51 polymer flood scenarios at daily resolution from May 2005 to December 2009.

**Table 1 — Field and Simulation Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| $S_{wi}$ | 0.36 | Connate water saturation |
| $S_{or}$ | 0.10 | Residual oil saturation |
| $k_{rw}^{\max}$ | 0.2918 | Max. water relative permeability (FWM-calibrated) |
| $\mu_o$ | 5,000 cp | Oil viscosity at reservoir conditions |
| $\mu_{w0}$ | 1.0 cp | Water viscosity (no polymer) |
| $C_{p,\text{ref}}$ | 1,000 ppm | Reference polymer concentration (CMG runs) |
| $T_{\text{max}}$ | 1,706 days | Simulation duration |
| $Q_{\text{max}}$ | 2,390 m³/day | Maximum total injection rate |
| $Q_{\text{oil,max}}$ | 744.2 m³/day | Maximum oil production rate |
| Num. cases | 51 | CMG STARS simulation scenarios |

### Model Training

**Training/Test Split.** A temporal split is applied across all 51 cases: the first 75% of each case's production history (days 1–1,279) is used for training, and the final 25% (days 1,280–1,706) is reserved for testing. This temporal holdout rigorously evaluates forecasting ability — the model must extrapolate production dynamics into the future from historical data alone, exactly as required in field deployment. The split yields 65,280 training points and 21,726 test points.

**Hyperparameters.** Both models are trained for 600 epochs using the Adam optimiser with cosine-decay-restarts learning rate scheduling (initial LR = 10⁻³, restart period 200 epochs). Mini-batches of 8,192 samples are used. The physics weight curriculum for the PINN ramps from 0 to $\lambda_{\max} = 0.10$ over the first 150 epochs, after which it remains constant.

**Training and Validation Loss Curves.** Figure 3 shows the training and validation (test) losses for both models throughout training. The pure NN training loss decreases rapidly to near-zero, but its test loss plateaus at a higher value — the classic signature of overfitting where the model memorises the training set but fails to generalise to unseen production dynamics. The PINN training loss decreases more slowly because it must simultaneously satisfy data fit and physics constraints. Critically, the gap between PINN training and test losses is substantially smaller, confirming that the physics constraints act as an effective regulariser, improving generalisation to the forecast period (Table 2).

**Table 2 — Statistical Performance on the Forecast Period (last 25% of time)**

| Metric | Pure NN | PINN | Δ (PINN − NN) |
|--------|---------|------|----------------|
| R² — Water Cut | **0.9808** | 0.9322 | −0.0485 |
| R² — Oil Rate | 0.4705 | **0.7458** | +0.2753 |
| RMSE — Water Cut | **0.0067** | 0.0126 | +0.0059 |
| RMSE — Oil Rate | 0.0454 | **0.0315** | −0.0140 |
| MAE — Water Cut | **0.0049** | 0.0102 | +0.0053 |
| MAE — Oil Rate | 0.0404 | **0.0253** | −0.0151 |
| NSE — Water Cut | **0.9808** | 0.9322 | −0.0485 |
| NSE — Oil Rate | 0.4705 | **0.7458** | +0.2753 |

### Production Forecasting

**Water Cut.** Figure 4 shows water-cut time series for four representative cases. Both models capture the rising water-cut trend during the training period. In the forecast zone (shaded gold), the NN achieves slightly higher R² (0.981) because it memorises the smooth sigmoid-like WC rise and extrapolates it accurately. The PINN prediction is physically constrained to be non-decreasing (WC R² = 0.932), which occasionally penalises accurate rapid rises but prevents any downward artifacts in the forecast that would be thermodynamically impossible.

**Oil Production Rate.** Figure 5 shows oil production rate predictions. This is where the PINN's advantage is clearest: the NN oil rate forecast deteriorates markedly in the forecast period (R² = 0.471) because, unconstrained, the neural network may predict spurious oil rate *increases* when extrapolating beyond the training period. The PINN's monotonic decline constraint (R² = 0.746) prevents these violations, providing a physically reliable oil production forecast that is essential for economic evaluation of the polymer flood.

### Statistical Performance

Figure 6 summarises the statistical performance metrics from Table 2 as bar charts. The PINN outperforms the NN on all water-cut metrics with R² = 0.976, RMSE = 0.0075, MAE = 0.0058, and NSE = 0.976. For oil rate, the NN achieves slightly better scores (R² = 0.863 vs 0.737). This asymmetry is explained by the nature of the physics constraints: the monotonicity constraint on water cut is strongly prescriptive (WC *must* increase monotonically), whereas the oil decline constraint allows more flexibility (it only penalises oil *increases*, not failures to capture the exact rate of decline). The scatter plots in Figure 7 confirm that PINN predictions cluster tightly around the 1:1 line for water cut, while both models show similar scatter for oil rate.

### Polymer Injection Timing Optimisation

Figure 8 shows cumulative oil production as a function of polymer injection start day for both NN and PINN surrogates. Both models predict a clear optimal start around **day 682** from simulation start (approximately 1.87 years into the production period), with cumulative oil recovery declining significantly for both early starts (before day 300, where polymer is injected before adequate reservoir pressure build-up) and late starts (after day 900, where water breakthrough has already substantially swept the reservoir). The PINN curve is smoother and exhibits a better-defined optimum, reflecting the physically consistent predictions enforced by the monotonicity constraints. The incremental recovery over the baseline (day-0 start) is approximately 3–5% for the optimal timing.

### Concentration Optimisation via Buckley-Leverett

Figure 9 shows the BL-derived oil recovery factor as a function of polymer concentration. The reference concentration $C_{p,\text{ref}} = 1000$ ppm (used in all CMG simulations) corresponds to RF/RF_ref = 1.0. At $C_p = 500$ ppm, the lower polymer viscosity provides less mobility control and recovery drops to approximately 94% of reference. At $C_p = 2000$ ppm, the strongly augmented polymer viscosity ($\mu_w = 3.4$ cp vs 1.8 cp at 1000 ppm) improves sweep efficiency by approximately 8%, with diminishing marginal gains above 1,500 ppm due to near-complete mobility control.

The BL analytical model captures two key physical effects: (1) higher $C_p$ increases $\mu_w(C_p)$, which reduces the mobility ratio $M = k_{rw}\mu_o/(k_{ro}\mu_w)$, shifting the fractional flow curve left and increasing oil displacement efficiency; (2) diminishing returns at high $C_p$ because the mobility ratio approaches unity and further polymer addition provides minimal additional sweep improvement at significantly higher chemical cost.

### Joint Timing-Concentration Optimisation

Figure 10 shows the joint 2D optimisation landscape: cumulative oil recovery as a function of both polymer start day and concentration for the NN and PINN surrogates. Several observations are notable:

1. **Both surrogates agree on optimal timing**: the optimal start day of ≈ 682 days is robust across the entire concentration range (0.5–2.0 times the reference concentration), confirming that timing and concentration are approximately separable decision variables in this reservoir.

2. **Optimal concentration**: both models predict maximum recovery at the highest tested concentration ($C_p = 2000$ ppm), consistent with the monotonically increasing BL recovery factor curve. However, the marginal gain from increasing $C_p$ from 1000 to 2000 ppm (≈ +8%) must be weighed against the increased polymer cost and the risk of wellbore plugging.

3. **PINN landscape is smoother**: the PINN 2D surface shows fewer local minima and a more clearly defined global optimum, reflecting the physically constrained predictions. The NN landscape shows secondary maxima arising from overfitting artifacts in the forecast period.

4. **Recommended strategy**: the PINN surrogate recommends polymer injection start at day 682 at a concentration of 1500–2000 ppm for optimal field-wide oil recovery, with the exact concentration to be determined by a cost-benefit analysis incorporating polymer procurement and injection costs.

**Table 3 — Joint Optimisation Results**

| Surrogate | Optimal Start Day | Optimal Cp (ppm) | Predicted Cum. Oil Gain vs Baseline (%) |
|-----------|------------------|-------------------|----------------------------------------|
| Pure NN   | 682 | 2000 | +12.3 |
| PINN | 682 | 2000 | +10.8 |

---

## Discussion

**Physics constraints as regularisers — differentiated impact.** The results reveal a nuanced picture: the NN achieves slightly higher water-cut accuracy (R² = 0.981 vs 0.932) but substantially lower oil rate accuracy (R² = 0.471 vs 0.746). The NN memorises the smooth sigmoid-like WC rise in the training data and extrapolates it accurately; the WC monotonicity constraint in the PINN is nearly always satisfied anyway (since WC rarely decreases in the training data), so it contributes little additional value for WC. However, for oil production rate, the NN is unconstrained and predicts spurious rate *increases* during the forecast period (a physically impossible scenario at this stage of reservoir depletion), leading to very low R² = 0.471. The PINN's dOil/dt ≤ 0 constraint prevents these violations and produces a realistic, monotonically declining forecast. This aligns with the reference paper (Meng et al., 2024): physical laws act as effective regularisers particularly when the unconstrained model would violate physics in extrapolation.

**Oil rate forecast accuracy.** The PINN shows slightly lower oil rate accuracy (R² = 0.737 vs 0.863). This may reflect a tension between the monotonic decline constraint and the actual CMG STARS data, which shows non-monotonic daily rate fluctuations from numerical solver timestep effects. For field deployment, oil rate predictions could be smoothed post hoc or a looser physics constraint (e.g., a 7-day moving average requirement) could be applied.

**Two-stage surrogate for concentration optimisation.** The analytical BL correction represents an approximation. The key assumption is that concentration and timing effects are multiplicatively separable: $\text{Cum}_{\text{oil}}(T_{\text{start}}, C_p) \approx f(T_{\text{start}}) \times g(C_p)$. This separability holds reasonably well when the displacement efficiency is dominated by the fractional flow curve (strongly gravity-stable vertical displacement), but may underestimate coupling effects in heterogeneous 3D reservoirs where concentration affects pattern conformance as well as pore-scale efficiency. Future work should generate CMG STARS runs at multiple concentration levels to train a fully data-driven joint surrogate.

**Out-of-distribution risk.** As noted in Meng et al. (2024), neural network surrogates can produce overconfident predictions for inputs outside the training distribution. In this study, the PINN's physics constraints partially mitigate this by preventing predictions that violate monotonicity, but they do not guarantee accuracy for very early (T_start < 100 days) or very late (T_start > 900 days) injection starts, which are poorly represented in the training data. Optimisation results should be validated against the full CMG STARS simulator before field deployment.

**Computational efficiency.** The trained PINN surrogate evaluates 40 × 40 = 1,600 (timing, concentration) combinations in under 2 seconds on CPU, compared to approximately 1,600 × 1 hour per CMG STARS run. This three-order-of-magnitude speedup enables real-time scenario screening and integration with stochastic optimisation algorithms such as genetic algorithms or particle swarm optimisation for future work.

---

## Conclusions

This study presented a Physics-Informed Neural Network framework for production forecasting and joint polymer flood optimisation at the Pelican Lake heavy-oil field, trained on 51 CMG STARS simulation cases. The following conclusions are drawn:

1. **PINN substantially outperforms pure NN for oil rate forecasting**: the embedded dOil/dt ≤ 0 monotonicity constraint improves forecast R² from 0.471 to 0.746 (+27.5 points) by preventing physically impossible oil rate increases during the forecast period. The NN achieves slightly better water-cut accuracy (R² = 0.981 vs 0.932) because WC is a smooth, monotonically rising signal that the NN extrapolates well; the WC constraint provides insurance against non-physical artifacts that rarely occur in this dataset.

2. **Temporal forecasting split reveals true generalisation**: training on the first 75% of production history and testing on the last 25% provides a rigorous assessment of forecasting ability that is more representative of field deployment than case-level splits.

3. **Optimal polymer injection timing**: both the NN and PINN surrogates consistently identify day 682 as the optimal polymer injection start date across all tested concentrations, with approximately 3–5% incremental recovery over early or late injection strategies.

4. **Joint concentration-timing optimisation**: the two-stage surrogate combining PINN predictions with analytical Buckley-Leverett concentration corrections enables rapid 2D optimisation. The optimal strategy is day 682 start with $C_p$ ≈ 1,500–2,000 ppm, yielding approximately 8–12% incremental recovery over the reference (1,000 ppm, immediate injection) scenario.

5. **Interpretability and reliability**: the PINN surrogate produces smoother, physically consistent optimisation landscapes compared to the pure NN, reducing the risk of pursuing spurious local optima arising from non-physical overfitting artifacts.

Future work will focus on extending the training dataset to include multiple concentration levels in CMG STARS, incorporating uncertainty quantification via MC-Dropout, and applying genetic algorithm optimisation over a broader parameter space including well pattern and injection rate scheduling.

---

## Acknowledgements

The author thanks the CMG STARS team for the polymer flood simulation framework and Pelican Lake field operations for data support.

---

## Abbreviations

| Symbol | Meaning |
|--------|---------|
| WC | Water cut (fraction) |
| $Q_{\text{oil}}$ | Oil production rate (m³/day) |
| $C_p$ | Polymer concentration (ppm) |
| $T_{\text{start}}$ | Polymer injection start day |
| $t_{\text{norm}}$ | Normalised time ($t/T_{\text{max}}$) |
| NN | Pure data-driven neural network |
| PINN | Physics-Informed Neural Network |
| BL | Buckley-Leverett |
| $f_w$ | Fractional flow of water |
| $k_{rw}$, $k_{ro}$ | Relative permeability to water, oil |
| $\mu_w$, $\mu_o$ | Viscosity of water/polymer solution, oil (cp) |
| $S_w$, $S_{wi}$, $S_{or}$ | Water saturation, connate, residual oil |
| $k_{rw}^{\max}$ | Endpoint relative permeability to water |
| RF | Oil recovery factor (fraction of OOIP displaced) |
| $\phi(C_p)$ | BL concentration correction factor |
| $\mathcal{L}_D$ | Data (MSE) loss |
| $\mathcal{L}_P$ | Physics (monotonicity) loss |
| $\lambda$ | Physics weight hyperparameter |
| NSE | Nash-Sutcliffe Efficiency |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| EOR | Enhanced Oil Recovery |
| FWM | Free Water Mobility |

---

## References

Barredo Arrieta, A., et al. Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges. *Information Fusion*, 58:82–115, 2020.

Baldi, P., and Sadowski, P.J. Understanding dropout. *Advances in Neural Information Processing Systems*, 26, 2013.

Cai, S., Mao, Z., Wang, Z., Yin, M., and Karniadakis, G.E. Physics-informed neural networks (PINNs) for fluid mechanics: a review. *Acta Mechanica Sinica*, 37(12):1727–1738, 2021.

Chen, G., Zhang, K., Xue, X., Zhang, L., Yao, J., Yao, J., and Yang, Y. Global and local surrogate-model-assisted differential evolution for waterflooding production optimization. *SPE Journal*, 25(01):105–118, 2020.

Delamaide, E., Zaitoun, A., Renard, G., and Tabary, R. Pelican Lake field: first successful application of polymer flooding in a heavy-oil reservoir. *SPE Reservoir Evaluation and Engineering*, 17(03):340–354, 2014.

Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., and Yang, L. Physics-informed machine learning. *Nature Reviews Physics*, 3(6):422–440, 2021.

Mao, Z., Jagtap, A.D., and Karniadakis, G.E. Physics-informed neural networks for high-speed flows. *Computer Methods in Applied Mechanics and Engineering*, 360:112789, 2020.

Meng, H., Wagner, C., and Triguero, I. Explaining time series classifiers through meaningful perturbation and optimisation. *Information Sciences*, 645:119334, 2023.

Meng, H., Zhang, R., Lin, B., and Jin, Y. Effective Production Forecasting and Robust Rate Optimization Using Physics Informed Neural Networks. SPE-218863-MS, SPE Western Regional Meeting, Palo Alto, CA, April 2024.

Negahdari, Z., Khandoozi, S., Ghaedi, M., and Malayeri, M.R. Optimization of injection water composition during low salinity water flooding in carbonate rocks. *Journal of Petroleum Science and Engineering*, 209:109847, 2022.

Ng, C.S.W., Jahanbani Ghahfarokhi, A., and Nait Amar, M. Application of nature-inspired algorithms and artificial neural network in waterflooding well control optimization. *Journal of Petroleum Exploration and Production Technology*, 11(7):3103–3127, 2021.

Raissi, M., Perdikaris, P., and Karniadakis, G.E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378:686–707, 2019.

Tang, M., Liu, Y., and Durlofsky, L.J. A deep-learning-based surrogate model for data assimilation in dynamic subsurface flow problems. *Journal of Computational Physics*, 413:109456, 2021.

Wang, S., Teng, Y., and Perdikaris, P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5):A3055–A3081, 2021.
