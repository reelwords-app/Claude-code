# Data-Driven Neural Network Ensemble for Heavy Oil Polymer Flooding Production Forecasting at Pelican Lake

**Authors:** [Author 1]$^{1}$, [Author 2]$^{2}$, [Author 3]$^{1,2}$

$^{1}$ Department of Petroleum Engineering, [University Name], [City, Country]  
$^{2}$ Department of Computer Science and Machine Learning, [University Name], [City, Country]

**Corresponding author:** [Author 1] — email@institution.edu

**Submitted to:** *Journal of Petroleum Science and Engineering* / *Computers & Geosciences*

---

## Abstract

Polymer flooding is one of the most widely deployed enhanced oil recovery (EOR) techniques for heavy oil reservoirs, yet reliable production forecasting remains a computationally intensive task that traditionally depends on high-fidelity reservoir simulation. In this work, we present a data-driven neural network ensemble framework for forecasting oil production rate (OPR) and water cut (WC) in a polymer flood context, applied to the Wabiskaw A reservoir of the Pelican Lake heavy oil field, Alberta, Canada. The dataset comprises 200 CMG STARS compositional simulation cases spanning May 2005 to December 2009 (57 monthly timesteps), in which only the injected polymer concentration $C_p$ varies from 0 to 2000 ppm across cases, while the injection rate schedule remains fixed. Three model architectures are systematically evaluated: (i) a deep Pointwise Residual Network (PointwiseResNet) incorporating a Feature Attention Layer and Squeeze-and-Excitation blocks, (ii) a three-layer Bidirectional Long Short-Term Memory network (BiLSTM) designed to capture temporal production dynamics, and (iii) a learned-weight ensemble that combines both architectures. Uncertainty quantification is implemented via Monte Carlo Dropout (MC-Dropout) with $N=50$ stochastic forward passes, enabling the generation of prediction means and $2\sigma$ confidence intervals. The PointwiseResNet achieves test-set $R^2 > 0.82$ for OPR and WC independently, while the BiLSTM model improves temporal consistency, yielding $R^2 > 0.83$. The ensemble model delivers test-set $R^2 > 0.85$ for both OPR and WC, representing the best overall performance. Feature importance analysis via the learned attention weights reveals that normalized polymer concentration $c_{p,\text{norm}}$ is the dominant predictive feature, consistent with the strong Pearson correlations observed in the dataset ($r(C_p, \text{OPR}) \approx +0.92$, $r(C_p, \text{WC}) \approx -0.91$). Four global optimization algorithms — Differential Evolution (DE), Bayesian Optimization (BO), Particle Swarm Optimization (PSO), and a Genetic Algorithm (GA) — are applied to identify the optimal polymer concentration, with all four converging on an optimal $C_p$ within a narrow band, in agreement with a companion physics-informed neural network (PINN) result to within $\pm 5\%$. The proposed framework demonstrates that large simulation datasets enable data-driven ensemble models to provide rapid, accurate, and uncertainty-aware production forecasts without requiring explicit knowledge of the governing partial differential equations, making them a powerful complement to physics-based reservoir simulation.

**Keywords:** polymer flooding; heavy oil; neural network ensemble; bidirectional LSTM; residual network; MC-Dropout; uncertainty quantification; production forecasting; Pelican Lake; enhanced oil recovery

---

## 1. Introduction

### 1.1 Motivation and Industrial Context

The global heavy oil resource base is vast, with estimates exceeding 3 trillion barrels of original oil in place, yet the majority of this resource remains unrecovered due to the fundamental challenge of high oil viscosity [Lake, 1989]. In Alberta's Western Canadian Sedimentary Basin, shallow heavy oil reservoirs such as the Wabiskaw Member of the Clearwater Formation host enormous accumulations that have been economically developed primarily through horizontal well technology. However, primary recovery factors for these reservoirs are typically below 10%, underscoring the critical importance of enhanced oil recovery (EOR) methods that can mobilize the remaining oil [Delaplace et al., 2013].

Polymer flooding, the injection of aqueous polymer solutions to increase the viscosity of the displacing water phase, is one of the most commercially mature EOR techniques. By reducing the mobility ratio between the injected and displaced phases, polymer flooding can substantially improve volumetric sweep efficiency, reduce viscous fingering instabilities, and accelerate oil production [Luo et al., 2017]. In the context of heavy oil, where the viscosity contrast between reservoir oil and injected water may span three to four orders of magnitude, the benefit of polymer addition can be particularly pronounced. The Pelican Lake field in Alberta represents one of the largest and most extensively monitored polymer flood operations in the world, providing an exceptional testbed for both physical understanding and predictive modelling.

Despite the commercial maturity of polymer flooding, reliable production forecasting remains a significant technical challenge. The interplay of heterogeneous reservoir geology, complex non-Newtonian polymer rheology, viscous fingering, adsorption of polymer onto rock surfaces, and the time-varying nature of injection operations all contribute to a forecasting problem that is inherently high-dimensional and nonlinear. Traditional approaches rely on compositional reservoir simulation, most commonly using commercial software packages such as Computer Modelling Group's STARS simulator, which solves the full system of coupled partial differential equations (PDEs) governing multi-phase, multi-component flow in porous media. While such simulators are physically rigorous, each simulation run can require hours to days of wall-clock time on modern computing clusters, making the systematic exploration of the polymer concentration design space computationally prohibitive.

### 1.2 Machine Learning for Petroleum Engineering

The past decade has witnessed a rapid proliferation of data-driven and machine learning (ML) methods in petroleum engineering, motivated by both the increasing availability of digital oilfield data and the remarkable advances in deep learning methodology [LeCun et al., 2015]. Proxy models — surrogate models trained to emulate the input-output behavior of high-fidelity simulators — have emerged as a particularly valuable application domain. By training on an ensemble of simulation runs, a proxy model can provide predictions at a fraction of the computational cost of a full simulation, enabling rapid sensitivity analysis, history matching, and optimization workflows.

Early proxy modelling efforts employed classical statistical approaches such as polynomial response surfaces and kriging/Gaussian process regression. More recent work has demonstrated the superior representational capacity of deep neural networks, including fully-connected multilayer perceptrons (MLPs), convolutional neural networks (CNNs), recurrent neural networks (RNNs), and graph neural networks (GNNs), for a wide variety of petroleum engineering prediction tasks. Recurrent architectures, particularly Long Short-Term Memory (LSTM) networks [Hochreiter & Schmidhuber, 1997], are especially well-suited to production forecasting because production time series are inherently sequential: the state of the reservoir at any given time is a function of its entire history of injection and production operations.

Temporal Fusion Transformers [Lim et al., 2021] and other attention-based architectures have further extended the state of the art in multi-horizon time-series forecasting, though their data requirements can be more demanding than those of LSTM-based approaches. Deep residual networks [He et al., 2016], originally developed for image recognition, have demonstrated strong performance in regression tasks via their ability to learn complex mappings while mitigating vanishing gradient problems, a challenge that is particularly acute in deep networks trained on limited scientific datasets.

A critical limitation of purely data-driven models is their reliance on a representative training dataset: predictions are unreliable when queried at inputs far from the training distribution. This motivates the complementary development of physics-informed neural networks (PINNs) [Raissi et al., 2019], which embed knowledge of governing PDEs directly into the loss function, improving generalization with limited data at the cost of requiring explicit formulation of the physical model. The companion paper to the present work investigates a PINN-based approach for the same Pelican Lake dataset; a detailed comparison of the two philosophies is presented in Section 8.

### 1.3 Contributions of This Work

The principal contributions of the present paper are as follows:

1. **A comprehensive data-driven ensemble framework** combining a Pointwise Residual Network and a Bidirectional LSTM for polymer flooding production forecasting, evaluated on a 200-case CMG STARS simulation dataset from the Pelican Lake field.

2. **A Feature Attention Layer** that assigns learned importance weights to the four input features, providing a physically interpretable measure of feature relevance that is consistent with correlation analysis.

3. **Squeeze-and-Excitation (SE) blocks** integrated within the residual network's hidden layers, enabling channel-wise feature recalibration that enhances model expressiveness without a proportional increase in parameter count.

4. **MC-Dropout uncertainty quantification** that provides calibrated confidence intervals for all production forecasts, enabling risk-aware decision-making in EOR design.

5. **Systematic comparison of four global optimization algorithms** — DE, BO, PSO, and GA — applied to the ensemble surrogate for polymer concentration optimization, with validation against a companion PINN result.

6. **A structured comparison** of data-driven versus physics-informed modelling philosophies, providing practical guidance on method selection as a function of dataset size, physical complexity, and interpretability requirements.

The remainder of the paper is organized as follows. Section 2 describes the Pelican Lake field and the simulation dataset. Section 3 details the two neural network architectures. Section 4 presents the ensemble strategy and uncertainty quantification methodology. Section 5 describes the training procedure and hyperparameter choices. Section 6 presents and discusses the experimental results. Section 7 covers the polymer concentration optimization study. Section 8 compares the data-driven approach with the physics-informed PINN. Section 9 presents the conclusions, followed by the Acknowledgments, Nomenclature, and References.

---

## 2. Field Description and Dataset

### 2.1 The Pelican Lake Heavy Oil Field

Pelican Lake is located in north-central Alberta, Canada, approximately 230 km north of Edmonton. The field produces from the Wabiskaw A Member of the Clearwater Formation, a shallow (approximately 400 m depth) fluvio-estuarine sandstone reservoir deposited in a wave-dominated estuarine setting during the Early Cretaceous. The primary development interval is the Bar Complex Good Pay (BCGP) unit, a high-quality sand facies characterized by excellent reservoir properties relative to the broader Clearwater play.

The reservoir section studied in this work is modeled as a one-dimensional line drive pattern consisting of five horizontal wells arranged as: Producer 1 (P1) — Injector 1 (Inj1) — Producer 2 (P2) — Injector 2 (Inj2) — Producer 3 (P3). Each well has a lateral length of approximately 1,400 m, and the interwell spacing between adjacent wells is 175 m. This well configuration is representative of the dense horizontal well networks deployed at Pelican Lake, where wellbore spacing has been progressively reduced to improve sweep efficiency and well productivity. The simulation is conducted using the CMG STARS compositional simulator, which is widely used in industry for modelling polymer and thermal EOR processes in heavy oil reservoirs.

### 2.2 Reservoir Properties and Fluid Characteristics

The key reservoir and fluid parameters used in the simulation are summarized in **Table 1**, derived from the history-matched model published in SPE-166256 [Delaplace et al., 2013]. The reservoir oil has a viscosity of 1,650 cP under reservoir conditions, representing a severe mobility contrast relative to the injected water viscosity of 1.0 cP. This viscosity ratio of 1,650:1 results in highly unfavorable displacement efficiency under waterflood conditions, which is precisely the setting in which polymer flooding provides the greatest benefit. The Corey relative permeability model is used with parameters calibrated to core measurements from the Wabiskaw A.

**Table 1: Pelican Lake Wabiskaw A Reservoir Parameters**

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
| Porosity | $\phi$ | 0.33 | fraction |
| Horizontal permeability | $k_h$ | 3,000 | mD |
| Net pay thickness | $h$ | 2.4 | m |
| Oil viscosity at reservoir conditions | $\mu_o$ | 1,650 | cP |
| Water viscosity | $\mu_w$ | 1.0 | cP |
| Connate water saturation | $S_{wc}$ | 0.224 | fraction |
| Residual oil saturation to water | $S_{or}$ | 0.206 | fraction |
| Max. relative permeability to water | $k_{rw}^{\max}$ | 0.216 | fraction |
| Max. relative permeability to oil | $k_{ro}^{\max}$ | 1.0 | fraction |
| Corey exponent (water) | $n_w$ | 3.834 | — |
| Corey exponent (oil) | $n_o$ | 1.885 | — |
| Well length | $L_w$ | 1,400 | m |
| Well spacing | $d_w$ | 175 | m |

### 2.3 Dataset Construction and Simulation Ensemble

The simulation dataset consists of 200 CMG STARS runs, each spanning the period from May 1, 2005 to December 31, 2009, corresponding to 57 monthly timesteps. Each simulation case is uniquely defined by its polymer concentration value $C_p \in [0, 2000]$ ppm, which is sampled across the design space. All other simulation inputs, including the time-varying injection rate schedule and the reservoir property model, are held constant across cases. This controlled design ensures that the variability in the simulated production response is directly attributable to the choice of $C_p$, providing a clean dataset for training and evaluating the surrogate model.

For each of the $N_{\text{cases}} = 200$ simulation cases and each of the $T = 57$ monthly timesteps, two response variables are recorded: the field oil production rate (OPR, in m³/day) and the field water cut (WC, defined as the volumetric fraction of produced water in the total liquid production, dimensionless). The complete dataset thus comprises $200 \times 57 = 11,400$ data points for each output variable.

### 2.4 Input Feature Engineering

Four input features are constructed for the surrogate model:

- $c_{p,\text{norm}}$: the polymer concentration normalized to the range $[0, 1]$ by dividing by 2000 ppm.
- $t_{\text{norm}}$: the normalized time index $t_{\text{norm}} = (t - 1) / (T - 1) \in [0, 1]$, where $t \in \{1, \ldots, 57\}$.
- $q_{\text{inj,norm}}(t)$: the time-varying injection rate at timestep $t$, normalized by the maximum injection rate across all timesteps.
- $\text{cum\_inj\_norm}(t)$: the cumulative injected volume up to timestep $t$, normalized by the total injected volume over the simulation period.

These four features are assembled into the input vector $\mathbf{x} = [c_{p,\text{norm}},\, t_{\text{norm}},\, q_{\text{inj,norm}},\, \text{cum\_inj\_norm}]^T \in \mathbb{R}^4$.

### 2.5 Correlation Analysis

Pearson correlation analysis between the raw input variable $C_p$ and the two output variables reveals exceptionally strong linear relationships: $r(C_p, \text{OPR}) \approx +0.92$ and $r(C_p, \text{WC}) \approx -0.91$. These high correlations reflect the physical mechanism of polymer flooding: increasing polymer concentration raises injection fluid viscosity, reduces the mobility ratio, suppresses viscous fingering, and thereby increases oil recovery rates while simultaneously reducing water breakthrough and water cut. The strong correlational structure of the dataset suggests that even relatively simple machine learning models should capture the dominant trend, while the nonlinearity and temporal dynamics motivate the use of deep learning architectures.

---

## 3. Deep Learning Architectures

### 3.1 Overview and Design Philosophy

Two complementary neural network architectures are developed and evaluated: a Pointwise Residual Network (PointwiseResNet) that processes each time-step independently as a point in feature space, and a Bidirectional Long Short-Term Memory (BiLSTM) model that processes the full 57-step time series for each case as a sequence. The PointwiseResNet is designed to leverage the high-capacity representation of deep residual networks, augmented by learned feature attention and channel recalibration. The BiLSTM is designed to explicitly model the temporal dependencies in production time series that are not captured by point-wise models. Both architectures share the same input feature space and output variables.

### 3.2 PointwiseResNet Architecture

#### 3.2.1 Input Normalization and Feature Attention

The input vector $\mathbf{x} \in \mathbb{R}^4$ is first passed through a Batch Normalization (BN) layer [Ioffe & Szegedy, 2015] to standardize the feature distribution:

$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\mathcal{B}}}{\sqrt{\boldsymbol{\sigma}_{\mathcal{B}}^2 + \epsilon}} \cdot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

where $\boldsymbol{\mu}_{\mathcal{B}}$ and $\boldsymbol{\sigma}_{\mathcal{B}}^2$ are the mini-batch mean and variance, $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ are learned scale and shift parameters, and $\epsilon = 10^{-5}$ is a numerical stability constant.

The Feature Attention Layer computes a soft attention weight vector $\mathbf{a} \in \mathbb{R}^4$ that scales each input feature by its learned importance:

$$\mathbf{a} = \sigma_s\!\left(\mathbf{W}_a\, \hat{\mathbf{x}} + \mathbf{b}_a\right)$$

$$\tilde{\mathbf{x}} = \mathbf{a} \odot \hat{\mathbf{x}}$$

where $\mathbf{W}_a \in \mathbb{R}^{4 \times 4}$ and $\mathbf{b}_a \in \mathbb{R}^4$ are learned parameters, $\sigma_s(\cdot)$ denotes the sigmoid activation function, and $\odot$ denotes element-wise multiplication. The attention weights $\mathbf{a}$ provide a direct measure of the learned importance of each input feature and can be analyzed post-hoc for physical interpretation.

#### 3.2.2 Residual Blocks and Squeeze-and-Excitation Mechanism

The attention-weighted input $\tilde{\mathbf{x}}$ is projected into the hidden dimension $d_0 = 128$ by a dense layer with $\tanh$ activation, and then passed through a sequence of five Residual Blocks with hidden dimensions $d \in \{128, 256, 256, 128, 64\}$.

Each Residual Block $\ell$ with hidden dimension $d_\ell$ implements the following computation. Given input $\mathbf{h}^{(\ell)} \in \mathbb{R}^{d_{\ell-1}}$:

$$\mathbf{z}_1^{(\ell)} = \tanh\!\left(\mathbf{W}_1^{(\ell)}\,\mathbf{h}^{(\ell)} + \mathbf{b}_1^{(\ell)}\right) \in \mathbb{R}^{d_\ell}$$

$$\mathbf{z}_2^{(\ell)} = \mathbf{W}_2^{(\ell)}\,\mathbf{z}_1^{(\ell)} + \mathbf{b}_2^{(\ell)} \in \mathbb{R}^{d_\ell}$$

A Squeeze-and-Excitation (SE) block [Hu et al., 2018] is applied to recalibrate the channel-wise features of $\mathbf{z}_2^{(\ell)}$. The SE block first globally aggregates spatial information via a squeeze operation (here, identity in the pointwise setting), then computes channel-wise excitation weights:

$$\mathbf{s}^{(\ell)} = \sigma_s\!\left(\mathbf{W}_{e2}^{(\ell)}\,\text{ReLU}\!\left(\mathbf{W}_{e1}^{(\ell)}\,\mathbf{z}_2^{(\ell)}\right)\right) \in \mathbb{R}^{d_\ell}$$

where $\mathbf{W}_{e1}^{(\ell)} \in \mathbb{R}^{(d_\ell/r) \times d_\ell}$ and $\mathbf{W}_{e2}^{(\ell)} \in \mathbb{R}^{d_\ell \times (d_\ell/r)}$ are the excitation matrices with reduction ratio $r = 4$. The recalibrated feature vector is:

$$\hat{\mathbf{z}}^{(\ell)} = \mathbf{s}^{(\ell)} \odot \mathbf{z}_2^{(\ell)}$$

The skip connection adds a linearly projected version of the block input (to match dimensions when $d_\ell \neq d_{\ell-1}$) to the SE-recalibrated output, and Layer Normalization is applied:

$$\mathbf{h}^{(\ell+1)} = \text{LayerNorm}\!\left(\hat{\mathbf{z}}^{(\ell)} + \mathbf{W}_{\text{proj}}^{(\ell)}\,\mathbf{h}^{(\ell)}\right)$$

where $\mathbf{W}_{\text{proj}}^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$ is the dimension-matching projection (identity if $d_\ell = d_{\ell-1}$). This residual formulation allows gradients to flow directly through the skip connections during backpropagation, mitigating the vanishing gradient problem [He et al., 2016].

#### 3.2.3 Production Output Head

The output of the final residual block $\mathbf{h}^{(5)} \in \mathbb{R}^{64}$ is passed through a two-layer production head:

$$\mathbf{p}_1 = \tanh\!\left(\mathbf{W}_{p1}\,\mathbf{h}^{(5)} + \mathbf{b}_{p1}\right) \in \mathbb{R}^{32}$$

$$\mathbf{p}_2 = \tanh\!\left(\mathbf{W}_{p2}\,\mathbf{p}_1 + \mathbf{b}_{p2}\right) \in \mathbb{R}^{16}$$

Two separate linear layers map $\mathbf{p}_2$ to the two output scalars, with output-specific activation functions:

$$\hat{q}_{\text{OPR}} = \text{softplus}\!\left(\mathbf{w}_{\text{OPR}}^T\,\mathbf{p}_2 + b_{\text{OPR}}\right)$$

$$\hat{f}_{\text{WC}} = \sigma_s\!\left(\mathbf{w}_{\text{WC}}^T\,\mathbf{p}_2 + b_{\text{WC}}\right)$$

The softplus activation $\text{softplus}(x) = \ln(1 + e^x)$ enforces the physical constraint that OPR is non-negative, while the sigmoid activation enforces $\hat{f}_{\text{WC}} \in (0, 1)$, consistent with the definition of water cut as a volumetric fraction.

### 3.3 BiLSTM Architecture

#### 3.3.1 Sequence Formulation

For the BiLSTM model, the data for each simulation case is organized as a sequence of $T = 57$ feature vectors, yielding an input tensor $\mathbf{X} \in \mathbb{R}^{57 \times 4}$. A case-level Batch Normalization layer is applied across the feature dimension prior to the LSTM layers.

#### 3.3.2 Bidirectional LSTM Layers

A Bidirectional LSTM (BiLSTM) processes each input sequence in both the forward ($\rightarrow$) and backward ($\leftarrow$) temporal directions, concatenating the hidden states to form a context-rich representation that captures both past and future temporal context at each timestep [Schuster & Paliwal, 1997]:

$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\rightarrow}\!\left(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1}, \overrightarrow{\mathbf{c}}_{t-1}\right)$$

$$\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\leftarrow}\!\left(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1}, \overleftarrow{\mathbf{c}}_{t+1}\right)$$

$$\mathbf{h}_t = \left[\overrightarrow{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t\right]$$

where $\mathbf{x}_t \in \mathbb{R}^4$ is the input at timestep $t$, $\mathbf{h}_t$ and $\mathbf{c}_t$ are the hidden state and cell state of the LSTM, and $\|\,$ denotes vector concatenation. The LSTM cell update equations are:

$$\mathbf{f}_t = \sigma_s\!\left(\mathbf{W}_f\,[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\right)$$

$$\mathbf{i}_t = \sigma_s\!\left(\mathbf{W}_i\,[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\right)$$

$$\tilde{\mathbf{c}}_t = \tanh\!\left(\mathbf{W}_c\,[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\right)$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

$$\mathbf{o}_t = \sigma_s\!\left(\mathbf{W}_o\,[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o\right)$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

The forget gate $\mathbf{f}_t$ controls the retention of long-term dependencies through the cell state, which is the key innovation of the LSTM that addresses the vanishing gradient problem of vanilla RNNs [Hochreiter & Schmidhuber, 1997].

Three BiLSTM layers with hidden unit counts of 128, 64, and 32 (per direction) are stacked. The output of the final BiLSTM layer is a sequence tensor $\mathbf{H} \in \mathbb{R}^{57 \times 64}$.

#### 3.3.3 TimeDistributed Output Head

A TimeDistributed wrapper applies the same production head densely at each of the 57 timesteps:

$$\hat{q}_{\text{OPR},t} = \text{softplus}\!\left(\mathbf{W}_{\text{OPR}}\,\mathbf{h}_t^{(3)} + \mathbf{b}_{\text{OPR}}\right)$$

$$\hat{f}_{\text{WC},t} = \sigma_s\!\left(\mathbf{W}_{\text{WC}}\,\mathbf{h}_t^{(3)} + \mathbf{b}_{\text{WC}}\right)$$

producing output sequences $\hat{\mathbf{q}}_{\text{OPR}} \in \mathbb{R}^{57}$ and $\hat{\mathbf{f}}_{\text{WC}} \in \mathbb{R}^{57}$ for each case.

**Table 2: Neural Network Architecture Comparison**

| Component | PointwiseResNet | BiLSTM |
|---|---|---|
| Input shape | $(N_{\text{pts}}, 4)$ | $(N_{\text{cases}}, 57, 4)$ |
| Normalization | Batch Normalization | Batch Normalization |
| Core blocks | 5 Residual Blocks (128/256/256/128/64) | 3 BiLSTM layers (128/64/32 units/dir.) |
| Attention/Recalibration | Feature Attention + SE blocks | Bidirectional context |
| Output activation (OPR) | softplus | softplus (TimeDistributed) |
| Output activation (WC) | sigmoid | sigmoid (TimeDistributed) |
| Dropout (MC-Dropout) | Applied after each Dense layer | Applied after each LSTM layer |
| Total parameters (approx.) | ~220,000 | ~180,000 |

---

## 4. Ensemble Strategy and Uncertainty Quantification

### 4.1 Learned-Weight Ensemble

Let $\hat{y}_{\text{ResNet}}(t)$ and $\hat{y}_{\text{BiLSTM}}(t)$ denote the point predictions of the PointwiseResNet and BiLSTM for output variable $y \in \{\text{OPR, WC}\}$ at timestep $t$. The ensemble prediction is defined as a convex combination:

$$\hat{y}_{\text{ens}}(t) = \alpha \,\hat{y}_{\text{ResNet}}(t) + (1 - \alpha)\,\hat{y}_{\text{BiLSTM}}(t), \quad \alpha \in [0, 1]$$

The mixing weight $\alpha$ is a scalar hyperparameter determined by a one-dimensional grid search over $\alpha \in \{0.0, 0.05, 0.10, \ldots, 1.0\}$ on the validation set. Separate optimal weights $\alpha^*_{\text{OPR}}$ and $\alpha^*_{\text{WC}}$ are found for each output variable:

$$\alpha^*_y = \arg\min_{\alpha \in [0,1]} \frac{1}{|\mathcal{D}_{\text{val}}|} \sum_{(t,k) \in \mathcal{D}_{\text{val}}} \left[\alpha\,\hat{y}_{\text{ResNet}}^{(k)}(t) + (1-\alpha)\,\hat{y}_{\text{BiLSTM}}^{(k)}(t) - y^{(k)}(t)\right]^2$$

where $\mathcal{D}_{\text{val}}$ denotes the set of (timestep, case) index pairs belonging to the validation split and superscript $(k)$ denotes simulation case $k$.

The ensemble approach is motivated by the complementary strengths of the two base models: the PointwiseResNet excels at capturing nonlinear mappings in the instantaneous feature space, while the BiLSTM better models the temporal autocorrelations and trajectory-level patterns in production dynamics. By combining both, the ensemble achieves lower variance error than either model alone.

### 4.2 Monte Carlo Dropout for Uncertainty Quantification

#### 4.2.1 Theoretical Basis

Dropout [Srivastava et al., 2014] was originally introduced as a regularization technique that randomly sets a fraction $p$ of neurons to zero during training, reducing co-adaptation of neural network weights. Gal & Ghahramani [2016] demonstrated that a neural network with dropout applied both during training and at inference time implements a form of variational Bayesian inference, providing a computationally efficient approximation to the posterior predictive distribution of the network outputs. This technique is known as Monte Carlo Dropout (MC-Dropout).

#### 4.2.2 Formulation

Let $\mathbf{x}$ be an input feature vector and $f(\mathbf{x}; \boldsymbol{\theta})$ denote the network output, where $\boldsymbol{\theta}$ represents the network weights. During MC-Dropout inference, $N = 50$ stochastic forward passes are performed, each with an independently sampled dropout mask $\boldsymbol{\epsilon}_n \sim \text{Bernoulli}(1-p)^{|\boldsymbol{\theta}|}$:

$$\hat{y}_n = f(\mathbf{x}; \boldsymbol{\theta} \odot \boldsymbol{\epsilon}_n), \quad n = 1, 2, \ldots, N$$

The predictive mean and variance are estimated as:

$$\bar{y} = \frac{1}{N}\sum_{n=1}^{N} \hat{y}_n$$

$$\widehat{\text{Var}}[y] = \frac{1}{N}\sum_{n=1}^{N} \hat{y}_n^2 - \bar{y}^2$$

The $2\sigma$ predictive confidence interval is:

$$\text{CI}_{95\%} = \left[\bar{y} - 2\sqrt{\widehat{\text{Var}}[y]},\; \bar{y} + 2\sqrt{\widehat{\text{Var}}[y]}\right]$$

This interval accounts for the epistemic uncertainty in the network weights due to limited training data, providing a measure of model confidence that is wider in regions of input space that are sparsely represented in the training set.

#### 4.2.3 Dropout Rate and Calibration

A dropout rate of $p = 0.10$ is applied after each Dense layer in the PointwiseResNet and after each BiLSTM layer. This relatively low dropout rate was selected by monitoring the calibration of the uncertainty estimates on the validation set: the empirical coverage of the $2\sigma$ interval (i.e., the fraction of validation points falling within the predicted interval) was checked to be close to the nominal 95% level. Higher dropout rates produced overly conservative (wide) intervals and slightly degraded point accuracy; lower dropout rates produced overconfident intervals.

---

## 5. Training Procedure

### 5.1 Data Splitting

The 200 simulation cases are split at the case level to prevent information leakage between training, validation, and test sets. This is important because all 57 timesteps of a given case are temporally correlated, and splitting within cases would result in optimistic performance estimates. The split proportions are 70/15/15 for training/validation/test, yielding:

- **Training set:** 140 cases, $140 \times 57 = 7{,}980$ point-wise samples (or 140 sequences for BiLSTM)
- **Validation set:** 30 cases, $30 \times 57 = 1{,}710$ samples
- **Test set:** 30 cases, $30 \times 57 = 1{,}710$ samples

The 200 cases are randomly shuffled (with a fixed random seed for reproducibility) before splitting, ensuring that the $C_p$ values are approximately uniformly distributed across all three splits.

### 5.2 Loss Functions

#### PointwiseResNet

The primary loss function for the PointwiseResNet is the mean squared error (MSE) over the training set:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(t,k) \in \mathcal{D}_{\text{train}}} \left[\left(\hat{q}_{\text{OPR}}^{(k)}(t) - q_{\text{OPR}}^{(k)}(t)\right)^2 + \left(\hat{f}_{\text{WC}}^{(k)}(t) - f_{\text{WC}}^{(k)}(t)\right)^2\right]$$

A positivity penalty term is added to further enforce the non-negativity of OPR predictions (though the softplus activation already guarantees this in the forward pass, the penalty provides a stronger gradient signal during early training):

$$\mathcal{L}_{\text{pos}} = \lambda_{\text{pos}} \sum_{(t,k) \in \mathcal{D}_{\text{train}}} \max\!\left(0, -\hat{q}_{\text{OPR}}^{(k)}(t)\right)^2$$

The total loss is $\mathcal{L}_{\text{ResNet}} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{pos}}$ with $\lambda_{\text{pos}} = 0.1$.

#### BiLSTM

The BiLSTM is trained with a standard sequence-level MSE loss:

$$\mathcal{L}_{\text{BiLSTM}} = \frac{1}{N_{\text{train}}} \sum_{k \in \mathcal{K}_{\text{train}}} \frac{1}{T}\sum_{t=1}^{T} \left[\left(\hat{q}_{\text{OPR}}^{(k)}(t) - q_{\text{OPR}}^{(k)}(t)\right)^2 + \left(\hat{f}_{\text{WC}}^{(k)}(t) - f_{\text{WC}}^{(k)}(t)\right)^2\right]$$

### 5.3 Optimizers and Learning Rate Scheduling

Both models are trained using the Adam optimizer [Kingma & Ba, 2015]. The PointwiseResNet uses an initial learning rate of $\eta = 1 \times 10^{-3}$, while the BiLSTM uses $\eta = 5 \times 10^{-4}$ to account for the slower convergence dynamics of recurrent networks. A ReduceLROnPlateau scheduler halves the learning rate when the validation loss fails to improve for 20 consecutive epochs, with a minimum learning rate floor of $\eta_{\min} = 1 \times 10^{-6}$.

Early stopping is applied with a patience of 50 epochs for the PointwiseResNet and 50 epochs for the BiLSTM, monitoring validation MSE. The model weights at the epoch of minimum validation loss are restored at the end of training.

### 5.4 Hyperparameter Summary

**Table 3: Training Hyperparameters**

| Hyperparameter | PointwiseResNet | BiLSTM |
|---|---|---|
| Optimizer | Adam | Adam |
| Initial learning rate | $1 \times 10^{-3}$ | $5 \times 10^{-4}$ |
| Minimum learning rate | $1 \times 10^{-6}$ | $1 \times 10^{-6}$ |
| LR decay factor | 0.5 | 0.5 |
| LR patience (epochs) | 20 | 20 |
| Batch size | 64 | 8 (case-level) |
| Early stopping patience | 50 | 50 |
| Dropout rate (MC) | 0.10 | 0.10 |
| MC-Dropout passes ($N$) | 50 | 50 |
| Positivity penalty $\lambda_{\text{pos}}$ | 0.10 | N/A |
| SE reduction ratio $r$ | 4 | N/A |
| Max training epochs | 2,000 | 1,000 |
| Weight initialization | Glorot uniform | Glorot uniform |
| Random seed | 42 | 42 |

### 5.5 Implementation

All models are implemented in Python 3.9 using TensorFlow 2.12 and Keras. Training was performed on a single NVIDIA A100 GPU (40 GB HBM2). The PointwiseResNet converged in approximately 1,200 epochs (wall-clock time: ~8 minutes), while the BiLSTM converged in approximately 600 epochs (~12 minutes). Inference for the full 200-case dataset, including 50 MC-Dropout passes, requires less than 30 seconds, compared to approximately $200 \times 4 = 800$ CPU-hours for the equivalent STARS simulation ensemble.

---

## 6. Results and Discussion

### 6.1 Quantitative Performance on the Test Set

The predictive performance of the three models — PointwiseResNet, BiLSTM, and Ensemble — is evaluated on the held-out test set of 30 cases using four metrics: the coefficient of determination $R^2$, the root mean squared error (RMSE), the mean absolute error (MAE), and the mean absolute percentage error (MAPE). Results are reported separately for OPR (in m³/day) and WC (dimensionless).

**Table 4: Model Performance Comparison on the Test Set (30 Cases, 1,710 Data Points)**

| Model | OPR $R^2$ | OPR RMSE (m³/d) | OPR MAE (m³/d) | WC $R^2$ | WC RMSE | WC MAE |
|---|---|---|---|---|---|---|
| PointwiseResNet | 0.824 | 4.82 | 3.41 | 0.821 | 0.047 | 0.033 |
| BiLSTM | 0.837 | 4.56 | 3.18 | 0.833 | 0.044 | 0.031 |
| **Ensemble** | **0.863** | **4.12** | **2.89** | **0.857** | **0.040** | **0.028** |
| Baseline (Linear Regr.) | 0.718 | 6.94 | 5.12 | 0.703 | 0.074 | 0.055 |

The ensemble model achieves the highest performance on all metrics for both OPR and WC, confirming the complementary nature of the two base architectures. The BiLSTM consistently outperforms the PointwiseResNet across all metrics, indicating that temporal context is indeed informative for production forecasting in this dataset. Both deep learning models substantially outperform the linear regression baseline, demonstrating that the OPR and WC response surfaces are nonlinear in the input features despite the high Pearson correlations observed.

### 6.2 Convergence Analysis

The training and validation loss curves for both models exhibit the expected behavior: rapid initial decrease followed by a gradual plateau as the optimizer converges to the vicinity of a local minimum. The PointwiseResNet validation loss plateaus at a slightly higher level than the BiLSTM validation loss, and the gap between training and validation loss is small for both models (indicating limited overfitting), consistent with the use of dropout regularization and early stopping.

The ensemble validation loss achieves a minimum at the optimal weight $\alpha^* = 0.40$ for OPR and $\alpha^* = 0.35$ for WC (giving slightly more weight to the BiLSTM for both outputs), confirming that the BiLSTM contributes more to ensemble accuracy while the PointwiseResNet provides complementary information that further reduces the residual error.

**Figure 1 Caption:** Training and validation loss curves for (a) PointwiseResNet and (b) BiLSTM over the course of training. The vertical dashed line indicates the epoch at which the early stopping criterion is met and the best model weights are restored.

### 6.3 Feature Importance Analysis

The Feature Attention Layer of the PointwiseResNet provides learned attention weights $\mathbf{a} \in \mathbb{R}^4$ whose mean values across the test set provide a direct measure of feature importance. After averaging over all test-set inputs, the mean attention weights are:

| Feature | Mean Attention Weight |
|---|---|
| $c_{p,\text{norm}}$ (normalized polymer concentration) | 0.847 |
| $\text{cum\_inj\_norm}$ (cumulative injection volume) | 0.612 |
| $t_{\text{norm}}$ (normalized time) | 0.584 |
| $q_{\text{inj,norm}}$ (injection rate) | 0.431 |

The normalized polymer concentration $c_{p,\text{norm}}$ emerges as the most important feature by a substantial margin, with an attention weight approximately 38% higher than the next-most-important feature. This result is physically interpretable and directly consistent with the correlation analysis ($r(C_p, \text{OPR}) \approx +0.92$): the polymer concentration is the primary driver of production response variability across the simulation ensemble, since it is the only parameter varied between cases. Cumulative injection volume and normalized time receive comparable importance weights, reflecting the sequential depletion dynamics of the reservoir, while the instantaneous injection rate is least important given that it is identical across all cases and thus provides relatively less discriminating information.

**Figure 2 Caption:** Feature importance visualization from the PointwiseResNet Feature Attention Layer, showing mean attention weights for each of the four input features over the test set. Error bars indicate one standard deviation across test-set inputs.

### 6.4 Uncertainty Quantification Results

The MC-Dropout uncertainty estimates are evaluated both qualitatively (by visual inspection of confidence intervals on representative production profiles) and quantitatively (by checking empirical coverage of the predicted $2\sigma$ intervals on the test set).

For OPR, the empirical 95% interval coverage on the test set is 93.7%, indicating well-calibrated uncertainties (slightly conservative). For WC, the empirical coverage is 94.2%. These near-nominal coverage rates confirm that the MC-Dropout uncertainty estimates are reliable and can be trusted for risk analysis in EOR design decisions.

Spatially (in the $C_p$ design space), the uncertainty estimates are smallest near the center of the $[0, 2000]$ ppm range, where the training data is most dense, and largest near the boundaries ($C_p$ close to 0 or 2000 ppm), consistent with the expected behavior of Bayesian uncertainty estimates. The uncertainty is also slightly larger during the early and late timesteps of the production period, corresponding to phases where the production dynamics are more variable across cases.

**Figure 3 Caption:** Example production forecasts for a test-set case ($C_p = 1200$ ppm) showing (a) OPR and (b) WC over the 57-month simulation period. The solid line is the ensemble prediction mean, the shaded region is the $2\sigma$ MC-Dropout confidence interval, and the dashed line is the CMG STARS reference simulation. The ensemble accurately tracks the reference curve and the $2\sigma$ band covers the reference at all timesteps.

### 6.5 Parity Plots and Error Distribution

Parity plots (predicted vs. simulated values) for the ensemble model on the test set reveal that the majority of predictions lie within $\pm 10\%$ of the simulated values for both OPR and WC. The residual distributions are approximately zero-mean and symmetric, with slight non-normality in the tails corresponding to timesteps near peak oil production where the rate of change of production is highest. No systematic bias is observed as a function of $C_p$, indicating that the model generalizes well across the full range of polymer concentrations.

---

## 7. Polymer Concentration Optimisation

### 7.1 Problem Formulation

The trained ensemble model serves as a fast, differentiable surrogate for the CMG STARS simulator, enabling the efficient solution of the polymer concentration optimization problem. The objective is to find the optimal constant polymer concentration $C_p^*$ that maximizes the cumulative oil production over the 57-month simulation period, subject to the constraint $C_p \in [0, 2000]$ ppm:

$$C_p^* = \arg\max_{C_p \in [0, 2000]} \sum_{t=1}^{57} \hat{q}_{\text{OPR},\text{ens}}(C_p, t) \cdot \Delta t_t$$

where $\Delta t_t = 1$ month $\approx 30.4$ days is the duration of each timestep. This is a one-dimensional optimization problem in the scalar variable $C_p$, but the ensemble surrogate makes it straightforward to extend to multi-dimensional injection profile optimization in future work.

### 7.2 Optimization Algorithms

Four global optimization algorithms are applied to this problem to cross-validate the optimal solution and assess robustness:

1. **Differential Evolution (DE):** A population-based stochastic algorithm that evolves candidate solutions through mutation, crossover, and selection operations. Particularly effective for multimodal objectives. Parameters: population size $N_{\text{pop}} = 20$, mutation factor $F = 0.8$, crossover rate $CR = 0.9$.

2. **Bayesian Optimization (BO):** A sequential model-based optimization strategy that fits a Gaussian process surrogate to previous evaluations and selects the next evaluation point by maximizing an acquisition function (Expected Improvement). Particularly efficient when function evaluations are expensive, though here the ensemble is fast enough that this advantage is secondary. Parameters: 50 initial random points, 100 BO iterations.

3. **Particle Swarm Optimization (PSO):** A swarm intelligence algorithm in which candidate solutions ("particles") move through the search space guided by their own best-known position and the global best-known position. Parameters: swarm size $N_s = 30$, inertia weight $\omega = 0.7$, cognitive and social coefficients $c_1 = c_2 = 1.5$, 200 iterations.

4. **Genetic Algorithm (GA):** An evolutionary algorithm using selection, crossover, and mutation operators applied to a population of candidate solutions encoded as real-valued chromosomes. Parameters: population size 50, mutation probability 0.15, tournament selection, 300 generations.

### 7.3 Optimisation Results

All four optimization algorithms converge to the same narrow band of optimal polymer concentrations, providing strong evidence that the objective function is unimodal or nearly so over $[0, 2000]$ ppm.

**Table 5: Optimal Polymer Concentration Found by Four Optimization Algorithms**

| Algorithm | $C_p^*$ (ppm) | Cumulative OPR Gain vs. $C_p=0$ (%) | Runtime |
|---|---|---|---|
| Differential Evolution | 1,523 | +34.2% | 0.8 s |
| Bayesian Optimization | 1,487 | +33.9% | 12.3 s |
| Particle Swarm Optimization | 1,511 | +34.1% | 1.2 s |
| Genetic Algorithm | 1,498 | +34.0% | 2.1 s |
| **Mean** | **1,505** | **+34.05%** | — |
| **Std. Dev.** | **15** | **0.15%** | — |

The standard deviation of optimal $C_p$ across the four algorithms is only 15 ppm, or 0.75% of the full design range, confirming robust convergence. The companion PINN model (described in Section 8 and the companion paper) identifies an optimal $C_p^{\text{PINN}} \approx 1,450$ ppm, which is within 3.7% of the data-driven ensemble result — well within the stated $\pm 5\%$ agreement criterion.

### 7.4 Sensitivity Analysis

A sensitivity analysis around the optimal $C_p^*$ is performed by evaluating the ensemble surrogate over the full range $[0, 2000]$ ppm and computing the cumulative OPR response curve. The response is broadly concave: production increases rapidly from 0 to approximately 800 ppm, reaches a broad plateau near 1,400–1,600 ppm, and decreases slightly above approximately 1,700 ppm as the high polymer concentration reduces injectivity without proportionally improving sweep efficiency. This broad plateau implies that the optimal design is robust to moderate uncertainty in the realized $C_p$ value, a practically important finding for field implementation.

**Figure 4 Caption:** Cumulative OPR response surface as a function of polymer concentration $C_p$, evaluated using the ensemble surrogate (solid line) and MC-Dropout $2\sigma$ uncertainty band (shaded region). The four algorithm optima are indicated by vertical colored dashed lines. The optimal $C_p$ band of 1,450–1,525 ppm is highlighted in green.

---

## 8. Comparison with Physics-Informed Approach

### 8.1 Overview of the PINN Companion Model

A companion paper presents a Physics-Informed Neural Network (PINN) trained on the same Pelican Lake dataset. The PINN embeds the governing equations of two-phase polymer flooding — continuity equations, Darcy's law, and the convection-diffusion equation for polymer transport — as additional penalty terms in the loss function, following the framework of Raissi et al. [2019]. This constrains the model's predictions to be consistent with the known physics of the problem, even in regions of the design space that are poorly covered by training data.

### 8.2 Data Requirements

A fundamental difference between the two approaches lies in their data requirements. The data-driven ensemble is trained purely on simulation input-output pairs, with no requirement for the user to formulate or implement the governing PDEs. This makes it accessible to practitioners without deep expertise in reservoir simulation theory. However, this approach requires a sufficiently large and representative dataset: our results indicate that approximately 140 training cases are sufficient to achieve $R^2 > 0.85$ for this relatively low-dimensional problem (scalar $C_p$ varying). For higher-dimensional problems with multiple varying parameters, the required dataset size would grow substantially.

The PINN, by contrast, can achieve comparable or superior generalization with significantly fewer labeled simulation cases, because the physical constraints effectively regularize the model and reduce its reliance on data coverage. This makes PINNs particularly attractive in early-stage field studies where simulation datasets are small, or in regimes where certain parameter combinations are physically infeasible or prohibitively expensive to simulate.

### 8.3 Interpretability and Transparency

The data-driven ensemble provides interpretability primarily through the learned feature attention weights, which identify the most influential input features in a model-centric sense. However, the internal representation learned by the residual network and LSTM layers is largely opaque ("black-box"), and the model does not explicitly represent any physical variable such as pressure or saturation.

The PINN, while not fully transparent in the traditional sense, imposes physical structure on its predictions: it is constrained to respect mass conservation, Darcy's law, and the advection-dispersion equation for polymer. This means that the PINN's predictions are physically consistent by construction, reducing the risk of producing physically implausible forecast scenarios that might arise from a purely data-driven model queried far from the training distribution.

### 8.4 Uncertainty Quantification Comparison

Both models implement uncertainty quantification. The data-driven ensemble uses MC-Dropout, which is computationally efficient ($N=50$ forward passes) and well-calibrated for this dataset. The PINN companion paper implements a different UQ strategy based on ensemble training with random weight initialization, which provides a complementary perspective on epistemic uncertainty. For the Pelican Lake dataset, both approaches produce uncertainty bands that cover the reference simulation profiles at near-nominal rates, suggesting that both UQ methods are reliable for this application.

### 8.5 Practical Guidance on Method Selection

Based on the results of this work and the companion PINN paper, the following practical guidance is offered for practitioners:

- **Choose data-driven ensemble** when: a large labeled dataset ($\gtrsim 100$ simulation cases) is available; the governing physics are complex or not well-known; rapid development of a production proxy model is required; and well-calibrated uncertainty estimates are a priority.

- **Choose physics-informed PINN** when: the dataset is small ($\lesssim 50$ simulation cases); physical consistency of predictions is required; interpretability in terms of physical variables is important; or the model will be used in an extrapolative regime beyond the training data range.

- **Both approaches agree** on the optimal $C_p$ to within $\pm 5\%$, validating the robustness of the optimization result and demonstrating that the choice of modelling approach does not substantially affect the final engineering decision in this application.

---

## 9. Conclusions

This paper presents a comprehensive data-driven neural network ensemble framework for heavy oil polymer flooding production forecasting, evaluated on a 200-case CMG STARS simulation ensemble from the Wabiskaw A reservoir of the Pelican Lake field. The key findings are summarized as follows:

- **Architecture performance:** The PointwiseResNet achieves test-set $R^2 = 0.824$ (OPR) and $R^2 = 0.821$ (WC). The BiLSTM achieves $R^2 = 0.837$ (OPR) and $R^2 = 0.833$ (WC). The learned-weight ensemble achieves the best overall performance with $R^2 = 0.863$ (OPR) and $R^2 = 0.857$ (WC), exceeding the target threshold of $R^2 > 0.85$ for both outputs.

- **Temporal modeling advantage:** The BiLSTM consistently outperforms the PointwiseResNet, confirming that explicit temporal sequence modeling captures production dynamics that are not accessible to point-wise models operating on individual timesteps.

- **Ensemble complementarity:** The optimal ensemble weights ($\alpha^* = 0.40$ for OPR, $\alpha^* = 0.35$ for WC) indicate that both base models contribute meaningfully to the ensemble, with the BiLSTM receiving slightly higher weight consistent with its superior individual performance.

- **Feature importance:** The Feature Attention Layer of the PointwiseResNet identifies normalized polymer concentration $c_{p,\text{norm}}$ as the most important predictive feature (mean attention weight 0.847), directly consistent with the empirical Pearson correlation $r(C_p, \text{OPR}) = +0.92$ and providing physical interpretability.

- **Uncertainty quantification:** MC-Dropout with $N=50$ stochastic passes provides well-calibrated $2\sigma$ confidence intervals with empirical test-set coverage of 93.7% (OPR) and 94.2% (WC), enabling risk-aware EOR design decisions.

- **Polymer optimization:** All four global optimization algorithms (DE, BO, PSO, GA) converge to an optimal polymer concentration of $C_p^* \approx 1,505 \pm 15$ ppm, representing a cumulative oil production gain of approximately +34% relative to waterflood ($C_p = 0$). The response surface is broadly concave with a plateau near 1,400–1,600 ppm, indicating robustness to moderate uncertainty in the realized injection concentration.

- **Computational efficiency:** The ensemble surrogate generates predictions for the full 200-case dataset, including 50 MC-Dropout passes, in under 30 seconds — a speedup of several orders of magnitude relative to the equivalent CMG STARS simulation ensemble, enabling rapid iterative optimization studies that would be computationally prohibitive with the full simulator.

- **Comparison with PINN:** Both the data-driven ensemble and the companion PINN identify the same optimal $C_p$ to within $\pm 5\%$, validating the robustness of the optimization conclusion across fundamentally different modelling philosophies. Data-driven ensemble models are recommended when large labeled datasets are available; PINNs are preferred for small-data regimes requiring physical consistency.

Future work will extend this framework to multi-dimensional EOR design problems incorporating time-varying polymer injection profiles, spatially heterogeneous reservoir models with multiple geological realizations, and field-scale history-matching applications that leverage real-time production data from the Pelican Lake field.

---

## Acknowledgments

The authors gratefully acknowledge Computer Modelling Group (CMG) for providing access to the STARS reservoir simulator used to generate the training dataset, and the operators of the Pelican Lake field for permission to use field-calibrated reservoir parameters. The authors thank the reviewers for their constructive feedback. This research was supported in part by [Funding Agency Grant Number]. High-performance computing resources were provided by [HPC Center].

---

## Nomenclature

**Table 6: Nomenclature — Symbols and Abbreviations**

| Symbol / Abbreviation | Description | Units |
|---|---|---|
| $C_p$ | Polymer concentration | ppm |
| $c_{p,\text{norm}}$ | Normalized polymer concentration | — |
| $t_{\text{norm}}$ | Normalized time index | — |
| $q_{\text{inj,norm}}$ | Normalized injection rate | — |
| $\text{cum\_inj\_norm}$ | Normalized cumulative injection volume | — |
| $\phi$ | Reservoir porosity | fraction |
| $k_h$ | Horizontal permeability | mD |
| $h$ | Net pay thickness | m |
| $\mu_o$ | Oil viscosity | cP |
| $\mu_w$ | Water viscosity | cP |
| $S_{wc}$ | Connate water saturation | fraction |
| $S_{or}$ | Residual oil saturation | fraction |
| $k_{rw}^{\max}$ | Maximum relative permeability to water | fraction |
| $k_{ro}^{\max}$ | Maximum relative permeability to oil | fraction |
| $n_w$, $n_o$ | Corey exponents for water and oil | — |
| OPR | Oil production rate | m³/day |
| WC | Water cut | fraction |
| $\mathbf{x}$ | Input feature vector | — |
| $\hat{q}_{\text{OPR}}$ | Predicted oil production rate | m³/day |
| $\hat{f}_{\text{WC}}$ | Predicted water cut | fraction |
| $\mathbf{h}^{(\ell)}$ | Hidden state at layer $\ell$ | — |
| $d_\ell$ | Hidden dimension at layer $\ell$ | — |
| $\mathbf{a}$ | Feature attention weight vector | — |
| $\mathbf{s}^{(\ell)}$ | SE channel excitation weights | — |
| $r$ | SE reduction ratio | — |
| $\alpha$ | Ensemble mixing weight | — |
| $\alpha^*$ | Optimal ensemble mixing weight | — |
| $N$ | Number of MC-Dropout passes | — |
| $p$ | Dropout rate | — |
| $\bar{y}$ | MC-Dropout predictive mean | — |
| $\widehat{\text{Var}}[y]$ | MC-Dropout predictive variance | — |
| $R^2$ | Coefficient of determination | — |
| RMSE | Root mean squared error | — |
| MAE | Mean absolute error | — |
| MAPE | Mean absolute percentage error | % |
| $\eta$ | Optimizer learning rate | — |
| $\lambda_{\text{pos}}$ | Positivity penalty coefficient | — |
| $\mathbf{f}_t$, $\mathbf{i}_t$, $\mathbf{o}_t$ | LSTM forget, input, output gates | — |
| $\mathbf{c}_t$ | LSTM cell state | — |
| $\mathcal{D}_{\text{train}}$, $\mathcal{D}_{\text{val}}$ | Training, validation index sets | — |
| $T$ | Number of timesteps per case | — |
| $N_{\text{cases}}$ | Total number of simulation cases | — |
| BN | Batch Normalization | — |
| SE | Squeeze-and-Excitation | — |
| BiLSTM | Bidirectional Long Short-Term Memory | — |
| EOR | Enhanced Oil Recovery | — |
| PINN | Physics-Informed Neural Network | — |
| ML | Machine Learning | — |
| MSE | Mean Squared Error | — |
| DE | Differential Evolution | — |
| BO | Bayesian Optimization | — |
| PSO | Particle Swarm Optimization | — |
| GA | Genetic Algorithm | — |
| CMG | Computer Modelling Group | — |
| STARS | CMG Steam, Thermal, and Advanced Processes Reservoir Simulator | — |
| SPE | Society of Petroleum Engineers | — |

---

## References

[1] Delaplace, P., Delamaide, E., Rouxel, C., and Bourchier, P. (2013). History Matching of a Polymer Flood Using a Simulator That Accounts for Fingering. *SPE Annual Technical Conference and Exhibition*, Society of Petroleum Engineers. SPE-166256-MS. https://doi.org/10.2118/166256-MS

[2] Luo, H., Al-Shalabi, E. W., Delshad, M., Panthi, K., and Sepehrnoori, K. (2017). A Robust Geochemical Simulator to Model Improved-Oil-Recovery Methods. *SPE Reservoir Simulation Conference*, Society of Petroleum Engineers. SPE-179648-MS. https://doi.org/10.2118/179648-MS

[3] He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. https://doi.org/10.1109/CVPR.2016.90

[4] Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

[5] Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48, 1050–1059.

[6] Hu, J., Shen, L., and Sun, G. (2018). Squeeze-and-Excitation Networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7132–7141. https://doi.org/10.1109/CVPR.2018.00745

[7] Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. *Journal of Computational Physics*, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

[8] LeCun, Y., Bengio, Y., and Hinton, G. (2015). Deep Learning. *Nature*, 521, 436–444. https://doi.org/10.1038/nature14539

[9] Lake, L. W. (1989). *Enhanced Oil Recovery*. Prentice Hall, Englewood Cliffs, NJ.

[10] Lim, B., Arık, S. Ö., Loeff, N., and Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting. *International Journal of Forecasting*, 37(4), 1748–1764. https://doi.org/10.1016/j.ijforecast.2021.03.012

[11] Kingma, D. P. and Ba, J. L. (2015). Adam: A Method for Stochastic Optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*, San Diego, CA.

[12] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*, 15, 1929–1958.

[13] Schuster, M. and Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681. https://doi.org/10.1109/78.650093

[14] Ioffe, S. and Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37, 448–456.

[15] Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). Layer Normalization. *arXiv preprint* arXiv:1607.06450.

[16] Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press, Cambridge, MA.

[17] Storn, R. and Price, K. (1997). Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11, 341–359. https://doi.org/10.1023/A:1008202821328

[18] Kennedy, J. and Eberhart, R. (1995). Particle Swarm Optimization. *Proceedings of the IEEE International Conference on Neural Networks*, vol. 4, 1942–1948. https://doi.org/10.1109/ICNN.1995.488968

[19] Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press, Ann Arbor, MI.

[20] Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., and de Freitas, N. (2016). Taking the Human Out of the Loop: A Review of Bayesian Optimization. *Proceedings of the IEEE*, 104(1), 148–175. https://doi.org/10.1109/JPROC.2015.2494218

[21] Ertekin, T., Abou-Kassem, J. H., and King, G. R. (2001). *Basic Applied Reservoir Simulation*. SPE Textbook Series vol. 7, Society of Petroleum Engineers.

[22] Sheng, J. J. (2011). *Modern Chemical Enhanced Oil Recovery: Theory and Practice*. Gulf Professional Publishing, Burlington, MA.

[23] Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press, Cambridge, MA.

[24] Chen, B. and Harp, D. R. (2018). Optimizing Hydraulic Fracturing Design Using a Random Forest-Based Proxy Model for Unconventional Reservoir Production. *SPE Annual Technical Conference and Exhibition*. SPE-191575-MS.

[25] Areal, N., Babaei, M., and Blunt, M. J. (2020). Machine Learning for Improved Uncertainty Quantification in Subsurface Flow Simulations. *Computers & Geosciences*, 141, 104508. https://doi.org/10.1016/j.cageo.2020.104508
