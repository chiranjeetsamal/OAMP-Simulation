# OAMP-Net for MIMO Detection

## 1. Overview

This project implements and evaluates a model-driven deep learning detector for MIMO systems based on OAMP-Net.

The repository contains:

- A real-valued MIMO simulation pipeline
- A baseline OAMP detector
- An unfolded OAMP-Net with `2T` trainable scalar parameters
- Training and evaluation scripts
- BER vs SNR plotting
- Support for Rayleigh and correlated MIMO channels
- Support for BPSK, QPSK, and 16-QAM in the equivalent real-valued domain

The main objective is to compare:

- `OAMP`: a model-based iterative detector
- `OAMP-Net`: a model-driven deep unfolding network derived from OAMP

Your latest successful run shows that the implementation is functioning correctly and that OAMP-Net improves BER over baseline OAMP.

## 2. Problem Statement

We consider the MIMO detection problem:

```math
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
```

where:

- `x` is the transmitted symbol vector
- `H` is the channel matrix
- `n` is additive white Gaussian noise
- `y` is the received signal vector

The task of the detector is to estimate `x` from `y` and `H`.

This problem becomes difficult when:

- the number of antennas increases
- the channel is ill-conditioned
- the SNR is moderate or low

## 3. Why OAMP and OAMP-Net

### 3.1 OAMP

Orthogonal Approximate Message Passing is an iterative detector that alternates between:

- a linear estimation step
- a nonlinear denoising step

It has strong theoretical grounding and relatively low parameter complexity.

### 3.2 OAMP-Net

OAMP-Net unfolds the iterative OAMP algorithm into a neural network with `T` layers.

Instead of learning a fully black-box network, it only learns:

- `gamma_t`
- `theta_t`

for each layer `t`.

So total trainable parameters are:

```math
2T
```

This is why OAMP-Net is called model-driven deep learning:

- it preserves the signal-processing structure
- it injects only a few trainable corrections

## 4. Real-Valued System Model

The original complex MIMO system is:

```math
\bar{\mathbf{y}} = \bar{\mathbf{H}}\bar{\mathbf{x}} + \bar{\mathbf{n}}
```

To use standard real-valued deep learning layers, the project converts the system into an equivalent real-valued model:

```math
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
```

with:

```math
\mathbf{x} =
\begin{bmatrix}
\Re(\bar{\mathbf{x}}) \\
\Im(\bar{\mathbf{x}})
\end{bmatrix},
\quad
\mathbf{y} =
\begin{bmatrix}
\Re(\bar{\mathbf{y}}) \\
\Im(\bar{\mathbf{y}})
\end{bmatrix}
```

and

```math
\mathbf{H} =
\begin{bmatrix}
\Re(\bar{\mathbf{H}}) & -\Im(\bar{\mathbf{H}}) \\
\Im(\bar{\mathbf{H}}) & \Re(\bar{\mathbf{H}})
\end{bmatrix}
```

This doubles the effective dimension:

- complex `N x N` becomes real `2N x 2N`

For QPSK, this is especially convenient because:

- the in-phase and quadrature components become two real antipodal components

## 5. Math Section

### 5.1 OAMP Update Equations

At iteration `t`, OAMP performs:

#### Linear update

```math
\mathbf{r}_t = \hat{\mathbf{x}}_t + \mathbf{W}_t(\mathbf{y} - \mathbf{H}\hat{\mathbf{x}}_t)
```

#### Nonlinear denoising

```math
\hat{\mathbf{x}}_{t+1} = \mathbb{E}\{\mathbf{x} \mid \mathbf{r}_t, \tau_t\}
```

#### Error variance estimate

```math
v_t^2 = \frac{\|\mathbf{y} - \mathbf{H}\hat{\mathbf{x}}_t\|_2^2 - M\sigma^2}{\mathrm{tr}(\mathbf{H}^T\mathbf{H})}
```

#### Equivalent noise variance

```math
\tau_t^2 = \frac{1}{2N}\mathrm{tr}(\mathbf{B}_t\mathbf{B}_t^T)v_t^2
+ \frac{1}{4N}\mathrm{tr}(\mathbf{W}_t\mathbf{W}_t^T)\sigma^2
```

where:

```math
\mathbf{B}_t = \mathbf{I} - \mathbf{W}_t\mathbf{H}
```

### 5.2 De-correlated LMMSE Matrix

The linear estimator uses an LMMSE-style matrix:

```math
\hat{\mathbf{W}}_t =
v_t^2 \mathbf{H}^T
\left(
v_t^2 \mathbf{H}\mathbf{H}^T + \frac{\sigma^2}{2}\mathbf{I}
\right)^{-1}
```

Then it is de-correlated as:

```math
\mathbf{W}_t =
\frac{2N}{\mathrm{tr}(\hat{\mathbf{W}}_t \mathbf{H})}\hat{\mathbf{W}}_t
```

This scaling is important because it enforces:

```math
\mathrm{tr}(\mathbf{W}_t\mathbf{H}) = 2N
```

In the real-valued implementation, this becomes the correct normalization for the unfolded system.

### 5.3 OAMP-Net Update Equations

OAMP-Net replaces the fixed OAMP scalars with trainable layer-wise parameters:

#### Learned linear update

```math
\mathbf{r}_t = \hat{\mathbf{x}}_t + \gamma_t \mathbf{W}_t(\mathbf{y} - \mathbf{H}\hat{\mathbf{x}}_t)
```

#### Learned variance update

```math
\tau_t^2 = \frac{1}{2N}\mathrm{tr}(\mathbf{C}_t\mathbf{C}_t^T)v_t^2
+ \frac{\theta_t^2 \sigma^2}{4N}\mathrm{tr}(\mathbf{W}_t\mathbf{W}_t^T)
```

where:

```math
\mathbf{C}_t = \mathbf{I} - \theta_t \mathbf{W}_t \mathbf{H}
```

So:

- `gamma_t` controls the step size of the mean update
- `theta_t` controls the effective variance update

### 5.4 MMSE Denoiser

For a real alphabet set:

```math
\mathcal{S} = \{s_1, s_2, \dots, s_K\}
```

the posterior mean estimator is:

```math
\mathbb{E}\{x_i \mid r_i, \tau_t\}
=
\frac{
\sum_{s \in \mathcal{S}} s \, \mathcal{N}(s; r_i, \tau_t^2) p(s)
}{
\sum_{s \in \mathcal{S}} \mathcal{N}(s; r_i, \tau_t^2) p(s)
}
```

In code, this is implemented as a softmax-weighted average over constellation points.

## 6. Channel and Modulation Models

### 6.1 Rayleigh Channel

The independent Rayleigh channel is generated as:

```math
\bar{\mathbf{H}} \sim \mathcal{N}_{\mathbb{C}}(0, 1/M)
```

This means each complex channel coefficient is zero-mean Gaussian with normalized variance.

### 6.2 Correlated Channel

The correlated MIMO channel follows the Kronecker model:

```math
\mathbf{H} = \mathbf{R}_R^{1/2}\mathbf{A}\mathbf{R}_T^{1/2}
```

where:

- `A` is a Rayleigh fading matrix
- `R_R` is receive correlation
- `R_T` is transmit correlation

The code uses exponential correlation:

```math
[\mathbf{R}]_{ij} = \rho^{|i-j|}
```

### 6.3 Modulations

Implemented:

- `BPSK`
- `QPSK`
- `16QAM`

QPSK is the primary setting used in the paper.

## 7. Training Setup

The network is trained using:

- loss: Mean Squared Error
- optimizer: Adam
- mini-batch training
- GPU support through PyTorch

In the code:

- training samples per epoch: `5000`
- validation samples per epoch: `1000`
- trainable variables: only `2T`

This makes OAMP-Net very lightweight compared to fully connected black-box detectors.

## 8. Interpretation of the Latest Experimental Output

The latest validated experiments are:

- `4x4`, QPSK, Rayleigh, `T=10`, `epochs=2000`, `num_eval_batches=50`
- `8x8`, QPSK, Rayleigh, `T=10`, `epochs=2000`, `num_eval_batches=50`
- `4x4`, QPSK, correlated channel with `rho_tx=rho_rx=0.5`, `T=10`, `epochs=2000`, `num_eval_batches=200`

### 8.1 4x4 Results

Selected points:

- `0 dB`: OAMP `2.5960e-01`, OAMP-Net `2.3714e-01`
- `10 dB`: OAMP `4.9886e-02`, OAMP-Net `4.1951e-02`
- `20 dB`: OAMP `4.8242e-03`, OAMP-Net `3.0469e-03`
- `30 dB`: OAMP `5.2734e-04`, OAMP-Net `3.2227e-04`

Interpretation:

- BER decreases steadily with SNR, so the detector is stable
- OAMP-Net is consistently better than OAMP
- the gain is moderate but real
- the system is already near very low BER at high SNR

### 8.2 8x8 Results

Selected points:

- `0 dB`: OAMP `2.5159e-01`, OAMP-Net `2.1960e-01`
- `10 dB`: OAMP `3.1860e-02`, OAMP-Net `2.5464e-02`
- `20 dB`: OAMP `1.4941e-03`, OAMP-Net `4.8828e-04`
- `24 dB`: OAMP `5.4199e-04`, OAMP-Net `1.8555e-04`
- `30 dB`: OAMP `1.2207e-04`, OAMP-Net `5.3711e-05`

Interpretation:

- OAMP-Net improvement is stronger in `8x8`
- the gain becomes clearer in the moderate-to-high SNR regime
- this matches the intuition behind model-driven learning: the learned parameters help more when the iterative baseline is close to good but still suboptimal

### 8.3 Correlated Channel Results

- `0 dB`: OAMP `3.0005e-01`, OAMP-Net `2.6660e-01`
- `10 dB`: OAMP `8.0627e-02`, OAMP-Net `6.8665e-02`
- `20 dB`: OAMP `1.0041e-02`, OAMP-Net `6.9253e-03`
- `30 dB`: OAMP `1.1230e-03`, OAMP-Net `7.2266e-04`

Interpretation:

- the correlated channel is clearly harder than the Rayleigh channel
- BER values are higher at the same SNR
- OAMP-Net still maintains a consistent advantage over OAMP
- this supports the robustness claim in the paper-style evaluation

### 8.4 SNR Gain Across Antenna Configurations

Using interpolation at target `BER = 10^{-3}`:

- `4x4` Rayleigh: OAMP-Net gain is about `2.13 dB`
- `8x8` Rayleigh: OAMP-Net gain is about `3.46 dB`

The latest `8x8` result shows the largest gain among the tested Rayleigh configurations.

## 9. File-by-File Explanation

## 9.1 `data.py`

Purpose:

- generates synthetic MIMO training and evaluation data

What it does:

- samples SNR values
- creates Rayleigh or correlated channels
- generates transmitted symbols for BPSK, QPSK, or 16-QAM
- adds AWGN noise
- converts complex systems to equivalent real-valued form

Key functions:

- `_sample_snr_db`
- `_rayleigh_complex`
- `_apply_kronecker_correlation`
- `_to_real_valued_system`
- `generate_mimo_batch`

Why it matters:

- this is the full simulation environment used by both training and testing

## 9.2 `oamp.py`

Purpose:

- implements the baseline OAMP detector and shared mathematical primitives

What it does:

- estimates signal variance `v_t^2`
- computes de-correlated LMMSE matrix `W_t`
- computes `tau_t^2`
- performs MMSE denoising over the constellation
- runs iterative OAMP detection

Key functions:

- `estimate_symbol_variance`
- `lmmse_matrix`
- `decorrelate_matrix`
- `compute_tau2`
- `mmse_denoiser`
- `oamp_update`
- `oamp_detect`

Why it matters:

- this is the mathematical core of the project
- OAMP-Net reuses this structure

## 9.3 `oamp_net.py`

Purpose:

- implements the unfolded neural detector

What it does:

- defines `OAMPNet`
- stores layer-wise trainable parameters `gamma` and `theta`
- runs unfolded iterative inference across `T` layers
- optionally defines `LMMSETISTA` for comparison

Key classes:

- `OAMPNet`
- `LMMSETISTA`

Why it matters:

- this is the model-driven deep learning component
- it preserves OAMP structure while learning only a few parameters

## 9.4 `train.py`

Purpose:

- trains OAMP-Net or LMMSE-TISTA

What it does:

- creates the model
- builds batches from `data.py`
- applies MSE loss
- optimizes with Adam
- tracks training and validation loss
- saves checkpoints

Key components:

- `TrainConfig`
- `train_model`
- `train_oamp_net`

Why it matters:

- this file turns the detector into a learnable network

## 9.5 `evaluate.py`

Purpose:

- evaluates BER performance

What it does:

- generates test batches across SNR values
- evaluates OAMP and OAMP-Net
- computes BER
- supports optional LMMSE-TISTA comparison

Key components:

- `EvalConfig`
- `evaluate_ber`

Why it matters:

- this is the performance-measurement layer of the project

## 9.6 `utils.py`

Purpose:

- utility functions for reproducibility, checkpoints, BER, and plots

What it does:

- sets random seeds
- selects CPU or GPU device
- computes BER
- saves and loads checkpoints
- plots training loss
- plots BER vs SNR curves
- forces a headless Matplotlib backend

Why it matters:

- this file keeps the project clean and reusable

## 9.7 `main.py`

Purpose:

- end-to-end experiment driver

What it does:

- parses CLI arguments
- loops over antenna configurations
- trains the model
- evaluates BER
- saves plots and JSON outputs

Why it matters:

- this is the single command entry point for the full project

## 9.8 `requirements.txt`

Purpose:

- lists Python dependencies

Main dependencies:

- `torch`
- `numpy`
- `matplotlib`

## 10. Output Artifacts

For each run, the project produces:

- model checkpoint `.pt`
- training loss plot `.png`
- BER curve plot `.png`
- BER data `.json`

These artifacts are useful for:

- academic reports
- presentations
- viva discussions
- portfolio demos

## 11. Viva Section

### 11.1 Short Viva Summary

This project implements a model-driven deep learning detector for MIMO systems.

The baseline algorithm is OAMP, which alternates between a de-correlated LMMSE linear estimator and a Bayesian MMSE denoiser.

OAMP-Net unfolds the OAMP iterations into neural network layers and introduces only two trainable scalar parameters per layer: `gamma_t` and `theta_t`.

This keeps the model lightweight, interpretable, and easier to train than a fully data-driven detector.

### 11.2 Expected Viva Questions and Answers

#### Q1. What is the main problem solved in this project?

This project solves MIMO detection, which means estimating transmitted symbols from the received signal in the presence of channel mixing and noise.

#### Q2. Why is MIMO detection difficult?

Because the receiver observes a mixture of all transmitted signals through the channel matrix, and the estimation becomes harder when the number of antennas increases or when the channel is poorly conditioned.

#### Q3. What is OAMP?

OAMP stands for Orthogonal Approximate Message Passing. It is an iterative detector that combines a linear estimation step with a nonlinear denoising step and uses variance tracking to approximate Bayesian MMSE detection.

#### Q4. What is OAMP-Net?

OAMP-Net is a deep unfolding version of OAMP where each iteration becomes one network layer, and a few scalar parameters are learned from data to improve performance.

#### Q5. Why is OAMP-Net called model-driven?

Because it is not a black-box neural network. It is derived directly from a known iterative algorithm and preserves the mathematical structure of that algorithm.

#### Q6. How many trainable parameters does OAMP-Net have?

If there are `T` layers, then it has `2T` trainable scalar parameters: one `gamma_t` and one `theta_t` per layer.

#### Q7. What is the role of `gamma_t`?

It scales the linear update step.

#### Q8. What is the role of `theta_t`?

It adjusts the variance recursion used to compute the equivalent AWGN noise variance seen by the denoiser.

#### Q9. Why do we convert the system into real-valued form?

Because standard neural network implementations are easier to build and train in the real domain, and the complex system can be represented exactly by an equivalent real-valued model.

#### Q10. What loss function was used?

Mean Squared Error between predicted symbols and ground-truth transmitted symbols.

#### Q11. What metric was used for performance evaluation?

Bit Error Rate versus SNR.

#### Q12. Why is BER plotted on a log scale?

Because BER often spans several orders of magnitude, and a log-scale plot makes performance differences clearer.

#### Q13. Why does OAMP-Net outperform OAMP?

Because the learned parameters adapt the iterative updates to the practical channel and data distribution, reducing mismatch in the fixed analytical updates.

#### Q14. Why were some BER points irregular at high SNR?

Because the evaluation used only a small number of batches, so the BER estimate had sampling noise.

### 11.3 Strong Viva Talking Points

- OAMP-Net is parameter-efficient because it learns only `2T` scalars
- It is interpretable because every learned parameter has a physical role
- It preserves the algorithmic prior of OAMP
- It avoids the heavy parameter count of fully connected detectors
- It improves BER while keeping the overall iterative structure

## 12. Practical Conclusion

Your current implementation is now in a good state:

- baseline OAMP works
- OAMP-Net works
- BER decreases with SNR
- OAMP-Net outperforms OAMP
- the code is modular and reusable

For the best final report or portfolio presentation, the next best step is:

- rerun evaluation with more batches for smoother BER curves
- optionally run the correlated-channel experiment
- include the generated plots in your final write-up

## 13. Suggested Final Statement

This project demonstrates that model-driven deep learning can improve classical iterative MIMO detection without abandoning the underlying signal-processing structure. OAMP-Net preserves the interpretability and efficiency of OAMP while using a very small number of trainable parameters to achieve better BER performance across multiple MIMO sizes and SNR regimes.
