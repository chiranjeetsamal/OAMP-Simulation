# OAMP and OAMP-Net for MIMO Detection

This project implements a paper-aligned, end-to-end simulation pipeline for:

- Baseline OAMP detection
- Unfolded OAMP-Net with `2T` trainable scalar parameters
- Optional LMMSE-TISTA comparison model
- Training with L2 loss and Adam
- BER vs SNR evaluation
- Rayleigh and Kronecker-correlated channel simulation
- BPSK, QPSK, and 16-QAM support

The implementation follows the structure of the OAMP-Net paper:

1. Real-valued MIMO model `y = Hx + n`
2. OAMP linear estimation with de-correlated LMMSE matrix
3. Variance recursion for `v_t^2` and `tau_t^2`
4. MMSE denoiser from the posterior mean formula in the paper
5. Unfolding `T` iterations into `T` layers with trainable `gamma_t` and `theta_t`

## Project structure

```text
project/
  data.py
  oamp.py
  oamp_net.py
  train.py
  evaluate.py
  utils.py
  main.py
  requirements.txt
```

## What is implemented from the paper

- Iterative OAMP baseline with de-correlated LMMSE estimation
- OAMP-Net layer unfolding with `2T` learned scalars
- Optional LMMSE-TISTA-style comparison path
- BER evaluation across `0` to `30` dB
- Multi-configuration runs such as `4x4` and `8x8`
- QPSK support via equivalent real-valued formulation
- Correlated MIMO channel support through a Kronecker model
- Training loss and BER plot generation
- Checkpoint save/load helpers

The exact per-layer equations implemented are:

- `r_t = x_hat_t + gamma_t W_t (y - H x_hat_t)`
- `v_t^2 = (||y - H x_hat_t||_2^2 - M sigma^2) / tr(H^T H)`
- `tau_t^2 = tr(C_t C_t^T) v_t^2 / (2N) + theta_t^2 sigma^2 tr(W_t W_t^T) / (4N)`
- `C_t = I - theta_t W_t H`
- `W_t = (2N / tr(W_hat_t H)) W_hat_t`
- `W_hat_t = v_t^2 H^T (v_t^2 H H^T + sigma^2 I / 2)^(-1)` in the paper

In this repo, the simulation works in an equivalent real-valued system, so the noise variance passed through the code is the real-domain variance. That keeps the implementation algebraically consistent with the paper after complex-to-real conversion.

## Install

Use Python `3.11` or `3.12` for best PyTorch compatibility.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Run the paper-style QPSK experiments for `4x4` and `8x8` i.i.d. Rayleigh channels:

```bash
cd project
python main.py --configs 4x4,8x8 --modulation qpsk --epochs 2000 --num_iters 10 --num_eval_batches 200
```

Run a correlated-channel experiment:

```bash
cd project
python main.py --configs 4x4 --modulation qpsk --channel_model correlated --rho_tx 0.5 --rho_rx 0.5 --epochs 2000
```

Run both channel models:

```bash
cd project
python main.py --configs 4x4,8x8 --modulation qpsk --channel_model both --rho_tx 0.5 --rho_rx 0.5
```

Run with the optional LMMSE-TISTA comparison:

```bash
cd project
python main.py --configs 4x4 --modulation qpsk --compare_tista --epochs 2000
```

## Outputs

Results are written under `artifacts/`:

- `oamp_net_*.pt`: model checkpoints
- `loss_*.png`: training loss plots
- `ber_*.png`: BER vs SNR plots
- `ber_*.json`: raw BER data and experiment metadata

## Latest validated results

Using the latest runs with larger evaluation budgets:

- `4x4` Rayleigh at `20 dB`: OAMP `4.8242e-03`, OAMP-Net `3.0469e-03`
- `8x8` Rayleigh at `20 dB`: OAMP `1.4941e-03`, OAMP-Net `4.8828e-04`
- `4x4` correlated at `20 dB`: OAMP `1.0041e-02`, OAMP-Net `6.9253e-03`

These results show:

- BER decreases with SNR as expected
- OAMP-Net consistently outperforms OAMP
- correlation makes detection harder, but OAMP-Net retains its advantage

## Notes

- The default modulation is `qpsk` because that is the primary setting used in the OAMP-Net paper, but `bpsk` is also supported.
- The paper uses `T=10`, QPSK, `10,000` epochs, `5,000` training samples and `1,000` validation samples per epoch, Adam with learning rate `1e-3`, and BER evaluation until more than `1,000` bit errors are observed. The repo now exposes these knobs directly.
- For QPSK, the detector works on the equivalent real-valued system, so BER is computed over the in-phase and quadrature bits.
- The trainable scalars are exposed as `gamma` and `theta` in the model. They correspond to the per-layer linear-update and variance-scaling parameters in the unfolded network.
