# OAMP-Simulation

Model-driven deep learning for MIMO detection with a full PyTorch implementation of:

- Baseline `OAMP`
- Learned `OAMP-Net`
- Rayleigh and correlated-channel experiments
- BER vs SNR evaluation
- MATLAB plotting assets
- Technical and student-friendly documentation

## Highlights

- Paper-aligned OAMP and OAMP-Net pipeline
- QPSK-focused experiments with support for `BPSK` and `16QAM`
- Rayleigh and Kronecker-correlated MIMO channel models
- BER plots, JSON outputs, MATLAB CSV files, and publication-style MATLAB scripts
- Viva-ready reports, simplified explanations, and a code walkthrough

## What This Repo Contains

```text
oamp_simulation/
  A-Model-Driven-Deep-Learning-Network-for-MIMO-Detection.pdf
  README.md
  project/
    README.md
    data.py
    oamp.py
    oamp_net.py
    train.py
    evaluate.py
    utils.py
    main.py
    requirements.txt
    artifacts/
    matlab_data/
    TECHNICAL_REPORT.md
    TECHNICAL_REPORT_SIMPLE.md
    FINDINGS_AND_RESULTS_SIMPLE.md
    CODE_WALKTHROUGH.md
```

## Core Idea

This project studies the MIMO detection problem:

```math
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
```

The key comparison is:

- `OAMP`: classical iterative detector
- `OAMP-Net`: unfolded OAMP with learned per-layer parameters

Why OAMP-Net matters:

- it keeps the mathematical structure of OAMP
- it learns only a small number of parameters
- it improves BER without becoming a large black-box network

## Latest Validated Results

### Rayleigh Channel

| Configuration | SNR | OAMP BER | OAMP-Net BER |
|---|---:|---:|---:|
| `4x4` | `20 dB` | `4.8242e-03` | `3.0469e-03` |
| `8x8` | `20 dB` | `1.4941e-03` | `4.8828e-04` |

### Correlated Channel

| Configuration | SNR | OAMP BER | OAMP-Net BER |
|---|---:|---:|---:|
| `4x4`, `rho=0.5` | `20 dB` | `1.0041e-02` | `6.9253e-03` |

### SNR Gain at BER = `10^-3`

| Configuration | OAMP-Net Gain |
|---|---:|
| `4x4` | `2.13 dB` |
| `8x8` | `3.46 dB` |

These results show:

- BER decreases correctly with SNR
- OAMP-Net consistently outperforms OAMP
- the correlated channel is harder, but OAMP-Net still keeps an advantage

## Quick Start

From the `project/` directory:

```bash
../.venv312/bin/python main.py --configs 4x4,8x8 --modulation qpsk --num_iters 10 --epochs 2000 --batch_size 256 --num_eval_batches 50 --out_dir artifacts
```

Correlated-channel run:

```bash
../.venv312/bin/python main.py --configs 4x4 --modulation qpsk --channel_model correlated --rho_tx 0.5 --rho_rx 0.5 --num_iters 10 --epochs 2000 --batch_size 256 --num_eval_batches 50 --out_dir artifacts
```

## Documentation Map

If you are new to the project:

- Start with [project/README.md](./project/README.md)

If you want a code-level explanation:

- Read [project/CODE_WALKTHROUGH.md](./project/CODE_WALKTHROUGH.md)

If you want a simple explanation:

- Read [project/TECHNICAL_REPORT_SIMPLE.md](./project/TECHNICAL_REPORT_SIMPLE.md)
- Read [project/FINDINGS_AND_RESULTS_SIMPLE.md](./project/FINDINGS_AND_RESULTS_SIMPLE.md)

If you want the full technical write-up:

- Read [project/TECHNICAL_REPORT.md](./project/TECHNICAL_REPORT.md)

If you want MATLAB figures:

- Use [project/matlab_data/plot_oampnet_results.m](./project/matlab_data/plot_oampnet_results.m)

## Outputs

The repository keeps generated outputs in version control, including:

- BER figures
- BER JSON files
- training loss plots
- model checkpoints
- MATLAB CSV files
- MATLAB export scripts

Main output folder:

- [project/artifacts](./project/artifacts)

## Implementation Features

- Real-valued equivalent MIMO system modeling
- De-correlated LMMSE-based OAMP updates
- Posterior-mean MMSE denoiser
- OAMP-Net with `2T` trainable scalars
- Optional LMMSE-TISTA comparison model
- Training and validation loss tracking
- Publication-style MATLAB plots

## Contributors

- [th3-ma3stro](https://github.com/th3-ma3stro)

## Suggested Repository Metadata

Suggested description:

> Paper-aligned PyTorch implementation of OAMP and OAMP-Net for MIMO detection with Rayleigh/correlated channels, BER evaluation, MATLAB plotting, and technical documentation.

Suggested topics:

- `mimo`
- `wireless-communications`
- `signal-processing`
- `deep-learning`
- `pytorch`
- `oamp`
- `oamp-net`
- `ber`
- `matlab`
- `communications`
