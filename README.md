# OAMP-Net MIMO Detection Project

This repository contains a complete end-to-end implementation of:

- baseline `OAMP` MIMO detection
- `OAMP-Net` as a model-driven deep unfolding network
- training and BER evaluation pipelines
- Rayleigh and correlated-channel experiments
- MATLAB-ready plotting assets
- technical reports, simple explanations, and viva support material

The implementation is based on:

- the local summary PDF: [A-Model-Driven-Deep-Learning-Network-for-MIMO-Detection.pdf](./A-Model-Driven-Deep-Learning-Network-for-MIMO-Detection.pdf)
- the OAMP-Net paper structure and equations

## Repository layout

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

## What is included

- Working Python implementation in PyTorch
- Baseline OAMP
- OAMP-Net with `2T` trainable parameters
- BER vs SNR plots and raw BER JSON outputs
- Correlated-channel experiment support
- MATLAB CSV files and plotting scripts
- Student-friendly and technical documentation

## Latest validated findings

- `4x4` Rayleigh at `20 dB`:
  - OAMP BER = `4.8242e-03`
  - OAMP-Net BER = `3.0469e-03`

- `8x8` Rayleigh at `20 dB`:
  - OAMP BER = `1.4941e-03`
  - OAMP-Net BER = `4.8828e-04`

- `4x4` correlated channel at `20 dB`:
  - OAMP BER = `1.0041e-02`
  - OAMP-Net BER = `6.9253e-03`

These results show that:

- BER decreases correctly with SNR
- OAMP-Net consistently outperforms OAMP
- the correlated channel is harder, but OAMP-Net still keeps an advantage

## Where to start

If you want the implementation details, start here:

- [project/README.md](./project/README.md)

If you want the code flow explanation:

- [project/CODE_WALKTHROUGH.md](./project/CODE_WALKTHROUGH.md)

If you want an easy explanation:

- [project/TECHNICAL_REPORT_SIMPLE.md](./project/TECHNICAL_REPORT_SIMPLE.md)
- [project/FINDINGS_AND_RESULTS_SIMPLE.md](./project/FINDINGS_AND_RESULTS_SIMPLE.md)

If you want MATLAB plotting:

- [project/matlab_data/plot_oampnet_results.m](./project/matlab_data/plot_oampnet_results.m)

## Running the project

From `project/`:

```bash
../.venv312/bin/python main.py --configs 4x4,8x8 --modulation qpsk --num_iters 10 --epochs 2000 --batch_size 256 --num_eval_batches 50 --out_dir artifacts
```

For more commands and options, see:

- [project/README.md](./project/README.md)

## Contributors

- [th3-ma3stro](https://github.com/th3-ma3stro)

## Suggested GitHub repo metadata

Suggested repository name:

- `oamp-net-mimo-detection`

Suggested repository description:

- `Paper-aligned PyTorch implementation of OAMP and OAMP-Net for MIMO detection with Rayleigh/correlated channels, BER evaluation, MATLAB plotting, and documentation.`
