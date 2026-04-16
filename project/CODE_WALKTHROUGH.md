# Code Walkthrough

## 1. Purpose of this document

This document explains how the code executes from the top-level script down to the OAMP and OAMP-Net detector internals.

If you want to understand the project as a pipeline, read the files in this order:

1. `main.py`
2. `train.py`
3. `data.py`
4. `oamp_net.py`
5. `oamp.py`
6. `evaluate.py`
7. `utils.py`

## 2. High-level execution flow

When you run:

```bash
python main.py ...
```

the project performs these stages:

1. parse command-line arguments
2. choose antenna configurations and channel models
3. train OAMP-Net
4. optionally train LMMSE-TISTA
5. evaluate BER for OAMP and OAMP-Net
6. save checkpoint, plots, and BER data

In short:

```text
main.py
  -> train.py
     -> data.py
     -> oamp_net.py
        -> oamp.py
  -> evaluate.py
     -> data.py
     -> oamp.py
     -> oamp_net.py
  -> utils.py
```

## 3. `main.py`: experiment driver

This is the entry point.

### What happens here

`main.py` is responsible for:

- reading CLI arguments
- deciding which experiments to run
- calling training
- calling evaluation
- saving plots and JSON outputs

### Main functions

#### `parse_args()`

Reads options such as:

- antenna configurations
- modulation
- number of iterations
- number of epochs
- batch size
- output directory
- channel model

#### `parse_antenna_configs()`

Converts a string like:

```text
4x4,8x8
```

into:

```text
[(4,4), (8,8)]
```

#### `iter_channel_models()`

Expands:

- `rayleigh`
- `correlated`
- `both`

into one or more actual experiment runs.

#### `run_single_experiment()`

This is the most important function in `main.py`.

It performs one complete run for one setting, such as:

- `4x4`, QPSK, Rayleigh

Inside it:

1. build training config
2. train OAMP-Net
3. optionally train LMMSE-TISTA
4. build evaluation config
5. evaluate BER
6. save plots and JSON results

#### `main()`

Loops through all requested combinations and calls `run_single_experiment()`.

## 4. `train.py`: model training

This file handles training.

### Key idea

Training here does not load a fixed dataset from disk.
Instead, it generates fresh synthetic MIMO data every epoch.

### Main parts

#### `TrainConfig`

Stores all training settings in one place:

- antennas
- modulation
- SNR range
- batch size
- epochs
- learning rate
- checkpoint path

#### `train_model()`

This is the main training loop.

What it does:

1. create the model
2. create optimizer and loss function
3. generate batches using `generate_mimo_batch()`
4. run forward pass
5. compute MSE loss
6. run backpropagation
7. update parameters
8. compute validation loss
9. save checkpoint at the end

### Training call chain

```text
train_model()
  -> generate_mimo_batch()
  -> model(H, y, sigma2)
     -> OAMPNet.forward()
        -> oamp_update()
```

## 5. `data.py`: synthetic MIMO data generation

This file creates the simulation data.

### What it generates

For each batch it produces:

- channel matrix `H`
- transmitted symbols `x`
- noisy received signal `y`
- noise variance `sigma2`

### Main helper functions

#### `_sample_snr_db()`

Either:

- uses one fixed SNR
- or samples a random SNR from a range

#### `_rayleigh_complex()`

Generates the complex Rayleigh fading channel matrix.

#### `_apply_kronecker_correlation()`

Applies correlation for the correlated-channel experiment.

#### `_to_real_valued_system()`

Converts the complex MIMO system into the equivalent real-valued system used by the rest of the pipeline.

#### `generate_mimo_batch()`

Main public function.

This is where the full synthetic sample is built.

What it does:

1. choose SNR
2. generate channel matrix
3. optionally apply correlation
4. generate modulation symbols
5. add noise
6. convert to real-valued form if needed
7. return `(H, x, y, sigma2)`

## 6. `oamp_net.py`: unfolded learned detector

This file contains the deep unfolding models.

### Main classes

#### `OAMPNet`

This is the main learned detector.

How it works:

1. initialize `x = 0`
2. estimate initial variance
3. repeat for `T` layers:
   - read learned `gamma_t`
   - read learned `theta_t`
   - call `oamp_update()`
4. return final estimate

This class does not implement all math directly.
It reuses the shared OAMP update logic from `oamp.py`.

#### `LMMSETISTA`

Optional comparison model.

It is similar to OAMP-Net, but ties `gamma_t` and `theta_t` into one shared step parameter per layer.

## 7. `oamp.py`: core detector math

This is the heart of the project.

It contains the mathematical operations used by both:

- baseline OAMP
- learned OAMP-Net

### Main functions

#### `estimate_symbol_variance()`

Computes the current symbol-error variance `v_t^2` from the residual.

#### `lmmse_matrix()`

Builds the linear MMSE estimator `W_hat`.

#### `decorrelate_matrix()`

Applies the OAMP normalization so the linear estimator becomes de-correlated.

#### `compute_tau2()`

Computes `tau_t^2`, which is the effective scalar noise variance used by the denoiser.

#### `mmse_denoiser()`

Performs scalar posterior-mean denoising over the real-valued constellation.

This is where the algorithm uses knowledge of the symbol alphabet.

#### `oamp_update()`

This is the key step.

It performs one full iteration/layer:

1. linear update
2. variance update
3. MMSE denoising
4. next variance estimate

This function is reused by:

- baseline OAMP
- OAMP-Net
- optional LMMSE-TISTA

#### `oamp_detect()`

Runs the baseline OAMP detector for a fixed number of iterations.

This is used in evaluation as the non-learned baseline.

## 8. `evaluate.py`: BER evaluation

This file measures final detector performance.

### Main logic

For each SNR:

1. generate test data
2. run baseline OAMP
3. run trained OAMP-Net
4. optionally run LMMSE-TISTA
5. compute BER
6. repeat for enough batches

### Main components

#### `EvalConfig`

Stores:

- antenna setup
- modulation
- channel model
- SNR list
- batch size
- stopping conditions

#### `evaluate_ber()`

Returns a dictionary containing:

- SNR values
- OAMP BER
- OAMP-Net BER
- optional LMMSE-TISTA BER

## 9. `utils.py`: support utilities

This file contains shared helper functions.

### Key functions

#### `set_seed()`

Sets random seeds for reproducibility.

#### `get_device()`

Chooses GPU if available, otherwise CPU.

#### `ber_from_logits()`

Converts predicted real-valued symbols into decisions and computes BER.

#### `save_checkpoint()` / `load_checkpoint()`

Save and restore model state.

#### `plot_training_loss()`

Plots training and validation loss.

#### `plot_ber_curves()`

Plots BER vs SNR.

## 10. End-to-end example

Suppose you run:

```bash
python main.py --configs 4x4 --modulation qpsk --num_iters 10 --epochs 2000
```

The actual flow is:

1. `main()` parses the arguments
2. `run_single_experiment()` builds a `TrainConfig`
3. `train_model()` creates `OAMPNet`
4. `generate_mimo_batch()` creates training data
5. `OAMPNet.forward()` runs `T` unfolded layers
6. each layer calls `oamp_update()`
7. training finishes and checkpoint is saved
8. `evaluate_ber()` runs OAMP and OAMP-Net on test SNRs
9. BER plot and JSON file are saved

## 11. Most important design idea

The most important design idea in the whole codebase is:

> `oamp_update()` is the shared mathematical unit.

That one function lets the project support:

- baseline OAMP
- OAMP-Net
- optional LMMSE-TISTA

This keeps the code:

- modular
- consistent
- easy to maintain

## 12. Final takeaway

If you want to understand the code quickly:

- `main.py` runs the experiment
- `train.py` learns the parameters
- `evaluate.py` measures BER
- `data.py` generates wireless samples
- `oamp.py` contains the core math
- `oamp_net.py` turns the math into a learned unfolded network
