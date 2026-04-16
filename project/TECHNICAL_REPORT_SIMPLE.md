# Simple Explanation of the OAMP-Net Project

## 1. What is this project about?

This project is about **detecting transmitted signals in a MIMO wireless system**.

In a MIMO system:

- multiple antennas transmit signals
- multiple antennas receive signals
- the received signals get mixed together by the wireless channel

So the receiver has to answer:

> "What symbols were actually transmitted?"

That is called the **MIMO detection problem**.

This project compares two methods:

- `OAMP`: a classical signal processing algorithm
- `OAMP-Net`: a smarter version of OAMP that learns a few parameters using deep learning

## 2. What does the system look like?

The system equation is:

```text
y = Hx + n
```

where:

- `x` = transmitted symbols
- `H` = channel matrix
- `n` = noise
- `y` = received signal

Simple meaning:

- `x` is what we send
- `H` mixes everything
- `n` adds random disturbance
- `y` is what we observe at the receiver

Our goal is to recover `x` from `y` and `H`.

## 3. Why is this difficult?

Because:

- many transmitted signals are mixed together
- the channel changes randomly
- noise is added
- larger antenna systems are harder to solve

So this is not a simple inverse problem.

## 4. What is OAMP?

OAMP stands for:

**Orthogonal Approximate Message Passing**

You do not need to memorize the full name for basic understanding.

Just remember:

- it is an **iterative detector**
- it improves the estimate step by step

In each iteration, it does two things:

1. **Linear step**
   - use the channel matrix to improve the current estimate

2. **Nonlinear denoising step**
   - push the estimate toward valid symbol values like QPSK points

So OAMP works like:

```text
rough guess -> improve -> denoise -> improve -> denoise -> ...
```

## 5. What is OAMP-Net?

OAMP-Net is the deep learning version of OAMP.

It is built by:

- taking each OAMP iteration
- turning it into one neural network layer

But it is **not** a fully black-box neural network.

It only learns a small number of parameters.

For each layer, it learns:

- `gamma_t`
- `theta_t`

If there are `T` layers, total trainable parameters are:

```text
2T
```

So if `T = 10`, the network learns only:

```text
20 parameters
```

That is very small compared to a normal deep neural network.

## 6. Why is it called a hybrid approach?

This is one of the most important ideas.

It is called **hybrid** because it combines:

- **classical signal processing**
- **deep learning**

The classical part is:

- OAMP structure
- linear estimator
- variance update
- denoiser

The learning part is:

- trainable `gamma_t`
- trainable `theta_t`

So this method is not:

- fully traditional
- fully neural

It is a mix of both.

That is why the paper calls it a **model-driven deep learning network**.

## 7. What is “model-driven”?

Model-driven means:

- we already know the mathematical structure of the problem
- we use that structure directly
- we only learn small corrections

This is different from a black-box neural network, where the model tries to learn everything from data.

In this project:

- the **model** is OAMP
- the **learning** adjusts only important step sizes

## 8. Why do we convert everything into real numbers?

Wireless signals are naturally complex-valued.

But deep learning code is easier to build in real-valued form.

So the project converts:

- complex vectors
- complex channel matrices

into an equivalent real-valued system.

This does **not** change the problem.

It just makes implementation easier.

## 9. What is BER?

BER means:

**Bit Error Rate**

It tells us:

> what fraction of bits were detected incorrectly

Lower BER is better.

Examples:

- `BER = 0.2` means 20% bits are wrong
- `BER = 0.001` means only 0.1% bits are wrong

So when the BER curve goes down, the detector is improving.

## 10. What does SNR mean?

SNR means:

**Signal-to-Noise Ratio**

It measures how strong the signal is compared to noise.

Higher SNR usually means:

- cleaner received signal
- easier detection
- lower BER

That is why BER is usually plotted against SNR.

## 11. What did your latest results show?

Your output showed:

- BER decreases as SNR increases
- OAMP-Net performs better than OAMP
- this happened for both `4x4` and `8x8`
- the same improvement also appeared in the correlated-channel experiment

That means the project is working correctly.

For example:

- at low SNR, both methods have higher BER
- at high SNR, BER becomes very small
- OAMP-Net gives lower BER than OAMP at many points

This is exactly the behavior you want.

## 12. Why is OAMP-Net better?

Because plain OAMP uses fixed update rules.

OAMP-Net keeps the same structure but learns better step sizes from data.

So OAMP-Net can:

- adapt better to practical channel conditions
- reduce mismatch in the original algorithm
- improve BER without needing a huge neural network

## 13. What is a correlated channel?

In an ideal Rayleigh channel:

- channel coefficients are independent

In a correlated channel:

- antenna paths are related to each other
- the channel becomes harder

This is more realistic in practice.

So correlated-channel experiments are useful because they test:

- robustness
- realism
- how well the method works when the channel is not ideal

Usually correlated channels give:

- worse BER than independent Rayleigh channels

But if OAMP-Net still beats OAMP there, that is a strong result.

## 14. What does each file do?

## `data.py`

This file creates the wireless data.

It:

- generates transmitted symbols
- creates channel matrices
- adds noise
- supports Rayleigh and correlated channels

Simple meaning:

- this file creates the simulation world

## `oamp.py`

This file contains the baseline OAMP algorithm.

It:

- computes the linear update
- computes the variance
- denoises the estimate
- repeats for multiple iterations

Simple meaning:

- this file is the classical detector

## `oamp_net.py`

This file contains OAMP-Net.

It:

- unfolds OAMP into layers
- adds trainable `gamma` and `theta`

Simple meaning:

- this file is the hybrid deep learning detector

## `train.py`

This file trains the network.

It:

- generates batches
- computes loss
- updates parameters using Adam

Simple meaning:

- this file teaches OAMP-Net how to improve itself

## `evaluate.py`

This file evaluates BER.

It:

- runs OAMP and OAMP-Net on test data
- compares predictions with true symbols
- reports BER for different SNR values

Simple meaning:

- this file checks how good the detector is

## `utils.py`

This file contains helper functions.

It:

- sets seeds
- saves models
- loads models
- computes BER
- plots graphs

Simple meaning:

- this file contains supporting tools

## `main.py`

This is the main entry point.

It:

- reads command-line settings
- trains the model
- evaluates BER
- saves plots and outputs

Simple meaning:

- this file runs the full project

## 15. What should you say in viva?

### Short explanation

This project solves the MIMO detection problem using a hybrid method called OAMP-Net.

OAMP is the baseline iterative detector. OAMP-Net unfolds OAMP into neural network layers and learns only two scalar parameters per layer, which improves detection performance while keeping the original algorithm structure.

### If asked “why hybrid?”

Say:

> It is hybrid because it combines the mathematical structure of OAMP with small trainable parameters learned from data.

### If asked “why not a fully connected neural network?”

Say:

> A fully connected neural network has many parameters and less interpretability. OAMP-Net is much lighter and preserves domain knowledge from signal processing.

### If asked “what metric did you use?”

Say:

> BER versus SNR.

### If asked “what is the main result?”

Say:

> OAMP-Net consistently achieves lower BER than OAMP, especially in moderate and high SNR regions.

## 16. Final takeaway

The main message of the project is:

> We can improve a classical MIMO detector by adding a very small amount of learning without losing the original mathematical structure.

That is why OAMP-Net is powerful:

- simple
- interpretable
- lightweight
- effective

## 17. One-line summary

This project shows that a smart combination of signal processing and deep learning can detect MIMO signals more accurately than the classical algorithm alone.
