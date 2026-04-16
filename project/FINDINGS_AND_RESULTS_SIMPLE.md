# Findings and Results Explained Simply

## 1. What was the goal?

The goal of this project was to compare two methods for MIMO detection:

- `OAMP`
- `OAMP-Net`

Both methods try to recover transmitted symbols from received noisy signals.

The main question was:

> Does OAMP-Net perform better than normal OAMP?

The answer from the results is:

> Yes. OAMP-Net performs better than OAMP in the tested cases.

## 2. What was tested?

The experiments were run on:

- `4x4` MIMO with QPSK
- `8x8` MIMO with QPSK
- `4x4` correlated-channel MIMO with QPSK

The system was tested across many SNR values from:

- `0 dB` to `30 dB`

The main measurement used was:

- `BER` or Bit Error Rate

Lower BER means better performance.

## 3. Main finding

The most important finding is:

> OAMP-Net consistently gives lower BER than OAMP.

This means the hybrid model works better than the classical baseline.

It also means the learning-based improvements in OAMP-Net are useful.

## 4. What happened in the Rayleigh channel results?

In the Rayleigh channel experiments:

- BER decreased as SNR increased
- OAMP-Net was better than OAMP across the whole SNR range

This is exactly the kind of behavior we expect from a working detector.

### Why is this good?

Because:

- at low SNR, detection is hard, so BER is higher
- at high SNR, the channel is cleaner, so BER becomes lower

A good detector should show a BER curve that drops as SNR increases.

That happened here.

## 5. 4x4 Rayleigh result explained

For `4x4`, the detector worked well and OAMP-Net improved the results.

Examples:

- At `0 dB`
  - OAMP BER = `0.2596`
  - OAMP-Net BER = `0.2371`

- At `20 dB`
  - OAMP BER = `0.004824`
  - OAMP-Net BER = `0.003047`

- At `30 dB`
  - OAMP BER = `0.000527`
  - OAMP-Net BER = `0.000322`

### What does this mean?

It means:

- both methods improve as SNR increases
- OAMP-Net makes fewer mistakes than OAMP
- the improvement is clear, especially at medium and high SNR

## 6. 8x8 Rayleigh result explained

For `8x8`, the improvement from OAMP-Net is even stronger.

Examples:

- At `0 dB`
  - OAMP BER = `0.2516`
  - OAMP-Net BER = `0.2196`

- At `20 dB`
  - OAMP BER = `0.001494`
  - OAMP-Net BER = `0.000488`

- At `30 dB`
  - OAMP BER = `0.000122`
  - OAMP-Net BER = `0.0000537`

### What does this mean?

This means:

- OAMP-Net is also better in the larger antenna case
- the gain becomes more visible in `8x8`
- the hybrid approach is especially useful when the system becomes more challenging

This is a strong result because larger MIMO systems are usually harder to detect.

## 7. SNR gain across antenna configurations

One useful way to compare methods is:

> How much less SNR does OAMP-Net need to reach the same BER?

Using a target BER of `10^-3`, the estimated SNR gains are:

- `4x4`: about `2.13 dB`
- `8x8`: about `3.46 dB`

### Why is this important?

Because it means:

- OAMP-Net reaches the same error performance with less signal power
- or, for the same SNR, it gives better reliability

In simple words:

> OAMP-Net is more efficient.

## 8. What happened in the correlated-channel result?

The correlated-channel experiment is important because it is harder and more realistic than the ideal Rayleigh case.

In correlated channels:

- antenna paths are related to each other
- the channel is more difficult
- detection usually becomes worse

That is exactly what we observed.

### Example at `20 dB`

For `4x4` correlated:

- OAMP BER = `0.01004`
- OAMP-Net BER = `0.00693`

Compare that to `4x4` Rayleigh at `20 dB`:

- OAMP BER = `0.004824`
- OAMP-Net BER = `0.003047`

### What does this tell us?

It tells us:

- correlated channels are harder than Rayleigh channels
- BER becomes worse in both methods
- but OAMP-Net still stays better than OAMP

This is a very useful finding.

It shows that OAMP-Net does not only work in the easy case.
It also works better in the more difficult channel case.

## 9. Why are these findings important?

These findings matter because they support the main idea of the project:

> A hybrid model-driven deep learning approach can improve classical MIMO detection.

This is important for three reasons:

### 1. Better performance

OAMP-Net lowers BER compared to OAMP.

### 2. Better robustness

It still performs better even when the channel is correlated.

### 3. Small number of trainable parameters

OAMP-Net does not use a huge neural network.
It only learns a few parameters per layer.

So it improves performance without becoming unnecessarily large or complex.

## 10. What do the training curves show?

The training convergence plots show:

- training loss decreases over time
- validation loss stays in a similar range
- the learning process is stable overall

This means:

- the network learns useful parameter values
- training is not exploding
- the model is not behaving randomly

For both `4x4` and `8x8`, the losses remain controlled and gradually improve.

That supports the conclusion that the training pipeline is working correctly.

## 11. What do the final graphs tell us?

### BER vs SNR graph

This graph tells us:

- how BER changes with SNR
- whether OAMP-Net beats OAMP
- whether performance improves smoothly

From your results:

- all BER curves go down as SNR increases
- OAMP-Net is below OAMP, which means it is better

### SNR gain graph

This graph tells us:

- how much SNR OAMP-Net saves compared to OAMP at the same BER

From your results:

- OAMP-Net gives noticeable SNR gain
- the gain is larger in `8x8`

### Training convergence graph

This graph tells us:

- whether the model is learning properly during training

From your results:

- the training process is stable
- the loss values are reasonable

## 12. Overall conclusion

The final conclusion of this project is:

> OAMP-Net works better than OAMP for the tested MIMO detection problems.

More specifically:

- it reduces BER in `4x4`
- it reduces BER in `8x8`
- it keeps improving performance even under correlated channels
- it does this while keeping the original OAMP structure

So the project successfully demonstrates the value of the hybrid approach.

## 13. What should you say if someone asks for the takeaway?

A simple answer is:

> The results show that OAMP-Net improves classical OAMP by learning a small number of important update parameters. This gives better detection accuracy in both ideal and more realistic channel conditions.

## 14. Short viva-style summary

If you want a short spoken explanation, say:

> In this work, I implemented both OAMP and OAMP-Net for MIMO detection. I tested them on Rayleigh and correlated channels using QPSK modulation. The BER curves show that OAMP-Net consistently outperforms OAMP, especially in larger systems like 8x8 and in moderate-to-high SNR ranges. The correlated-channel results also show that OAMP-Net remains better even in harder channel conditions. So the main result is that a model-driven deep learning approach improves detection without losing the structure of the classical algorithm.

## 15. Final one-line summary

This project proves that OAMP-Net gives more accurate MIMO detection than OAMP, and it stays effective even when the channel becomes more difficult.
