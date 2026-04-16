from typing import Dict, Optional, Tuple

import torch

from data import real_constellation


Tensor = torch.Tensor

# Small floor to keep variance estimates numerically stable.
EPS = 1e-9


def _batch_eye(size: int, batch_size: int, ref: Tensor) -> Tensor:
    # Build a batched identity matrix on the same device/dtype as the reference
    # tensor so all later matrix operations stay type-consistent.
    return torch.eye(size, device=ref.device, dtype=ref.dtype).unsqueeze(0).expand(
        batch_size, -1, -1
    )


def estimate_symbol_variance(H: Tensor, y: Tensor, x: Tensor, sigma2: Tensor) -> Tensor:
    """Estimate the current symbol-error variance ``v_t^2``.

    Args:
        H: Real-valued channel matrices with shape ``[B, M, N]``.
        y: Received vectors with shape ``[B, M, 1]``.
        x: Current symbol estimates with shape ``[B, N, 1]``.
        sigma2: Per-sample noise variance with shape ``[B, 1, 1]``.

    Returns:
        A batched variance estimate with shape ``[B, 1, 1]``.
    """
    # This is the paper's residual-based estimate of the current symbol error
    # variance v_t^2.
    residual = y - H @ x
    power = residual.pow(2).sum(dim=1, keepdim=True)
    m = H.shape[1]
    h_energy = H.pow(2).sum(dim=(1, 2), keepdim=True).clamp_min(EPS)
    v2 = (power - m * sigma2) / h_energy
    return v2.clamp_min(EPS)


def lmmse_matrix(H: Tensor, v2: Tensor, sigma2: Tensor) -> Tensor:
    """Compute the pre-scaled LMMSE linear estimator ``W_hat``.

    Args:
        H: Real-valued channel matrices with shape ``[B, M, N]``.
        v2: Current symbol-error variance estimate ``[B, 1, 1]``.
        sigma2: Noise variance ``[B, 1, 1]``.

    Returns:
        Batched linear estimator matrices with shape ``[B, N, M]``.
    """
    # Compute the linear MMSE matrix before the de-correlation scaling step.
    batch_size, m, _n = H.shape
    eye_m = _batch_eye(m, batch_size, H)
    hht = H @ H.transpose(1, 2)
    cov = v2 * hht + sigma2 * eye_m
    inv_cov = torch.linalg.solve(cov, eye_m)
    return v2 * H.transpose(1, 2) @ inv_cov


def decorrelate_matrix(W_hat: Tensor, H: Tensor) -> Tensor:
    """Apply the OAMP de-correlation scaling to ``W_hat``.

    Args:
        W_hat: Pre-scaled linear estimator with shape ``[B, N, M]``.
        H: Channel matrices with shape ``[B, M, N]``.

    Returns:
        De-correlated linear estimator ``W`` with shape ``[B, N, M]``.
    """
    # Scale W_hat so that tr(WH) = n in the real-valued system. This is what
    # gives OAMP its de-correlated linear estimator.
    n = H.shape[2]
    wh = W_hat @ H
    trace_wh = wh.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    scale = n / trace_wh.clamp_min(EPS)
    return scale * W_hat


def compute_tau2(
    H: Tensor,
    W: Tensor,
    v2: Tensor,
    sigma2: Tensor,
    theta: Optional[Tensor] = None,
) -> Tensor:
    """Compute the equivalent scalar AWGN variance ``tau_t^2``.

    Args:
        H: Channel matrices with shape ``[B, M, N]``.
        W: De-correlated linear estimator with shape ``[B, N, M]``.
        v2: Current symbol-error variance estimate ``[B, 1, 1]``.
        sigma2: Noise variance ``[B, 1, 1]``.
        theta: Optional learned OAMP-Net variance-scaling parameter.

    Returns:
        A batched scalar variance tensor with shape ``[B, 1, 1]``.
    """
    # tau_t^2 is the effective AWGN variance seen by the scalar denoiser.
    batch_size, _m, n = H.shape
    eye_n = _batch_eye(n, batch_size, H)

    if theta is None:
        theta = torch.ones_like(v2)

    c_mat = eye_n - theta * (W @ H)
    ww_t = W @ W.transpose(1, 2)
    cc_t = c_mat @ c_mat.transpose(1, 2)

    trace_cc = cc_t.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
    trace_ww = ww_t.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)

    # Paper-aligned OAMP-Net variance recursion.
    tau2 = (trace_cc / (2.0 * n)) * v2 + (theta.pow(2) * trace_ww / (4.0 * n)) * sigma2
    return tau2.clamp_min(EPS)


def mmse_denoiser(z: Tensor, tau2: Tensor, modulation: str = "qpsk") -> Tensor:
    """Apply the posterior-mean scalar MMSE denoiser.

    Args:
        z: Effective AWGN observations with shape ``[B, N, 1]``.
        tau2: Equivalent scalar variance ``[B, 1, 1]``.
        modulation: Modulation type that selects the real-valued alphabet.

    Returns:
        Denoised symbol estimates with the same shape as ``z``.
    """
    # Posterior-mean denoiser over the real-valued constellation points.
    levels = torch.tensor(
        real_constellation(modulation),
        device=z.device,
        dtype=z.dtype,
    ).view(1, 1, -1)
    z_flat = z.reshape(z.shape[0], -1, 1)
    tau2_flat = tau2.reshape(tau2.shape[0], 1, 1).clamp_min(EPS)

    # Compute unnormalized log posteriors under an AWGN observation model.
    logits = -((z_flat - levels) ** 2) / (2.0 * tau2_flat)
    weights = torch.softmax(logits, dim=-1)
    post_mean = (weights * levels).sum(dim=-1, keepdim=True)
    return post_mean.reshape_as(z)


def oamp_update(
    H: Tensor,
    y: Tensor,
    x: Tensor,
    v2: Tensor,
    sigma2: Tensor,
    modulation: str = "bpsk",
    gamma: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Execute one OAMP or OAMP-Net update step.

    Args:
        H: Channel matrices with shape ``[B, M, N]``.
        y: Received vectors ``[B, M, 1]``.
        x: Current symbol estimate ``[B, N, 1]``.
        v2: Current symbol-error variance ``[B, 1, 1]``.
        sigma2: Noise variance ``[B, 1, 1]``.
        modulation: Modulation type for the scalar denoiser.
        gamma: Optional learned linear-update scaling parameter.
        theta: Optional learned variance-update scaling parameter.

    Returns:
        A tuple ``(x_next, v2_next, stats)`` containing the next symbol
        estimate, next symbol-error variance, and selected intermediate values.
    """
    # In plain OAMP both gamma and theta are fixed to one. In OAMP-Net they are
    # learned per layer, so we accept them as optional inputs here.
    if gamma is None:
        gamma = torch.ones_like(v2)
    if theta is None:
        theta = torch.ones_like(v2)

    # 1) Linear estimator
    w_hat = lmmse_matrix(H, v2, sigma2)
    w = decorrelate_matrix(w_hat, H)
    residual = y - H @ x
    z = x + gamma * (w @ residual)

    # 2) Equivalent noise variance for the scalar AWGN denoiser
    tau2 = compute_tau2(H, w, v2, sigma2, theta=theta)

    # 3) Nonlinear denoiser
    x_next = mmse_denoiser(z, tau2, modulation=modulation)

    # 4) Refresh the symbol variance estimate for the next layer/iteration
    v2_next = estimate_symbol_variance(H, y, x_next, sigma2)

    stats = {
        "z": z,
        "w": w,
        "tau2": tau2,
        "v2": v2_next,
    }
    return x_next, v2_next, stats


@torch.no_grad()
def oamp_detect(
    H: Tensor,
    y: Tensor,
    sigma2: Tensor,
    num_iters: int = 10,
    modulation: str = "bpsk",
) -> Tensor:
    """Run the baseline OAMP detector for a fixed number of iterations.

    Args:
        H: Channel matrices with shape ``[B, M, N]``.
        y: Received vectors ``[B, M, 1]``.
        sigma2: Noise variance ``[B, 1, 1]``.
        num_iters: Number of OAMP iterations.
        modulation: Modulation type for the denoiser.

    Returns:
        Final detected symbols with shape ``[B, N, 1]``.
    """
    # Start from the all-zero estimate, which matches the paper's setup.
    batch_size, _m, n = H.shape
    x = torch.zeros(batch_size, n, 1, device=H.device, dtype=H.dtype)
    v2 = estimate_symbol_variance(H, y, x, sigma2)

    for _ in range(num_iters):
        # Each OAMP iteration reuses the same analytical update with gamma=1
        # and theta=1.
        x, v2, _stats = oamp_update(
            H=H,
            y=y,
            x=x,
            v2=v2,
            sigma2=sigma2,
            modulation=modulation,
        )

    return x
