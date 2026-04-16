import math
from typing import Tuple, Union

import torch


# All data generation functions return tensors in the equivalent real-valued
# domain so that the rest of the pipeline can stay purely real-valued.
Tensor = torch.Tensor


def _sample_snr_db(
    batch_size: int,
    snr_db: Union[float, Tuple[float, float]],
    device: torch.device,
) -> Tensor:
    # During training we often sample a range of SNRs. During evaluation we
    # usually pass a single fixed SNR value.
    if isinstance(snr_db, tuple):
        low, high = snr_db
        return low + (high - low) * torch.rand(batch_size, 1, 1, device=device)
    return torch.full((batch_size, 1, 1), float(snr_db), device=device)


def _exponential_correlation(
    size: int,
    rho: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    # Exponential correlation model used in the correlated MIMO experiments:
    # [R]_{ij} = rho^{|i-j|}
    idx = torch.arange(size, device=device, dtype=torch.float32)
    dist = torch.abs(idx[:, None] - idx[None, :])
    corr = (rho ** dist).to(dtype)
    return corr


def _complex_matrix_sqrt_psd(matrix: Tensor) -> Tensor:
    # The Kronecker channel model needs a matrix square root of a Hermitian
    # positive semidefinite correlation matrix.
    evals, evecs = torch.linalg.eigh(matrix)
    evals = evals.clamp_min(0.0)
    sqrt_evals = torch.diag_embed(torch.sqrt(evals)).to(matrix.dtype)
    return evecs @ sqrt_evals @ evecs.conj().transpose(-2, -1)


def _rayleigh_complex(
    batch_size: int,
    n_rx: int,
    n_tx: int,
    device: torch.device,
) -> Tensor:
    # Normalize by sqrt(n_rx) so received power stays well-scaled as the
    # antenna configuration changes.
    real = torch.randn(batch_size, n_rx, n_tx, device=device)
    imag = torch.randn(batch_size, n_rx, n_tx, device=device)
    return (real + 1j * imag) / math.sqrt(2.0 * n_rx)


def _apply_kronecker_correlation(
    Hc: Tensor,
    rho_tx: float,
    rho_rx: float,
) -> Tensor:
    # Kronecker correlation applies separable spatial correlation at the
    # transmitter and receiver.
    batch_size, n_rx, n_tx = Hc.shape
    device = Hc.device
    dtype = Hc.dtype

    r_tx = _exponential_correlation(n_tx, rho_tx, device, dtype)
    r_rx = _exponential_correlation(n_rx, rho_rx, device, dtype)

    sqrt_tx = _complex_matrix_sqrt_psd(r_tx).unsqueeze(0).expand(batch_size, -1, -1)
    sqrt_rx = _complex_matrix_sqrt_psd(r_rx).unsqueeze(0).expand(batch_size, -1, -1)
    return sqrt_rx @ Hc @ sqrt_tx


def _to_real_valued_system(
    Hc: Tensor,
    xc: Tensor,
    yc: Tensor,
    sigma2: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Convert the complex system into the standard equivalent real-valued form:
    # [Re(y); Im(y)] = [[Re(H), -Im(H)]; [Im(H), Re(H)]] [Re(x); Im(x)] + n.
    Hr = Hc.real
    Hi = Hc.imag
    xr = torch.cat([xc.real, xc.imag], dim=1)
    yr = torch.cat([yc.real, yc.imag], dim=1)

    top = torch.cat([Hr, -Hi], dim=2)
    bot = torch.cat([Hi, Hr], dim=2)
    H_real = torch.cat([top, bot], dim=1)
    # After conversion to the real-valued system, each real component sees half
    # of the original complex noise variance.
    sigma2_real = sigma2 / 2.0
    return H_real, xr, yr, sigma2_real


def antipodal_amplitude(modulation: str) -> float:
    """Return the per-component symbol amplitude for antipodal constellations.

    Args:
        modulation: Modulation name such as ``"bpsk"`` or ``"qpsk"``.

    Returns:
        The real-domain amplitude used for one scalar component.

    Raises:
        ValueError: If the modulation is not supported by this helper.
    """
    # BPSK symbols lie on {-1, +1}; QPSK real and imaginary components lie on
    # {-1/sqrt(2), +1/sqrt(2)} so total symbol energy stays normalized.
    modulation = modulation.lower()
    if modulation == "bpsk":
        return 1.0
    if modulation == "qpsk":
        return 1.0 / math.sqrt(2.0)
    raise ValueError(f"Unsupported modulation '{modulation}'.")


def real_constellation(modulation: str):
    """Return the real-valued alphabet used by the scalar MMSE denoiser.

    Args:
        modulation: Modulation name such as ``"bpsk"``, ``"qpsk"``, or
            ``"16qam"``.

    Returns:
        A Python list containing the real-valued constellation points.

    Raises:
        ValueError: If the modulation is not implemented.
    """
    # The denoiser operates on the real-valued alphabet of each component.
    modulation = modulation.lower()
    if modulation == "bpsk":
        return [-1.0, 1.0]
    if modulation == "qpsk":
        amp = 1.0 / math.sqrt(2.0)
        return [-amp, amp]
    if modulation == "16qam":
        scale = 1.0 / math.sqrt(10.0)
        return [-3.0 * scale, -1.0 * scale, 1.0 * scale, 3.0 * scale]
    raise ValueError(f"Unsupported modulation '{modulation}'.")


def generate_mimo_batch(
    batch_size: int,
    n_rx: int,
    n_tx: int,
    snr_db: Union[float, Tuple[float, float]],
    modulation: str = "bpsk",
    channel_model: str = "rayleigh",
    rho_tx: float = 0.0,
    rho_rx: float = 0.0,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate a batch of synthetic MIMO samples in real-valued form.

    Args:
        batch_size: Number of independent channel/symbol realizations.
        n_rx: Number of receive antennas in the original complex system.
        n_tx: Number of transmit antennas in the original complex system.
        snr_db: Either one fixed SNR value or a ``(low, high)`` range.
        modulation: Modulation type.
        channel_model: ``"rayleigh"`` or ``"correlated"``.
        rho_tx: Transmit-side correlation coefficient for correlated channels.
        rho_rx: Receive-side correlation coefficient for correlated channels.
        device: Target PyTorch device.

    Returns:
        A tuple ``(H, x, y, sigma2)`` where:
            - ``H`` has shape ``[B, M, N]``
            - ``x`` has shape ``[B, N, 1]``
            - ``y`` has shape ``[B, M, 1]``
            - ``sigma2`` has shape ``[B, 1, 1]``
    """
    device = torch.device(device)
    snr_batch_db = _sample_snr_db(batch_size, snr_db, device)
    snr_lin = 10.0 ** (snr_batch_db / 10.0)
    sigma2 = 1.0 / snr_lin

    modulation = modulation.lower()
    channel_model = channel_model.lower()
    if modulation not in {"bpsk", "qpsk", "16qam"}:
        raise ValueError(f"Unsupported modulation '{modulation}'.")
    if channel_model not in {"rayleigh", "correlated"}:
        raise ValueError(f"Unsupported channel_model '{channel_model}'.")

    # Start from an i.i.d. Rayleigh fading matrix and optionally apply spatial
    # correlation.
    Hc = _rayleigh_complex(batch_size, n_rx, n_tx, device)
    if channel_model == "correlated":
        Hc = _apply_kronecker_correlation(Hc, rho_tx=rho_tx, rho_rx=rho_rx)

    if modulation == "bpsk":
        # BPSK stays entirely real, so we can work directly with the real part
        # of the channel matrix without building the doubled system.
        bits = torch.randint(
            0,
            2,
            (batch_size, n_tx, 1),
            device=device,
            dtype=torch.float32,
        )
        xc = 2.0 * bits - 1.0
        noise_c = torch.randn(batch_size, n_rx, 1, device=device) * torch.sqrt(sigma2)
        yc = Hc.real @ xc + noise_c
        return Hc.real, xc, yc, sigma2

    if modulation == "qpsk":
        # QPSK is generated in the complex domain first and then converted to
        # the equivalent real-valued system.
        amp = antipodal_amplitude("qpsk")
        bits_i = torch.randint(
            0,
            2,
            (batch_size, n_tx, 1),
            device=device,
            dtype=torch.float32,
        )
        bits_q = torch.randint(
            0,
            2,
            (batch_size, n_tx, 1),
            device=device,
            dtype=torch.float32,
        )
        xr = amp * (2.0 * bits_i - 1.0)
        xi = amp * (2.0 * bits_q - 1.0)
        xc = xr + 1j * xi
    else:
        # 16-QAM is represented as two independent 4-PAM components.
        levels = torch.tensor(real_constellation("16qam"), device=device, dtype=torch.float32)
        idx_i = torch.randint(0, levels.numel(), (batch_size, n_tx, 1), device=device)
        idx_q = torch.randint(0, levels.numel(), (batch_size, n_tx, 1), device=device)
        xr = levels[idx_i]
        xi = levels[idx_q]
        xc = xr + 1j * xi

    noise_c = (
        torch.randn(batch_size, n_rx, 1, device=device)
        + 1j * torch.randn(batch_size, n_rx, 1, device=device)
    ) * torch.sqrt(sigma2 / 2.0)
    yc = Hc @ xc + noise_c
    return _to_real_valued_system(Hc, xc, yc, sigma2)


def default_train_snr_range(modulation: str) -> Tuple[float, float]:
    """Return a sensible default training SNR range for a modulation type.

    Args:
        modulation: Modulation name.

    Returns:
        A ``(low_snr_db, high_snr_db)`` tuple used for random SNR sampling
        during training.

    Raises:
        ValueError: If the modulation is not supported.
    """
    # The wider 16-QAM range shifts upward because denser constellations need
    # higher SNR for meaningful training targets.
    modulation = modulation.lower()
    if modulation == "bpsk":
        return (0.0, 20.0)
    if modulation == "qpsk":
        return (0.0, 20.0)
    if modulation == "16qam":
        return (5.0, 25.0)
    raise ValueError(f"Unsupported modulation '{modulation}'.")
