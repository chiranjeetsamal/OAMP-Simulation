import os
import random
from pathlib import Path
from typing import Dict, Optional

# Force a non-interactive Matplotlib backend so experiments can run on remote
# shells and headless servers without display errors.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random generators.

    Args:
        seed: Random seed value.
    """
    # Seed Python, NumPy, and PyTorch to make runs more reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the preferred execution device."""
    # Prefer GPU when available, otherwise fall back to CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ber_from_logits(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    return_counts: bool = False,
):
    """Compute BER from real-valued detector outputs.

    Args:
        x_pred: Predicted real-valued symbols.
        x_true: Ground-truth real-valued symbols.
        return_counts: When ``True``, also return raw error and bit counts.

    Returns:
        Either the BER as a float, or ``(ber, bit_errors, total_bits)``.
    """
    # Convert soft outputs into symbol decisions by taking the sign of each
    # real-valued component. This matches BPSK/QPSK bit decisions in the
    # equivalent real-valued system.
    xh = torch.sign(x_pred)
    xh[xh == 0] = 1
    xt = torch.sign(x_true)
    xt[xt == 0] = 1
    error_mask = (xh != xt)
    bit_errors = int(error_mask.sum().item())
    total_bits = int(error_mask.numel())
    ber = bit_errors / max(1, total_bits)
    if return_counts:
        return ber, bit_errors, total_bits
    return ber


def ensure_dir(path: str) -> None:
    """Create a directory path if it does not already exist.

    Args:
        path: Directory path to create.
    """
    # Create parent directories before saving figures, checkpoints, or JSON.
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: str,
) -> None:
    """Save a model checkpoint to disk.

    Args:
        model: Model whose weights should be saved.
        optimizer: Optional optimizer state to save alongside the model.
        epoch: Epoch number to record in the checkpoint.
        path: Destination file path.
    """
    # Save everything needed to resume training or evaluate the trained model.
    ensure_dir(str(Path(path).parent))
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict:
    """Load a model checkpoint from disk.

    Args:
        model: Model instance that will receive the checkpoint weights.
        path: Checkpoint file path.
        optimizer: Optional optimizer that should be restored.
        map_location: Device mapping passed to ``torch.load``.

    Returns:
        The raw checkpoint dictionary.
    """
    # Optionally restore optimizer state if the caller wants to continue
    # training from the checkpoint.
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def plot_training_loss(losses, out_path: str) -> None:
    """Save a training-loss figure to disk.

    Args:
        losses: Either a single loss sequence or a dict with ``train`` and
            ``val`` loss histories.
        out_path: Destination figure path.
    """
    # Accept either a single loss list or a dict with separate train/val curves.
    ensure_dir(str(Path(out_path).parent))
    plt.figure(figsize=(7, 5))
    if isinstance(losses, dict):
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        if train_loss:
            plt.plot(train_loss, linewidth=2.0, color="#0f5c5c", label="Train")
        if val_loss:
            plt.plot(val_loss, linewidth=2.0, color="#b91c1c", label="Validation")
        if train_loss or val_loss:
            plt.legend()
    else:
        plt.plot(losses, linewidth=2.0, color="#0f5c5c")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("OAMP-Net Training Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ber_curves(snr_db_list, ber_dict, out_path: str, title: str) -> None:
    """Save one or more BER curves as a figure.

    Args:
        snr_db_list: SNR grid in dB.
        ber_dict: Mapping from curve label to BER values.
        out_path: Destination figure path.
        title: Figure title.
    """
    # Plot one or more BER curves on a log-scale y-axis, which is standard for
    # digital communication performance figures.
    ensure_dir(str(Path(out_path).parent))
    plt.figure(figsize=(7.5, 5.5))
    colors = ["#1d4ed8", "#b91c1c", "#047857", "#7c3aed"]
    for idx, (label, ber_vals) in enumerate(ber_dict.items()):
        plt.semilogy(
            snr_db_list,
            ber_vals,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=label,
            color=colors[idx % len(colors)],
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
