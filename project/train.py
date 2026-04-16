from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from data import default_train_snr_range, generate_mimo_batch
from oamp_net import LMMSETISTA, OAMPNet
from utils import get_device, save_checkpoint


# Training settings are grouped into a dataclass so experiments are easy to
# reproduce and serialize.
@dataclass
class TrainConfig:
    """Configuration container for model training."""
    n_rx: int = 8
    n_tx: int = 8
    modulation: str = "qpsk"
    channel_model: str = "rayleigh"
    rho_tx: float = 0.0
    rho_rx: float = 0.0
    num_iters: int = 10
    snr_train: Union[float, Tuple[float, float]] = (0.0, 20.0)
    batch_size: int = 256
    epochs: int = 2000
    lr: float = 1e-3
    train_samples_per_epoch: int = 5000
    val_samples_per_epoch: int = 1000
    save_path: str = "artifacts/oamp_net.pt"
    print_every: int = 100

    def __post_init__(self):
        # If the caller does not specify a training SNR range, pick a sensible
        # default based on modulation difficulty.
        if self.snr_train is None:
            self.snr_train = default_train_snr_range(self.modulation)


def train_oamp_net(cfg: TrainConfig):
    """Train the default OAMP-Net model.

    Args:
        cfg: Training configuration.

    Returns:
        A tuple ``(model, losses)`` where ``losses`` contains train/validation
        loss histories.
    """
    return train_model(cfg, model_type="oampnet")


def train_model(cfg: TrainConfig, model_type: str = "oampnet"):
    """Train one of the supported unfolded detectors.

    Args:
        cfg: Training configuration.
        model_type: Either ``"oampnet"`` or ``"lmmse_tista"``.

    Returns:
        A tuple ``(model, losses)`` where ``losses`` is a dictionary with
        ``"train"`` and ``"val"`` histories.
    """
    device = get_device()
    # The training loop can build either the main OAMP-Net or the optional
    # LMMSE-TISTA comparison model.
    if model_type == "oampnet":
        model = OAMPNet(num_iters=cfg.num_iters, modulation=cfg.modulation).to(device)
    elif model_type == "lmmse_tista":
        model = LMMSETISTA(num_iters=cfg.num_iters, modulation=cfg.modulation).to(device)
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    losses: List[float] = []
    val_losses: List[float] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Each epoch uses freshly generated synthetic samples instead of a fixed
        # dataset stored on disk.
        num_train_batches = max(1, cfg.train_samples_per_epoch // cfg.batch_size)
        for _ in range(num_train_batches):
            H, x, y, sigma2 = generate_mimo_batch(
                batch_size=cfg.batch_size,
                n_rx=cfg.n_rx,
                n_tx=cfg.n_tx,
                snr_db=cfg.snr_train,
                modulation=cfg.modulation,
                channel_model=cfg.channel_model,
                rho_tx=cfg.rho_tx,
                rho_rx=cfg.rho_rx,
                device=device,
            )

            pred = model(H, y, sigma2)
            loss = criterion(pred, x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        # Store average epoch loss so plots are easy to interpret later.
        train_loss = epoch_loss / num_train_batches
        losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0
            # Validation also uses synthetic batches, but without gradient
            # updates, to track whether the learned parameters are improving.
            num_val_batches = max(1, cfg.val_samples_per_epoch // cfg.batch_size)
            for _ in range(num_val_batches):
                H, x, y, sigma2 = generate_mimo_batch(
                    batch_size=cfg.batch_size,
                    n_rx=cfg.n_rx,
                    n_tx=cfg.n_tx,
                    snr_db=cfg.snr_train,
                    modulation=cfg.modulation,
                    channel_model=cfg.channel_model,
                    rho_tx=cfg.rho_tx,
                    rho_rx=cfg.rho_rx,
                    device=device,
                )
                pred = model(H, y, sigma2)
                epoch_val_loss += float(criterion(pred, x).item())
            val_loss = epoch_val_loss / num_val_batches
            val_losses.append(val_loss)

        if epoch % cfg.print_every == 0 or epoch == 1:
            print(
                f"[Train] Epoch {epoch:5d}/{cfg.epochs} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

    # Save both the learned weights and the optimizer state so training can be
    # resumed later if needed.
    save_checkpoint(model, optimizer, cfg.epochs, cfg.save_path)
    return model, {"train": losses, "val": val_losses}
