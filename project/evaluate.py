from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from data import generate_mimo_batch
from oamp import oamp_detect
from oamp_net import OAMPNet
from utils import ber_from_logits, get_device


# Evaluation settings are kept separate from training settings because test-time
# SNR sweeps and stopping rules differ from training.
@dataclass
class EvalConfig:
    """Configuration container for BER evaluation."""
    n_rx: int = 8
    n_tx: int = 8
    modulation: str = "qpsk"
    channel_model: str = "rayleigh"
    rho_tx: float = 0.0
    rho_rx: float = 0.0
    num_iters: int = 10
    snr_db_list: Optional[List[int]] = None
    num_batches: int = 200
    batch_size: int = 256
    min_error_bits: int = 1000

    def __post_init__(self):
        if self.snr_db_list is None:
            self.snr_db_list = list(range(0, 31, 2))


@torch.no_grad()
def evaluate_ber(model: OAMPNet, cfg: EvalConfig, model_tista=None) -> Dict[str, List[float]]:
    """Evaluate BER over an SNR sweep.

    Args:
        model: Trained OAMP-Net model.
        cfg: Evaluation configuration.
        model_tista: Optional trained comparison model.

    Returns:
        A dictionary containing the SNR grid and BER curves for the evaluated
        detectors.
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    # The optional comparison model is evaluated on the same batches so all
    # methods see identical test data.
    if model_tista is not None:
        model_tista = model_tista.to(device)
        model_tista.eval()

    ber_oamp: List[float] = []
    ber_oampnet: List[float] = []
    ber_tista: List[float] = []

    for snr_db in cfg.snr_db_list:
        # We accumulate raw bit errors so the final BER estimate uses the full
        # tested bit count across multiple random batches.
        err_oamp = 0.0
        err_net = 0.0
        err_tista = 0.0
        total_bits = 0
        batches_used = 0

        while True:
            H, x, y, sigma2 = generate_mimo_batch(
                batch_size=cfg.batch_size,
                n_rx=cfg.n_rx,
                n_tx=cfg.n_tx,
                snr_db=float(snr_db),
                modulation=cfg.modulation,
                channel_model=cfg.channel_model,
                rho_tx=cfg.rho_tx,
                rho_rx=cfg.rho_rx,
                device=device,
            )

            x_oamp = oamp_detect(
                H=H,
                y=y,
                sigma2=sigma2,
                num_iters=cfg.num_iters,
                modulation=cfg.modulation,
            )
            x_net = model(H, y, sigma2)
            x_tista = model_tista(H, y, sigma2) if model_tista is not None else None

            batch_ber_oamp, batch_errors_oamp, batch_bits = ber_from_logits(
                x_oamp, x, return_counts=True
            )
            batch_ber_net, batch_errors_net, _batch_bits = ber_from_logits(
                x_net, x, return_counts=True
            )
            if x_tista is not None:
                _batch_ber_tista, batch_errors_tista, _ = ber_from_logits(
                    x_tista, x, return_counts=True
                )
                err_tista += batch_errors_tista
            err_oamp += batch_errors_oamp
            err_net += batch_errors_net
            total_bits += batch_bits
            batches_used += 1

            # Stop once we have seen enough batches or enough total errors. The
            # second condition is especially useful at high SNR.
            enough_batches = batches_used >= cfg.num_batches
            enough_errors = min(err_oamp, err_net) >= cfg.min_error_bits
            if enough_batches or enough_errors:
                break

        # Convert accumulated error counts into BER values for the current SNR.
        ber_oamp.append(err_oamp / total_bits)
        ber_oampnet.append(err_net / total_bits)
        if model_tista is not None:
            ber_tista.append(err_tista / total_bits)
        print(
            f"[Eval] {cfg.channel_model:10s} | SNR={snr_db:2d} dB | "
            f"OAMP BER={ber_oamp[-1]:.4e} | "
            f"{'LMMSE-TISTA BER=' + format(ber_tista[-1], '.4e') + ' | ' if model_tista is not None else ''}"
            f"OAMP-Net BER={ber_oampnet[-1]:.4e}"
        )

    results = {
        "snr_db": cfg.snr_db_list,
        "oamp": ber_oamp,
        "oamp_net": ber_oampnet,
    }
    if model_tista is not None:
        results["lmmse_tista"] = ber_tista
    return results
