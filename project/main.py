import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

from evaluate import EvalConfig, evaluate_ber
from train import TrainConfig, train_model
from utils import ensure_dir, plot_ber_curves, plot_training_loss, set_seed


def parse_antenna_configs(raw: str) -> List[Tuple[int, int]]:
    """Parse comma-separated antenna configurations from the CLI.

    Args:
        raw: String such as ``"4x4,8x8"``.

    Returns:
        A list of ``(n_rx, n_tx)`` tuples.
    """
    # The CLI accepts comma-separated configs such as "4x4,8x8".
    configs: List[Tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        n_rx_str, n_tx_str = item.split("x")
        configs.append((int(n_rx_str), int(n_tx_str)))
    if not configs:
        raise ValueError("At least one antenna configuration is required.")
    return configs


def parse_args():
    """Build and parse command-line arguments for the experiment driver."""
    parser = argparse.ArgumentParser(
        description="Paper-aligned OAMP vs OAMP-Net simulation"
    )
    parser.add_argument("--configs", type=str, default="4x4,8x8")
    parser.add_argument(
        "--modulation",
        type=str,
        default="qpsk",
        choices=["bpsk", "qpsk", "16qam"],
    )
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_eval_batches", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--channel_model",
        type=str,
        default="rayleigh",
        choices=["rayleigh", "correlated", "both"],
    )
    parser.add_argument("--rho_tx", type=float, default=0.0)
    parser.add_argument("--rho_rx", type=float, default=0.0)
    parser.add_argument("--compare_tista", action="store_true")
    return parser.parse_args()


def iter_channel_models(channel_model: str) -> Iterable[str]:
    """Expand the requested channel-model option into one or more runs.

    Args:
        channel_model: CLI option value.

    Yields:
        One channel-model name at a time.
    """
    # "both" expands into two independent experiment runs.
    if channel_model == "both":
        yield "rayleigh"
        yield "correlated"
        return
    yield channel_model


def build_train_config(
    n_rx: int,
    n_tx: int,
    modulation: str,
    channel_model: str,
    rho_tx: float,
    rho_rx: float,
    num_iters: int,
    batch_size: int,
    epochs: int,
    save_path: str,
) -> TrainConfig:
    """Construct a training configuration for one experiment.

    Args:
        n_rx: Number of receive antennas.
        n_tx: Number of transmit antennas.
        modulation: Modulation type.
        channel_model: Channel-model name.
        rho_tx: Transmit correlation coefficient.
        rho_rx: Receive correlation coefficient.
        num_iters: Number of unfolded layers or OAMP iterations.
        batch_size: Mini-batch size.
        epochs: Number of training epochs.
        save_path: Output checkpoint path.

    Returns:
        A populated :class:`TrainConfig` instance.
    """
    # Centralize TrainConfig creation so the main model and optional comparison
    # model use the same experiment settings.
    return TrainConfig(
        n_rx=n_rx,
        n_tx=n_tx,
        modulation=modulation,
        channel_model=channel_model,
        rho_tx=rho_tx,
        rho_rx=rho_rx,
        num_iters=num_iters,
        batch_size=batch_size,
        epochs=epochs,
        save_path=save_path,
        print_every=max(1, epochs // 20),
    )


def run_single_experiment(
    out_dir: str,
    n_rx: int,
    n_tx: int,
    modulation: str,
    channel_model: str,
    rho_tx: float,
    rho_rx: float,
    num_iters: int,
    epochs: int,
    batch_size: int,
    num_eval_batches: int,
    compare_tista: bool,
) -> None:
    """Run one full train-evaluate-save experiment.

    Args:
        out_dir: Output directory for checkpoints, plots, and JSON.
        n_rx: Number of receive antennas.
        n_tx: Number of transmit antennas.
        modulation: Modulation type.
        channel_model: Channel model name.
        rho_tx: Transmit correlation coefficient.
        rho_rx: Receive correlation coefficient.
        num_iters: Number of unfolded layers or OAMP iterations.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        num_eval_batches: Maximum number of evaluation batches per SNR point.
        compare_tista: Whether to train/evaluate the comparison model too.
    """
    ensure_dir(out_dir)
    tag = f"{n_rx}x{n_tx}_{modulation}_{channel_model}"

    # 1) Train OAMP-Net
    ckpt_path = str(Path(out_dir) / f"oamp_net_{tag}.pt")
    train_cfg = build_train_config(
        n_rx=n_rx,
        n_tx=n_tx,
        modulation=modulation,
        channel_model=channel_model,
        rho_tx=rho_tx,
        rho_rx=rho_rx,
        num_iters=num_iters,
        batch_size=batch_size,
        epochs=epochs,
        save_path=ckpt_path,
    )

    model, losses = train_model(train_cfg, model_type="oampnet")
    loss_plot = str(Path(out_dir) / f"loss_{tag}.png")
    plot_training_loss(losses, loss_plot)

    tista_model = None
    if compare_tista:
        # 2) Optionally train the paper's LMMSE-TISTA-style comparison model
        tista_ckpt_path = str(Path(out_dir) / f"lmmse_tista_{tag}.pt")
        tista_cfg = build_train_config(
            n_rx=n_rx,
            n_tx=n_tx,
            modulation=modulation,
            channel_model=channel_model,
            rho_tx=rho_tx,
            rho_rx=rho_rx,
            num_iters=num_iters,
            batch_size=batch_size,
            epochs=epochs,
            save_path=tista_ckpt_path,
        )
        tista_model, tista_losses = train_model(tista_cfg, model_type="lmmse_tista")
        tista_loss_plot = str(Path(out_dir) / f"loss_lmmse_tista_{tag}.png")
        plot_training_loss(tista_losses, tista_loss_plot)

    # 3) Evaluate BER over the requested SNR sweep
    eval_cfg = EvalConfig(
        n_rx=n_rx,
        n_tx=n_tx,
        modulation=modulation,
        channel_model=channel_model,
        rho_tx=rho_tx,
        rho_rx=rho_rx,
        num_iters=num_iters,
        snr_db_list=list(range(0, 31, 2)),
        num_batches=num_eval_batches,
        batch_size=batch_size,
    )
    results = evaluate_ber(model, eval_cfg, model_tista=tista_model)

    # 4) Save figures and raw numeric outputs for later analysis/reporting
    ber_plot = str(Path(out_dir) / f"ber_{tag}.png")
    ber_curves = {"OAMP": results["oamp"]}
    if "lmmse_tista" in results:
        ber_curves["LMMSE-TISTA"] = results["lmmse_tista"]
    ber_curves["OAMP-Net"] = results["oamp_net"]
    plot_ber_curves(
        results["snr_db"],
        ber_curves,
        ber_plot,
        title=f"BER vs SNR ({n_rx}x{n_tx}, {modulation.upper()}, {channel_model})",
    )

    result_payload = {
        "config": {
            "n_rx": n_rx,
            "n_tx": n_tx,
            "modulation": modulation,
            "channel_model": channel_model,
            "rho_tx": rho_tx,
            "rho_rx": rho_rx,
            "num_iters": num_iters,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_eval_batches": num_eval_batches,
        },
        "results": results,
    }
    result_path = str(Path(out_dir) / f"ber_{tag}.json")
    with open(result_path, "w", encoding="ascii") as f:
        json.dump(result_payload, f, indent=2)

    print("\nRun complete.")
    print(f"- Checkpoint: {ckpt_path}")
    print(f"- Loss plot:  {loss_plot}")
    print(f"- BER plot:   {ber_plot}")
    print(f"- BER data:   {result_path}")


def main():
    """Entry point for command-line experiments."""
    args = parse_args()
    set_seed(args.seed)

    # Run one full train-evaluate-save pipeline for each requested channel model
    # and antenna configuration.
    configs = parse_antenna_configs(args.configs)
    for channel_model in iter_channel_models(args.channel_model):
        for n_rx, n_tx in configs:
            print(
                f"\n===== Running {n_rx}x{n_tx} | {args.modulation.upper()} | "
                f"{channel_model} ====="
            )
            run_single_experiment(
                out_dir=args.out_dir,
                n_rx=n_rx,
                n_tx=n_tx,
                modulation=args.modulation,
                channel_model=channel_model,
                rho_tx=args.rho_tx,
                rho_rx=args.rho_rx,
                num_iters=args.num_iters,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_eval_batches=args.num_eval_batches,
                compare_tista=args.compare_tista,
            )


if __name__ == "__main__":
    main()
