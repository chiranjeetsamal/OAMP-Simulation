import torch
import torch.nn as nn

from oamp import estimate_symbol_variance, oamp_update


# This file contains the learned unfolded detectors. Each network keeps the
# same update structure as OAMP and only learns a very small number of scalar
# parameters per layer.
class OAMPNet(nn.Module):
    """
    Unfolded OAMP-Net with 2T trainable scalar parameters.

    The paper parameterizes each layer with:
      - gamma_t: linear update scaling
      - theta_t: equivalent noise scaling in tau_t^2

    These correspond to the requested per-iteration step-size and denoising-scale
    parameters while preserving the OAMP structure.
    """

    def __init__(self, num_iters: int = 10, modulation: str = "bpsk"):
        """Initialize the unfolded OAMP-Net model.

        Args:
            num_iters: Number of unfolded OAMP layers.
            modulation: Modulation type used by the scalar denoiser.
        """
        super().__init__()
        self.num_iters = num_iters
        self.modulation = modulation

        # One learned scalar for the linear mean update and one for the variance
        # update in each layer.
        self.gamma = nn.Parameter(torch.ones(num_iters))
        self.theta = nn.Parameter(torch.ones(num_iters))

    @property
    def alpha(self) -> nn.Parameter:
        return self.gamma

    @property
    def beta(self) -> nn.Parameter:
        return self.theta

    def forward(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """Run the unfolded OAMP-Net forward pass.

        Args:
            H: Channel matrices with shape ``[B, M, N]``.
            y: Received vectors ``[B, M, 1]``.
            sigma2: Noise variance ``[B, 1, 1]``.

        Returns:
            Final detected symbols with shape ``[B, N, 1]``.
        """
        # Layer 1 starts from the same zero initialization as the original OAMP
        # algorithm.
        batch_size, _m, n = H.shape
        x = torch.zeros(batch_size, n, 1, device=H.device, dtype=H.dtype)
        v2 = estimate_symbol_variance(H, y, x, sigma2)

        for t in range(self.num_iters):
            # Broadcast each layer's learned scalars to the batch.
            gamma_t = self.gamma[t].view(1, 1, 1)
            theta_t = self.theta[t].view(1, 1, 1)
            x, v2, _stats = oamp_update(
                H=H,
                y=y,
                x=x,
                v2=v2,
                sigma2=sigma2,
                modulation=self.modulation,
                gamma=gamma_t,
                theta=theta_t,
            )

        return x


class LMMSETISTA(nn.Module):
    """
    LMMSE-TISTA style variant used in the paper's comparisons.

    This ties gamma_t and theta_t into a single learned scalar per layer while
    still using the LMMSE linear estimator.
    """

    def __init__(self, num_iters: int = 10, modulation: str = "bpsk"):
        """Initialize the LMMSE-TISTA comparison model.

        Args:
            num_iters: Number of unfolded layers.
            modulation: Modulation type used by the scalar denoiser.
        """
        super().__init__()
        self.num_iters = num_iters
        self.modulation = modulation

        # TISTA-style tied step size: one scalar per layer shared across the
        # mean and variance updates.
        self.step = nn.Parameter(torch.ones(num_iters))

    def forward(
        self,
        H: torch.Tensor,
        y: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """Run the LMMSE-TISTA-style forward pass.

        Args:
            H: Channel matrices with shape ``[B, M, N]``.
            y: Received vectors ``[B, M, 1]``.
            sigma2: Noise variance ``[B, 1, 1]``.

        Returns:
            Final detected symbols with shape ``[B, N, 1]``.
        """
        batch_size, _m, n = H.shape
        x = torch.zeros(batch_size, n, 1, device=H.device, dtype=H.dtype)
        v2 = estimate_symbol_variance(H, y, x, sigma2)

        for t in range(self.num_iters):
            # This comparison model ties gamma_t and theta_t together.
            step_t = self.step[t].view(1, 1, 1)
            x, v2, _stats = oamp_update(
                H=H,
                y=y,
                x=x,
                v2=v2,
                sigma2=sigma2,
                modulation=self.modulation,
                gamma=step_t,
                theta=step_t,
            )
        return x
