from typing import Union

import numpy as np

import torch
import torchmetrics


class TokenPerplexity(torchmetrics.Metric):
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("nll_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tok_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self, loss_sum: torch.Tensor, n_tokens: Union[int, torch.Tensor]
    ) -> None:
        loss_sum = loss_sum.detach()

        # Convert int -> tensor on correct device/dtype
        if not torch.is_tensor(n_tokens):
            n_tokens = torch.tensor(
                n_tokens,
                device=loss_sum.device,
                dtype=loss_sum.dtype,
            )
        else:
            n_tokens = n_tokens.detach().to(loss_sum.device, loss_sum.dtype)

        self.nll_sum += loss_sum
        self.tok_sum += n_tokens

    def compute(self) -> torch.Tensor:
        mean_nll = self.nll_sum / self.tok_sum.clamp_min(1.0)
        return torch.exp(mean_nll.clamp(max=50.0))


class ApproxBPS(torchmetrics.Metric):
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("nll_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_tokens", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("seconds_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        loss_sum: torch.Tensor,
        num_seconds: Union[float, torch.Tensor],
        num_tokens: Union[int, torch.Tensor],
    ) -> None:
        loss_sum = loss_sum.detach()

        # Convert int -> tensor on correct device/dtype
        if not torch.is_tensor(num_seconds):
            num_seconds = torch.tensor(
                num_seconds,
                device=loss_sum.device,
                dtype=loss_sum.dtype,
            )
        else:
            num_seconds = num_seconds.detach().to(loss_sum.device, loss_sum.dtype)

        if not torch.is_tensor(num_tokens):
            num_tokens = torch.tensor(
                num_tokens,
                device=loss_sum.device,
                dtype=loss_sum.dtype,
            )
        else:
            num_tokens = num_tokens.detach().to(loss_sum.device, loss_sum.dtype)

        self.nll_sum += loss_sum
        self.num_tokens += num_tokens
        self.seconds_sum += num_seconds

    def compute(self) -> torch.Tensor:
        mean_nll = self.nll_sum / self.num_tokens.clamp_min(1.0)
        approx_bps = mean_nll * np.log2(np.e) * (self.num_tokens / self.seconds_sum)
        return approx_bps
