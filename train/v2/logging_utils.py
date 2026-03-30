import io
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
)
import wandb

from anticipation.v2.sample import generate_ar_simple


def log_bytes_at_step(run: wandb.sdk.wandb_run.Run, data: bytes, step: int) -> None:
    if wandb.run is None:
        # raise RuntimeError("No active wandb run. Make sure WandbLogger has initialized (e.g. after on_fit_start).")
        return

    with tempfile.TemporaryDirectory() as td:
        # write this file
        td_path = Path(td)
        f_name = f"midi_step_{step:08d}.mid"
        temp_file = td_path / f_name
        temp_file.write_bytes(data)

        # wandb needs a real file, not buffered io for some reason...
        # below code is really finicky
        art = wandb.Artifact(name=f"inference-midi", type="custom")
        art.add_file(temp_file, name=f_name)
        wandb.log_artifact(art, aliases=[f"inference-midi-step-{step}", "latest"])
        wandb.log(
            {
                "payload/artifact_name": art.name,
            },
            step=step,
        )


@dataclass
class SampleConfig:
    start_after_step: int = 5_000
    num_events_to_generate: int = 100
    top_p: float = 1.0


class GenerateSamplesOnValEnd(pl.Callback):
    def __init__(
        self,
        cfg: SampleConfig,
        *,
        prompt_fn: Optional[Callable[[pl.LightningModule], Any]] = None,
    ):
        self.cfg = cfg
        self.prompt_fn = prompt_fn

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Only once globally in DDP
        if not trainer.is_global_zero:
            return
        if trainer.sanity_checking:
            return

        step = trainer.global_step
        if step < self.cfg.start_after_step:
            # too early
            return

        # https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#on-validation-epoch-end
        # hoping that this callback is called only ONCE per validation, instead of once per
        # validation batch.
        prompts = self.prompt_fn(pl_module) if self.prompt_fn else None
        pl_module.eval()
        with torch.no_grad():
            sample = self._generate(pl_module, prompts)
        self._log_samples(trainer, sample, step)

    def _generate(self, pl_module: pl.LightningModule, prompts) -> bytes:
        model = getattr(pl_module, "model", pl_module)
        settings = getattr(pl_module, "anticipation_settings", None)
        assert settings

        # use our simple autoregressive generation function to
        # get a small midi file
        midi_file = generate_ar_simple(
            model,
            settings,
            num_events_to_generate=self.cfg.num_events_to_generate,
            top_p=self.cfg.top_p,
        )
        bytes_io = io.BytesIO()
        midi_file.save(file=bytes_io)
        return bytes_io.getvalue()

    def _log_samples(self, trainer: pl.Trainer, sample, step: int) -> None:
        logger = trainer.logger
        if logger is None:
            return

        # if not using wandb, this is a noop
        log_bytes_at_step(logger.experiment, sample, step)


class MaxStepProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._persistent_bar = None

    def init_train_tqdm(self):
        if self._persistent_bar is None:
            bar = super().init_train_tqdm()
            bar.set_description("Training Progress")
            self._persistent_bar = bar

        return self._persistent_bar

    def on_train_epoch_start(self, trainer, pl_module):
        if self._persistent_bar is not None:
            self._persistent_bar.set_description("Training Progress")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current = trainer.global_step
        total = trainer.max_steps
        self.train_progress_bar.n = current
        self.train_progress_bar.total = total
        self._persistent_bar.refresh()
