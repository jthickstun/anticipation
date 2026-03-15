import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# keep this!!!
torch.set_float32_matmul_precision("high")

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from transformers import PretrainedConfig, GPT2LMHeadModel, GPT2Config

from anticipation.v2.config import AnticipationV2Settings
from train.v2.custom_metrics import TokenPerplexity, ApproxBPS
from train.v2.dataset_utils import PreTokenizedDataset
from train.v2.logging_utils import (
    GenerateSamplesOnValEnd,
    SampleConfig,
    MaxStepProgressBar,
)


class GPT2LightningModule(pl.LightningModule):
    def __init__(
        self,
        data_dir: Path,
        settings: AnticipationV2Settings,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        pretrained_checkpoint: str = None,
        config: PretrainedConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir

        if pretrained_checkpoint:
            self.model = GPT2LMHeadModel.from_pretrained(
                pretrained_checkpoint, config=config
            )
            rank_zero_info(f"Loaded pre-trained model from {pretrained_checkpoint}")
        else:
            self.model = GPT2LMHeadModel(config)

        self.model.gradient_checkpointing_enable()
        self.model.config.bos_token_id = self.model.config.eos_token_id = 0
        self.anticipation_settings = settings

        # --- validation split metrics ----
        self.ppl = TokenPerplexity()

        # all triples
        self.event_ppl = TokenPerplexity()

        # parts of the triple
        self.onset_ppl = TokenPerplexity()
        self.onset_ppl_no_ticks = TokenPerplexity()
        self.dur_ppl = TokenPerplexity()
        self.dur_ppl_no_ticks = TokenPerplexity()
        self.note_instr_ppl = TokenPerplexity()
        self.note_instr_ppl_no_ticks = TokenPerplexity()

        # ticks only
        self.tick_ppl = TokenPerplexity()

        # approximate the bps using the number of ticks to roughly estimate total seconds
        self.approx_bps = ApproxBPS()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        labels = batch.pop("labels")
        outputs = self(**batch)

        # keep this upcast!
        # https://x.com/jwthickstun/status/1737134520141246938
        logits = outputs.logits.float()  # upcast logits and compute loss in fp32
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train_loss", loss.detach(), prog_bar=True, logger=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        labels = batch.pop("labels")
        outputs = self(**batch)

        # keep this upcast!
        # https://x.com/jwthickstun/status/1737134520141246938
        logits = outputs.logits.float()  # upcast logits and compute loss in fp32
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log(
            "val_loss",
            loss.detach(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # ----- new metrics -----
        v = self.anticipation_settings.vocab
        tokens = batch["input_ids"]

        shift_logits = logits[:, :-1, :]
        targets = tokens[:, 1:]
        per_tok_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)  # [bs, L-1]

        # we don't really care about loss on the controls since we put those in ourselves
        controls = (
            (targets == v.SEPARATOR)
            | (targets == v.ANTICIPATE)
            | (targets == v.AUTOREGRESS)
        )

        # exclude the controls form overall perplexity
        ppl_overall_mask: torch.Tensor = ~controls  # type: ignore
        total_loss_sum = per_tok_ce[ppl_overall_mask].sum()
        total_tokens = ppl_overall_mask.sum()
        self.ppl.update(loss_sum=total_loss_sum, n_tokens=total_tokens)
        self.log("ppl", self.ppl, on_epoch=True, prog_bar=True, sync_dist=True)

        ticks = targets == v.TICK
        tick_mask: torch.Tensor = ticks  # type: ignore
        num_ticks = tick_mask.sum()

        num_seconds_approx = (
            num_ticks * self.anticipation_settings.tick_token_frequency_in_midi_ticks
        ) / self.anticipation_settings.time_resolution
        self.approx_bps.update(
            loss_sum=total_loss_sum,
            num_seconds=num_seconds_approx,
            num_tokens=total_tokens,
        )
        self.log(
            "approx_bps",
            self.approx_bps,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # events are everything that isn't a control and isn't a tick
        events: torch.Tensor = (~controls) & (~ticks)  # type: ignore

        # Event index within each sequence: 0,1,2,... for event positions
        # (non-event positions will have junk values, but we always AND with `events`)
        event_idx = torch.cumsum(events.to(torch.int64), dim=1) - 1

        # How many event tokens per sequence, and how many to keep (truncate tail to multiple of 3)
        n_event_tokens = events.sum(dim=1).to(torch.int64)
        n_keep_tokens = (n_event_tokens // 3) * 3
        n_events = n_keep_tokens // 3
        num_events_total = n_events.sum()

        # drops incomplete / truncated last triple per seq if necessary
        keep_events = events & (event_idx < n_keep_tokens.unsqueeze(1))

        # isolate each part of a triple (time aka onset, duration, note x instrument)
        onsets = keep_events & ((event_idx % 3) == 0)
        durs = keep_events & ((event_idx % 3) == 1)
        note_instrs = keep_events & ((event_idx % 3) == 2)

        onset_loss_sum = per_tok_ce[onsets].sum()
        dur_loss_sum = per_tok_ce[durs].sum()
        note_instr_loss_sum = per_tok_ce[note_instrs].sum()
        tick_loss_sum = per_tok_ce[tick_mask].sum()

        # does not include the tick
        # formulation here is approximate, following the v1 eval script
        self.event_ppl.update(loss_sum=(3 * per_tok_ce[keep_events].mean()), n_tokens=1)
        self.log(
            "event_ppl",
            self.event_ppl,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # each part of the triple (time, duration, note x instrument)
        # both with and without the CE contributed by the tick
        self.onset_ppl.update(
            loss_sum=(onset_loss_sum + tick_loss_sum), n_tokens=num_events_total
        )
        self.log(
            "onset_ppl",
            self.onset_ppl,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.onset_ppl_no_ticks.update(
            loss_sum=onset_loss_sum, n_tokens=num_events_total
        )
        self.log(
            "onset_ppl_no_ticks",
            self.onset_ppl_no_ticks,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.dur_ppl.update(
            loss_sum=(dur_loss_sum + tick_loss_sum), n_tokens=num_events_total
        )
        self.log("dur_ppl", self.dur_ppl, on_epoch=True, prog_bar=True, sync_dist=True)
        self.dur_ppl_no_ticks.update(loss_sum=dur_loss_sum, n_tokens=num_events_total)
        self.log(
            "dur_ppl_no_ticks",
            self.dur_ppl_no_ticks,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.note_instr_ppl.update(
            loss_sum=(note_instr_loss_sum + tick_loss_sum),
            n_tokens=num_events_total,
        )
        self.log(
            "note_instr_ppl",
            self.note_instr_ppl,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.note_instr_ppl_no_ticks.update(
            loss_sum=note_instr_loss_sum, n_tokens=num_events_total
        )
        self.log(
            "note_instr_ppl_no_ticks",
            self.note_instr_ppl_no_ticks,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.tick_ppl.update(loss_sum=tick_loss_sum, n_tokens=num_ticks)
        self.log(
            "tick_ppl", self.tick_ppl, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
        )

        train_dataloader = self.train_dataloader()
        if hasattr(train_dataloader, "dataset"):
            dataset_size = len(train_dataloader.dataset)
            pct_start = min(0.3, self.hparams.warmup_steps / self.trainer.max_steps)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.max_steps,
                pct_start=pct_start,
                div_factor=100.0,
                final_div_factor=0.1,
                anneal_strategy="cos",
                three_phase=False,
                cycle_momentum=False,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler_config]

        return optimizer

    def train_dataloader(self) -> DataLoader:
        num_devices = max(1, self.trainer.num_devices)
        per_device_batch_size = self.hparams.train_batch_size // num_devices
        dataset = PreTokenizedDataset(self.data_dir / "train.npy")
        return DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        num_devices = max(1, self.trainer.num_devices)
        per_device_batch_size = self.hparams.train_batch_size // num_devices
        dataset = PreTokenizedDataset(self.data_dir / "valid.npy")
        return DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )


class HuggingFaceCheckpoint(ModelCheckpoint):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        if (trainer.global_step % self._every_n_train_steps) != 0:
            return

        step_dir = os.path.join(self.dirpath, f"step-{trainer.global_step}")

        raw_model = pl_module.model if hasattr(pl_module, "model") else pl_module
        raw_model.save_pretrained(
            step_dir, safe_serialization=True, max_shard_size="2GB"
        )


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed)
    tokenized_dataset_path = Path(args.data_dir)
    assert tokenized_dataset_path.exists()
    assert tokenized_dataset_path.is_dir()
    settings_file = next(tokenized_dataset_path.glob("settings_*.json"), None)
    if not settings_file:
        raise RuntimeError("Unable to find settings")
    settings = AnticipationV2Settings.load_from_disk(settings_file)
    model_config = GPT2Config(
        vocab_size=settings.vocab.total_tokens(),
        n_positions=settings.context_size,
        n_embd=args.hidden_dim,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        embd_pdrop=args.embed_pdrop,
        resid_pdrop=args.resid_pdrop,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=True,
        use_cache=False,
    )
    model = GPT2LightningModule(
        data_dir=tokenized_dataset_path,
        settings=settings,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        pretrained_checkpoint=args.checkpoint_path,
        config=model_config,
    )

    checkpoint_callback = HuggingFaceCheckpoint(
        config=model.model.config,
        dirpath=args.output_dir,
        filename="{step}",
        save_top_k=0,
        monitor=None,
        save_last=False,
        every_n_train_steps=args.steps_per_checkpoint,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"

    logger_dict = {}
    if args.use_wandb:
        logger_dict["logger"] = WandbLogger(
            project="gpt-anticipation-2.0",
            name=f"run-{int(time.time())}",
            save_dir=args.output_dir,
            config=vars(args),
        )

    ddp_params = {}
    if not args.no_ddp:
        ddp_params["strategy"] = DDPStrategy(
            find_unused_parameters=False, static_graph=False
        )

    trainer = pl.Trainer(
        max_steps=args.num_train_steps,
        # always use gpu and then thrown an error if it's unavailable - that's preferrable
        # to defaulting to cpu imo
        accelerator="gpu",
        devices=args.gpus_per_node,
        num_nodes=args.num_nodes,
        **ddp_params,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            MaxStepProgressBar(),
            GenerateSamplesOnValEnd(
                SampleConfig(
                    start_after_step=args.save_midi_output_after_step,
                    num_events_to_generate=args.num_events_to_generate_for_midi_inference,
                )
            ),
        ],
        enable_progress_bar=True,
        precision="bf16-mixed" if args.bf16 else 32,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=10,
        val_check_interval=args.steps_per_eval,
        **logger_dict,
    )
    trainer.fit(model)


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT-2 training script using PyTorch Lightning"
    )
    parser.add_argument(
        "--data_dir", type=str, help="Directory where tokenized dataset is located"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Initialize model weights from this checkpoint",
    )
    # Model parameters
    parser.add_argument(
        "--hidden_dim", type=int, default=768, help="Model hidden dimensions"
    )  # 768
    parser.add_argument(
        "--num_heads", type=int, default=12, help="Model number of attention heads"
    )  # 12
    parser.add_argument(
        "--num_layers", type=int, default=12, help="Model number of layers"
    )  # 12
    parser.add_argument(
        "--embed_pdrop", type=float, default=0.1, help="Apply embedding dropout"
    )
    parser.add_argument(
        "--resid_pdrop", type=float, default=0.1, help="Apply residual dropout"
    )

    # Optimization parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_train_steps", type=int, default=100000, help="Number of training steps"
    )  # 100000
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=512, help="Batch size for training"
    )  # keep 512 batch size
    parser.add_argument(
        "--eval_batch_size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=200, help="Number of warmup steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 mixed precision training"
    )
    parser.add_argument(
        "--steps_per_eval",
        type=int,
        default=1000,
        help="Number of steps between validation evals",
    )
    parser.add_argument(
        "--steps_per_checkpoint",
        type=int,
        default=1000,
        help="Number of steps between checkpoints",
    )  # set back to 1000

    parser.add_argument(
        "--save_midi_output_after_step",
        type=int,
        default=5000,
        help="After this number of steps, at the end of a validation epoch, we will otout MIDI to wandb.",
    )
    parser.add_argument(
        "--num_events_to_generate_for_midi_inference",
        type=int,
        default=100,
        help="Number of EVENTS, not tokens. An event is a triple or tick.",
    )

    # System parameters
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--gpus_per_node", type=int, default=1, help="Number of GPUs per node"
    )  # 4 gpus

    parser.add_argument(
        "--use_wandb", action="store_true", help="whether to use wandb logging"
    )
    parser.add_argument(
        "--no_ddp",
        action="store_true",
        help="whether to ignore using DDP - just for testing really.",
    )
    return parser


if __name__ == "__main__":
    ap = get_argparser()
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
