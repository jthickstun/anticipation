import argparse
import gc
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.util import save_text
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset

from train.v2.custom_metrics import ApproxBPS, TokenPerplexity
from train.v2.dataset_utils import PreTokenizedDataset
from train.v2.hf_gpt2_rewrite import (
    GPT2ConfigLite,
    GPT2LMHeadModelLite,
    build_model_meta,
    estimate_flops,
    get_num_scaling_params,
    get_scaling_analysis_data,
    print0,
)
from train.v2.logging_utils import (
    GenerateSamplesOnValEnd,
    MaxStepProgressBar,
    SampleConfig,
)

# keep this!!!
torch.set_float32_matmul_precision("high")


class TipFilter(logging.Filter):
    """
    Credit to: https://github.com/Lightning-AI/pytorch-lightning/issues/21294#issuecomment-3410770397
    """

    def filter(self, record):
        m = record.getMessage()
        return "💡 Tip:" not in m


logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(TipFilter())


class GPT2LightningModule(pl.LightningModule):
    def __init__(
        self,
        data_dir: Path,
        settings: AnticipationV2Settings,
        config: GPT2ConfigLite,
        num_flops_per_token: int,
        learning_rate: float = 5e-5,
        warmup_percent: float = 0.1,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        pretrained_checkpoint: str = None,
        no_cuda_graphs: bool = False,
        skip_all_validation_during_training: bool = False,
        do_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # these are saved by .save_hyperparameters, but I want to
        # emphasize that they are indeed used
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__train_batch_size = train_batch_size
        self.__eval_batch_size = eval_batch_size
        self.__warmup_percent = warmup_percent

        self.data_dir = data_dir
        self.num_flops_per_token = num_flops_per_token
        self.no_cuda_graphs = no_cuda_graphs
        self.do_gradient_checkpointing = do_gradient_checkpointing

        if pretrained_checkpoint:
            self.model = GPT2LMHeadModelLite.from_pretrained(
                pretrained_checkpoint, config=config
            )
            rank_zero_info(f"Loaded pre-trained model from {pretrained_checkpoint}")
        else:
            self.model = GPT2LMHeadModelLite(config)
            if config.do_torch_compile:
                if no_cuda_graphs:
                    # OOM can happen on GB
                    self.model = torch.compile(
                        self.model, dynamic=False, mode="max-autotune-no-cudagraphs"
                    )
                else:
                    self.model = torch.compile(self.model, dynamic=False)

        if self.do_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.config.bos_token_id = self.model.config.eos_token_id = 0
        self.anticipation_settings = settings

        # --- validation metrics and settings ----
        self.skip_all_validation_during_training = skip_all_validation_during_training
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
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model(**inputs)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        labels = batch.pop("labels")
        # logits = self(**batch)
        outputs = self(**batch)
        # labels = batch.pop("labels")
        # loss = self(idx=batch["input_ids"], targets=labels)

        # keep this upcast!
        # https://x.com/jwthickstun/status/1737134520141246938
        logits = outputs.logits.float()  # upcast logits and compute loss in fp32
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train_loss", loss.detach(), prog_bar=True, logger=True)

        # things that are logged in nanochat training,
        # see: https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/scripts/base_train.py#L539

        # NB:
        # in lightning 'global_step' corresponds to an optimizer step
        # step = self.global_step
        # 'training_step' runs each microbatch... so it increases even if gradient
        # accululation is not 1

        # if self.num_flops_per_token != -1:
        ## DDP stuff is taken care of for us
        # num_tokens_per_step = (
        # self.hparams.train_batch_size * self.anticipation_settings.context_size
        # )
        # flops_per_step = self.num_flops_per_token * num_tokens_per_step
        # flops_so_far = step * flops_per_step
        # self.log("flops_so_far", flops_so_far, prog_bar=False, logger=True)

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # logits = self(idx=batch["input_ids"])
        # labels = batch.pop("labels")
        labels = batch.pop("labels")
        outputs = self(**batch)
        # logits = self(**batch)

        # keep this upcast!
        # https://x.com/jwthickstun/status/1737134520141246938
        logits = outputs.logits.float()  # upcast logits and compute loss in fp32
        # logits = logits.float()  # upcast logits and compute loss in fp32
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
            num_ticks * self.anticipation_settings.tick_token_every_n_ticks
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
        special = ["resid_lambdas", "x0_lambdas"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (
                        not any(nd in n for nd in no_decay)
                        and not any(nd in n for nd in special)
                    )
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
            # --- new nanochat groups ---
            # see: https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/nanochat/gpt.py#L348
            {
                "params": [self.model.transformer.resid_lambdas],
                "lr": 0.5 * 0.01,
                "weight_decay": 0.0,
                "eps": 1e-10,
            },
            {
                "params": [self.model.transformer.x0_lambdas],
                "lr": 0.5,
                "weight_decay": 0.0,
                "betas": (0.96, 0.95),
                "eps": 1e-10,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
            fused=True,
        )

        train_dataloader = self.train_dataloader()
        if hasattr(train_dataloader, "dataset"):
            dataset_size = len(train_dataloader.dataset)
            pct_start = self.__warmup_percent
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

        if self.skip_all_validation_during_training and (
            self.trainer is not None
            and getattr(self.trainer.state, "fn", None) == "fit"
        ):
            # hack for short term, I can't figure out a reliable way
            # to get lightning simply to SKIP the validation step during
            # training.
            empty = Subset(dataset, [])
            return DataLoader(empty, batch_size=1)

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

        if trainer.global_step == 0:
            # don't save if we just started
            return

        step_dir = os.path.join(self.dirpath, f"step-{trainer.global_step}")
        raw_model = pl_module.model if hasattr(pl_module, "model") else pl_module
        raw_model.save_pretrained(
            step_dir, safe_serialization=True, max_shard_size="2GB"
        )


def compute_max_steps_from_flop_budget(
    flop_budget: float,
    flops_per_token: float,
    seq_len: int,
    per_gpu_batch_size: int,
    num_gpus: int,
    grad_accum_steps: int = 1,
) -> int:
    import math

    if flop_budget <= 0:
        raise ValueError("flop_budget must be > 0")
    if flops_per_token <= 0:
        raise ValueError("flops_per_token must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if per_gpu_batch_size <= 0:
        raise ValueError("per_gpu_batch_size must be > 0")
    if num_gpus <= 0:
        raise ValueError("num_gpus must be > 0")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")

    tokens_per_update = seq_len * per_gpu_batch_size * num_gpus * grad_accum_steps
    token_budget = flop_budget / flops_per_token
    max_steps = math.floor(token_budget / tokens_per_update)
    return max(1, max_steps)


def main(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed)
    tokenized_dataset_path = Path(args.data_dir)
    assert tokenized_dataset_path.exists()
    assert tokenized_dataset_path.is_dir()
    anticipation_settings_file = next(
        tokenized_dataset_path.glob("settings_*.json"), None
    )
    if not anticipation_settings_file:
        raise RuntimeError("Unable to find settings")
    anticipation_settings = AnticipationV2Settings.load_from_disk(
        anticipation_settings_file
    )

    print(f"Saving outputs to: {args.output_dir}")

    if args.num_train_steps > -1:
        # TODO: warn
        pass

    depth = args.depth
    flops = args.flops
    target_param_data_ratio = args.target_param_data_ratio
    if depth > -1 and (flops > -1 or target_param_data_ratio > -1):
        # doing a 'scaling law' kind of experiment
        model_config: GPT2ConfigLite = build_model_meta(
            depth,
            anticipation_settings=anticipation_settings,
            # these dropouts are very important for us
            embd_pdrop=args.embed_pdrop,
            resid_pdrop=args.resid_pdrop,
            aspect_ratio=args.aspect_ratio,
            head_dim=args.head_dim,
            window_pattern=args.window_pattern,
            layer_norm_epsilon=args.layer_norm_epsilon,
            pos_emb=args.pos_emb,
            embedding_and_lm_head_weight_tying=not args.no_weight_tie,
            use_value_embeds=args.use_value_embeds,
            do_torch_compile=not args.no_torch_compile,
            # mlp_style=args.mlp_style,
        )
        with torch.device("meta"):
            model = GPT2LMHeadModelLite(model_config)
            train_info = get_scaling_analysis_data(
                model,
                args.train_batch_size,
                num_iterations=-1,
                target_param_data_ratio=target_param_data_ratio,
                target_flops=flops,
            )
            model_conf_json = asdict(model_config)
            csv_row = {
                **train_info,
                **model_conf_json,
            }
            total_tokens = csv_row["total_tokens"]
            dataset = PreTokenizedDataset(tokenized_dataset_path / "train.npy")

            # https://arxiv.org/abs/2305.16264v5
            # TL;DR of 'scaling data constrained language models (2025)', a model can
            # backprop over the same data up to 4 times more or less before reaching
            # diminishing returns, when it is roughly on the efficient frontier
            token_multiplier = 4
            total_tokens_available = token_multiplier * dataset.num_tokens
            if not (total_tokens <= total_tokens_available):
                print(
                    f"Not enough tokens to meet FLOP or ratio budget. "
                    f"Need: {total_tokens:,}, Have: {dataset.num_tokens:,}"
                )
                # exit the program without killing
                return None

            effective_num_iterations = csv_row["num_iterations"]
            num_flops_per_token = csv_row["num_flops_per_token"]

            # save the settings if rank0
            save_to_path = Path(args.output_dir) / "scaling_analysis_info.json"
            save_text(save_to_path, json.dumps(csv_row))
    else:
        if args.checkpoint_path:
            model_config = GPT2ConfigLite.from_json(
                str(Path(args.checkpoint_path) / "config.json")
            )
        else:
            model_config = GPT2ConfigLite(
                vocab_size=anticipation_settings.vocab.total_tokens(),
                n_positions=anticipation_settings.context_size,
                n_embd=args.hidden_dim,
                n_inner=args.intermediate_dim,
                n_layer=args.num_layers,
                n_head=args.num_heads,
                embd_pdrop=args.embed_pdrop,
                resid_pdrop=args.resid_pdrop,
                # as of right now doesn't do anything, we always use gelu
                activation_function="gelu_new",
                layer_norm_epsilon=args.layer_norm_epsilon,
                pos_emb=args.pos_emb,
                window_pattern=args.window_pattern,
                scale_attn_weights=True,
                scale_attn_by_inverse_layer_idx=True,
                use_cache=False,
                embedding_and_lm_head_weight_tying=not args.no_weight_tie,
                use_value_embeds=args.use_value_embeds,
                do_torch_compile=not args.no_torch_compile,
                mlp_style=args.mlp_style,
            )

        with torch.device("meta"):
            model = GPT2LMHeadModelLite(model_config)
            num_flops_per_token = estimate_flops(model)
            param_counts_by_purpose = {
                "num_params_" + x: y for x, y in get_num_scaling_params(model).items()
            }
            print0(json.dumps(param_counts_by_purpose, indent=4))
            print0("Num flops per token:")
            print0(str(num_flops_per_token))

        effective_num_iterations = args.num_train_steps
        csv_row = {"num_flops_per_token": num_flops_per_token}

    model = GPT2LightningModule(
        data_dir=tokenized_dataset_path,
        settings=anticipation_settings,
        config=model_config,
        learning_rate=args.learning_rate,
        warmup_percent=args.warmup_percent,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        pretrained_checkpoint=args.checkpoint_path,
        num_flops_per_token=num_flops_per_token,
        no_cuda_graphs=args.no_cuda_graphs,
        skip_all_validation_during_training=args.skip_all_validation_during_training,
        do_gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    checkpoint_callback = HuggingFaceCheckpoint(
        config=model.model.config,
        dirpath=args.output_dir,
        filename="{step}",
        save_top_k=0,
        monitor=None,
        save_last=True,
        every_n_train_steps=args.steps_per_checkpoint,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"

    logger_dict = {}
    logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(TipFilter())
    if args.use_wandb:
        if args.wandb_tag:
            tags = [str(args.wandb_tag).strip()]
        else:
            tags = []

        if tags == ["scaling"]:
            run_name = f"scaling-run-{int(time.time())}"
        else:
            run_name = f"run-{int(time.time())}"

        logger_dict["logger"] = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            save_dir=args.output_dir,
            config={
                **vars(args),
                # this is already saved to disk, but associate it with the run
                # just for convenience
                **csv_row,
            },
            tags=tags,
        )

    ddp_params = {}
    if not args.no_ddp:
        ddp_params["strategy"] = DDPStrategy(
            find_unused_parameters=False, static_graph=False
        )

    # clean up some stuff from setup
    gc.collect()

    num_devices = max(1, args.gpus_per_node)
    per_device_batch_size = args.train_batch_size // num_devices
    max_steps = compute_max_steps_from_flop_budget(
        flop_budget=args.flops,
        flops_per_token=num_flops_per_token,
        seq_len=model_config.n_positions,
        per_gpu_batch_size=per_device_batch_size,
        num_gpus=num_devices,
        grad_accum_steps=args.gradient_accumulation_steps,
    )
    print0("Max steps")
    print0(str(max_steps))

    trainer = pl.Trainer(
        max_steps=max_steps,
        # always use gpu and then thrown an error if it's unavailable - that's preferable
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
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.steps_per_eval,
        **logger_dict,
    )
    trainer.fit(model)

    # ensure we run validation at the very end too
    trainer.validate(model)
    return None


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
        "--intermediate_dim", type=int, default=1792, help="Inner dim of MLPs"
    )
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
    parser.add_argument(
        "--warmup_percent",
        type=float,
        default=0.01,
        help="Percentage of maximum steps to use in the warmup for cosine schedule.",
    )
    parser.add_argument(
        "--pos_emb",
        type=str,
        default="absolute",
        choices=["absolute", "rope"],
        help=(
            "The positional embedding choice: either `absolute` (vanilla) or RoPE. (rope)"
        ),
    )
    parser.add_argument(
        "--mlp_style",
        type=str,
        default="GPT2",
        choices=["GPT2", "Llama"],
        help=("The kind of MLP to use"),
    )
    parser.add_argument(
        "--layer_norm_epsilon", type=float, help="Layer Norm epsilon", default=1e-5
    )
    parser.add_argument(
        "--window_pattern",
        type=str,
        help="Sliding window pattern for long/short attention. Use only S and L, S = partial, L = long/full context.",
        default="SSSL",
    )
    parser.add_argument(
        "--no_weight_tie",
        action="store_true",
        help="whether apply weight tying to the GPT LM head and token embeddings.",
    )
    parser.add_argument(
        "--use_value_embeds",
        action="store_true",
        help="Whether to use alternating value embeddings (nanochat design)",
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
        "--no_torch_compile",
        action="store_true",
        help="Disable calling torch.compile on model instance",
    )
    parser.add_argument(
        "--no_cuda_graphs",
        action="store_true",
        help="Disable cuda graphs in torch compile.",
    )
    parser.add_argument(
        "--skip_all_validation_during_training",
        action="store_true",
        help="Disable ALL validation steps during training, regardless of the steps_per_eval setting.",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )

    # Logging Parameters
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Frequency for logging during training.",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="whether to use wandb logging"
    )
    parser.add_argument(
        "--wandb_tag", type=str, help="A tag to associate with this run"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="The project to save this run ",
        default="gpt-anticipation-2.0",
    )

    # Testing Parameters
    parser.add_argument(
        "--no_ddp",
        action="store_true",
        help="whether to ignore using DDP - just for testing really.",
    )

    # Nanochat Scaling Parameters
    parser.add_argument(
        "--depth",
        type=int,
        default=-1,
        help="The depth of the model, defaults to -1, which means model dimensions are chosen manually. If not -1, then model params are determined by `build_model_meta`. ",
    )
    parser.add_argument(
        "--flops",
        type=float,
        default=-1.0,
        help="The FLOP budget for training, defaults to -1, which does nothing. If not -1, then training parameters are fit accordingly for scaling laws analysis.",
    )
    parser.add_argument(
        "--target_param_data_ratio",
        type=float,
        default=-1.0,
        help="The ratio of params to tokens (D:N in chinchilla). defaults to -1, which does nothing. If not -1, then number of iterations are dynamically determined by this.",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=int,
        default=64,
        help="model_dim = depth * aspect_ratio. Default is 64, same as nanochat",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=128,
        help="The dimension of head in MHA, only used for nanochat related stuff.",
    )

    return parser


if __name__ == "__main__":
    _args = get_argparser().parse_args()

    os.makedirs(_args.output_dir, exist_ok=True)
    main(_args)
