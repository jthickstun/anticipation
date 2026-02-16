import os, time
import argparse
import torch

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('high')

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from transformers import PretrainedConfig, GPT2LMHeadModel, GPT2Config


class PretokenizedDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        self.line_offsets = [0]
        self.length = 0
        with open(file_path, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.length += 1
                self.line_offsets.append(f.tell())

            f.seek(self.line_offsets[0])
            context_length = len(f.readline().strip().split())

        rank_zero_info(f"Loaded a dataset {file_path}")
        rank_zero_info(f"  - Sequence length: {context_length}")
        rank_zero_info(f"  - Number of sequences: {self.length}")
        rank_zero_info(f"  - Number of tokens: {context_length * self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            tokens = [int(token) for token in f.readline().strip().split()]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels = torch.roll(labels, shifts=-1, dims=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class GPT2LightningModule(pl.LightningModule):
    def __init__(
            self,
            data_dir: str = "",
            learning_rate: float = 5e-5,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            pretrained_checkpoint: str = None,
            config: PretrainedConfig = None,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir

        if pretrained_checkpoint:
            self.model = GPT2LMHeadModel.from_pretrained(pretrained_checkpoint, config=config)
            rank_zero_info(f"Loaded pre-trained model from {pretrained_checkpoint}")
        else:
            self.model = GPT2LMHeadModel(config)

        # if pretrained_checkpoint:
        #     rank_zero_info(f"Loading pre-trained model from {pretrained_checkpoint}")
        #     self.model = GPT2LMHeadModel.from_pretrained(
        #         pretrained_checkpoint,
        #         config=config,
        #         low_cpu_mem_usage=True,
        #         torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        #         device_map=None   # <- load fully on CPU first
        #     )
        # else:
        #     self.model = GPT2LMHeadModel(config)

        self.model.gradient_checkpointing_enable()
        self.model.config.bos_token_id = self.model.config.eos_token_id = 0

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        outputs = self(**batch)

        logits = outputs.logits.float()  # upcast logits and compute loss in fp32
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        outputs = self(**batch)

        logits = outputs.logits.float()  # upcast logits and compute loss in fp32
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95)
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
                anneal_strategy='cos',
                three_phase=False,
                cycle_momentum=False
            )

            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }

            return [optimizer], [scheduler_config]

        return optimizer

    def train_dataloader(self):
        num_devices = max(1, self.trainer.num_devices)
        per_device_batch_size = self.hparams.train_batch_size // num_devices

        dataset = PretokenizedDataset(os.path.join(self.data_dir, "train.txt"))

        return DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        num_devices = max(1, self.trainer.num_devices)
        per_device_batch_size = self.hparams.train_batch_size // num_devices

        dataset = PretokenizedDataset(os.path.join(self.data_dir, "valid.txt"))

        return DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True
        )


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
            step_dir,
            safe_serialization=True,
            max_shard_size="2GB"
        )


def main(args):
    pl.seed_everything(args.seed)

    model_config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.seq_len,
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
        data_dir=args.data_dir,
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
        every_n_train_steps=args.steps_per_checkpoint
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"

    trainer = pl.Trainer(
        max_steps=args.num_train_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus_per_node,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(
            find_unused_parameters=False,
            static_graph=False
        ),
        callbacks=[checkpoint_callback,
                   LearningRateMonitor(logging_interval="step"),
                   MaxStepProgressBar()],
        enable_progress_bar=True,
        precision="bf16-mixed" if args.bf16 else 32,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=10,
        val_check_interval=args.steps_per_eval,
        logger=WandbLogger(
            project="gpt-anticipation-2.0",
            name=f"run-{int(time.time())}",
            save_dir=args.output_dir,
            config=vars(args)
        ),
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 training script using PyTorch Lightning")
    parser.add_argument("--data_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Initialize model weights from this checkpoint")

    # Dataset parameters
    parser.add_argument("--vocab_size", type=int, default=35329, help="Dataset vocabulary size")
    parser.add_argument("--seq_len", type=int, default=1024, help="Dataset sequence length")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=768, help="Model hidden dimensions")  # 768
    parser.add_argument("--num_heads", type=int, default=12, help="Model number of attention heads")  # 12
    parser.add_argument("--num_layers", type=int, default=12, help="Model number of layers")  # 12
    parser.add_argument("--embed_pdrop", type=float, default=0.1, help="Apply embedding dropout")
    parser.add_argument("--resid_pdrop", type=float, default=0.1, help="Apply residual dropout")

    # Optimization parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_train_steps", type=int, default=100000, help="Number of training steps")  # 100000
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=512,
                        help="Batch size for training")  # keep 512 batch size
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision training")
    parser.add_argument("--steps_per_eval", type=int, default=1000, help="Number of steps between validation evals")
    parser.add_argument("--steps_per_checkpoint", type=int, default=1000,
                        help="Number of steps between checkpoints")  # set back to 1000

    # System parameters
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of GPUs per node")  # 4 gpus

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)