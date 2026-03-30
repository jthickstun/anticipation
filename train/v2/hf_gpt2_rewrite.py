import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.nanochat.flash_attention import flash_attn
from torch import nn
from torch.utils.checkpoint import checkpoint


def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        print(s, **kwargs)


@dataclass
class CausalLMOutputLite:
    loss: torch.Tensor | None
    logits: torch.Tensor
    # Optionally: past_key_values / hidden_states / attentions


@dataclass
class GPT2ConfigLite:
    vocab_size: int = 50257
    n_positions: int = 1024
    # hidden size
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    # intermediate size, the size of MLP inner dim
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False

    # Positional embedding choice (non-HF extension)
    pos_emb: str = "absolute"  # "absolute" or "rope"
    rope_theta: float = 10000.0
    rope_pct: float = 1.0  # fraction of head_dim to rotate
    window_pattern: str = "L"
    embedding_and_lm_head_weight_tying: bool = True
    use_value_embeds: bool = True
    mlp_style: str = "GPT2"
    do_torch_compile: bool = True

    @classmethod
    def from_json(cls, path: str) -> "GPT2ConfigLite":
        d = json.loads(open(path, "r", encoding="utf-8").read())
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def __post_init__(self) -> None:
        # validation logic for parameters
        assert self.mlp_style in ("GPT2", "Llama")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONV1D_SUFFIXES = (
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
)


def remap_state_dict(
    sd, *, rename_rules=None, drop_prefixes=(), drop_exact=(), conv1d_to_linear=False
):
    """
    sd: dict[str, Tensor]
    rename_rules: list[tuple[pattern, repl]] where pattern is regex
    """
    rename_rules = rename_rules or []
    out = {}

    for k, v in sd.items():
        if k in drop_exact:
            continue
        if any(k.startswith(p) for p in drop_prefixes):
            continue

        new_k = k
        for pat, repl in rename_rules:
            new_k = re.sub(pat, repl, new_k)

        # Optional: convert HF Conv1D weights to nn.Linear convention.
        if conv1d_to_linear and any(new_k.endswith(suf) for suf in CONV1D_SUFFIXES):
            # Conv1D stores [in, out]; nn.Linear wants [out, in]
            v = v.t().contiguous()

        out[new_k] = v

    return out


def load_hf_state_dict_into_model(model, hf_sd, *, strict=False, **remap_kwargs):
    sd = remap_state_dict(hf_sd, **remap_kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return missing, unexpected


def save_hf_style_checkpoint(save_dir, model, config_dict):
    import json
    from pathlib import Path

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "config.json").write_text(
        json.dumps(config_dict, indent=4, sort_keys=True)
    )
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")


def rotary_cos_sin(seq_len: int, dim: int, theta: float, device, dtype):
    # dim must be even
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, dim/2]
    cos = freqs.cos()[None, None, :, :]  # [1,1,T,dim/2]
    sin = freqs.sin()[None, None, :, :]  # [1,1,T,dim/2]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; rotate first D_rope = 2 * cos.size(-1)
    d2 = cos.size(-1)
    x1 = x[..., : 2 * d2]
    x2 = x[..., 2 * d2 :]
    x1_even = x1[..., 0::2]
    x1_odd = x1[..., 1::2]
    rot_even = x1_even * cos - x1_odd * sin
    rot_odd = x1_even * sin + x1_odd * cos
    x1_rot = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
    return torch.cat([x1_rot, x2], dim=-1)


class Conv1D(nn.Module):
    def __init__(self, nf: int, nx: int, *, init_std: float = 0.02):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class GPT2AttentionLite(nn.Module):
    def __init__(
        self, config: GPT2ConfigLite, layer_idx: int, ve_gate_channels: int = 32
    ):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads: int = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim

        self.scale_attn_weights = config.scale_attn_weights

        # doesn't actually do anything
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx

        self.layer_idx = layer_idx

        self.c_attn = Conv1D(
            3 * self.embed_dim, self.embed_dim, init_std=config.initializer_range
        )
        self.c_proj = Conv1D(
            self.embed_dim, self.embed_dim, init_std=config.initializer_range
        )
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        max_pos = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_pos, max_pos, dtype=torch.uint8)).view(
                1, 1, max_pos, max_pos
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.use_rope = config.pos_emb == "rope"
        self.rope_theta = config.rope_theta
        self.rope_pct = config.rope_pct

        self.ve_gate_channels = ve_gate_channels
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.num_heads, bias=False)
            if (config.use_value_embeds and has_ve(layer_idx, config.n_layer))
            else None
        )

    def _split_heads(self, x):
        b, t, c = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B,H,T,D]

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        b, t, h, d = x.shape
        return x.view(b, t, h * d)

    def forward(
        self,
        x,
        ve,
        window_size,
        attention_mask=None,
    ):
        # note: not using the attention mask because we always use causal attention
        # and that is controlled by a single variable in flash_attn
        b, t, _ = x.shape
        qkv = self.c_attn(x)  # [B,T,3*C]
        q, k, v = qkv.split(self.embed_dim, dim=2)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)  # (b, n_heads, t, d)

        if ve is not None:
            ve = ve.view(b, t, self.num_heads, self.head_dim)
            gate = 2 * torch.sigmoid(
                self.ve_gate(x[..., : self.ve_gate_channels])
            )  # (B, T, n_head), range (0, 2)
            _to_add = gate.unsqueeze(-1) * ve
            _to_add = _to_add.permute(0, 2, 1, 3).contiguous()
            v = v + _to_add
            v = v.to(q.dtype)

        if self.use_rope:
            rope_dim = int(self.rope_pct * self.head_dim)
            rope_dim = rope_dim - (rope_dim % 2)
            cos, sin = rotary_cos_sin(t, rope_dim, self.rope_theta, x.device, x.dtype)
            q = torch.cat(
                [apply_rope(q[..., :rope_dim], cos, sin), q[..., rope_dim:]], dim=-1
            )
            k = torch.cat(
                [apply_rope(k[..., :rope_dim], cos, sin), k[..., rope_dim:]], dim=-1
            )

        # B,H,T,D -> B,T,H,D
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        assert v.dtype == q.dtype == k.dtype, (
            f"These must all be equal: v.dtype: {v.dtype}, q.dtype: {q.dtype}, k.dtype: {k.dtype}"
        )
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # B T H D -> B H T D
        y = y.permute(0, 2, 1, 3)

        y = self._merge_heads(y)  # [B,T,C]
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2ConfigLite) -> None:
        super().__init__()
        d_ff = config.n_inner or (4 * config.n_embd)
        d_model = config.n_embd
        self.c_fc = Conv1D(d_ff, d_model, init_std=config.initializer_range)
        self.c_proj = Conv1D(d_model, d_ff, init_std=config.initializer_range)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = torch.nn.functional.gelu(
            x
        )  # keep minimal; HF supports multiple activations
        x = self.c_proj(x)
        return self.dropout(x)


class LlamaMLP(nn.Module):
    def __init__(self, config: GPT2ConfigLite) -> None:
        super().__init__()
        d_ff = config.n_inner or (4 * config.n_embd)
        d_model = config.n_embd
        # gated MLP, with silu
        self.gate_proj = Conv1D(d_ff, d_model, init_std=config.initializer_range)
        self.up_proj = Conv1D(d_ff, d_model, init_std=config.initializer_range)
        self.down_proj = Conv1D(d_model, d_ff, init_std=config.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GPT2BlockLite(nn.Module):
    def __init__(self, config: GPT2ConfigLite, layer_idx: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2AttentionLite(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        if config.mlp_style == "GPT2":
            self.mlp = GPT2MLP(config)
        elif config.mlp_style == "Llama":
            self.mlp = LlamaMLP(config)
        else:
            raise ValueError(f"Unsupported MLP style, got: {self.config.mlp_style}")

    def forward(self, x, *, window_size, ve, attention_mask=None):
        x = x + self.attn(
            self.ln_1(x), ve=ve, window_size=window_size, attention_mask=attention_mask
        )
        x = x + self.mlp(self.ln_2(x))
        return x


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


class GPT2ModelLite(nn.Module):
    def __init__(self, config: GPT2ConfigLite) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPT2BlockLite(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.window_sizes = self._compute_window_sizes(config)

        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.full((config.n_layer,), 0.1))

        if self.config.use_value_embeds:
            self.value_embeds = nn.ModuleDict(
                {
                    str(i): nn.Embedding(config.vocab_size, config.n_embd)
                    for i in range(config.n_layer)
                    if has_ve(i, config.n_layer)
                }
            )
        else:
            self.value_embeds = {}

        self.reset_parameters()

    def _compute_window_sizes(self, config: GPT2ConfigLite):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        # Map characters to window sizes
        long_window = config.n_positions
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def reset_parameters(self):
        # Mirrors GPT2PreTrainedModel _init_weights: normal_(std=initializer_range) and LN=1/bias=0.
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding, Conv1D)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.normal_(m.weight, std=self.config.initializer_range)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        # pos = torch.arange(0, t, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids)  # + self.wpe(pos)
        x = self.drop(x)

        # Convert attention_mask (B,T) with 1=keep, 0=mask to additive mask (B,1,1,T) with 0 / -10000.
        attn_bias = None
        if attention_mask is not None:
            m = attention_mask[:, None, None, :].to(dtype=x.dtype)
            attn_bias = (1.0 - m) * -10000.0

        x0 = x
        block: GPT2BlockLite
        for i, block in enumerate(self.h):
            window_size = self.window_sizes[i]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = (
                self.value_embeds[str(i)](input_ids)
                if str(i) in self.value_embeds
                else None
            )

            if self.gradient_checkpointing:

                def _custom_forward(
                    _x, block=block, ve=ve, window_size=window_size, attn_bias=attn_bias
                ):
                    return block(
                        _x, ve=ve, window_size=window_size, attention_mask=attn_bias
                    )

                x = checkpoint(
                    _custom_forward,
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, ve=ve, window_size=window_size, attention_mask=attn_bias)

        return self.ln_f(x)


class GPT2LMHeadModelLite(nn.Module):
    def __init__(self, config: GPT2ConfigLite) -> None:
        super().__init__()
        self.config = config
        self.transformer = GPT2ModelLite(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.embedding_and_lm_head_weight_tying:
            # Tie weights like HF tie_weights: lm_head.weight points to wte.weight.
            self.lm_head.weight = self.transformer.wte.weight

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.transformer.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
        **kwargs,
    ):
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if return_dict:
            return CausalLMOutputLite(loss=loss, logits=logits)
        return (loss, logits) if loss is not None else (logits,)

    @property
    def device(self) -> torch.device:
        return self.transformer.wte.weight.device

    def save_pretrained(
        self, step_dir, safe_serialization: bool = True, max_shard_size: str = "2GB"
    ) -> None:
        save_hf_style_checkpoint(
            save_dir=step_dir, model=self, config_dict=self.config.to_dict()
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[GPT2ConfigLite] = None,
        map_location: str | torch.device | dict | None = "cpu",
    ) -> "GPT2LMHeadModelLite":
        assert isinstance(checkpoint_path, str)
        p = Path(checkpoint_path)
        assert p.exists()
        assert p.is_dir()

        if config is None:
            config_path = p / "config.json"
            assert config_path.exists()
            assert config_path.is_file()
            config = GPT2ConfigLite.from_json(str(config_path.absolute()))

        weights_path = p / "pytorch_model.bin"
        assert weights_path.exists()
        assert weights_path.is_file()

        instance = cls(config=config)
        sd = torch.load(weights_path, map_location=map_location, weights_only=True)
        load_hf_state_dict_into_model(instance, sd)
        return instance


def build_model_meta(
    depth: int,
    anticipation_settings: AnticipationV2Settings,
    embd_pdrop: float = 0.1,
    resid_pdrop: float = 0.1,
    aspect_ratio: int = 64,
    head_dim: int = 128,
    window_pattern: str = "SSSL",
    activation_function: str = "gelu_new",
    layer_norm_epsilon: float = 1e-5,
    scale_attn_weights: bool = True,
    scale_attn_by_inverse_layer_idx: bool = True,
    pos_emb: str = "rope",
    embedding_and_lm_head_weight_tying: bool = True,
    use_value_embeds: bool = False,
    do_torch_compile: bool = True,
) -> GPT2ConfigLite:
    """
    From: https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/scripts/base_train.py#L125
    """
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return GPT2ConfigLite(
        vocab_size=anticipation_settings.vocab.total_tokens(),
        n_positions=anticipation_settings.context_size,
        # --- leave these below ---
        n_embd=model_dim,
        n_layer=depth,
        n_head=num_heads,
        # ----------------------
        embd_pdrop=embd_pdrop,
        resid_pdrop=resid_pdrop,
        activation_function=activation_function,
        layer_norm_epsilon=layer_norm_epsilon,
        scale_attn_weights=scale_attn_weights,
        scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
        use_cache=False,
        pos_emb=pos_emb,
        window_pattern=window_pattern,
        embedding_and_lm_head_weight_tying=embedding_and_lm_head_weight_tying,
        use_value_embeds=use_value_embeds,
        do_torch_compile=do_torch_compile,
    )


def get_num_scaling_params(gpt: GPT2LMHeadModelLite) -> dict[str, int]:
    # https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/nanochat/gpt.py#L319
    # Count each group separately (mirrors the grouping in setup_optimizers)
    wte = sum(p.numel() for p in gpt.transformer.wte.parameters())
    if gpt.transformer.value_embeds:
        value_embeds = sum(p.numel() for p in gpt.transformer.value_embeds.parameters())
    else:
        value_embeds = 0

    lm_head = sum(p.numel() for p in gpt.lm_head.parameters())
    transformer_matrices = sum(p.numel() for p in gpt.transformer.h.parameters())
    scalars = gpt.transformer.resid_lambdas.numel() + gpt.transformer.x0_lambdas.numel()

    # these two are unique to us, nanochat removes these - we will keep for now. Have
    # not yet ablated them to see if removing them improves for us
    # wpe = sum(p.numel() for p in gpt.transformer.wpe.parameters())
    ln_f = sum(p.numel() for p in gpt.transformer.ln_f.parameters())

    total = (
        wte + value_embeds + lm_head + transformer_matrices + scalars + ln_f
    )  # + wpe

    if gpt.config.embedding_and_lm_head_weight_tying:
        # if using weight tying the params are shared
        total -= wte

    actual = sum(p.numel() for p in gpt.parameters())
    assert total == actual, (
        f"Parameter count mismatch. Expected: {total:,}, Actual: {actual:,}."
    )
    return {
        "wte": wte,
        "value_embeds": value_embeds,
        "lm_head": lm_head,
        "transformer_matrices": transformer_matrices,
        "scalars": scalars,
        "total": total,
    }


def estimate_flops(gpt: GPT2LMHeadModelLite) -> int:
    """
    Return the estimated FLOPs per token for the model (forward + backward).
    Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward,
        and 2X that in backward => 2+4=6.

    Cleanest explanation of this:
        https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4

    On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.

    With sliding windows, effective_seq_len varies per layer (capped by window size).

    Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).

    This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
    - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
    - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
    """
    nparams = sum(p.numel() for p in gpt.parameters())

    # Exclude non-matmul params: embeddings and per-layer scalars
    if gpt.transformer.value_embeds:
        value_embeds_numel = sum(
            ve.weight.numel() for ve in gpt.transformer.value_embeds.values()
        )
    else:
        value_embeds_numel = 0

    nparams_exclude = (
        gpt.transformer.wte.weight.numel()
        # + gpt.transformer.wpe.weight.numel()
        + value_embeds_numel
        + gpt.transformer.resid_lambdas.numel()
        + gpt.transformer.x0_lambdas.numel()
    )
    h = gpt.config.n_head
    q = gpt.config.n_embd // gpt.config.n_head
    t = gpt.config.n_positions

    # Sum attention FLOPs per layer, accounting for sliding window
    attn_flops = 0
    for window_size in gpt.transformer.window_sizes:
        window = window_size[0]  # (left, right) tuple, we use left
        effective_seq = t if window < 0 else min(window, t)
        attn_flops += 12 * h * q * effective_seq
    num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
    return num_flops_per_token


def get_scaling_params(m: GPT2LMHeadModelLite) -> int:
    # From: https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/scripts/base_train.py#L258
    # As for which params to use exactly, transformer matrices + lm_head gives
    # cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = get_num_scaling_params(m)
    _scaling_params = params_counts["transformer_matrices"] + params_counts["lm_head"]
    return _scaling_params


def get_scaling_analysis_data(
    gpt: GPT2LMHeadModelLite,
    num_sequences_per_batch: int,
    num_iterations: int = -1,
    target_param_data_ratio: float = -1.0,
    target_flops: int = -1,
) -> dict:
    """Determine the number of steps to train for given some target.

    We may define the number of steps during training in a few ways:
    - explicitly as an argument `num_iterations`, which is basically a hard override
    - target_param_data_ratio, which is the quantity D:N reported in Chinchilla paper
        we need to discover the optimal ratio for ourselves, do not just look at the paper and put in
        the value of 20, which is what they found to be optimal in their analysis.
    - target_flops, which is the total number of algorithmic flops we want to spend on
        training the model.

    Any of these can be given, the function will return the number of training steps required to meet that
    budget.

    To discover the optimal ratio D:N, we want to sweep over:
    - FLOP budgets
    - Model Depth, which in term changes the number of parameters

    The script used to do this for nanochat is here:
        https://github.com/karpathy/nanochat/blob/master/runs/scaling_laws.sh

    From here:
        https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/scripts/base_train.py#L326

    :param gpt: The model instance we wish to use.
        This object holds the number of parameters, estimated FLOPs, and the depth of the model.
    :param num_sequences_per_batch: The batch size we wish to use in SEQUENCES (not tokens)
        in nanochat, they choose this dynamically - but only after discovering the optimal D:N ratio.
    :param num_iterations:
        the number of iterations to train for. If set to -1, determine the effective number of iterations
        based on the other arguments, otherwise this parameter is exactly the number of iterations to train
    :param target_param_data_ratio:
        D:N in chinchilla, set to -1 if we want to determine the number of iterations by FLOPs.
    :param target_flops: The total number of FLOP we have to train this model. In the nanochat scaling laws
        experiment, they choose:
            - 1e18
            - 2.15e18
            - 4.64e18
            - 1e19

    :return:
    """
    # Calculate the number of iterations we will train for and set up the various schedulers
    total_batch_size = num_sequences_per_batch * gpt.config.n_positions

    # num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
    assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0

    num_scaling_params = get_scaling_params(gpt)
    num_flops_per_token = estimate_flops(gpt)

    if num_iterations > 0:
        # Override num_iterations to a specific value if given
        effective_num_iterations = num_iterations
        print0(
            f"Using user-provided number of iterations: {effective_num_iterations:,}"
        )
    elif target_flops > 0:
        # Calculate the number of iterations from the target flops (used in scaling laws
        # analysis, e.g. runs/scaling_laws.sh)
        effective_num_iterations = round(
            target_flops / (num_flops_per_token * total_batch_size)
        )
        print0(
            f"Calculated number of iterations from target FLOPs: {effective_num_iterations:,}"
        )
    elif target_param_data_ratio > 0:
        # target_tokens is the optimal number of tokens for the model we are about to train
        target_tokens = int(target_param_data_ratio * num_scaling_params)

        # Calculate the number of iterations from the target param data ratio (the most common use case)
        effective_num_iterations = target_tokens // total_batch_size
        print0(
            f"Calculated number of iterations from target data:param ratio: {effective_num_iterations:,}"
        )
    else:
        raise ValueError("No training horizon specified")

    # the actual number of tokens we will train for
    total_tokens = total_batch_size * effective_num_iterations
    print0(f"Compute-Optimal number of training tokens: {total_tokens:,}")

    # e.g. Chinchilla was ~20
    print0(
        f"Tokens : Scaling params ratio: {total_batch_size * effective_num_iterations / num_scaling_params:.2f}"
    )
    print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

    # If we are doing scaling law stuff, then we want to produce 3 plots similar to Chinchilla:
    # 1. IsoFLOP Curves: Final Validation Metric vs. FLOPs
    #       do a quadratic fit and then pick the model that achieves lowest loss for fixed FLOPs
    #       we need to know, for that model, its parameter count and number of training tokens
    #       for each FLOP budget, use each dimension: parameter count, token count to create the
    #       next two graphs...
    # 2. Compute Optimal Model Size: Optimal Parameters vs. FLOPs
    # 3. Compute Optimal Training Tokens: Optimal Training Tokens vs. FLOPs
    param_counts_by_purpose = {
        "num_params_" + x: y for x, y in get_num_scaling_params(gpt).items()
    }
    return {
        "num_iterations": effective_num_iterations,
        "num_scaling_params": num_scaling_params,
        "num_flops_per_token": num_flops_per_token,
        "target_flops": target_flops,
        "total_tokens": total_tokens,
        "effective_flops": num_flops_per_token * total_tokens,
        "total_batch_size": total_batch_size,
        **param_counts_by_purpose,
    }


def get_compute_optimal_settings(
    gpt: GPT2LMHeadModelLite,
    anticipation_settings: AnticipationV2Settings,
    target_param_data_ratio: float = 10.5,
    total_batch_size: int = -1,
    weight_decay: float = 0.2,
):
    """Get parameters for scaling laws analysis.

    All credit here goes to nanochat:
        https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/scripts/base_train.py#L255
        https://github.com/karpathy/nanochat/blob/c7ba25214276d165eeefca7cb2060587975db189/runs/scaling_laws.sh#L69

    :param gpt: The model we want to train using compute-optimal settings.
    :param anticipation_settings: The AMT settings.
    :param target_param_data_ratio: The target param to data ratio, from the Chinchilla paper this is N:D.
    :param total_batch_size: The total batch size to use during training. If this is -1 then we will
        determine the optimal batch size for fixed FLOPs. If not -1, then we use that value.
    :param weight_decay: set to 0.2, which is the default argument in the nanochat codebase.
    :return:
    """

    # 1) Use scaling laws to determine the optimal training horizon in tokens
    # The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally
    # via scaling laws analysis).
    # We've already initialized the model so we have Params.
    # Optimal Tokens is now simply: --target-param-data-ratio * Params

    scaling_params = get_scaling_params(gpt)

    # optimal tokens for the model we are about to train
    target_tokens = int(target_param_data_ratio * scaling_params)

    with torch.device("meta"):
        d12_ref_config = build_model_meta(
            depth=12,
            anticipation_settings=anticipation_settings,
            aspect_ratio=64,
            head_dim=128,
            window_pattern="SSSL",
        )
        d12_ref = GPT2LMHeadModelLite(d12_ref_config)

        # compute-optimal d12 training horizon in tokens (measured empirically)
        D_REF = target_param_data_ratio * get_scaling_params(d12_ref)

        # optimal batch size at d12 ~= 524,288 tokens (measured empirically)
        B_REF = 2**19

    # 2) Now that we have the token horizon, we can calculate the optimal batch size
    # We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
    # The optimal batch size grows as approximately D^0.383,
    # so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
    if total_batch_size == -1:
        # when this is set to -1, then we auto compute it - otherwise it is a user-override.
        batch_size_ratio = target_tokens / D_REF
        predicted_batch_size = B_REF * batch_size_ratio**0.383

        # clamp the batch size to the nearest power of 2 for efficiency
        total_batch_size = 2 ** round(math.log2(predicted_batch_size))
        print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

    # 3) Knowing the batch size, we can now calculate a learning rate correction
    # (bigger batch size allows higher learning rates)
    batch_lr_scale = 1.0
    batch_ratio = total_batch_size / B_REF  # B/B_ref
    if batch_ratio != 1.0:
        # SGD: linear scaling with batch size is standard (not used in nanochat)
        # AdamW: sqrt scaling is standard: η ∝ \sqrt{(B/B_ref)}
        # Muon: we will use the same scaling for Muon as for AdamW: η ∝ \sqrt{(B/B_ref)}
        #   (not studied carefully, assumption!)
        # NOTE: for AMT, we do not use Muon so far

        # η ∝ \sqrt{(B/B_ref)}
        batch_lr_scale = batch_ratio**0.5
        print0(
            f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})"
        )

    # 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
    # We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
    # Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
    # Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math
    # to derive that to keep T_epoch constant, we need:
    #   λ = λ_ref · √(B/B_ref) · (D_ref/D)
    #
    # Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling
    # hoping it ~works for Muon too.
    weight_decay_scaled = (
        weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    )
    if weight_decay_scaled != weight_decay:
        print0(
            f"Scaling weight decay from {weight_decay:.6f} to {weight_decay_scaled:.6f} for depth 12."
        )
