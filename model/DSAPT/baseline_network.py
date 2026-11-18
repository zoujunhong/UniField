"""Official Utonia and a condition-free, native-width two-head query decoder.

The encoder uses the checkpoint's untouched architecture and all final tokens.
Full/LoRA/frozen policies share decoder initialization and inference semantics.
"""
from contextlib import contextmanager, nullcontext
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.third_party.utonia_dsapt.model import PointTransformerV3, Block


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.multiplier = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        low = F.linear(self.dropout(x), self.lora_A)
        return self.base(x) + F.linear(low, self.lora_B) * self.multiplier


class UtoniaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config["mode"]
        if self.mode not in ("full", "lora", "frozen"):
            raise ValueError(self.mode)
        checkpoint = torch.load(config["encoder_checkpoint"], map_location="cpu", weights_only=True)
        self.checkpoint_config = dict(checkpoint["config"])
        self.official_config = dict(self.checkpoint_config, stride=tuple(config["encoder_stride"]),
                                   enc_patch_size=(config["encoder_patch_size"],)*5)
        assert tuple(self.official_config["enc_channels"]) == (18,36,72,144,216)
        assert tuple(self.official_config["enc_depths"]) == (2,2,2,6,2)
        self.backbone = PointTransformerV3(**self.official_config)
        self.backbone.load_state_dict(checkpoint["state_dict"], strict=True)
        stride = tuple(config["encoder_stride"])
        assert stride == (2,1,1,1)
        stages = list(self.backbone.enc.children())
        assert len(stages) == 5
        for stage, value in zip(stages[1:], stride):
            stage.down.stride = value
        self.official_config = dict(self.official_config, stride=stride)
        self.backbone.requires_grad_(self.mode == "full")
        self.padding_multiple = 64
        self.lora_targets = []
        if self.mode == "lora":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.get("seed", 200099) + 1)
                # Materialize the block list before adding modules.
                blocks = [(n, b) for n, b in self.backbone.named_modules() if isinstance(b, Block)]
                for name, block in blocks:
                    for parent, attr in ((block.attn, "qkv"), (block.attn, "proj"),
                                         (block.mlp[0], "fc1"), (block.mlp[0], "fc2")):
                        setattr(parent, attr, LoRALinear(getattr(parent, attr),
                                config.get("lora_rank", 8), config.get("lora_alpha", 16),
                                config.get("lora_dropout", 0.0)))
                    self.lora_targets.extend(name + "." + suffix for suffix in
                                             ("attn.qkv", "attn.proj", "mlp.0.fc1", "mlp.0.fc2"))
            if len(self.lora_targets) != 4 * sum(self.official_config["enc_depths"]):
                raise RuntimeError(f"Unexpected official LoRA target count: {len(self.lora_targets)}")
        self.stage_counts = [None] * 5
        for i, stage in enumerate(stages):
            stage.register_forward_hook(lambda module, args, output, index=i: self.stage_counts.__setitem__(index, int(output.feat.shape[0])))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Disable pretrained drop-path and stateful running-statistic updates;
        # eval mode intentionally does not disable full/LoRA autograd.
        self.backbone.eval()
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module.dropout.train(mode)
        return self

    def forward(self, point):
        with torch.no_grad() if self.mode == "frozen" else nullcontext():
            out = self.backbone(point)
            counts = out.offset.diff(prepend=out.offset.new_zeros(1)).tolist()
            if out.feat.shape[-1] != 216 or not counts or min(counts) <= 0:
                raise RuntimeError("Expected nonempty official Utonia final features of width 216")
            width = math.ceil(max(counts) / self.padding_multiple) * self.padding_multiple
            features = nn.utils.rnn.pad_sequence(out.feat.split(counts), batch_first=True)
            positions = nn.utils.rnn.pad_sequence(out.coord.split(counts), batch_first=True)
            features = F.pad(features, (0, 0, 0, width - features.shape[1]))
            positions = F.pad(positions, (0, 0, 0, width - positions.shape[1]))
            valid = torch.arange(width, device=out.feat.device)[None, :] < torch.tensor(counts, device=out.feat.device)[:, None]
        return features, positions, valid, counts


class PositionEmbedding(nn.Module):
    def __init__(self, dim, bands=6, normal=False):
        super().__init__()
        self.register_buffer("frequencies", math.pi * 2.0 ** torch.arange(bands))
        self.normal = normal
        self.proj = nn.Linear(3 + 6 * bands + (3 if normal else 0), dim)

    def forward(self, query):
        xyz = query[..., :3]
        phase = (xyz.unsqueeze(-1) * self.frequencies).flatten(-2)
        parts = (xyz, phase.sin(), phase.cos(), query[..., 3:6]) if self.normal else (xyz, phase.sin(), phase.cos())
        return self.proj(torch.cat(parts, dim=-1))


class CrossBlock(nn.Module):
    def __init__(self, dim=576, heads=6, ratio=4):
        super().__init__()
        if dim % heads:
            raise ValueError("Attention width must be divisible by head count")
        self.heads, self.head_dim = heads, dim // heads
        self.qnorm = nn.LayerNorm(dim)
        self.knorm = nn.LayerNorm(dim)
        self.fnorm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ratio * dim), nn.GELU(), nn.Linear(ratio * dim, dim))

    def forward(self, x, memory, valid):
        b, n, dim = x.shape
        q = self.q(self.qnorm(x)).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.knorm(memory)).view(b, memory.shape[1], 2, self.heads, self.head_dim)
        k, v = kv.unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        # Torch 2.6 masked efficient SDPA needs head_dim divisible by 8.
        # Zero extension preserves the 36-D dot products, values and scaling.
        extra = -self.head_dim % 8
        if extra:
            q, k, v = (F.pad(t, (0, extra)) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v,
                                          attn_mask=valid[:, None, None, :], dropout_p=0.0,
                                          scale=self.head_dim ** -0.5)[..., :self.head_dim]
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, dim))
        return x + self.ffn(self.fnorm(x))


class QueryDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.get("decoder_dim", 576)
        if dim != 216:
            raise ValueError("Use the native 216-dimensional Tiny encoder features")
        self.surface_queries = config.get("surface_queries", 16384)
        self.volume_queries = config.get("volume_queries", 16384)
        self.query_position = PositionEmbedding(dim, config.get("fourier_bands", 6), normal=True)
        self.memory_position = PositionEmbedding(dim, config.get("fourier_bands", 6))
        self.memory_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(CrossBlock(dim, config.get("heads", 6), 4)
                                    for _ in range(config.get("decoder_depth", 4)))
        self.output_norm = nn.LayerNorm(dim)
        self.surface_head = nn.Linear(dim, 1)

    def forward(self, features, positions, valid, query):
        if query.shape[-1] != 6:
            raise ValueError("Surface queries must provide xyz + normal")
        memory = self.memory_norm(features) + self.memory_position(positions)
        x = self.query_position(query)
        for block in self.blocks:
            x = block(x, memory, valid)
        x = self.output_norm(x)
        return {"surface_p": self.surface_head(x)}


def losses(prediction, batch):
    parts = {key: F.mse_loss(prediction[key].float(), batch["target_" + key].float())
             for key in ("surface_p",)}
    return sum(parts.values()), parts


def make_models(config, device):
    encoder = UtoniaEncoder(config).to(device)
    # Neither adapter construction nor official model allocation changes the
    # random decoder's initialization across the three controlled experiments.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.get("seed", 200099))
        decoder = QueryDecoder(config).to(device)
    return encoder, decoder


class Lion(torch.optim.Optimizer):
    """Unmodified Lion update equations, vectorized over each parameter group."""
    def __init__(self, params, lr, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid Lion hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            if any(g.is_sparse for g in gradients):
                raise RuntimeError("Lion requires dense gradients")
            for p in params:
                if not self.state[p]:
                    self.state[p]["exp_avg"] = torch.zeros_like(p)
            momenta = [self.state[p]["exp_avg"] for p in params]
            beta_update, beta_state = group["betas"]
            direction = torch._foreach_mul(momenta, beta_update)
            torch._foreach_add_(direction, gradients, alpha=1 - beta_update)
            torch._foreach_sign_(direction)
            torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
            torch._foreach_add_(params, direction, alpha=-group["lr"])
            torch._foreach_mul_(momenta, beta_state)
            torch._foreach_add_(momenta, gradients, alpha=1 - beta_state)
        return loss


def _uncompiled(module):
    return getattr(module, "_orig_mod", module)


def make_optimizer(encoder, decoder, config):
    groups = []
    for prefix, module, scale, weight_decay in (
        ("decoder", _uncompiled(decoder), 1.0, config.get("weight_decay", 0.05)),
        ("encoder" if config["mode"] == "full" else "lora", encoder,
         0.1 if config["mode"] == "full" else 1.0,
         config.get("weight_decay", 0.05) if config["mode"] == "full" else 0.0),
    ):
        decay, exempt = [], []
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                (exempt if parameter.ndim <= 1 or name.endswith(".bias") or weight_decay == 0 else decay).append(parameter)
        for suffix, params, wd in (("decay", decay, weight_decay), ("no_decay", exempt, 0.0)):
            if params:
                groups.append(dict(params=params, name=prefix + "_" + suffix,
                                   lr_scale=scale, weight_decay=wd))
    initial_lr = config.get("peak_lr", 1e-4) / (config.get("warmup_steps", 10179) + 1)
    for group in groups:
        group["lr"] = initial_lr * group["lr_scale"]
    return Lion(groups, lr=initial_lr, betas=tuple(config.get("betas", (0.9, 0.99))))


class EMA:
    """FP32 EMA of trainable parameters; frozen parameters remain shared.

    Validation temporarily swaps parameter storage, restoring original tensors
    even if evaluation fails. No additional full model is built on CPU/GPU.
    """
    def __init__(self, encoder, decoder, decay=0.9999):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        modules = {"encoder": _uncompiled(encoder), "decoder": _uncompiled(decoder)}
        self.parameters = {prefix + "." + name: p for prefix, module in modules.items()
                           for name, p in module.named_parameters() if p.requires_grad}
        self.buffers = {prefix + "." + name: b for prefix, module in modules.items()
                        for name, b in module.named_buffers()}
        if any(p.dtype != torch.float32 for p in self.parameters.values()):
            raise TypeError("This BF16-AMP experiment keeps optimizer parameters in FP32")
        self.shadow = {name: p.detach().clone() for name, p in self.parameters.items()}
        self.buffer_shadow = {name: b.detach().clone() for name, b in self.buffers.items()}
        self.num_updates = 0
        self._applied = False

    @torch.no_grad()
    def update(self):
        if self._applied:
            raise RuntimeError("Cannot update EMA while its weights are applied")
        if self.parameters:
            torch._foreach_lerp_(list(self.shadow.values()), list(self.parameters.values()), 1 - self.decay)
        for name, buffer in self.buffers.items():
            self.buffer_shadow[name].copy_(buffer)
        self.num_updates += 1

    def state_dict(self):
        if self._applied:
            raise RuntimeError("Checkpoint outside the EMA validation context")
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow=self.shadow, buffers=self.buffer_shadow)

    @torch.no_grad()
    def load_state_dict(self, state):
        if self._applied:
            raise RuntimeError("Cannot restore EMA during validation")
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay changed on resume")
        for key, destination in (("shadow", self.shadow), ("buffers", self.buffer_shadow)):
            if destination.keys() != state[key].keys():
                raise ValueError(f"EMA {key} keys changed on resume")
            for name, tensor in destination.items():
                saved = state[key][name]
                if tensor.shape != saved.shape or tensor.dtype != saved.dtype:
                    raise ValueError(f"EMA tensor metadata changed: {name}")
                tensor.copy_(saved)
        self.num_updates = int(state["num_updates"])

    @contextmanager
    def apply(self):
        if self._applied:
            raise RuntimeError("Nested EMA application is not supported")
        entries = [(p, self.shadow[name]) for name, p in self.parameters.items()]
        entries.extend((b, self.buffer_shadow[name]) for name, b in self.buffers.items())
        previous = []
        self._applied = True
        try:
            for tensor, shadow in entries:
                previous.append((tensor, tensor.data))
                tensor.data = shadow
            yield
        finally:
            for tensor, original in previous:
                tensor.data = original
            self._applied = False
