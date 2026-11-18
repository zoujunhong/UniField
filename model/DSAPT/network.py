"""Coarse-to-fine Tiny encoders with global K/V in every fine window."""
import json
import math
import types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from model.third_party.utonia_dsapt.model import PointTransformerV3, Block, SerializedAttention
from .baseline_network import LoRALinear, Lion, EMA, losses as main_losses
from .decoder import QueryDecoder
from .idw import interpolate_global
from . import local_compile


def losses(prediction, batch, config=None):
    """Standardized main pressure MSE plus optional IDW pressure supervision.

    Both heads use the same normalized target and scoring points. Logged
    parts remain unweighted MSE values; only the returned total applies the
    auxiliary coefficient. Evaluation returns the original main prediction.
    """
    total, parts = main_losses(prediction, batch)
    if "auxiliary_surface_p" in prediction:
        weight = float((config or {}).get("auxiliary_loss_weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("auxiliary_loss_weight must be finite and nonnegative")
        auxiliary = F.mse_loss(prediction["auxiliary_surface_p"].float(),
                               batch["target_surface_p"].float())
        parts["auxiliary_surface_p"] = auxiliary
        total = total + weight * auxiliary
    return total, parts


def global_tensor_attention(self, feat, coord, order, inverse, length, global_feat, global_coord):
    h, c = self.num_heads, self.channels
    d = c // h
    qkv = self.qkv(feat)[order]
    q, k, v = qkv.reshape(-1, 3, h, d).unbind(1)
    q, k = self.rope(q, k, coord[order])
    context = self.global_projection(global_feat)
    _, gk, gv = self.qkv(context).reshape(-1, 3, h, d).unbind(1)
    _, gk = self.rope(gk, gk, global_coord)
    q, k, v = (x.to(getattr(self, 'attention_dtype', torch.float16)).reshape(-1, length, h, d).transpose(1, 2)
               for x in (q, k, v))
    gk, gv = (x.to(getattr(self, 'attention_dtype', torch.float16)).transpose(0, 1)[None].expand(q.shape[0], -1, -1, -1)
              for x in (gk, gv))
    k, v = torch.cat((k, gk), dim=2), torch.cat((v, gv), dim=2)
    extra = -d % 8
    if extra:
        q, k, v = (F.pad(x, (0, extra)) for x in (q, k, v))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
    y = y[..., :d].transpose(1, 2).reshape(-1, c).to(qkv.dtype)[inverse]
    return self.proj_drop(self.proj(y))


def global_forward(self, point):
    if point.offset.numel() != 1:
        raise ValueError('Validated for one vehicle per GPU only')
    pad, unpad, _ = self.get_padding_and_inverse(point)
    order = point.serialized_order[self.order_index][pad]
    inverse = unpad[point.serialized_inverse[self.order_index]]
    point.feat = self._tensor_attention(point.feat, point.coord, order, inverse,
        min(self.patch_size, point.feat.shape[0]), point.context['feat'], point.context['coord'])
    return point


class Branch(nn.Module):
    def __init__(self, config, stride, branch_seed):
        super().__init__()
        self.mode = config['mode']
        architecture = dict(config['architecture']) if 'architecture' in config else json.loads(Path(config.get('architecture_file', Path(__file__).with_name('architecture.json'))).read_text())
        architecture.update(stride=tuple(stride), enc_patch_size=(config['encoder_patch_size'],)*5)
        assert architecture['enc_mode'] and tuple(architecture['enc_channels']) == (18,36,72,144,216)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(branch_seed)
            self.backbone = PointTransformerV3(**architecture)
        if self.mode == 'lora':
            checkpoint = torch.load(config['encoder_checkpoint'], map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            self.backbone.requires_grad_(False)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(branch_seed + 100)
                for block in [m for m in self.backbone.modules() if isinstance(m, Block)]:
                    for parent, attr in ((block.attn,'qkv'), (block.attn,'proj'),
                                         (block.mlp[0],'fc1'), (block.mlp[0],'fc2')):
                        setattr(parent, attr, LoRALinear(getattr(parent, attr), config['lora_rank'],
                            config['lora_alpha'], config['lora_dropout']))
        elif self.mode != 'scratch':
            raise ValueError(self.mode)
        self.stage_counts = [0] * 5
        for i, stage in enumerate(self.backbone.enc.children()):
            stage.register_forward_hook(lambda m,a,o,index=i: self.stage_counts.__setitem__(index,int(o.feat.shape[0])))

    def forward(self, point):
        # Each backbone constructs its own Point and serialization metadata.
        return self.backbone(dict(point))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Drop-path disabled in both controlled variants; autograd remains enabled.
        return self


class DualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.coarse = Branch(config, (2,2,2,2), config['seed']+1000)
        self.fine = Branch(config, (2,1,1,1), config['seed']+2000)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed']+3000)
            for module in self.fine.modules():
                if isinstance(module, SerializedAttention):
                    module.global_projection = nn.Sequential(nn.LayerNorm(216),
                        nn.Linear(216,module.channels), nn.LayerNorm(module.channels))
        precision = str(config.get('encoder_precision', 'float16')).lower()
        if precision not in {'float16', 'fp16', 'bfloat16', 'bf16'}:
            raise ValueError(f'Unsupported DSAPT encoder precision: {precision}')
        attention_dtype = torch.bfloat16 if precision in {'bfloat16', 'bf16'} else torch.float16
        for module in self.modules():
            if isinstance(module, SerializedAttention):
                module.attention_dtype = attention_dtype
        self.enable_execution(config.get('encoder_compile', True))
        self.train()

    def enable_execution(self, compile_dense):
        # Point organization, CPE sparse convolution and pooling stay eager.
        local_compile.patch(self.coarse, dense_compile=compile_dense)
        for module in list(self.fine.modules()):
            if isinstance(module, SerializedAttention):
                fn = types.MethodType(global_tensor_attention, module)
                module._tensor_attention = torch.compile(fn,fullgraph=True,dynamic=True) if compile_dense else fn
                module.forward = types.MethodType(global_forward, module)
            elif isinstance(module, Block) and compile_dense:
                module.mlp[0].forward = torch.compile(module.mlp[0].forward,fullgraph=True,dynamic=True)

    @property
    def stage_counts(self):
        return dict(coarse=self.coarse.stage_counts, fine=self.fine.stage_counts)

    def forward(self, batch):
        point = batch['point'] if 'point' in batch else batch
        if point['offset'].numel() != 1:
            raise ValueError('One vehicle per forward is required')
        coarse = self.coarse(point)
        fine_point = dict(point, context=dict(feat=coarse.feat, coord=coarse.coord))
        fine = self.fine(fine_point)
        features = torch.cat((coarse.feat, fine.feat), dim=0)[None]
        positions = torch.cat((coarse.coord, fine.coord), dim=0)[None]
        # Keep native encoder order. Full-global attention and IDW are
        # permutation-invariant and need no additional Morton serialization.
        return features, positions, int(coarse.feat.shape[0]), [features.shape[1]]


def prepare_query_initial_features(features, positions, coarse_count, query, config):
    """Global IDW of only the fine branch, computed once outside compilation.

    Geometry is fixed (no coordinate gradient); token-feature gradients flow
    through IDW back into both encoders via the fine branch's global context.
    """
    if not config.get("idw_residual", False):
        return None
    if features.shape[0] != 1 or not 0 < coarse_count < features.shape[1]:
        raise ValueError('Expected nonempty coarse and fine token groups for one vehicle')
    return interpolate_global(
        features[:, coarse_count:], query[..., :3], positions[:, coarse_count:],
        query_chunk=config.get('idw_query_chunk', 512),
        power=config.get('idw_power', 2.0), eps=config.get('idw_eps', 1e-12),
        distance_backend='triton', matmul_dtype=config.get('idw_matmul_dtype', 'float32'))


def make_models(config, device='cuda'):
    encoder = DualEncoder(config).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        decoder = QueryDecoder(config).to(device)
    return encoder, decoder


def make_optimizer(encoder, decoder, config):
    groups = []
    for prefix, module in (('encoder', encoder), ('decoder', decoder)):
        buckets = {'lora': [], 'matrix': [], 'no_decay': []}
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            kind = 'lora' if name.endswith(('lora_A', 'lora_B')) else ('matrix' if p.ndim > 1 and not name.endswith('.bias') else 'no_decay')
            buckets[kind].append(p)
        for kind, parameters in buckets.items():
            if parameters:
                wd = config['weight_decay']  # Global decay includes LoRA, biases and norms.
                groups.append(dict(params=parameters, name=prefix+'_'+kind, lr_scale=1.0, weight_decay=wd))
    lr = config['peak_lr'] / (config['warmup_steps']+1)
    optimizer_name = str(config.get('optimizer', 'AdamW')).lower()
    if optimizer_name == 'ranger':
        from utils.ranger2020 import Ranger
        return Ranger(groups, lr=lr, betas=tuple(config['betas']), **config['ranger'])
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(config.get('betas', (0.9, 0.999))))
    if optimizer_name == 'lion':
        return Lion(groups, lr=lr, betas=tuple(config.get('betas', (0.9, 0.99))))
    raise ValueError(f'Unsupported DSAPT optimizer: {optimizer_name}')


def training_loss(prediction, batch, config=None):
    """Supervise available fields with the configured per-car objective."""
    from utils.dsapt_metrics import relative_l2
    loss_type = str((config or {}).get('loss_type', 'relative_l2')).lower()
    if loss_type not in {'relative_l2', 'rel_l2', 'rell2', 'mse'}:
        raise ValueError(f'Unsupported DSAPT loss_type: {loss_type}')
    fields = [field for field in ('surface_p', 'volume_p', 'volume_u')
              if field in prediction and 'target_' + field in batch
              and prediction[field].numel() > 0]
    if not fields:
        raise ValueError('DSAPT requires at least one nonempty supervised field.')
    parts, objectives, mse_values, mae_values = {}, [], [], []
    for field in fields:
        pred = prediction[field].float()
        target = batch['target_' + field].float()
        rel = relative_l2(pred, target).mean()
        mse = (pred - target).square().mean()
        parts[field + '_rel_l2'] = rel
        field_config = (config or {}).get('field_loss_config', {}).get(field, {})
        field_loss_type = str(field_config.get('loss_type', loss_type)).lower()
        if field_loss_type not in {'relative_l2', 'rel_l2', 'rell2', 'mse'}:
            raise ValueError(f'Unsupported DSAPT loss_type for {field}: {field_loss_type}')
        weight = float(field_config.get('weight', 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f'Invalid DSAPT supervision weight for {field}: {weight}')
        objective = mse if field_loss_type == 'mse' else rel
        parts[field] = objective
        objectives.append(objective if weight == 1.0 else weight * objective)
        mse_values.append(mse)
        mae_values.append((pred - target).abs().mean())
    parts['mse'] = sum(mse_values)
    parts['mae'] = sum(mae_values)
    return sum(objectives), parts
