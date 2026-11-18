"""One compiled dense region per eval block; sparse convolution remains eager."""
import types
import torch
from .local_compile import tensor_attention
from .network import global_tensor_attention
from .optimized_attention_metadata import attention_metadata

def body(self, feat, conv_feat, coord, order, inverse, length, context_feat, context_coord):
    conv, linear, cpe_norm, norm1, norm2, ls1, ls2, mlp, drop_path, attn = self._dense_children
    feat = feat + cpe_norm(linear(conv_feat))
    shortcut = feat
    if self.pre_norm: feat = norm1(feat)
    if hasattr(attn, 'global_projection'):
        feat = global_tensor_attention(attn, feat, coord, order, inverse, length, context_feat, context_coord)
    else:
        feat = tensor_attention(attn, feat, coord, order, inverse, length)
    feat = shortcut + drop_path(ls1(feat))
    if not self.pre_norm: feat = norm1(feat)
    shortcut = feat
    if self.pre_norm: feat = norm2(feat)
    feat = shortcut + drop_path(ls2(mlp(feat)))
    if not self.pre_norm: feat = norm2(feat)
    return feat

def forward(self, point):
    if self.training: raise RuntimeError('Encoder blocks must remain eval mode')
    sparse = self._dense_children[0](point.sparse_conv_feat)
    context = point.context if hasattr(self._dense_children[-1], 'global_projection') else None
    attn = self._dense_children[-1]
    order, inverse, length = attention_metadata(point, attn.patch_size, attn.order_index)
    point.feat = self._dense_body(point.feat, sparse.features, point.coord, order, inverse, length,
        context['feat'] if context is not None else None,
        context['coord'] if context is not None else None)
    point.sparse_conv_feat = sparse.replace_feature(point.feat)
    return point

def patch(encoder):
    from model.third_party.utonia_dsapt.model import Block, SerializedAttention
    # Only token counts vary: symbolic channel/head integers break pad backward
    # shape reasoning in torch 2.6 when the entire dense block is traced.
    torch._dynamo.mark_static(SerializedAttention)
    torch._dynamo.mark_static(Block)
    count=0
    for module in list(encoder.modules()):
        if not isinstance(module, Block): continue
        assert not module.training
        children=(module.cpe[0],module.cpe[1],module.cpe[2],module.norm1[0],module.norm2[0],
                  module.ls1[0],module.ls2[0],module.mlp[0],module.drop_path[0],module.attn)
        module._dense_children=children
        module._dense_body=torch.compile(types.MethodType(body,module),fullgraph=True,dynamic=True)
        module.forward=types.MethodType(forward,module)
        count+=1
    return dict(blocks=count,scope='CPE Linear+LayerNorm+residual+attention+FFN in one tensor region; sparse convolution and point organization eager')

def sparse_fp16(encoder):
    import spconv.pytorch as spconv
    changed=[]
    for name,module in list(encoder.named_modules()):
        if not isinstance(module,spconv.SubMConv3d): continue
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError('This probe only converts frozen sparse-convolution weights')
        original=module.forward
        module.half()
        module.fp32_accum=True
        def wrapped(self, point, _original=original):
            dtype=point.features.dtype
            with torch.autocast('cuda',enabled=False):
                out=_original(point.replace_feature(point.features.to(torch.float16)))
            return out.replace_feature(out.features.to(dtype))
        module.forward=types.MethodType(wrapped,module)
        changed.append(name)
    return dict(layers=changed,weight_input_output='fp16 inside sparse conv',
                fp32_accum_requested=True,return_dtype='original feature dtype',rest='BF16 AMP, original FP32 frozen dense weights')
