"""Batch-one exact serialized patch boundaries, original padding/RoPE/scale."""
import math,types
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

def tensor_attention(self,feat,coord,order,inverse,kernel_length):
    H=self.num_heads;C=self.channels;D=C//H
    qkv=self.qkv(feat)[order];q,k,v=qkv.reshape(-1,3,H,D).unbind(1)
    q,k=self.rope(q,k,coord[order].clone())
    # Match the official explicit BF16 cast after RoPE, including V.
    q,k,v=(x.to(getattr(self, 'attention_dtype', torch.float16)).reshape(-1,kernel_length,H,D).transpose(1,2) for x in (q,k,v))
    extra=-D%8
    if extra:q,k,v=(F.pad(x,(0,extra)) for x in (q,k,v))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.attn_drop if self.training else 0.0,scale=self.scale)
    y=y[...,:D].transpose(1,2).reshape(-1,C).to(qkv.dtype)[inverse]
    return self.proj_drop(self.proj(y))

def forward(self,point):
    if point.offset.numel()!=1:raise ValueError('This equivalence probe supports batch1; do not merge different cars into one patch')
    pad,unpad,cu=self.get_padding_and_inverse(point)
    order=point.serialized_order[self.order_index][pad];inverse=unpad[point.serialized_inverse[self.order_index]]
    length=min(self.patch_size,point.feat.shape[0])
    point.feat=self._tensor_attention(point.feat,point.coord,order,inverse,length)
    return point

def patch(encoder,dense_compile=False):
    from model.third_party.utonia_dsapt.model import SerializedAttention,Block
    if dense_compile:
        torch._dynamo.mark_static(SerializedAttention)
    for module in encoder.modules():
        if isinstance(module,SerializedAttention):
            assert module.enable_flash and not module.enable_rpe
            fn=types.MethodType(tensor_attention,module)
            module._tensor_attention=torch.compile(fn,fullgraph=True,dynamic=True) if dense_compile else fn
            module.forward=types.MethodType(forward,module)
    if dense_compile:
        for module in list(encoder.modules()):
            if isinstance(module,Block):
                # Compile the tensor MLP, not the Point/spconv container.
                module.mlp[0].forward=torch.compile(module.mlp[0].forward,fullgraph=True,dynamic=True)
