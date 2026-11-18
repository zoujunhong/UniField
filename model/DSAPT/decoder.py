"""Preserve the source query/memory embeddings; add only the specified adaptation."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from .baseline_network import QueryDecoder as BaseDecoder, CrossBlock as BaseBlock

class RoPE3D(nn.Module):
    def __init__(self, head_dim, base=10.0):
        super().__init__()
        if head_dim % 6:
            raise ValueError("3D RoPE requires a head width divisible by six")
        self.axis_dim = head_dim // 3
        self.register_buffer("inv_freq", base ** (-torch.arange(0, self.axis_dim, 2).float() / self.axis_dim))

    def phases(self, xyz):
        angles = xyz.float().unsqueeze(-1) * self.inv_freq
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)

    def forward(self, tensor, phases):
        shape = tensor.shape
        tensor = tensor.float().reshape(*shape[:-1], 3, self.axis_dim)
        a, b = tensor.chunk(2, dim=-1)
        rotated = torch.cat((-b, a), dim=-1)
        return (tensor * phases[0] + rotated * phases[1]).reshape(shape)



class CrossBlock(BaseBlock):
    def forward(self, x, memory, valid, rope=None, query_phases=None, memory_phases=None):
        if rope is None:
            return super().forward(x, memory, valid)
        b, n, dim = x.shape
        q = self.q(self.qnorm(x)).view(b,n,self.heads,self.head_dim).transpose(1,2)
        k,v = self.kv(self.knorm(memory)).view(b,memory.shape[1],2,self.heads,self.head_dim).unbind(2)
        k,v = k.transpose(1,2),v.transpose(1,2)
        dtype=q.dtype
        q,k=rope(q,query_phases).to(dtype),rope(k,memory_phases).to(dtype)
        extra=-self.head_dim % 8
        if extra:q,k,v=(F.pad(t,(0,extra)) for t in (q,k,v))
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.,scale=self.head_dim**-.5)[...,:self.head_dim]
        x=x+self.proj(y.transpose(1,2).reshape(b,n,dim))
        return x+self.ffn(self.fnorm(x))

class QueryDecoder(BaseDecoder):
    def __init__(self,config):
        super().__init__(config)
        dim,heads=config['decoder_dim'],config['heads']
        self.surface_count=config['surface_queries']
        self.joint_fields = bool(config.get('joint_fields', config.get('volume_queries', 0) > 0))
        if self.joint_fields:
            self.volume_pressure_head=nn.Linear(dim,1)
            self.volume_velocity_head=nn.Linear(dim,3)
        self.blocks=nn.ModuleList(CrossBlock(dim,heads,4) for _ in range(config['decoder_depth']))
        self.rope=RoPE3D(dim//heads,config.get('decoder_rope_base',10.)) if config['decoder_rope'] else None
        self.idw_residual=bool(config.get('idw_residual',False))
        if self.idw_residual:
            self.idw_residual_mlp=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
            nn.init.zeros_(self.idw_residual_mlp[-1].weight)
            nn.init.zeros_(self.idw_residual_mlp[-1].bias)
    def forward(self,features,positions,valid,query,query_initial_features=None,surface_count=None):
        if valid is None:valid=torch.ones(features.shape[:2],device=features.device,dtype=torch.bool)
        memory=self.memory_norm(features)+self.memory_position(positions)
        x=self.query_position(query)
        if self.idw_residual:
            if query_initial_features is None:raise ValueError('IDW residual requires interpolated features')
            x=x+self.idw_residual_mlp(query_initial_features)
        qp=self.rope.phases(query[...,:3]) if self.rope is not None else None
        mp=self.rope.phases(positions) if self.rope is not None else None
        # Batch-one memory has no padding: the all-True mask is redundant.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for block in self.blocks:x=block(x,memory,valid,self.rope,qp,mp)
        x=self.output_norm(x)
        ns=self.surface_count if surface_count is None else surface_count
        result = {'surface_p': self.surface_head(x[:, :ns])}
        if self.joint_fields:
            result.update(volume_p=self.volume_pressure_head(x[:, ns:]), volume_u=self.volume_velocity_head(x[:, ns:]))
        elif ns != x.shape[1]:
            raise ValueError('Volume queries require joint_fields=True.')
        return result
