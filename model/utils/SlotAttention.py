import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import farthest_point_sample, knn

class SlotAttention(nn.Module):
    def __init__(
        self,
        slot_size, 
        mlp_size, 
        feat_size,
        num_slots=11,
        epsilon=1e-6,
    ):
        super().__init__()
        self.slot_size = slot_size 
        self.epsilon = epsilon
        self.num_iters = 3

        self.norm_feature = nn.LayerNorm(feat_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        
        self.project_k = nn.Linear(feat_size, slot_size, bias=False)
        self.project_v = nn.Linear(feat_size, slot_size, bias=False)
        self.project_q = nn.Linear(slot_size, slot_size, bias=False)

        self.slots_init = nn.Embedding(num_slots, slot_size)
        nn.init.xavier_uniform_(self.slots_init.weight)

        self.gru = nn.GRUCell(self.slot_size, self.slot_size)

        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, slot_size))

    def forward(self, features, sigma):
        B = features.shape[0]
        mu = self.slots_init.weight.expand(B, -1, -1)
        z = torch.randn_like(mu).type_as(features)
        slots_init = mu + z * sigma * mu.detach()
        # `feature` has shape [batch_size, num_feature, inputs_size].
        features = self.norm_feature(features)  
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        B, N, D = v.shape
        slots = slots_init
        # Multiple rounds of attention.
        for i in range(self.num_iters):
            if i == self.num_iters - 1:
                slots = slots.detach() + slots_init - slots_init.detach()
                
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            # Attention
            scale = D ** -0.5
            attn_logits= torch.einsum('bid,bjd->bij', q, k) * scale
            attn = F.softmax(attn_logits, dim=1)

            # Weighted mean
            attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
            attn_wm = attn / attn_sum 
            updates = torch.einsum('bij, bjd->bid', attn_wm, v)            

            # Update slots
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )
            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn
    

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, rel_pos):
        return self.mlp(rel_pos)

class PointTransformerAttention(nn.Module):
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.pos_enc = PositionalEncoding(dim)
        self.softmax = nn.Softmax(dim=-1)  # over k dimension

    def forward(self, q_feat, k_feat, q_pos, k_pos, return_attn=False):
        """
        q_feat: (B, S, D) - slot features
        k_feat: (B, N, D) - point features
        q_pos:  (B, S, 3) - slot positions
        k_pos:  (B, N, 3) - point positions
        """
        B, S, D = q_feat.shape
        N = k_feat.shape[1]

        q = self.q_proj(q_feat)  # (B, S, D)
        k = self.k_proj(k_feat)  # (B, N, D)
        v = self.v_proj(k_feat)  # (B, N, D)

        if self.k is None:
            # 全局注意力
            rel_pos = q_pos.unsqueeze(2) - k_pos.unsqueeze(1)     # (B, S, N, 3)
            pos_enc = self.pos_enc(rel_pos)                       # (B, S, N, D)
            attn = q.unsqueeze(2) - k.unsqueeze(1) + pos_enc      # (B, S, N, D)
            attn = F.softmax(attn.sum(-1), dim=1)                 # (B, S, N)
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True)+1e-6)   # (B, S, N)
            weighted_v = attn_norm.unsqueeze(-1) * v.unsqueeze(1)      # (B, S, N, D)
            out = weighted_v.sum(2)                               # (B, S, D)
        else:
            # 局部注意力：kNN
            idx = knn(k_pos, q_pos, self.k)  # (B, S, k)，索引的是 k_pos/k_feat/v

            # 使用 idx 采样
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, D)     # (B, S, k, D)
            k_neighbors = torch.gather(k.unsqueeze(1).expand(-1, S, -1, -1), 2, idx_exp)             # (B, S, k, D)
            v_neighbors = torch.gather(v.unsqueeze(1).expand(-1, S, -1, -1), 2, idx_exp)             # (B, S, k, D)

            # 相对位置
            idx_pos_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3) # (B, S, k, 3)
            k_pos_neighbors = torch.gather(k_pos.unsqueeze(1).expand(-1, S, -1, -1), 2, idx_pos_exp) # (B, S, k, 3)
            rel_pos = q_pos.unsqueeze(2) - k_pos_neighbors        # (B, S, k, 3)

            # 注意力
            pos_enc = self.pos_enc(rel_pos)                       # (B, S, k, D)
            attn_norm = q.unsqueeze(2) - k_neighbors + pos_enc    # (B, S, k, D)
            attn_norm = self.softmax(attn_norm.sum(-1))           # (B, S, k)
            attn = attn_norm / (attn_norm.sum(dim=1, keepdim=True)+1e-6)
            weighted_v = attn.unsqueeze(-1) * v_neighbors         # (B, S, k, D)
            out = weighted_v.sum(2)                               # (B, S, D)
            
        if return_attn:
            if self.k is None:
                return out, attn 
            else:
                one_hot_idx = torch.zeros((B, idx.shape[1], self.k, k_feat.shape[1]), device=k_feat.device).scatter_(-1, idx.unsqueeze(-1), 1.)
                return out, torch.matmul(attn.unsqueeze(2), one_hot_idx).squeeze(2)
        else:
            return out


class PointSlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, hidden_dim=128, k=None):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.dim = dim

        self.attn = PointTransformerAttention(dim, k=k)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs, pos, return_attn=False):
        """
        inputs: (B, N, D)  特征
        pos:    (B, N, 3)  坐标
        """
        B, N, D = inputs.shape
        inputs = self.norm_inputs(inputs)

        # FPS 初始化 slots
        idx = farthest_point_sample(pos, self.num_slots)  # (B, num_slots)
        batch_indices = torch.arange(B, device=inputs.device).unsqueeze(-1)
        slots = inputs[batch_indices, idx]  # (B, num_slots, D)
        slot_pos = pos[batch_indices, idx]  # (B, num_slots, 3)

        for i in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            # 使用 Point Transformer 的注意力机制
            if return_attn:
                updates, attn = self.attn(slots_norm, inputs, slot_pos, pos, return_attn=True)
            else:
                updates = self.attn(slots_norm, inputs, slot_pos, pos)
                
            # print(attn.shape)
            # GRU + MLP 更新
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, -1, D)

            slots = slots + self.mlp(self.norm_mlp(slots))

        if return_attn:
            return slots, slot_pos, attn
        return slots, slot_pos  # (B, num_slots, D)
