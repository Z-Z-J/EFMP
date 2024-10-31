# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:03:16 2024

@author: ZZJ
"""
import torch
import torch.nn as nn

from functools import partial

from einops import rearrange

from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, A_s, A_v, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 5, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(3*dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        
        self.M = nn.Parameter(torch.zeros(size=(1, A_s.size(0), A_v.size(0), dim), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)
        
        
        self.A_s = A_s.view(1, 1, A_s.shape[0], A_s.shape[1])
        self.A_v = A_v.view(1, 1, A_v.shape[0], A_v.shape[1])
        
        self.adj_s = nn.Parameter(torch.ones_like(self.A_s))
        nn.init.constant_(self.adj_s, 1e-6)
        
        self.adj_v = nn.Parameter(torch.ones_like(self.A_v))
        nn.init.constant_(self.adj_v, 1e-6)
        
        
    def forward(self, x):
        B, T, J, N, C = x.shape
        x = rearrange(x, 'b t j n c -> (b t) j n c')   # [BT, J, N, C]
        
        # KTE
        qkv = self.qkv(x)
        #qkv = self.M * qkv
        qkv = qkv.view(B*T, J, N, C, 5).permute(4, 0, 1, 2, 3)  # [3, BT, J, N, C]
        q, k, vs, vv, vsv = qkv[0], qkv[1], qkv[2], qkv[3], qkv[4]   # [BT, J, N, C]
        
        q = q * self.M
        k = k * self.M
        vs = vs * self.M
        vv = vv * self.M
        vsv = vsv * self.M
    
        # PKE
        q_s = rearrange(q, 'bt j n (h c) -> (bt h) n j c', h=self.num_heads)  # [(BT,H), N, J, C//8]
        k_s = rearrange(k, 'bt j n (h c) -> (bt h) n c j', h=self.num_heads)  # [(BT,H), N, C//8, J]
        
        q_v = rearrange(q, 'bt j n (h c) -> (bt h) j n c', h=self.num_heads)  # [(BT,H), J, N, C//8]
        k_v = rearrange(k, 'bt j n (h c) -> (bt h) j c n', h=self.num_heads)  # [(BT,H), J, C//8, N]
        
        att_s_o = (q_s @ k_s) * self.scale  # [(BT,H),N, J, J]
        att_v_o = (q_v @ k_v) * self.scale  # [(BT,H),J, N, N]
        
        att_s = att_s_o.softmax(-1) 
        att_v = att_v_o.softmax(-1)
       
        # SVJA
        att_sv = att_s.unsqueeze(-1).repeat(1, 1, 1, 1, N)  # [(BT,H), N, J, J, N]
        att_vs = att_v.unsqueeze(-2).repeat(1, 1, 1, J, 1)  # [(BT,H), J, N, J, N]
        
        attn = att_sv.permute(0,2,1,3,4) * att_vs  # [(BT, H), J, N, J, N]
        attn = self.attn_drop(attn)
        
        vsv = rearrange(vsv, 'bt j n (h c) -> (bt h) j n c', h=self.num_heads)  # [(BT,H), J, N, C//8]
        x_vsv = attn.unsqueeze(-1) * vsv.unsqueeze(1).unsqueeze(1)
        x_vsv = x_vsv.sum(dim=[-2,-3])
        x_vsv = rearrange(x_vsv, '(bt h) j n c -> bt j n (h c) ', h=self.num_heads)  # [BT, J, N, C]
        
        # Local
        adj_s = self.A_s + self.adj_s
        adj_v = self.A_v + self.adj_v
        
        adj_s = (adj_s.transpose(-2, -1) + adj_s) / 2.
        adj_v = (adj_v.transpose(-2, -1) + adj_v) / 2.
        
        att_s = (adj_s + att_s_o).softmax(-1) 
        att_v = (adj_v + att_v_o).softmax(-1)
        
        # local spatial
        vs = rearrange(vs, 'bt j n (h c) -> (bt h) n j c', h=self.num_heads)
        x_vs = torch.matmul(att_s, vs)
        x_vs = rearrange(x_vs, '(bt h) n j c -> bt j n (h c)', h=self.num_heads)
        
        # local view
        vv = rearrange(vv, 'bt j n (h c) -> (bt h) j n c', h=self.num_heads)
        x_vv = torch.matmul(att_v, vv)
        x_vv = rearrange(x_vv, '(bt h) j n c -> bt j n (h c)', h=self.num_heads)
    
        x = torch.cat([x_vsv, x_vs, x_vv], dim=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = rearrange(x, '(b t) j n c -> b t j n c', t=T)
        
        return x

class Block(nn.Module):
    def __init__(self, A_s, A_v, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            A_s, A_v, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class SVJFormer(nn.Module):
    def __init__(self, A_s, A_v, num_frames=9, num_joints=17, num_views=4, in_chans=2, out_chans=3, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints, num_views, embed_dim_ratio))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth
        
        self.SVJblocks = nn.ModuleList([
            Block(
                A_s, A_v, dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
           
        self.norm = norm_layer(embed_dim_ratio)
    
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, out_chans),
        )    
    
    def forward(self, x):
        b, t, j, c, n = x.shape
        x = rearrange(x, 'b t j c n -> b t j n c')    
        
        # Linear mapping
        x = self.patch_to_embedding(x)
        
        # Position Embedding
        x += self.pos_embed
        x = self.pos_drop(x)
        
        # Self-Attention
        for blk in self.SVJblocks:
            x = blk(x)
            
        x = self.head(x)
        x = rearrange(x, 'b t j n c -> b t j c n')
        return x
  

if __name__ == "__main__":
    from common.graph_utils import adj_mx_from_skeleton, adj_mx_from_view
    
    hm36_parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    
    A_s = adj_mx_from_skeleton(17, hm36_parent)
    A_v = adj_mx_from_view(2)
    
    model_pos = SVJFormer(A_s, A_v, num_frames=1, num_joints=17, num_views=2, in_chans=2, out_chans=3,
             embed_dim_ratio=128, depth=6, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)
            
    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params/1000000, 'Million')        
            
    x = torch.randn(1,1,17,2,2)
    y = model_pos(x)
    
    from thop import profile
    # flops = 2*macs
    macs, params = profile(model_pos, inputs=(x,))
    print(2*macs /(1000000))
    print(params /(1000000))    
        
        