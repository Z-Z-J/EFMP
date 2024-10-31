# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 11:54:45 2023

@author: ZZJ
"""

import torch

import itertools

from einops import rearrange



def infer_rotation_get_error(pos_3d):
    """
    Args:
        pos_3d: (B, T, J, C, N)
    """
    B, T, J, C, N = pos_3d.shape
    index = torch.tensor(list(itertools.combinations(range(N),2)))
    source = pos_3d[...,index[:,0]].clone()
    target = pos_3d[...,index[:,1]].clone()
    tmp_source = pos_3d[...,index[:,0]].clone()
    tmp_target = pos_3d[...,index[:,1]].clone()
    
    source = rearrange(source, 'b t j c n -> (b n) (j t) c')
    target = rearrange(target, 'b t j c n -> (b n) (j t) c')
    
    muX = torch.mean(target, dim=1, keepdims=True)
    muY = torch.mean(source, dim=1, keepdims=True)
    X0 = target - muX
    Y0 = source - muY
    
    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdims=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdims=True))
    X0 = X0 / normX
    Y0 = Y0 / normY
    
    H = torch.matmul(X0.permute(0, 2, 1), Y0)
    U, s, Vt = torch.linalg.svd(H)
    V = Vt.permute(0, 2, 1)
    R = torch.matmul(V, U.permute(0, 2, 1))
    #out = torch.matmul(source, R)
    R = R.permute(0,2,1)  # [b,3,3]
    source = source.unsqueeze(-1)
    out = torch.einsum('bcd, bndh -> bnch', R, source)
    out = out.squeeze(-1)
    
    out = rearrange(out, '(b n) (j t) c -> b t j c n', b=B, t=T, j=J, c=C)
    
    error = torch.mean(torch.norm(out - tmp_target, dim=len(tmp_target.shape) - 2))
    
    return R, error


def infer_rotation_index(pos_3d, index):
    """
    Args:
        pos_3d: (B, T, J, C, N)
    """
    B, T, J, C, N = pos_3d.shape
    
    target = pos_3d[...,index:index+1].clone()
    target = target.repeat(1, 1, 1, 1, N)
    source = pos_3d.clone()
 
    source = rearrange(source, 'b t j c n -> (b n) (j t) c')
    target = rearrange(target, 'b t j c n -> (b n) (j t) c')
    
    muX = torch.mean(target, dim=1, keepdims=True)
    muY = torch.mean(source, dim=1, keepdims=True)
    X0 = target - muX
    Y0 = source - muY
    
    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdims=True))
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdims=True))
    X0 = X0 / normX
    Y0 = Y0 / normY
    
    H = torch.matmul(X0.permute(0, 2, 1), Y0)
    U, s, Vt = torch.linalg.svd(H)
    V = Vt.permute(0, 2, 1)
    R = torch.matmul(V, U.permute(0, 2, 1))
    R = R.permute(0,2,1)  # [b,3,3]
    
    R = R.contiguous()
    R = R.view(B, -1, 3, 3)
    return R