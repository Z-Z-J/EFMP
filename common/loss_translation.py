# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:42:17 2023

@author: ZZJ
"""
import torch

import itertools

from einops import rearrange

def infer_translation_get_error(traj, rotation):
    """
    Args:
        traj: (B, T, 1, C, N)
        rotation:
    """
    B, T, J, C, N = traj.shape 
    index = torch.tensor(list(itertools.combinations(range(N),2)))
    source = traj[...,index[:,0]].clone()
    target = traj[...,index[:,1]].clone()
    tmp_source = traj[...,index[:,0]].clone()
    tmp_target = traj[...,index[:,1]].clone()
    
    source = rearrange(source, 'b t j c n -> (b n) (j t) c')
    target = rearrange(target, 'b t j c n -> (b n) (j t) c')
    
    source = source.unsqueeze(-1)
    out = torch.einsum('bcd, bndh -> bnch', rotation, source)
    out = out.squeeze(-1)
    
    translation = torch.mean(target - out, dim=-2).unsqueeze(1)
    out = out + translation
    
    out = rearrange(out, '(b n) (j t) c -> b t j c n', b=B, t=T, j=J, c=C)
    error = torch.mean(torch.norm(tmp_target - out, dim=len(tmp_target.shape) - 2))    
    return translation, error



def infer_translation_index(traj, rotation, index):
    """
    Args:
        traj: [B, T, 1, C, N]
        rotation: [b N 3 3]
    """
    B, T, J, C, N = traj.shape 
    
    target = traj[...,index:index+1].clone()
    target = target.repeat(1, 1, 1, 1, N)
    source = traj.clone()
    
    source = rearrange(source, 'b t j c n -> (b n) (j t) c')
    target = rearrange(target, 'b t j c n -> (b n) (j t) c')
    rotation = rotation.view(-1, 3, 3)
    
    source = source.unsqueeze(-1)

    out = torch.einsum('bcd, bndh -> bnch', rotation, source)
    out = out.squeeze(-1)
    translation = torch.mean(target - out, dim=-2).unsqueeze(1)
    translation = translation.contiguous()
    translation = translation.view(B, -1, 1, 3)
    return translation
