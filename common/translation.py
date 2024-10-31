# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:42:17 2023

@author: ZZJ
"""
import torch

import itertools

from einops import rearrange

def infer_translation(traj, rotation):
    """
    Args:
        traj: (B, T, 1, C, N)
        rotation:
    """
    B, T, J, C, N = traj.shape 
    index = torch.tensor(list(itertools.combinations(range(N),2)))
    source = traj[...,index[:,0]].clone()
    target = traj[...,index[:,1]].clone()
    
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


def infer_translation_index(traj, rotation, index):
    """
    Args:
        traj: [B, T, 1, C, N]
        rotation: [b N 3 3]
    """
    B, T, J, C, N = traj.shape 
    
    source = traj[...,index:index+1].clone()
    source = source.repeat(1, 1, 1, 1, N)
    target = traj.clone()
    
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


def infer_relative_translation(traj, relative_rotation, t=0):
    """
    Args:
        traj: (B, T, 1, C, N)
        rotation: (B,N,N,3,3)
    Returns:
        translation: (B,N,N,1,3)
    """
    B, T, J, C, N = traj.shape
    device = traj.device
    
    relative_translation = torch.zeros([B,N,N,1,3], dtype=torch.float32).to(device)
    
    for i in range(N):
        for j in range(N):
            source = traj[:,t,:,:,i].clone()
            target = traj[:,t,:,:,j].clone()
            R = relative_rotation[:,i,j,:,:].clone()
            
            source = source.unsqueeze(-1)

            out = torch.einsum('bcd, bndh -> bnch', R, source)
            out = out.squeeze(-1)
            translation = target - out
            
            relative_translation[:,i,j,:,:] = translation
    
    return relative_translation



        
            
            
            
            
            
    
    
    