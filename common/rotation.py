# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 11:54:45 2023

@author: ZZJ
"""

import torch

import itertools

from einops import rearrange

def infer_rotation(pos_3d):
    """
    Args:
        pos_3d: (B, T, J, C, N)
    """
    B, T, J, C, N = pos_3d.shape
    index = torch.tensor(list(itertools.combinations(range(N),2)))
    source = pos_3d[...,index[:,0]].clone()
    target = pos_3d[...,index[:,1]].clone()
 
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

def infer_rotation_index(pos_3d, index):
    """
    Args:
        pos_3d: (B, T, J, C, N)
    """
    B, T, J, C, N = pos_3d.shape
    
    source = pos_3d[...,index:index+1].clone()
    source = source.repeat(1, 1, 1, 1, N)
    target = pos_3d.clone()
 
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

def infer_relative_rotation(pos_3d, t=0):
    """
    Args:
        pos_3d: (B,T,J,C,N)
    Returns:
        rotation: (B,N,N,3,3)
    """
    B, T, J, C, N = pos_3d.shape
    device = pos_3d.device
    
    relative_rotation = torch.zeros([B,N,N,3,3], dtype=torch.float32).to(device)
    
    for i in range(N):
        for j in range(N):
            # b,j,c
            source = pos_3d[:,t,:,:,i].clone()
            target = pos_3d[:,t,:,:,j].clone()
            
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
            R = R.permute(0,2,1).contiguous()  # [b,3,3]
            
            relative_rotation[:,i,j] = R
    
    return relative_rotation


            
    
    
    
    
    
    
    
    
    
    
