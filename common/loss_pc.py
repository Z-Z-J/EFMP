# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:30:04 2022

@author: ZZJ
"""

import torch
from einops import rearrange

def infer_zxy_get_error(points3d, ray):
    """
    points3d: [b,t,j,3,n]
    ray: [b,t,j,2,n]
    """
    Bs,T,J,_,N = points3d.shape
    device = points3d.device
    
    points3d = rearrange(points3d, 'b t j c n -> (b n) t j c')
    ray = rearrange(ray, 'b t j c n -> (b n) t j c')
    
    b,t,j,_ = points3d.shape
    
    points3d = points3d.contiguous().view(-1,j,3)
    ray = ray.contiguous().view(-1,j,2)
    
    #
    x_coeff = torch.zeros(b*t,j-1,1, dtype=torch.float32).to(device)
    x_coeff[:,:,0] = ray[:,1:,0] - ray[:,0:1,0]
    
    
    y_coeff = torch.zeros(b*t,j-1,1, dtype=torch.float32).to(device)
    y_coeff[:,:,0] = ray[:,1:,1] - ray[:,0:1,1]
    
    A = torch.cat([x_coeff, y_coeff], dim=1)                                   # [b,2*n,3]
    #
    x_cons = points3d[:,1:,0] - ray[:,1:,0] * points3d[:,1:,2]
    
    y_cons = points3d[:,1:,1] - ray[:,1:,1] * points3d[:,1:,2]
    
    B = torch.cat([x_cons, y_cons], dim=1).unsqueeze(-1)                       # [b,2*n,1]
    
    # solve
    x = torch.matmul(A.permute(0,2,1), A)                                      # [b,t+4,t+4]
    x = torch.matmul(torch.inverse(x), A.permute(0,2,1))
    x = torch.matmul(x,B)
    #x,_,_,_ = torch.linalg.lstsq(A,B, driver='gels')
    error = torch.mean(torch.abs(torch.matmul(A,x)-B))
    
    ray = ray.view(b,t,j,2)
    c_z = x.view(b,t,1,1)
    c_x = c_z * ray[:,:,0:1,0:1]
    c_y = c_z * ray[:,:,0:1,1:2]
    
    traj = torch.cat([c_x, c_y, c_z], dim=-1)
    points3d = points3d.view(b,t,j,-1)
    out_ray_x = (traj[:,:,:,0:1]+points3d[:,:,:,0:1]) / (traj[:,:,:,2:3]+points3d[:,:,:,2:3])
    out_ray_y = (traj[:,:,:,1:2]+points3d[:,:,:,1:2]) / (traj[:,:,:,2:3]+points3d[:,:,:,2:3])
    out_pray = torch.cat([out_ray_x, out_ray_y], dim=-1)
  
    traj = rearrange(traj, '(b n) t j c -> b t j c n', b=Bs, n=N)
    out_pray = rearrange(out_pray, '(b n) t j c -> b t j c n', b=Bs, n=N)
    
    
    return traj, out_pray, error


    
    