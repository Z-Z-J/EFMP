# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:36:19 2024

@author: ZZJ
"""
import torch

from common.loss_pc import infer_zxy_get_error
from common.loss_rotation import infer_rotation_get_error, infer_rotation_index
from common.loss_translation import infer_translation_get_error, infer_translation_index
from common.bone_decomposition import get_bl_from_pos, get_bd_from_pos, get_pos_from_bl_bd, estimate_bl, estimate_bd

def BPMA(p2d_pred, p3d_pred, iter_step=1):
    """
    p2d_pred: [b t j c n]
    p3d_pred: [b t j c n]
    """
    #
    traj_pred, _, loss_pc = infer_zxy_get_error(p3d_pred, p2d_pred)            # [b t 1 c n]
    _, loss_rotation = infer_rotation_get_error(p3d_pred)          
    # 
    R_pred = infer_rotation_index(p3d_pred, index=0)                           # [b n 3 3]
    T_pred = infer_translation_index(traj_pred, R_pred, index=0)               # [b n 1 3]
    #
    p3d_abs_pred = p3d_pred + traj_pred
    rp2d_pred = p3d_abs_pred[...,0:2,:] / p3d_abs_pred[...,2:3,:]
    #
    error = torch.norm(rp2d_pred - p2d_pred, dim=len(p2d_pred.shape) - 2, keepdim=True)
    W = (1 / (error + 1e-6)) / ( 1 / (error + 1e-6)).sum(dim=-1, keepdim=True)
    p3d_abs_pred_single = torch.einsum('bncd, btjdn -> btjcn', R_pred, p3d_abs_pred)
    p3d_abs_pred_single = p3d_abs_pred_single + T_pred.permute(0,2,3,1).unsqueeze(1)
    p3d_abs_pred_single = (W * p3d_abs_pred_single).sum(dim=-1)
    #
    traj_pred_single = p3d_abs_pred_single[:,:,0:1,:]
    p3d_pred_single = p3d_abs_pred_single - traj_pred_single
    #
    bl_init = get_bl_from_pos(p3d_abs_pred_single)
    bd_init = get_bd_from_pos(p3d_abs_pred_single)

    bl_list = [bl_init]
    bd_list = [bd_init]

    for i in range(2):
        bl_optimize = estimate_bl(bd_list[-1], p3d_pred_single)
        bl_list.append(bl_optimize)
        bd_optimize = estimate_bd(bl_list[-1], p3d_pred_single)
        bd_list.append(bd_optimize)
        
    p3d_pred_single_sym = get_pos_from_bl_bd(bl_list[-1], bd_list[-1])
    
    p3d_abs_pred_single_sym = p3d_pred_single_sym + traj_pred_single
    
    
    loss_con = loss_pc + loss_rotation
    
    return p3d_abs_pred_single_sym, R_pred, T_pred, loss_con
    
    