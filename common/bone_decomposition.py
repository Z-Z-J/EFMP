# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:37:13 2024

@author: ZZJ
"""
import torch


def distance(position1, position2):
    vector = torch.abs(position1 - position2)
    return torch.norm(vector, dim=-1)


def estimate_bl(bd, points3d):
    ############### Estimate bone bl ##########
    """
    Inputs:
        bd: [B T 16 3]
        points3d: [B T J C]
    Outputs:
        bl: [B T 10]
    """
    Bs, T, J, C = points3d.shape
    device = points3d.device
    
    p_pos = points3d - points3d[:,:,0:1,:]  
    p_pos = p_pos.view(Bs*T, J*C)
    bd = bd.view(Bs*T, -1, C)

    A = torch.zeros(Bs*T, (J-1)*3, 10, dtype=torch.float32).to(device)
     
    A[:,0:9,0] = bd[:,0].repeat(1,3)  
    A[:,9:18,0] = bd[:,3].repeat(1,3)
    
    A[:,3:9,1] = bd[:,1].repeat(1,2) 
    A[:,12:18,1] = bd[:,4].repeat(1,2)
    
    A[:,6:9,2] = bd[:,2]
    A[:,15:18,2] = bd[:,5]
    
    A[:,18:30,3] = bd[:,6].repeat(1,4)
    
    A[:,21:30,4] = bd[:,7].repeat(1,3)
    
    A[:,24:30,5] = bd[:,8].repeat(1,2)
    
    A[:,27:30,6] = bd[:,9]
    
    A[:,30:39,3] = bd[:,6].repeat(1,3)
    A[:,30:39,4] = bd[:,7].repeat(1,3)
    A[:,30:39,7] = bd[:,10].repeat(1,3)
    A[:,33:39,8] = bd[:,11].repeat(1,2)
    A[:,36:39,9] = bd[:,12]
    
    A[:,39:48,3] = bd[:,6].repeat(1,3)
    A[:,39:48,4] = bd[:,7].repeat(1,3)
    A[:,39:48,7] = bd[:,13].repeat(1,3)
    A[:,42:48,8] = bd[:,14].repeat(1,2)
    A[:,45:48,9] = bd[:,15]
    
    B = p_pos[:,3:].unsqueeze(-1)
        
    # solve
    x = torch.matmul(A.permute(0,2,1), A)                                      # [b,t+4,t+4]
    x = torch.matmul(torch.inverse(x), A.permute(0,2,1))
    x = torch.matmul(x,B)                                                      # [b, 10, 1] 
    
    bl = x.view(Bs, T, 10)
    return bl


def estimate_bd(bl, p3d_abs_cpn):
    ############### Estimate bone direction ##########
    """
    Inputs:
        bl: [B T 10]
        p3d_abs_cpn: [B T J C]
    Outputs:
        bd: [B T 16 3]
    """
    p_pos = p3d_abs_cpn - p3d_abs_cpn[:,:,0:1,:]
    #p_pos = p3d_abs_cpn
    # 0
    q_pos_0 = p_pos[:,:,0:1,:].clone()
    # 1
    p1_q0_bd = (p_pos[...,1:2,:] - q_pos_0) / torch.norm(p_pos[...,1:2,:] - q_pos_0, dim=-1, keepdim=True)
    q_pos_1 = q_pos_0 + p1_q0_bd * bl[:,:,0:1].unsqueeze(-1)
    # 2
    p2_q1_bd = (p_pos[...,2:3,:] - q_pos_1) / torch.norm(p_pos[...,2:3,:] - q_pos_1, dim=-1, keepdim=True)
    q_pos_2 = q_pos_1 + p2_q1_bd * bl[:,:,1:2].unsqueeze(-1) 
    # 3
    p3_q2_bd = (p_pos[...,3:4,:] - q_pos_2) / torch.norm(p_pos[...,3:4,:] - q_pos_2, dim=-1, keepdim=True)
    q_pos_3 = q_pos_2 + p3_q2_bd * bl[:,:,2:3].unsqueeze(-1) 
    
    # 4
    p4_q0_bd = (p_pos[...,4:5,:] - q_pos_0) / torch.norm(p_pos[...,4:5,:] - q_pos_0, dim=-1, keepdim=True)
    q_pos_4 = q_pos_0 + p4_q0_bd * bl[:,:,0:1].unsqueeze(-1)
    # 5
    p5_q4_bd = (p_pos[...,5:6,:] - q_pos_4) / torch.norm(p_pos[...,5:6,:] - q_pos_4, dim=-1, keepdim=True)
    q_pos_5 = q_pos_4 + p5_q4_bd * bl[:,:,1:2].unsqueeze(-1)
    # 6
    p6_q5_bd = (p_pos[...,6:7,:] - q_pos_5) / torch.norm(p_pos[...,6:7,:] - q_pos_5, dim=-1, keepdim=True)
    q_pos_6 = q_pos_5 + p6_q5_bd * bl[:,:,2:3].unsqueeze(-1)
    
    # 7
    p7_q0_bd = (p_pos[...,7:8,:] - q_pos_0) / torch.norm(p_pos[...,7:8,:] - q_pos_0, dim=-1, keepdim=True)
    q_pos_7 = q_pos_0 + p7_q0_bd * bl[:,:,3:4].unsqueeze(-1)
    # 8
    p8_q7_bd = (p_pos[...,8:9,:] - q_pos_7) / torch.norm(p_pos[...,8:9,:] - q_pos_7, dim=-1, keepdim=True)
    q_pos_8 = q_pos_7 + p8_q7_bd * bl[:,:,4:5].unsqueeze(-1)
    # 9
    p9_q8_bd = (p_pos[...,9:10,:] - q_pos_8) / torch.norm(p_pos[...,9:10,:] - q_pos_8, dim=-1, keepdim=True)
    q_pos_9 = q_pos_8 + p9_q8_bd * bl[:,:,5:6].unsqueeze(-1)
    # 10
    p10_q9_bd = (p_pos[...,10:11,:] - q_pos_9) / torch.norm(p_pos[...,10:11,:] - q_pos_9, dim=-1, keepdim=True)
    q_pos_10 = q_pos_9 + p10_q9_bd * bl[:,:,6:7].unsqueeze(-1)
    
    # 11
    p11_q8_bd = (p_pos[...,11:12,:] - q_pos_8) / torch.norm(p_pos[...,11:12,:] - q_pos_8, dim=-1, keepdim=True)
    q_pos_11 = q_pos_8 + p11_q8_bd * bl[:,:,7:8].unsqueeze(-1)
    # 12
    p12_q11_bd = (p_pos[...,12:13,:] - q_pos_11) / torch.norm(p_pos[...,12:13,:] - q_pos_11, dim=-1, keepdim=True)
    q_pos_12 = q_pos_11 + p12_q11_bd * bl[:,:,8:9].unsqueeze(-1)
    # 13
    p13_q12_bd = (p_pos[...,13:14,:] - q_pos_12) / torch.norm(p_pos[...,13:14,:] - q_pos_12, dim=-1, keepdim=True)
    q_pos_13 = q_pos_12 + p13_q12_bd * bl[:,:,9:10].unsqueeze(-1)
    
    # 14
    p14_q8_bd = (p_pos[...,14:15,:] - q_pos_8) / torch.norm(p_pos[...,14:15,:] - q_pos_8, dim=-1, keepdim=True)
    q_pos_14 = q_pos_8 + p14_q8_bd * bl[:,:,7:8].unsqueeze(-1)
    # 15
    p15_q14_bd = (p_pos[...,15:16,:] - q_pos_14) / torch.norm(p_pos[...,15:16,:] - q_pos_14, dim=-1, keepdim=True)
    q_pos_15 = q_pos_14 + p15_q14_bd * bl[:,:,8:9].unsqueeze(-1)
    # 16
    p16_q15_bd = (p_pos[...,16:17,:] - q_pos_15) / torch.norm(p_pos[...,16:17,:] - q_pos_15, dim=-1, keepdim=True)
    q_pos_16 = q_pos_15 + p16_q15_bd * bl[:,:,9:10].unsqueeze(-1)
    
    bd = torch.cat([p1_q0_bd, p2_q1_bd, p3_q2_bd, p4_q0_bd, p5_q4_bd, p6_q5_bd, p7_q0_bd, p8_q7_bd,
                    p9_q8_bd, p10_q9_bd, p11_q8_bd, p12_q11_bd, p13_q12_bd, p14_q8_bd, p15_q14_bd, p16_q15_bd], dim=-2)
    
    return bd


def get_bl_from_pos(position_3d):
    """
    Function:
        Get bone length from points3d.
    Arguments:
        points3d: [b, t, 17, 3]
    Returns:
        bl: [b,t,10]
    """
    b,t,j,c = position_3d.shape
    device = position_3d.device
    length = torch.zeros(b, t, 10, dtype=torch.float32).to(device)
    # root-hip
    length[:, :, 0] = (distance(position_3d[:, :, 0], position_3d[:, :, 1]) + distance(position_3d[:, :, 0], position_3d[:, :, 4])) / 2
    # hip-knee
    length[:, :, 1] = (distance(position_3d[:, :, 1], position_3d[:, :, 2]) + distance(position_3d[:, :, 4], position_3d[:, :, 5])) / 2
    # knee-foot
    length[:, :, 2] = (distance(position_3d[:, :, 2], position_3d[:, :, 3]) + distance(position_3d[:, :, 5], position_3d[:, :, 6])) / 2
    # root-spine
    length[:, :, 3] =  distance(position_3d[:, :, 0], position_3d[:, :, 7])
    # spine-spine1
    length[:, :, 4] =  distance(position_3d[:, :, 7], position_3d[:, :, 8])
    # spine1-neck
    length[:, :, 5] =  distance(position_3d[:, :, 8], position_3d[:, :, 9])
    # neck-head
    length[:, :, 6] =  distance(position_3d[:, :, 9], position_3d[:, :, 10])
    # neck-shoulder
    length[:, :, 7] = (distance(position_3d[:, :, 8], position_3d[:, :, 11]) + distance(position_3d[:, :, 8], position_3d[:, :, 14])) / 2
    # shoulder-elbow
    length[:, :, 8] = (distance(position_3d[:, :, 11], position_3d[:, :, 12]) + distance(position_3d[:, :, 14], position_3d[:, :, 15])) / 2
    # elbow-wrist
    length[:, :, 9] = (distance(position_3d[:, :, 12], position_3d[:, :, 13]) + distance(position_3d[:, :, 15], position_3d[:, :, 16])) / 2
    
    return length


def get_bd_from_pos(position_3d):
    """
    Function:
        Get bone length from points3d.
    Arguments:
        points3d: [b, t, 17, 3]
    Returns:
        bl: [b,t,16,3]
    """
    b,t,j,c = position_3d.shape
    device = position_3d.device
    direction = torch.zeros(b, t, 16, 3, dtype=torch.float32).to(device)
    # 0-1
    direction[:, :, 0] =  (position_3d[:, :, 1] - position_3d[:, :, 0]) / distance(position_3d[:, :, 0], position_3d[:, :, 1]).unsqueeze(-1)
    # 1-2
    direction[:, :, 1] =  (position_3d[:, :, 2] - position_3d[:, :, 1]) / distance(position_3d[:, :, 1], position_3d[:, :, 2]).unsqueeze(-1)
    # 2-3
    direction[:, :, 2] =  (position_3d[:, :, 3] - position_3d[:, :, 2]) / distance(position_3d[:, :, 2], position_3d[:, :, 3]).unsqueeze(-1)
    # 0-4
    direction[:, :, 3] =  (position_3d[:, :, 4] - position_3d[:, :, 0]) / distance(position_3d[:, :, 0], position_3d[:, :, 4]).unsqueeze(-1)
    # 4-5
    direction[:, :, 4] =  (position_3d[:, :, 5] - position_3d[:, :, 4]) / distance(position_3d[:, :, 4], position_3d[:, :, 5]).unsqueeze(-1)
    # 5-6
    direction[:, :, 5] =  (position_3d[:, :, 6] - position_3d[:, :, 5]) / distance(position_3d[:, :, 5], position_3d[:, :, 6]).unsqueeze(-1)
    # 0-7
    direction[:, :, 6] =  (position_3d[:, :, 7] - position_3d[:, :, 0]) / distance(position_3d[:, :, 0], position_3d[:, :, 7]).unsqueeze(-1)
    # 7-8
    direction[:, :, 7] =  (position_3d[:, :, 8] - position_3d[:, :, 7]) / distance(position_3d[:, :, 7], position_3d[:, :, 8]).unsqueeze(-1)
    # 8-9
    direction[:, :, 8] =  (position_3d[:, :, 9] - position_3d[:, :, 8]) / distance(position_3d[:, :, 8], position_3d[:, :, 9]).unsqueeze(-1)
    # 9-10
    direction[:, :, 9] =  (position_3d[:, :, 10] - position_3d[:, :, 9]) / distance(position_3d[:, :, 9], position_3d[:, :, 10]).unsqueeze(-1)
    # 8-11
    direction[:, :, 10] =  (position_3d[:, :, 11] - position_3d[:, :, 8]) / distance(position_3d[:, :, 8], position_3d[:, :, 11]).unsqueeze(-1)
    # 11-12
    direction[:, :, 11] =  (position_3d[:, :, 12] - position_3d[:, :, 11]) / distance(position_3d[:, :, 11], position_3d[:, :, 12]).unsqueeze(-1)
    # 12-13
    direction[:, :, 12] =  (position_3d[:, :, 13] - position_3d[:, :, 12]) / distance(position_3d[:, :, 12], position_3d[:, :, 13]).unsqueeze(-1)
    # 8-14
    direction[:, :, 13] =  (position_3d[:, :, 14] - position_3d[:, :, 8]) / distance(position_3d[:, :, 8], position_3d[:, :, 14]).unsqueeze(-1)
    # 14-15
    direction[:, :, 14] =  (position_3d[:, :, 15] - position_3d[:, :, 14]) / distance(position_3d[:, :, 14], position_3d[:, :, 15]).unsqueeze(-1)
    # 15-16
    direction[:, :, 15] =  (position_3d[:, :, 16] - position_3d[:, :, 15]) / distance(position_3d[:, :, 15], position_3d[:, :, 16]).unsqueeze(-1)
 
    return direction


def get_pos_from_bl_bd(bl, bd):
    """
    Function:
        Get points3d from bone length and direction.
    Arguments:
        bl: [b t 10]
        bd: [b, t, 16, 3]
    Returns:
        position_3d: [b,t,17,3]
    """
    b,t,j,c = bd.shape
    device = bd.device
    position_3d = torch.zeros(b, t, 17, 3, dtype=torch.float32).to(device)
    # 1
    position_3d[:,:,1,:] = position_3d[:,:,0,:] + bl[:,:,0:1] * bd[:,:,0,:]
    # 2
    position_3d[:,:,2,:] = position_3d[:,:,1,:] + bl[:,:,1:2] * bd[:,:,1,:]
    # 3
    position_3d[:,:,3,:] = position_3d[:,:,2,:] + bl[:,:,2:3] * bd[:,:,2,:]
    # 4
    position_3d[:,:,4,:] = position_3d[:,:,0,:] + bl[:,:,0:1] * bd[:,:,3,:]
    # 5
    position_3d[:,:,5,:] = position_3d[:,:,4,:] + bl[:,:,1:2] * bd[:,:,4,:]
    # 6
    position_3d[:,:,6,:] = position_3d[:,:,5,:] + bl[:,:,2:3] * bd[:,:,5,:]
    # 7
    position_3d[:,:,7,:] = position_3d[:,:,0,:] + bl[:,:,3:4] * bd[:,:,6,:]
    # 8
    position_3d[:,:,8,:] = position_3d[:,:,7,:] + bl[:,:,4:5] * bd[:,:,7,:]
    # 9
    position_3d[:,:,9,:] = position_3d[:,:,8,:] + bl[:,:,5:6] * bd[:,:,8,:]
    # 10
    position_3d[:,:,10,:] = position_3d[:,:,9,:] + bl[:,:,6:7] * bd[:,:,9,:]
    # 11
    position_3d[:,:,11,:] = position_3d[:,:,8,:] + bl[:,:,7:8] * bd[:,:,10,:]
    # 12
    position_3d[:,:,12,:] = position_3d[:,:,11,:] + bl[:,:,8:9] * bd[:,:,11,:]
    # 13
    position_3d[:,:,13,:] = position_3d[:,:,12,:] + bl[:,:,9:10] * bd[:,:,12,:]
    # 14
    position_3d[:,:,14,:] = position_3d[:,:,8,:] + bl[:,:,7:8] * bd[:,:,13,:]
    # 15
    position_3d[:,:,15,:] = position_3d[:,:,14,:] + bl[:,:,8:9] * bd[:,:,14,:]
    # 16
    position_3d[:,:,16,:] = position_3d[:,:,15,:] + bl[:,:,9:10] * bd[:,:,15,:]
    
    return position_3d
