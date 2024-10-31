# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:10:13 2024

@author: ZZJ
"""

import os
import logging
from tqdm import tqdm

import torch
import torch.utils.data
import torch.optim as optim

from model.svjformer_m5 import SVJFormer

from common.opt import opts
from common.load_data_h36m import Fusion
from common.loss_pos import define_mpjpe_error_list, mpjpe_cal, test_mpjpe_calculation

from common.loss_pc import infer_zxy_get_error
from common.loss_rotation import infer_rotation_get_error
from common.loss_translation import infer_translation_get_error
from common.multi_view_aggregation import BPMA
from common.bone_decomposition import get_bl_from_pos, get_bd_from_pos 

from common.rotation import infer_rotation_index
from common.translation import infer_translation_index
from common.intrinsic_encoding_decoding import intrinsic_parameters_encoding 
from common.graph_utils import adj_mx_from_skeleton, adj_mx_from_view
from common.utils import define_actions, AccumLoss, get_varialbe, print_error

from common.set_seed import set_seed
import random

def train(opt, actions, train_loader, model, optimizer, epoch):
    model_train = model['pos_train'] 
    return step('train', opt, actions, train_loader, model_train, optimizer, epoch)

def val(opt, actions, test_loader, model):
    model_train = model['pos_train']
    model_test = model['pos_test']
    with torch.no_grad():
        model_test.load_state_dict(model_train.state_dict())
        return step('test', opt, actions, test_loader, model_test)

def step(split, opt, actions, dataloader, model_pos, optimizer=None, epoch=None):
    
    loss_all = {
        'loss_p2d_cpn': AccumLoss(),
        'loss_p2d_pred': AccumLoss(),
        'loss_p3d_pred': AccumLoss(),

        'loss_con': AccumLoss(),
        'loss_p3d_s_pred': AccumLoss(),
        'loss_traj_s_pred': AccumLoss(),
        }
    
    action_2d_cpn_mpjpe_sum = define_mpjpe_error_list(actions)
    action_2d_pred_mpjpe_sum = define_mpjpe_error_list(actions)
    action_3d_pred_mpjpe_sum = define_mpjpe_error_list(actions) 
    
    action_3d_pred_s_mpjpe_sum = define_mpjpe_error_list(actions)
    action_traj_pred_s_mpjpe_sum = define_mpjpe_error_list(actions) 
       
    if split == 'train':
        model_pos.train()
    else:
        model_pos.eval()
    
    t = tqdm(dataloader, 0)
    for i, data in enumerate(t):
        cameras, input_poses, action = data
        inputs_poses, cameras = get_varialbe(split, [input_poses, cameras])
        
        #------------------------Data------------------------------------------
        # shuffle 
        index = [0, 1, 2, 3]
        random.shuffle(index)
        
        # gt_2d
        p2d_gt = inputs_poses[...,:2,index].clone()
        # pre_2d
        p2d_cpn = inputs_poses[...,2:4,index].clone()
        # gt traj
        traj_gt = inputs_poses[...,0:1,4:7,index].clone()
        # gt_3d
        p3d_gt = inputs_poses[...,4:7,index].clone()
        p3d_gt[...,0:1,:,:] = 0.
        # cam
        cam = cameras[...,index]
        
        if split == 'train':
            #----------------Input Normalization--------------------
            ## gt 2d & pre 2d & vis & gt 3d : [b t j c n]
            p2d_gt = intrinsic_parameters_encoding(p2d_gt, cam)
            p2d_cpn = intrinsic_parameters_encoding(p2d_cpn, cam)
                
            #----------------Output Normalization--------------------
            R_gt = infer_rotation_index(p3d_gt, index=0)            # [b n 3 3]
            T_gt = infer_translation_index(traj_gt, R_gt, index=0)  # [b n 1 3]
            
            bl_gt = get_bl_from_pos(p3d_gt[...,0])              # [b t 10]
            factor = bl_gt[...,1:2].unsqueeze(-1).unsqueeze(-1)
            p3d_gt = p3d_gt / factor
            traj_gt = traj_gt / factor
            
            #---------------Model----------------------------------------------
            pred = model_pos(p2d_cpn)  # [b t j c n]
            p2d_pred = p2d_cpn + pred[...,0:2,:]
            p3d_pred = pred[...,2:,:]

        else:
            p2d_gt, p2d_cpn, p2d_pred, cam, p3d_gt, traj_gt, p3d_pred = input_augmentation(p2d_gt, p2d_cpn, cam, p3d_gt, traj_gt, model_pos)
     
        

        p3d_abs_pred_s, R_pred, T_pred, loss_con = BPMA(p2d_pred, p3d_pred, iter_step=2)
        traj_pred_s = p3d_abs_pred_s[:,:,0:1,:]
        p3d_pred_s = p3d_abs_pred_s - traj_pred_s
        
        # loss 
        loss_p2d_cpn = mpjpe_cal(p2d_gt, p2d_cpn)
        loss_p2d_pred = mpjpe_cal(p2d_gt, p2d_pred)
        loss_p3d_pred = mpjpe_cal(p3d_gt, p3d_pred)
        
        loss_p3d_s_pred = mpjpe_cal(p3d_gt[...,0:1], p3d_pred_s.unsqueeze(-1))
        loss_traj_s_pred = mpjpe_cal(traj_gt[...,0:1], traj_pred_s.unsqueeze(-1))
        
        loss_con_np = loss_con.detach().cpu().numpy()
        loss_p2d_cpn_np = loss_p2d_cpn.detach().cpu().numpy()
        loss_p2d_pred_np = loss_p2d_pred.detach().cpu().numpy()
        loss_p3d_pred_np = loss_p3d_pred.detach().cpu().numpy()
        
        loss_p3d_s_pred_np = loss_p3d_s_pred.detach().cpu().numpy()
        loss_traj_s_pred_np = loss_traj_s_pred.detach().cpu().numpy()
        
        B, T, J, _, N = p3d_gt.shape
        Np = B * T * J * N
        Nc = B * T * (J-1) * 2 * N
        Ntraj = B * T * 1 * N
      
        loss_all['loss_con'].update(loss_con_np * Nc, Nc)
        loss_all['loss_p2d_cpn'].update(loss_p2d_cpn_np * Np, Np)
        loss_all['loss_p2d_pred'].update(loss_p2d_pred_np * Np, Np)
        loss_all['loss_p3d_pred'].update(loss_p3d_pred_np * Np, Np) 
        
        loss_all['loss_p3d_s_pred'].update(loss_p3d_s_pred_np * Np, Np) 
        loss_all['loss_traj_s_pred'].update(loss_traj_s_pred_np * Ntraj, Ntraj) 
        
        
        # loss prompting
        t.set_description('2d_cpn({0:,.4f}), 2d_pred({1:,.4f}), 3d_pred({2:,.4f}), con({3:,.4f}), 3d_s_pred({4:,.4f}), traj_s_pred({5:,.4f})'.format(loss_all["loss_p2d_cpn"].avg, loss_all["loss_p2d_pred"].avg, 
                                                                                                       loss_all["loss_p3d_pred"].avg, loss_all["loss_con"].avg, 
                                                                                                       loss_all["loss_p3d_s_pred"].avg, loss_all["loss_traj_s_pred"].avg))
        
        if split == 'train':
            if epoch <= 10:
                loss = loss_p3d_pred + loss_p2d_pred
            else:
                loss = loss_p3d_pred + loss_p2d_pred + loss_con
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            action_2d_cpn_mpjpe_sum = test_mpjpe_calculation(p2d_gt, p2d_cpn, action, action_2d_cpn_mpjpe_sum)
            action_2d_pred_mpjpe_sum = test_mpjpe_calculation(p2d_gt, p2d_pred, action, action_2d_pred_mpjpe_sum)
            action_3d_pred_mpjpe_sum = test_mpjpe_calculation(p3d_gt, p3d_pred, action, action_3d_pred_mpjpe_sum)
            
            action_3d_pred_s_mpjpe_sum = test_mpjpe_calculation(p3d_gt[...,0:1], p3d_pred_s.unsqueeze(-1), action, action_3d_pred_s_mpjpe_sum)
            action_traj_pred_s_mpjpe_sum = test_mpjpe_calculation(traj_gt[...,0:1], traj_pred_s.unsqueeze(-1), action, action_traj_pred_s_mpjpe_sum)
            
    if split == 'train':
        return loss_all
    else:
        mpjpe_2d_cpn = print_error(action_2d_cpn_mpjpe_sum, 'MPJPE', opt.train)
        mpjpe_2d_pred = print_error(action_2d_pred_mpjpe_sum, 'MPJPE', opt.train)
        mpjpe_3d_pred = print_error(action_3d_pred_mpjpe_sum, 'MPJPE', opt.train)
        
        mpjpe_3d_s_pred = print_error(action_3d_pred_s_mpjpe_sum, 'MPJPE', opt.train)
        mpjpe_traj_s_pred = print_error(action_traj_pred_s_mpjpe_sum, 'MPJPE', opt.train)
        
        return mpjpe_2d_cpn, mpjpe_2d_pred, mpjpe_3d_pred, mpjpe_3d_s_pred, mpjpe_traj_s_pred
        
def input_augmentation(p2d_gt, p2d_cpn, cameras, p3d_gt, traj_gt, model_pos):
    joints_left = [4, 5, 6, 11, 12, 13] 
    joints_right = [1, 2, 3, 14, 15, 16]
    
    # gt 2d   
    p2d_gt_non_flip = p2d_gt[:, 0] 
    p2d_gt_flip = p2d_gt[:, 1]
    # cpn 2d
    p2d_cpn_non_flip = p2d_cpn[:, 0] 
    p2d_cpn_flip = p2d_cpn[:, 1]
    # gt 3d
    p3d_gt_non_flip = p3d_gt[:, 0]
    p3d_gt_flip = p3d_gt[:, 1]
    # gt traj
    traj_gt_non_flip = traj_gt[:, 0]
    traj_gt_flip = traj_gt[:, 1]
    # cameras
    cameras_non_flip = cameras[:,0]
    cameras_flip = cameras[:,1]
    
    #-------------------------Non flip-----------------------------------------
    #----------------Intrinsic parameter decoupling--------------------
    p2d_gt_non_flip = intrinsic_parameters_encoding(p2d_gt_non_flip, cameras_non_flip)
    p2d_cpn_non_flip = intrinsic_parameters_encoding(p2d_cpn_non_flip, cameras_non_flip)
    
    #----------------Output Normalization--------------------
    R_gt_non_flip = infer_rotation_index(p3d_gt_non_flip, index=0)              # [b n 3 3]
    T_gt_non_flip = infer_translation_index(traj_gt_non_flip, R_gt_non_flip, index=0)  # [b n 1 3]
    
    
    bl_gt_non_flip = get_bl_from_pos(p3d_gt_non_flip[...,0])              # [b t 10]
    factor_non_flip = bl_gt_non_flip[...,1:2].unsqueeze(-1).unsqueeze(-1)
    p3d_gt_non_flip = p3d_gt_non_flip / factor_non_flip
    traj_gt_non_flip = traj_gt_non_flip / factor_non_flip
    
    #---------------Model----------------------------------------------
    pred_non_flip = model_pos(p2d_cpn_non_flip)  # [b t j c n]
    
    p2d_pred_non_flip = p2d_cpn_non_flip + pred_non_flip[...,0:2,:]
    p3d_pred_non_flip = pred_non_flip[...,2:,:]
    
    return p2d_gt_non_flip, p2d_cpn_non_flip, p2d_pred_non_flip, cameras_non_flip, p3d_gt_non_flip, traj_gt_non_flip, p3d_pred_non_flip
        

if __name__ == '__main__':
    # ----------------------------opt & gpu & seed & log-----------------------
    opt = opts().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    print(opt)
    
    set_seed()
   
    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
    
    
    # ----------------------------------dataset--------------------------------
    actions = define_actions(opt.actions)
    if opt.train:
        train_data = Fusion(opt=opt, is_train=True)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    
    test_data = Fusion(opt=opt, is_train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    
    # ----------------------------------Model----------------------------------
    hm36_parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    A_s = adj_mx_from_skeleton(17, hm36_parent).cuda()
    A_v = adj_mx_from_view(4).cuda()
    
    model = {}
    model['pos_train'] = SVJFormer(A_s, A_v, num_frames=1, num_joints=17, num_views=4, in_chans=2, out_chans=5, embed_dim_ratio=128, depth=6,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1).cuda()
    model['pos_test'] = SVJFormer(A_s, A_v, num_frames=1, num_joints=17, num_views=4, in_chans=2, out_chans=5, embed_dim_ratio=128, depth=6,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.0).cuda()
    
    model_params = 0
    for parameter in model['pos_train'].parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params/1000000, 'Million') 
    
    # ----------------------------------Load-----------------------------------
    model_dict = model['pos_train'].state_dict()
    if opt.resume or opt.test:
        chk_filename = opt.previous_dir + '/epoch_best.pth'
        pre_dict = torch.load(chk_filename)
        model['pos_train'].load_state_dict(pre_dict['model'], strict=False)
    
    # ---------------------------------Optimizer------------------------------- 
    all_param = []
    lr = opt.lr
    all_param += list(model['pos_train'].parameters())
    
    optimizer= optim.Adam(all_param, lr=opt.lr, amsgrad=True)
    
    # ---------------------------------Train-----------------------------------
    for epoch in range(1, opt.nepoch):
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch)
        
        mpjpe_p2d_cpn, mpjpe_p2d_pred, mpjpe_p3d_pred, mpjpe_p3d_pred_s, mpjpe_traj_pred_s = val(opt, actions, test_dataloader, model)
        
        data_threshold = mpjpe_p3d_pred_s + mpjpe_p2d_pred
        if opt.train and data_threshold < opt.previous_best_threshold: 
            chk_path = os.path.join(opt.checkpoint, 'epoch_best.pth')
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model': model['pos_train'].state_dict(),
            }, chk_path)
            opt.previous_best_threshold = data_threshold
    
        if not opt.train:
            print('p3d_refine: %.2f' % (mpjpe_p3d_pred))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, p2d_cpn: %.2f, p2d_pred: %.2f, p3d_pred: %.2f, p3d_pred_s: %.2f, traj_pred_s: %.2f' % (epoch, lr, mpjpe_p2d_cpn, mpjpe_p2d_pred, mpjpe_p3d_pred, mpjpe_p3d_pred_s, mpjpe_traj_pred_s))
            logging.info('epoch: %d, lr: %.7f, p2d_cpn: %.2f, p2d_pred: %.2f, p3d_pred: %.2f, p3d_pred_s: %.2f, traj_pred_s: %.2f' % (epoch, lr, mpjpe_p2d_cpn, mpjpe_p2d_pred, mpjpe_p3d_pred, mpjpe_p3d_pred_s, mpjpe_traj_pred_s))
            print('e: %d, lr: %.7f, p2d_cpn: %.2f, p2d_pred: %.2f, p3d_pred: %.2f, p3d_pred_s: %.2f, traj_pred_s: %.2f' % (epoch, lr, mpjpe_p2d_cpn, mpjpe_p2d_pred, mpjpe_p3d_pred, mpjpe_p3d_pred_s, mpjpe_traj_pred_s))
            print('e: %d, lr: %.7f, p2d_cpn: %.2f, p2d_pred: %.2f, p3d_pred: %.2f, p3d_pred_s: %.2f, traj_pred_s: %.2f' % (epoch, lr, mpjpe_p2d_cpn, mpjpe_p2d_pred, mpjpe_p3d_pred, mpjpe_p3d_pred_s, mpjpe_traj_pred_s))
        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_deacy_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay                
