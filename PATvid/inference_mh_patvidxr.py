import os
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
import cv2 as cv
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import scipy.io as sio

from utils import weights_init_xavier
from mh_models import PATvidx4
from mh_utils import MultiSeqSet, PatvidxrSlowflowSet, _project_rescale_grid
from mh_utils import _to_npimg, _to_tensor, _2d_patch_crop, _2d_patch_combine, move_pixel_value

# prepare netwrk
device = 'cuda:0'
weight_file = './log/sep12vid_L1/best.pth.tar'
pretrained_dict = torch.load(weight_file)
print("Loading weight from: {:s}. It's epoch {:d} with validation PSNR {:.2f}".format(weight_file, pretrained_dict['epoch'], pretrained_dict['psnr']))

net = PATvidx4().to(device)
net = nn.DataParallel(net)
net.eval()
cudnn.benchmark = True
net.load_state_dict(pretrained_dict['state_dict'])

# prepare target folder
target_supf = './results/Sep13_patvidxr_L1/'
if not os.path.exists(target_supf):
    os.mkdir(target_supf)
print("Save all predictions to folder: {}".format(target_supf))

# compressive ratio
csratio = 8 # compressive ratio
res_gap = 4
# patch dividing param
tgt_mn = (2, 2)
m, n = tgt_mn
pad = 48
# attention search parameters
lr_hw = 4
hr_hw = 12

# loop through scenes
for src_folder, src_idx_list in [
    ['/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/skatepark_000021', np.arange(80, 88)], 
    ['/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/motocross_000001', np.arange(104, 112)],
    ['/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/kids__000000', np.arange(232, 240)],
    ['/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/horses__000028', np.arange(232, 240)],
    ['/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/spitzberglauf_000076', np.arange(168, 176)],
    ['/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/ball_000000', np.arange(264, 272)],
    ]:
    ####################
    ### Parameters
    ####################
    print("Loading images from: {}".format(src_folder))
    ptd = PatvidxrSlowflowSet(os.path.join(src_folder, 'gt_rgb'), csratio, 
                              lr_down_fac=res_gap, hr_down_fac=1/np.sqrt(1-csratio/res_gap**2)) # data loader
    _, _, gtimg, _ = ptd[src_idx_list[0]]
    gt_wh = (gtimg.shape[1], gtimg.shape[0])
    del gtimg

    # saving parameter
    tgt_folder = os.path.join(target_supf, os.path.split(src_folder)[1]) # target data folder
    if not os.path.exists(tgt_folder):
        os.mkdir(tgt_folder)
    for tgt_subf in ['lr', 'hr', 'gt', 'pred', 'pred_tuned']:
        tgt_subp = os.path.join(tgt_folder, tgt_subf)
        if not os.path.exists(tgt_subp):
            os.mkdir(tgt_subp)
    print("Saving predicted images to: {:s}".format(tgt_folder))

    ####################
    ### Inference frames
    ####################
    for a in tqdm(src_idx_list):
        #print("Frame {:07d}...".format(a))
        # load and parse input
        lr_list, hr, gt, lr_channel_list = ptd[a]
        lr_trivial_stack = np.stack(lr_list, axis=-1)[...,lr_channel_list]
        u_list = np.mean(lr_trivial_stack, axis=(0,1))
        s_list = np.std(lr_trivial_stack, axis=(0,1))

        # crop images to patches
        lr_stack_list = [_2d_patch_crop(img, tgt_mn, pad//res_gap) for img in lr_list]
        hr_gt_stack = [_2d_patch_crop(img, tgt_mn, pad) for img in [hr, gt]]
        five_stack_list = (*lr_stack_list, *hr_gt_stack)
        
        # find project grid
        bs, h, w = five_stack_list[0].shape
        lr_yys, lr_xxs = _project_rescale_grid((w,h), (w,h), lr_hw, 1)
        hr_yys, hr_xxs = _project_rescale_grid((w,h), (w,h), hr_hw, 1)
        pos_tuple = (np.stack([lr_xxs, lr_yys], axis=0).astype(np.int16), 
                     np.stack([lr_xxs, lr_yys], axis=0).astype(np.int16), 
                     np.stack([hr_xxs, hr_yys], axis=0).astype(np.int16))
        
        # go through network
        torch.cuda.empty_cache()
        # loop through patches
        predp_list = []
        for lr1p, lr2p, lr3p, hrp, gtp in zip(*five_stack_list):
            # convert one patch set to tensors
            lr_ins = [_to_tensor(img)[None].to(device) for img in (lr1p, lr2p, lr3p)]
            hr_in = _to_tensor(hrp)[None].to(device)
            gt_in = _to_tensor(gtp)[None].to(device)
            poss = [torch.from_numpy(pos[np.newaxis,...]).to(device) for pos in pos_tuple]
            # go through network
            with torch.no_grad():
                predp, _ = net(lr_ins, hr_in, 0, poss) # (1, 3, ph, pw)
                predp_list.append(_to_npimg(predp[0].cpu()))
        # clean GPU memory
        torch.cuda.empty_cache()
        
        # combine pred patches
        pred = _2d_patch_combine(np.array(predp_list), tgt_mn, pad)
        # put back channels
        pred = pred[...,np.array([1,0,2])[lr_channel_list]]
        # scale channels
        pred_t = np.copy(pred)
        for b,u,s in zip(range(3), u_list, s_list):
            pred_t[b] = move_pixel_value(pred_t[b], u, s)
        # clip
        pred = np.clip(pred, 0, 1)
        pred_t = np.clip(pred_t, 0, 1)

        # save lr, hr, pred, gt image
        lr_img = np.zeros_like(lr_trivial_stack)
        lr_img[..., lr_channel_list[0]] = lr_list[0]
        imsave(os.path.join(tgt_folder, 'lr', "{:03d}.png".format(a)), 
               img_as_ubyte(np.clip(lr_img,0,1)), check_contrast=False)
        imsave(os.path.join(tgt_folder, 'hr', "{:03d}.png".format(a)), img_as_ubyte(np.clip(hr,0,1)))
        imsave(os.path.join(tgt_folder, 'gt', "{:03d}.png".format(a)), img_as_ubyte(gt))
        imsave(os.path.join(tgt_folder, 'pred', "{:03d}.png".format(a)), img_as_ubyte(pred))
        imsave(os.path.join(tgt_folder, 'pred_tuned', "{:03d}.png".format(a)), img_as_ubyte(pred_t))