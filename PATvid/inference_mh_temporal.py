import os
import json

import numpy as np
import cv2 as cv
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import PAT
from mh_utils import PatTempData, img_pair_to_dual_input, go_through_net, combine_with_pad, move_pixel_value

####################
### Parameters
####################
# source data
src_folder = '/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow_small/horses__000028' # source data folder
src_idx_list = np.arange(224, 248)
csratio = 8 # compressive ratio
ptd = PatTempData(src_folder, csratio) # data loader
_, _, gtimg = ptd[src_idx_list[0]]
gt_wh = (gtimg.shape[1], gtimg.shape[0])
del gtimg

# divide into patches
tgt_mn = (8,16)
m, n = tgt_mn
pad = 24

# attention search parameters
hw = 21
s = 1

# saving parameter
tgt_folder = os.path.join('results/Aug30_slowflow_pred', os.path.split(src_folder)[1]) # target data folder
if not os.path.exists(tgt_folder):
    os.mkdir(tgt_folder)
    
# weight file
weight_file = os.path.join(os.path.split(tgt_folder)[0], 'Aug26_ftd4c_2.pth.tar') # weight file

####################
### Network
####################
net = PAT(1, in_channel=3, num_input=2).to('cuda')
net = nn.DataParallel(net)
cudnn.benchmark = True
pretrained_dict = torch.load(weight_file)
net.load_state_dict(pretrained_dict['state_dict'])
net.eval()
print("Loading data from: {:s}".format(ptd.data_folder))
print("Loading weight from: {:s}. Its validation PSNR: {:.2f}".format(weight_file, pretrained_dict['psnr']))
print("Saving predicted images to: {:s}".format(tgt_folder))


####################
### Inference frames
####################
for a in tqdm(src_idx_list):
    print("Frame {:07d}...".format(a))
    # process input
    limg, rimg, gt = ptd[a]
    #gt_list.append(gt)
    lin, rin, cmean_list, cstd_list = img_pair_to_dual_input(limg, rimg, gt_wh, tgt_mn, pad)
    lin, rin = lin.astype(np.float32), rin.astype(np.float32)

    # go through network
    out_list = []
    for b in range(len(lin)):
        out = go_through_net(net, lin[b:b+1], rin[b:b+1], hw=hw, s=s)
        out_list.append(out[0])
    torch.cuda.empty_cache() # clean GPU memory

    # concatenate patches
    out_list = np.array(out_list) #(m*n, ph, pw, 3)
    ph, pw = out_list.shape[1:3]
    pred = out_list.reshape(m,n,ph,pw,3).transpose(0,2,1,3,4) #(m, ph, n, pw, 3)
    pred = combine_with_pad(pred, pad) #(h, n, pw, 3)
    pred = pred.transpose(1, 2, 0, 3) #(n, pw, h, 3)
    pred = combine_with_pad(pred, pad) #(w, h, 3)
    pred = pred.transpose(1, 0, 2)
    
    # scale back (maybe not needed)
    pred = [move_pixel_value(uci, u, sig) for uci, u, sig in zip(pred.transpose(2, 0, 1), cmean_list, cstd_list)]
    pred = np.stack(pred, axis=-1)
    
    # save pred image
    pred = np.clip(pred, 0, 1)
    imsave(os.path.join(tgt_folder, "{:07d}.png".format(a)), img_as_ubyte(pred))
    
####################
### Parameter storage
####################
# Need 6 hours to finish, run it before sleep
# src_folder = '/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/motocross_000001' # source data folder
# src_idx_list = np.arange(70, 190)
# csratio = 8 # compressive ratio
# ptd = PatTempData(src_folder, csratio) # data loader
# _, _, gtimg = ptd[0]
# gt_wh = (gtimg.shape[1], gtimg.shape[0])
# del gtimg

# # divide into patches
# tgt_mn = (16,16)
# m, n = tgt_mn
# pad = 24

# # attention search parameters
# hw = 22
# s = 1

# # source data
# Took 2 hours
# src_folder = '/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/kids__000000' # source data folder
# src_idx_list = np.arange(200, 264)
# csratio = 8 # compressive ratio
# ptd = PatTempData(src_folder, csratio) # data loader
# _, _, gtimg = ptd[0]
# gt_wh = (gtimg.shape[1], gtimg.shape[0])
# del gtimg

# # divide into patches
# tgt_mn = (8,16)
# m, n = tgt_mn
# pad = 24

# # attention search parameters
# hw = 20
# s = 1

# # source data
# src_folder = '/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/horses__000028' # source data folder
# src_idx_list = np.arange(200, 264)
# csratio = 8 # compressive ratio
# ptd = PatTempData(src_folder, csratio) # data loader
# _, _, gtimg = ptd[0]
# gt_wh = (gtimg.shape[1], gtimg.shape[0])
# del gtimg

# # divide into patches
# tgt_mn = (8,16)
# m, n = tgt_mn
# pad = 24

# # attention search parameters
# hw = 21
# s = 1

# # source data
# src_folder = '/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/spitzberglauf_000076' # source data folder
# src_idx_list = np.arange(120, 184)
# csratio = 8 # compressive ratio
# ptd = PatTempData(src_folder, csratio) # data loader
# _, _, gtimg = ptd[0]
# gt_wh = (gtimg.shape[1], gtimg.shape[0])
# del gtimg

# # divide into patches
# tgt_mn = (16,16)
# m, n = tgt_mn
# pad = 24

# # attention search parameters
# hw = 22
# s = 1

# # source data
# src_folder = '/xdisk/djbrady/mh432/slowflow_dataset/pat_slowflow/ball_000000' # source data folder
# src_idx_list = np.arange(240, 360)
# csratio = 8 # compressive ratio
# ptd = PatTempData(src_folder, csratio) # data loader
# _, _, gtimg = ptd[0]
# gt_wh = (gtimg.shape[1], gtimg.shape[0])
# del gtimg

# # divide into patches
# tgt_mn = (16,16)
# m, n = tgt_mn
# pad = 24

# # attention search parameters
# hw = 22
# s = 1