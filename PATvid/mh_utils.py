import os
import json

import numpy as np
import cv2 as cv
from skimage.io import imread
from skimage.util import img_as_float32, img_as_ubyte, img_as_float

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

##########
### Image dataloaders
##########
class ImgBareSet():
    """
    A barebone loading method to a image dataset
    """
    def __init__(self, data_folder, img_fn, *,
                 n_samples=None, start_ind=0, as_float=False):
        # parse inputs
        assert os.path.isdir(data_folder), \
            "{} is not a valid folder".format(data_folder)
        self.data_folder = data_folder
        self.img_fn = img_fn
        max_n = len(os.listdir(self.data_folder))
        if n_samples is None:
            n_samples = max_n
        assert n_samples <= max_n, \
            "The folder {} contains only {} files, less than requested {}".format(self.data_folder, max_n, n_samples)
        self.n_samples = n_samples
        self.start_ind = start_ind
        self.as_float = as_float
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, ind):
        full_path, _ = self._name_parser(self.data_folder, self.img_fn, ind+self.start_ind)
        img = imread(full_path)
        assert img is not None, "Can not read {}".format(full_path)
        if self.as_float:
            img = img_as_float(img)
        return img
    
    @staticmethod
    def _name_parser(data_folder, fn, ind):
        # return absolute path and file name
        n = fn.format(ind)
        return os.path.join(data_folder,n), n
    
class SeqSet(Dataset):
    """
    Read one video sequence, gives network input for video reconstruct PAT network
    Net input: a tuple with 7 elements
        LR alpha, (h//4, w//4) single channel image, sample 2nd-channel. Suppose it's frame N
        LR n1,    (h//4, w//4) single channel image, sample 1st-channel. Frame N-1
        LR p1,    (h//4, w//4) single channel image, sample 3rd-channel. Frame N+1
        HR,       (h, w)       single channel image, grayscale. Frame N+d, d in [-3, 3] by default
        pos n1/p1/HR: position cube for LR n1, LR p1, HR
    Net gt: (h, w, 3) 3-channel image. High-res 3-channel frame N.
    """
    def __init__(self, folder, img_fn, hr_dist, *, 
                 rng=None, lr_down_fac=4, hr_down_fac=np.sqrt(2),
                 lr_hw=3, hr_hw=9):
        """
        folder: a folder containing only one image sequence
        img_fn: format string for frame filename
        hr_dist: high-res frame distance from gt frame
        rng: numpy random generator
        lr_down_fac: low-res resolution gap from gt
        hr_down_fac: high-res resolution gap from gt. Note that high-res will be scaled to original size
        lr_hw: low-res search range, affect the position cube for attention module
        hr_hw: high-res search range
        """
        super().__init__()
        self.hr_dist = hr_dist
        self.margin = max(self.hr_dist, 1)
        self.lr_down_fac = lr_down_fac
        self.hr_down_fac = hr_down_fac
        self.lr_hw = lr_hw
        self.hr_hw = hr_hw
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.img_set = ImgBareSet(folder, img_fn)
        
    def __len__(self):
        return len(self.img_set) - 2*self.hr_dist
    
    def __getitem__(self, idx):
        assert idx>=0 and idx<len(self), "Index {} out of range".format(idx)
        # fetch triplet, shuffle channels
        idx = idx + self.margin
        triplet = [img_as_float32(self.img_set[idx+a]) for a in range(-1,2)]
        clist = np.arange(3)
        self.rng.shuffle(clist)
        triplet = [img[...,clist] for img in triplet]
        # fetch hr, grayscale, downscale, then scale back
        hr_d = self.rng.integers(-self.hr_dist, self.hr_dist+1, None)
        hr = img_as_float32(self.img_set[idx+hr_d])
        hr = cv.cvtColor(hr, cv.COLOR_RGB2GRAY)
        gt_wh = (hr.shape[1],hr.shape[0])
        hr = cv.resize(hr, None, None, 1/self.hr_down_fac, 1/self.hr_down_fac, cv.INTER_AREA)
        hr = cv.resize(hr, gt_wh, interpolation=cv.INTER_CUBIC)
        # random rotation, flip
        rot_k = self.rng.integers(0, 4, None)
        flip_k = self.rng.integers(0, 2, 2)
        triplet_hr = triplet+[hr,]
        triplet_hr = [np.rot90(img, rot_k) for img in triplet_hr]
        for axidx in range(2):
            if flip_k[axidx]:
                triplet_hr = [np.flip(img, axidx) for img in triplet_hr]
        # pick GT, extract color channel, downscale LRs
        gt = triplet_hr[1]
        hr = triplet_hr[-1]
        lr_triplet = [cv.resize(triplet_hr[a][...,a], None, None, 
                                1/self.lr_down_fac, 1/self.lr_down_fac, cv.INTER_AREA) for a in range(3)]
        # order the images
        img_tuple = (lr_triplet[1], lr_triplet[0], lr_triplet[2], hr, gt)
        img_tuple = tuple([np.clip(img,0,1) for img in img_tuple])

        # prepare position cubes
        ########## NOTE that Qian uses swapped x,y ##########
        h, w = img_tuple[0].shape
        lr_yys, lr_xxs = _project_rescale_grid((w,h), (w,h), self.lr_hw, 1) #### swapped x,y
        hr_yys, hr_xxs = _project_rescale_grid((w,h), (w,h), self.hr_hw, 1) #### swapped x,y
        pos_tuple = (np.stack([lr_xxs, lr_yys], axis=0).astype(np.int16), 
                     np.stack([lr_xxs, lr_yys], axis=0).astype(np.int16), 
                     np.stack([hr_xxs, hr_yys], axis=0).astype(np.int16))

        # turn to tensor, return value
        img_tuple = tuple([_to_tensor(img) for img in img_tuple])
        pos_tuple = tuple([torch.from_numpy(pos) for pos in pos_tuple])
        
        return (*img_tuple[:4], *pos_tuple), img_tuple[4]
    
def MultiSeqSet(src_folder, *, 
                rng=None, img_fn='{:03d}.png', hr_dist=3, 
                lr_down_fac=4, hr_down_fac=np.sqrt(2),
                lr_hw=3, hr_hw=9):
    # find rng
    if rng is None:
        rng = np.random.default_rng()
    # find all sequence folder
    subpath_list = [os.path.join(src_folder, subf) for subf in os.listdir(src_folder)]
    subpath_list = [subp for subp in subpath_list if os.path.isdir(subp)]
    subpath_list = sorted(subpath_list)
    # form a list of sequence set
    seq_set_list = [SeqSet(subp, img_fn, hr_dist, rng=rng, 
                           lr_down_fac=lr_down_fac, hr_down_fac=hr_down_fac,
                           lr_hw=lr_hw, hr_hw=hr_hw) for subp in subpath_list]
    # concatenate and return
    return ConcatDataset(seq_set_list)
    
class PatvidxrSlowflowSet():
    def __init__(self, data_folder, tratio, *, 
                 n_samples=None, start_ind=0,
                 lr_down_fac=4, hr_down_fac=np.sqrt(2)):
        # parse inputs
        assert os.path.isdir(data_folder), \
            "{} is not a valid folder".format(data_folder)
        self.data_folder = data_folder
        self.tratio = tratio
        self.lr_down_fac = lr_down_fac
        self.hr_down_fac = hr_down_fac
        self.ibs = ImgBareSet(self.data_folder, "{:07d}.png", as_float=False)
        max_n = len(self.ibs)
        if n_samples is None:
            n_samples = max_n
        assert n_samples <= max_n, \
        "The folder {} contains only {} files, less than requested {}".format(self.data_folder, max_n, n_samples)
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, ind):
        if ind<0 or ind>=len(self):
            raise RuntimeError("Index out of range")
        
        # find indexing
        gt_ind = ind
        hr_ind = (ind//self.tratio)*self.tratio + self.tratio//2
        if ind==0:
            lr_ind_list = [ind, ind+1, ind+2]
        elif ind==len(self)-1:
            lr_ind_list = [ind, ind-1, ind-2]
        else:
            lr_ind_list = [ind, ind-1, ind+1]
        lr_channel_list = [lr_ind%3 for lr_ind in lr_ind_list]
        
        # load lr list
        lr_list = []
        for lr_ind, lr_channel in zip(lr_ind_list, lr_channel_list):
            lr = img_as_float32(self.ibs[lr_ind][..., lr_channel])
            lr = cv.resize(lr, None, None, 1/self.lr_down_fac, 1/self.lr_down_fac, cv.INTER_AREA)
            lr_list.append(lr)
        # load hr
        hr = img_as_float32(self.ibs[hr_ind])
        hr = cv.cvtColor(hr, cv.COLOR_RGB2GRAY)
        gt_wh = (hr.shape[1],hr.shape[0])
        hr = cv.resize(hr, None, None, 1/self.hr_down_fac, 1/self.hr_down_fac, cv.INTER_AREA)
        hr = cv.resize(hr, gt_wh, interpolation=cv.INTER_CUBIC)
        # load gt
        gt = img_as_float32(self.ibs[gt_ind])
        
        # return
        return lr_list, hr, gt, np.array(lr_channel_list)
    
##########
### pre/post process functions
##########
def _project_rescale_grid(src_wh, tgt_wh, hw, s):
    """
    Suppose target is the rescaled version of src_wh
    hw, s: half-width and stride
    """
    # parse inputs
    src_w, src_h = src_wh
    tgt_w, tgt_h = tgt_wh
    # find source points 
    x_src = np.arange(src_w)
    y_src = np.arange(src_h)
    # find target points meshgrid
    x_tgt = np.linspace(-0.5, tgt_w-0.5, src_w*2+1)[1:-1:2]
    y_tgt = np.linspace(-0.5, tgt_h-0.5, src_h*2+1)[1:-1:2]
    xy_mesh = np.meshgrid(x_tgt, y_tgt, indexing='xy')
    # expand to cube
    w_vec = np.arange(-hw, hw+1, 1)*s
    xy_w = np.meshgrid(w_vec, w_vec, indexing='xy')
    x_cube = xy_mesh[0].reshape(src_h, src_w, 1) + xy_w[0].reshape(1, 1, -1)
    y_cube = xy_mesh[1].reshape(src_h, src_w, 1) + xy_w[1].reshape(1, 1, -1)
    # round and clip within target wh
    x_cube = np.clip(np.round(x_cube).astype(int), 0, tgt_w-1)
    y_cube = np.clip(np.round(y_cube).astype(int), 0, tgt_h-1)
    # return
    return x_cube, y_cube

def move_pixel_value(img, mean, std):
    return (img-np.mean(img))/np.std(img.flatten())*std+mean

def crop_with_pad(vec, n, pad):
    l = len(vec) # length
    pl = (l+(n-1)*2*pad)//n # patch length
    assert (l+(n-1)*2*pad)%n == 0, "Better pad width to {}".format((pl+1)*n-(n-1)*2*pad) 
    x0_list = np.arange(n)*(pl-2*pad) # starting index
    patch_list = [vec[x0:x0+pl] for x0 in x0_list]
    return patch_list

def combine_with_pad(p_list, pad):
    if len(p_list)==0:
        raise RuntimeError("Empty input!")
    elif len(p_list)==1:
        return p_list[0]
    head = p_list[0][:-pad]
    tail = p_list[-1][pad:]
    mid_list = [p[pad:-pad] for p in p_list[1:-1]]
    rec = np.concatenate([head,]+mid_list+[tail,], axis=0)
    return rec

def _2d_patch_crop(img, mn, pad):
    m,n = mn
    if len(img.shape)==2:
        img = img[...,np.newaxis]
    assert len(img.shape)==3, "image shape/channel not proper"
    c = img.shape[-1]
    # crop row
    irow_list = np.array(crop_with_pad(img, m, pad)) #(m, ph, w, c)
    # crop colomn
    patch_list = np.array(crop_with_pad(irow_list.transpose(2,0,1,3), 
                                        n, pad)) #(n, pw, m, ph, c)
    # reshape
    patch_stack = patch_list.transpose(2,0,3,1,4) #(m, n, ph, pw, 3)
    ph, pw = patch_stack.shape[2:4]
    patch_stack = patch_stack.reshape(m*n,ph,pw,c)
    if c==1:
        patch_stack = patch_stack[...,-1]
    # return
    return patch_stack

def _2d_patch_combine(patch_stack, mn, pad):
    m,n = mn
    ph, pw, c = patch_stack[0].shape
    img = patch_stack.reshape(m,n,ph,pw,c).transpose(0,2,1,3,4) #(m, ph, n, pw, 3)
    img = combine_with_pad(img, pad) #(h, n, pw, 3)
    img = img.transpose(1, 2, 0, 3) #(n, pw, h, 3)
    img = combine_with_pad(img, pad) #(w, h, 3)
    img = img.transpose(1, 0, 2)
    return img

def _to_tensor(img):
    """
    Assumptions for the input: a RGB or grayscale image, float32, 0-1
    """
    if len(img.shape)==2:
        img = img[...,np.newaxis]
    assert len(img.shape)==3
    img = torch.from_numpy(img.transpose(2,0,1))
    return img

def _to_npimg(img):
    """
    Assumptions for input: a RGB or grascale cpu tensor, detached, float32, 0-1
    """
    img = img.numpy().transpose(1,2,0)
    img = np.squeeze(img)
    return img


########################################################
###################### Deprecated ######################
########################################################

##########
### Image dataloaders
##########
class PatTempData():
    def __init__(self, data_folder, tratio, *, 
                 n_samples=None, start_ind=0):
        # parse inputs
        assert os.path.isdir(data_folder), \
            "{} is not a valid folder".format(data_folder)
        self.data_folder = data_folder
        self.tratio = tratio
        # form 3 image loaders
        self.gt_set = ImgBareSet(os.path.join(self.data_folder, 'gt_rgb'), 
                                 "{:07d}.png", as_float=True)
        self.limg_set = ImgBareSet(os.path.join(self.data_folder, 'lr_gray'), 
                                   "{:07d}.png", as_float=True)
        self.rimg_set = ImgBareSet(os.path.join(self.data_folder, 'hr_rgb'), 
                                   "{:07d}.png", as_float=True)
        max_n = np.min([len(self.gt_set), len(self.limg_set), len(self.rimg_set)])
        if n_samples is None:
            n_samples = max_n
        assert n_samples <= max_n, \
            "The subfolders in folder {} contains only {} files, less than requested {}".format(self.data_folder, max_n, n_samples)
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, ind):
        gt = self.gt_set[ind]
        limg = self.limg_set[ind]
        rind = (ind//self.tratio)*self.tratio + self.tratio//2
        rimg = self.rimg_set[rind]
        #print(ind, rind)
        return limg, rimg, gt
    
##########
### pre/post process functions
##########
def img_pair_to_dual_input(limg, rimg, tgt_wh=None, tgt_mn=(1,1), pad=0):
    """
    limg is a low resolution grayscale image
    rimg is a high resolution rgb image
    """    
    # scale images
    if tgt_wh is None:
        tgt_wh = (rimg.shape[1], rimg.shape[0])
    w, h = tgt_wh
    limg = cv.resize(limg, tgt_wh, interpolation=cv.INTER_CUBIC)
    rimg = cv.resize(rimg, tgt_wh, interpolation=cv.INTER_CUBIC)
    limg = limg[..., np.newaxis].repeat(3, axis=2)
    
    # record mean and std
    cmean_list = np.mean(rimg, axis=(0,1))
    cstd_list = np.std(rimg, axis=(0,1))
    
    # crop to patches
    m, n = tgt_mn
    lr_patch_stack = []
    for img in (limg, rimg):
        irow_list = np.array(crop_with_pad(img, m, pad)) #(m, ph, w, 3)
        patch_list = np.array(crop_with_pad(irow_list.transpose(2,0,1,3), 
                                            n, pad)) #(n, pw, m, ph, 3)
        patch_stack = patch_list.transpose(2,0,3,1,4) #(m, n, ph, pw, 3)
        ph, pw = patch_stack.shape[2:4]
        lr_patch_stack.append(patch_stack.reshape(m*n,ph,pw,3))
    
    return *lr_patch_stack, cmean_list, cstd_list

def go_through_net(net, img0_list, img1_list, *, 
                   hw=22, s=1, device=torch.device('cuda')):
    # parameters
    bs, h, w, c = img0_list.shape
    bsp, hp, wp, cp = img1_list.shape
    assert bs==bsp
    assert c==cp and c==3
    
    # prepare xxs, yys
    # these are the possible key coordinateds
    yys, xxs = _project_rescale_grid((w,h), (wp,hp), hw, s)
    xxs = xxs[np.newaxis].repeat(bs, axis=0)
    yys = yys[np.newaxis].repeat(bs, axis=0)
    
    # turn to tensor
    x_left = torch.from_numpy(img0_list.transpose((0, 3, 1, 2))).to(device)
    x_right = torch.from_numpy(img1_list.transpose((0, 3, 1, 2))).to(device)
    Po = (torch.from_numpy(xxs), torch.from_numpy(yys))
    
    # send through network
    with torch.no_grad():
        x_left = net.module.init_feature(x_left)
        x_right = net.module.init_feature(x_right)
        buffer_left = net.module.pam.rb(x_left)
        buffer_right = net.module.pam.rb(x_right)
        Q = net.module.pam.b1(buffer_left)
        S = net.module.pam.b2s[0](buffer_right)
        R = net.module.pam.b3s[0](buffer_right)

        buffer_out, _ = net.module.pam.fe_pam(Q, S, R, Po, False)
        fused_feature = torch.cat((buffer_out, x_left), 1)
        out = net.module.pam.fusion(fused_feature)
        out = net.module.upscale(out)
        
    # back to numpy
    out_img = out.cpu().numpy().transpose(0, 2, 3, 1)
    
    # clean and return
    return out_img