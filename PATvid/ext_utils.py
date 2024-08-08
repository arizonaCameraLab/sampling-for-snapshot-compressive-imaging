########################################
### Extra utilities for this simplicity of inference tasks
### Note that this file is different from the utils.py in original PAT repo
########################################

import numpy as np
import cv2 as cv

def find_good_matching_points(img1, img2):
    """
    This function finds corresponding feature points between two input images
    Inputs:
        img1, img2 - 2d images
    Outputs:
        pts1, pts2 - feature points
    Notes:
        Using SIFT detector and FLANN matcher, parameters determined inside
    """
    # sift detector
    # SURF is said to be patented and removed. BAD!
    sift = cv.SIFT_create()
    # flann matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)

    # find, match, and ratio test as per Lowe's paper
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    matches = flann.knnMatch(des1,des2,k=2)
    # draw only good matches and get points
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
    pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ])
    pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ])

    return pts1, pts2

def _project_mesh_grid_to_indices_cube(src_wh, tgt_wh, M, hw, s):
    """
    source width/height, target width/height, homography matrix, half window width, stride
    """
    # form source grid
    src_w, src_h = src_wh
    x_src, y_src = np.arange(src_w), np.arange(src_h)
    # project to target
    xy_src = np.stack(np.meshgrid(x_src, y_src, indexing='xy'), axis=-1).astype(np.float32)
    xy_tgt = cv.perspectiveTransform(xy_src.reshape(-1,1,2), M).squeeze()
    xy_cube = np.round(xy_tgt.reshape(src_h, src_w, 2)).astype(int)
    # expand to cube
    w_vec = (np.arange(-hw, hw+1, 1)*s).astype(int)
    xy_w = np.meshgrid(w_vec, w_vec, indexing='xy')
    x_cube = xy_cube[..., 0].reshape(src_h, src_w, 1) + xy_w[0].reshape(1, 1, -1)
    y_cube = xy_cube[..., 1].reshape(src_h, src_w, 1) + xy_w[1].reshape(1, 1, -1)
    # clip within target wh
    tgt_w, tgt_h = tgt_wh
    x_cube = np.clip(x_cube, 0, tgt_w-1)
    y_cube = np.clip(y_cube, 0, tgt_h-1)
    # return
    return x_cube, y_cube

def move_pixel_value(img, mean, std):
    return (img-np.mean(img))/np.std(img.flatten())*std+mean