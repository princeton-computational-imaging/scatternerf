# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF++ (https://github.com/Kai-46/nerfplusplus)
# Copyright (c) 2020 the NeRF++ authors. All Rights Reserved.
# ------------------------------------------------------------------------------------
#From a copy of tnt
import glob
import os
from typing import *

import imageio
import numpy as np
import cv2
import pickle
from PIL import Image
from scipy.signal import convolve2d

def local_depth_variance(depth_map, window_size = 7):
    # Define a function to calculate the variance of a neighborhood
    def neighborhood_variance(arr):
        return np.var(arr)

    # Define a sliding window of size (window_size, window_size)
    window = np.ones((window_size, window_size), dtype=np.float32)

    # Compute the sum and sum of squares of the depth map using the sliding window
    depth_sum = convolve2d(depth_map, window, mode='same', boundary='symm')
    depth_sum2 = convolve2d(depth_map**2, window, mode='same', boundary='symm')

    # Compute the local variance of depth using the depth sum and sum of squares
    var_map = (depth_sum2 - depth_sum**2 / (window_size ** 2)) / ((window_size ** 2) - 1)

    return (var_map>50).astype(np.float)


def get_unc_defog(im, load_path = True) : 
    if load_path :
        im = Image.open(im )
    a = im.convert('HSV')
    np_im = np.array(im)
    np_a = np.array(a)
    hue = np_a[:,:,0]
    np_im_c = np.stack([np_im/255.0,1-np_im/255.0 ] )
    sh = np.max(np_im_c, axis = 0)

    hue = hue/360
    diff = np.sum(np.abs(sh - hue[...,None]), axis = -1  ) / 3

    return diff


def find_files(dir, exts):
    if os.path.isdir(dir):
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []


def similarity_from_cameras(c2w):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, scale


def load_dense_data(
    datadir: str,
    scene_name: str,
    train_skip: int,
    val_skip: int,
    test_skip: int,
    cam_scale_factor: float,
    near: Optional[float],
    far: Optional[float],
):

    basedir = os.path.join(datadir, scene_name)

    def parse_txt(filename ):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
         
        camera = np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)
        return camera

    # camera parameters files
    intrinsics_files = find_files(
        "{}/train/intrinsics".format(basedir), exts=["*.txt"]
    )[::train_skip]
    intrinsics_files += find_files(
        "{}/validation/intrinsics".format(basedir), exts=["*.txt"]
    )[::val_skip]
    intrinsics_files += find_files(
        "{}/test/intrinsics".format(basedir), exts=["*.txt"]
    )[::test_skip]
    pose_files = find_files("{}/train/pose".format(basedir), exts=["*.txt"])[
        ::train_skip
    ]
    pose_files += find_files("{}/validation/pose".format(basedir), exts=["*.txt"])[
        ::val_skip
    ]
    pose_files += find_files("{}/test/pose".format(basedir), exts=["*.txt"])[
        ::test_skip
    ]
    cam_cnt = len(pose_files)

    # img files
    img_files = find_files("{}/rgb".format(basedir), exts=["*.png", "*.jpg", "*.JPG"])
    if len(img_files) > 0:
        assert len(img_files) == cam_cnt
    else:
        img_files = [
            None,
        ] * cam_cnt

    # assume all images have the same size as training image
    train_imgfile = find_files("{}/train/rgb".format(basedir), exts=["*.png", "*.jpg", "*.JPG"])[
        ::train_skip
    ]
    val_imgfile = find_files(
        "{}/validation/rgb".format(basedir), exts=["*.png", "*.jpg", "*.JPG"]
    )[::val_skip]
    test_imgfile = find_files("{}/test/rgb".format(basedir), exts=["*.png", "*.jpg", "*.JPG"])[
        ::test_skip
    ]
    i_train = np.arange(len(train_imgfile))
    i_val = np.arange(len(val_imgfile)) + len(train_imgfile)
    i_test = np.arange(len(test_imgfile)) + len(train_imgfile) + len(val_imgfile)
    i_all = np.arange(len(train_imgfile) + len(val_imgfile) + len(test_imgfile))
    
    i_split = (i_train, i_val, i_test, i_all)

    SCALING_FACTOR = 4 #hardcoded

    images = (
        np.stack(
            [
                cv2.resize(imageio.imread(imgfile), (0,0), fx= 1/4, fy= 1/4 )
                for imgfile in train_imgfile + val_imgfile + test_imgfile
            ]
        )
        / 255.0
    )
    h, w = images[0].shape[:2]

    #defogged : 
    defogged = (
        np.stack(
            [
                cv2.resize(imageio.imread(imgfile.replace("/rgb/","/defogged/")), (0,0), fx= 1/SCALING_FACTOR, fy= 1/SCALING_FACTOR )
                for imgfile in train_imgfile + val_imgfile + test_imgfile
            ]
        )
        / 255.0
    )    
    print(images.shape, defogged.shape)
    assert(images.shape == defogged.shape)    

    #depth files:
    train_depthfile = find_files("{}/train/depths".format(basedir), exts=["*.npy"])[
        ::train_skip
    ]
    val_depthfile = find_files(
        "{}/validation/depths".format(basedir), exts=["*.npy"]
    )[::val_skip]
    test_depthfile = find_files("{}/test/depths".format(basedir), exts=["*.npy"])[
        ::test_skip
    ]

    assert(len(train_depthfile) == len(train_imgfile) )
    assert(len(val_depthfile) == len(val_imgfile) )
    assert(len(test_depthfile) == len(test_imgfile) )
    depths = (
        np.stack(
            [
                cv2.resize(np.load(depthfile.replace("/DSC","/_DSC"))[..., None] , (0,0), fx= 1/SCALING_FACTOR, fy= 1/SCALING_FACTOR, interpolation = cv2.INTER_NEAREST )
                for depthfile in train_depthfile + val_depthfile + test_depthfile
            ]
        )
    )
    unc_depth = np.stack([local_depth_variance(depths[i]) for i in range(depths.shape[0]) ])
    
    #Since the poses estimated with COLMAP are not in meters, need to adjust the depths:
    add = ""
    if "Sequence00" in basedir or "toy" in basedir:
        factor_to_multipli_depth = 0.10725175722224041
    elif "Sequence04" in basedir : 
        factor_to_multipli_depth = 0.09344572282517981
    elif "Sequence06"  in basedir : 
        factor_to_multipli_depth = 0.16727339933271204
    elif "Sequence08"  in basedir :
        factor_to_multipli_depth = 0.13657559984008424
    elif "Sequence09"  in basedir :
        factor_to_multipli_depth = 0.08368170889627297
    elif "Sequence10"  in basedir :        
        factor_to_multipli_depth = 0.14332983994009932
    elif "Sequence11"  in basedir :        
        factor_to_multipli_depth  = 0.1672806999421341
    elif "Sequence12"  in basedir :       
        factor_to_multipli_depth = 0.09684113609210428
    elif "Sequence13"  in basedir :       
        factor_to_multipli_depth = 0.18877780664812285
    elif "Sequence14"  in basedir :     
        factor_to_multipli_depth = 0.09478544646535393
    elif "Nikon" in basedir :
        factor_to_multipli_depth = 1
        add = "_"
    else:
        throw_error
    depths *= factor_to_multipli_depth


    intrinsics = np.stack(
        [parse_txt(intrinsics_file) for intrinsics_file in intrinsics_files]
    )
    intrinsics[:, :2, :3] /= SCALING_FACTOR

    extrinsics = np.stack([parse_txt(pose_file) for pose_file in pose_files])

    if cam_scale_factor > 0:
        T, sscale = similarity_from_cameras(extrinsics)
        extrinsics = np.einsum("nij, ki -> nkj", extrinsics, T)
        scene_scale = cam_scale_factor * sscale
        extrinsics[:, :3, 3] *= scene_scale
        depths *= scene_scale
        print("scene_scale:",scene_scale)
        print("So, the total factor is ", scene_scale*factor_to_multipli_depth)



    num_frame = len(extrinsics)

    image_sizes = np.array([[h, w] for i in range(num_frame)])

    near = 0.0 if near is None else near
    far = 1.0 if far is None else far

    render_poses = extrinsics
    images = np.concatenate([images, depths[...,None], defogged], axis = -1)


    # Getting the pre-estimated airlights:
    airlights = (
        np.stack(
            [
                cv2.resize(np.load(imgfile.replace("/rgb/","/airlight/"+add).replace(".png",".npy").replace(".JPG",".npy")) , (0,0), fx= 1/SCALING_FACTOR, fy= 1/SCALING_FACTOR, interpolation = cv2.INTER_NEAREST )
                for imgfile in train_imgfile + val_imgfile + test_imgfile
            ]
        )
    )

    #Other estimated defogged, used for visualization. Can be replaced with zeros if you dont have.
    unc_defog = (
        np.stack(
            [
                cv2.resize(get_unc_defog(imgfile.replace("/rgb/","/defogged/"))[..., None], (0,0), fx= 1/SCALING_FACTOR, fy= 1/SCALING_FACTOR)  
                for imgfile in train_imgfile + val_imgfile + test_imgfile
            ]
        )
    )    
    assert(unc_depth.shape == unc_defog.shape)

    images = np.concatenate([images, airlights, unc_defog[...,None], unc_depth[...,None]  ], axis = -1)

    return (
        images,
        depths,
        intrinsics,
        extrinsics,
        image_sizes,
        near,
        far,
        (-1, -1),
        i_split,
        render_poses,
    )
