# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os

import imageio
import numpy as np
from PIL import Image


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return to8b(x)


def store_image(dirpath, rgbs, appendix=""):
    for (i, rgb) in enumerate(rgbs):
        if appendix == "gt" : 
            imgname = f"image_gt{str(i).zfill(3)}.jpg"
        elif appendix == "defog_target" : 
            imgname = f"image_defog_target{str(i).zfill(3)}.jpg"   
        elif appendix == "depth" :
            imgname = f"image_depth{str(i).zfill(3)}.jpg"   
        elif appendix == "depth" :
            imgname = f"image_depth{str(i).zfill(3)}.jpg"   
                        
        else:
            imgname = f"image{str(i).zfill(3)}.jpg"
        rgbimg = Image.fromarray(to8b(rgb.detach().cpu().numpy()))
        imgpath = os.path.join(dirpath, imgname)
        rgbimg.save(imgpath)

def store_depth(dirpath, depths, appendix=""):
    print(len(depths))
    
    for (i, depth) in enumerate(depths):

        imgname = f"depth{str(i).zfill(3)}.jpg"
        imgname = imgname.replace("depth", "depth"+appendix)

        numpy_8b_img_3 = to8b(depth.detach().cpu().numpy())

        rgbimg = Image.fromarray(numpy_8b_img_3)
        imgpath = os.path.join(dirpath, imgname)
        rgbimg.save(imgpath)
        if True:
            np.save( imgpath.replace(".jpg",".npy"), depth.detach().cpu().numpy())        

def store_video(dirpath, rgbs, depths):
    rgbimgs = [to8b(rgb.cpu().detach().numpy()) for rgb in rgbs]
    video_dir = os.path.join(dirpath, "videos")
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, "images.mp4"), rgbimgs, fps=20, quality=8)
