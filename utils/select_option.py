# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import *

from src.data.litdata import (
    LitDataBlender,
    LitDataBlenderMultiScale,
    LitDataLF,
    LitDataLLFF,
    LitDataNeRF360V2,
    LitDataRefNeRFReal,
    LitDataShinyBlender,
    LitDataTnT,
    LitDataDense,
    LitDataDenseRaw
)

from src.model.nerf.model import LitNeRF
from src.model.scatternerf.model import LitScatterNeRF


def select_model(
    model_name: str,
):

    if model_name == "nerf":
        return LitNeRF() 
    elif model_name == "scatternerf":
        return LitScatterNeRF()                
    else:
        raise f"Unknown model named {model_name}"


def select_dataset(
    dataset_name: str,
    datadir: str,
    scene_name: str,
):
    if dataset_name == "blender":
        data_fun = LitDataBlender
    elif dataset_name == "blender_multiscale":
        data_fun = LitDataBlenderMultiScale
    elif dataset_name == "llff":
        data_fun = LitDataLLFF
    elif dataset_name == "tanks_and_temples":
        data_fun = LitDataTnT
    elif dataset_name == "dense":
        data_fun = LitDataDense        
    elif dataset_name == "lf":
        data_fun = LitDataLF
    elif dataset_name == "nerf_360_v2":
        data_fun = LitDataNeRF360V2
    elif dataset_name == "shiny_blender":
        data_fun = LitDataShinyBlender
    elif dataset_name == "refnerf_real":
        data_fun = LitDataRefNeRFReal
    elif dataset_name == 'dense_raw':
        data_fun = LitDataDenseRaw

    return data_fun(
        datadir=datadir,
        scene_name=scene_name,
    )


def select_callback(model_name):

    callbacks = []

    if model_name == "plenoxel":
        import src.model.plenoxel.model as model

        callbacks += [model.ResampleCallBack()]

    if model_name == "dvgo":
        import src.model.dvgo.model as model

        callbacks += [
            model.Coarse2Fine(),
            model.ProgressiveScaling(),
            model.UpdateOccupancyMask(),
        ]

    return callbacks
