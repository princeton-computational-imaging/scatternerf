# ScatterNeRF: Seeing Through Fog with Physically-Based Inverse Neural Rendering (ICCV23)


<p align="center">
<b> <a href="https://light.princeton.edu/publication/scatternerf/">Project Page</a></b> | <b><a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Ramazzina_ScatterNeRF_Seeing_Through_Fog_with_Physically-Based_Inverse_Neural_Rendering_ICCV_2023_paper.pdf">Paper</a></b>
</p>

<p align="center"> 
 <img src="scatternerf_teaser.gif" alt="animated" height=300/>
</p>


Official code implementation of [ScatterNeRF](https://light.princeton.edu/publication/scatternerf/). This work is being built on top of the great [NeRF-Factory](https://github.com/kakaobrain/nerf-factory) codebase. 



## Requirements
```
conda create -n nerf_factory -c anaconda python=3.8
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip3 install -r requirements.txt

## Or you could directly build from nerf_factory.yml
conda env create --file nerf_factory.yml
```
## Dataset
We provide a ready-to-use sequence dataset sample, which can be downloaded here

## Start the training
Both single and multiple GPU supported
```bash
python3 -m run --ginc configs/[model]/[data].gin
# ex) CUDA_VISIBLE_DEVICES=1,2,3 python3 run.py --ginc configs/scatternerf/tnt.gin --scene_name Sequence00_left_right
```
 
## Render results
(Currently, only single-gpu supported)
```bash
python3 run.py --ginc configs/[model]/[data].gin --scene [scene] --ginb run.run_train=False
# ex) CUDA_VISIBLE_DEVICES=0 python3 run.py --ginc configs/scatternerf/tnt.gin --scene_name Sequence00_left_right --ginb run.run_train=False
```

### License
Copyright (c) 2022 POSTECH, KAIST, and Kakao Brain Corp. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (see [LICENSE](https://github.com/kakaobrain/NeRF-Factory/tree/main/LICENSE) for details)
