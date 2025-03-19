#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def get_max_contributor(raster_settings, means3D, means2D, opacities, scales = None, rotations = None, cov3D_precomp = None):
    if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
        raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
    shs = torch.Tensor([])
    colors_precomp = torch.Tensor([])

    if scales is None:
        scales = torch.Tensor([])
    if rotations is None:
        rotations = torch.Tensor([])
    if cov3D_precomp is None:
        cov3D_precomp = torch.Tensor([])

    # Restructure arguments the way that the C++ lib expects them
    args = (
        raster_settings.bg, 
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings.scale_modifier,
        cov3D_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        shs,
        raster_settings.sh_degree,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.debug
    )

    # Invoke C++/CUDA rasterizer
    if raster_settings.debug:
        cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        try:
            num_rendered, max_contributor, max_contribute, radii, geomBuffer, binningBuffer, imgBuffer = _C.get_max_contributor(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_fw.dump")
            print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
            raise ex
    else:
        num_rendered, max_contributor, max_contribute, radii, geomBuffer, binningBuffer, imgBuffer = _C.get_max_contributor(*args)

    return max_contributor, max_contribute, radii