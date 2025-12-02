import argparse
from collections import OrderedDict
from enum import Enum
from typing import Protocol, Union, Tuple, List

import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize
from nnssl.configuration import ANISO_THRESHOLD


class ResamplingProtocol(Protocol):
    def __call__(
        self,
        data: np.ndarray,
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        is_seg: bool = False,
        axis: Union[None, int] = None,
        order: int = 3,
        do_separate_z: bool = False,
        order_z: int = 0,
    ) -> np.ndarray: ...


class ResamplingSchemes(Enum):
    DATA_TO_SHAPE = "resample_data_or_seg_to_shape"
    DATA_TO_SPACING = "resample_data_or_seg_to_spacing"


def resampling_scheme_type(value):
    try:
        return ResamplingSchemes(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid option")


def get_resampling_scheme(
    resampling_scheme: ResamplingSchemes | str,
) -> ResamplingProtocol:
    if isinstance(resampling_scheme, str):
        resampling_scheme = resampling_scheme_type(resampling_scheme)
    if resampling_scheme == ResamplingSchemes.DATA_TO_SHAPE:
        return resample_data_or_seg_to_shape
    elif resampling_scheme == ResamplingSchemes.DATA_TO_SPACING:
        return resample_data_or_seg_to_spacing
    else:
        raise ValueError(f"Unknown resampling scheme: {resampling_scheme}")


def get_do_separate_z(
    spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    anisotropy_threshold=ANISO_THRESHOLD,
):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[
        0
    ]  # find which axis is anisotropic
    return axis


def compute_new_shape(
    old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array(
        [int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)]
    )
    return new_shape


def resample_data_or_seg_to_spacing(
    data: np.ndarray,
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
):
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the same spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    shape = np.array(data[0].shape)
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)

    data_reshaped = resample_data_or_seg(
        data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z
    )
    return data_reshaped


def resample_data_or_seg_to_shape(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
):
    """
    needed for segmentation export. Stupid, I know. Maybe we can fix that with Leos new resampling functions
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the same spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(
        data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z
    )
    return data_reshaped


def torch_resize_3d(arr, new_shape, is_seg=False):
    """
    arr: (C, X, Y, Z) numpy array
    new_shape: (X2, Y2, Z2)
    Returns numpy array (C, X2, Y2, Z2)
    """
    mode = "nearest" if is_seg else "trilinear"
    arr_t = torch.from_numpy(arr[None]).float()  # (1, C, X, Y, Z)

    out = F.interpolate(
        arr_t,
        size=new_shape.tolist(),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )

    return out[0].cpu().numpy()


def torch_resize_2d(arr, new_shape, is_seg=False):
    """
    arr: (C, H, W)
    """
    mode = "nearest" if is_seg else "bilinear"
    arr_t = torch.from_numpy(arr[None]).float()  # (1, C, H, W)

    out = F.interpolate(
        arr_t,
        size=new_shape.tolist(),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )

    return out[0].cpu().numpy()


def _torch_device(prefer_cuda: bool = True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_torch_tensor(x: np.ndarray, device: torch.device):
    t = torch.from_numpy(x)
    # float tensors for interpolation; keep ints for segs handled separately
    return t.to(device)


def _torch_interpolate(
    tensor: torch.Tensor, size: tuple[int, ...], mode: str, align_corners: bool | None
):
    # tensor shape expected: (N, C, ...) where ... is spatial dims
    return F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)


def _batched_2d_resize_slices(
    vol_swapped: np.ndarray,
    target_hw: tuple[int, int],
    is_seg: bool,
    device: torch.device,
):
    """
    vol_swapped: (D, H, W) numpy
    returns: (D, H2, W2) numpy
    Single batched 2D interpolation over all slices.
    """
    # create tensor shape (D, 1, H, W) for batch-of-slices
    x = torch.from_numpy(vol_swapped[:, None]).float().to(device)  # (D,1,H,W)
    mode = "nearest" if is_seg else "bilinear"
    align_corners = None if is_seg else False
    out = _torch_interpolate(
        x, size=target_hw, mode=mode, align_corners=align_corners
    )  # (D,1,H2,W2)
    out_np = out.squeeze(1).cpu().numpy()  # (D, H2, W2)
    return out_np


def _batched_3d_resize(
    vol_3d: np.ndarray,
    new_shape: tuple[int, int, int],
    is_seg: bool,
    device: torch.device,
):
    """
    vol_3d: (D, H, W) numpy -> will be treated as (1, 1, D, H, W) for interpolate
    returns: (D2, H2, W2) numpy
    """
    x = torch.from_numpy(vol_3d[None, None]).float().to(device)  # (1,1,D,H,W)
    mode = "nearest" if is_seg else "trilinear"
    align_corners = None if is_seg else False
    out = _torch_interpolate(
        x, size=tuple(new_shape), mode=mode, align_corners=align_corners
    )  # (1,1,D2,H2,W2)
    return out[0, 0].cpu().numpy()


def resample_data_or_seg(
    data: np.ndarray,
    new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    axis: Union[None, int] = None,
    order: int = 3,
    do_separate_z: bool = False,
    order_z: int = 0,
    use_torch: bool = True,  # <---- NEW OPTION
):
    """
    data: (C, X, Y, Z)
    """
    assert data.ndim == 4
    assert len(new_shape) == 3

    dtype_data = data.dtype
    shape = np.array(data.shape[1:])
    new_shape = np.array(new_shape)

    # -------------------------
    # Case 1: Torch fast path
    # -------------------------
    if use_torch and not do_separate_z:
        if np.any(shape != new_shape):
            data_f = data.astype(np.float32)
            out = torch_resize_3d(data_f, new_shape, is_seg=is_seg)
            return out.astype(dtype_data)
        else:
            return data

    # -------------------------
    # Case 2: Torch path but separate Z
    # (slice-wise)
    # -------------------------
    if use_torch and do_separate_z:
        assert len(axis) == 1
        axis = axis[0]

        reshaped = []
        for c in range(data.shape[0]):
            channel = data[c]

            # Move the chosen axis to the front
            channel_swapped = np.moveaxis(channel, axis, 0)

            resized_slices = []
            for sl in channel_swapped:
                resized_slices.append(
                    torch_resize_2d(
                        sl[None],
                        new_shape[[i for i in range(3) if i != axis]],
                        is_seg=is_seg,
                    )[0]
                )
            resized_slices = np.stack(resized_slices, axis=0)

            # Now resize along the separated axis
            if shape[axis] != new_shape[axis]:
                # 1D interpolate with nearest/linear
                resized_axis = torch_resize_3d(
                    resized_slices[None],  # (1, D, H, W)
                    new_shape.tolist(),
                    is_seg=is_seg,
                )[0]
                reshaped.append(resized_axis[None])
            else:
                reshaped.append(np.moveaxis(resized_slices, 0, axis)[None])

        return np.vstack(reshaped).astype(dtype_data)

    # -------------------------
    # Case 3: Original slow path (skimage)
    # -------------------------
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {"mode": "edge", "anti_aliasing": False}

    if np.any(shape != new_shape):
        data_f = data.astype(float)

        if do_separate_z:
            assert len(axis) == 1
            axis = axis[0]

            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data_f.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(
                            resize_fn(
                                data_f[c, slice_id], new_shape_2d, order, **kwargs
                            )
                        )
                    elif axis == 1:
                        reshaped_data.append(
                            resize_fn(
                                data_f[c, :, slice_id], new_shape_2d, order, **kwargs
                            )
                        )
                    else:
                        reshaped_data.append(
                            resize_fn(
                                data_f[c, :, :, slice_id], new_shape_2d, order, **kwargs
                            )
                        )
                reshaped_data = np.stack(reshaped_data, axis)

                if shape[axis] != new_shape[axis]:
                    rows, cols, dim = new_shape
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(
                            map_coordinates(
                                reshaped_data, coord_map, order=order_z, mode="nearest"
                            )[None]
                        )
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for cl in unique_labels:
                            hot = (reshaped_data == cl).astype(float)
                            warped = map_coordinates(
                                hot, coord_map, order=order_z, mode="nearest"
                            )
                            reshaped[warped > 0.5] = cl

                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            reshaped = []
            for c in range(data_f.shape[0]):
                reshaped.append(resize_fn(data_f[c], new_shape, order, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)

        return reshaped_final_data.astype(dtype_data)

    return data
