import cv2
import numpy as np
from functools import partial
import warnings
from nptyping import Int, NDArray, Shape
from typing import Tuple, Optional, Union
from ._utils import COLORS_RGB_DICT


from ._utils import type_decorator, _relative_check, _relative_handle, _process_color, _handle_rect_coords
from .utils import xywh2xyxy, ccwh2xyxy, yyxx2xyxy

__all__ = [
    'vflip',
    'hflip',
    'dflip',
    'transform',
    'rotate',
    'rotate90',
    'rotate180',
    'rotate270',
    'scale',
    'shift',
    'translate',
    'xshift',
    'xtranslate',
    'yshift',
    'ytranslate',
    'resize',
    'crop',
    'pad',
    'copyMakeBorder'
]


def _inter_flag_match(flag: str):
    inter_mapping = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'cubic': cv2.INTER_CUBIC,
        'lanczos4': cv2.INTER_LANCZOS4
    }
    if flag not in inter_mapping.keys() or not isinstance(flag, str):
        raise ValueError(f'Parameter "inter" has to be a string from: {list(inter_mapping.keys())}')
    return inter_mapping[flag]


def _border_flag_match(flag: str):
    border_mapping = {
        'constant': cv2.BORDER_CONSTANT,
        'replicate': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT,
        'wrap': cv2.BORDER_WRAP,
        'default': cv2.BORDER_DEFAULT
    }
    if flag not in border_mapping.keys() or not isinstance(flag, str):
        raise ValueError(f'Parameter `border` has to be a string from: {list(border_mapping.keys())}')
    return border_mapping[flag]


def _color_value_check(color: Optional[Union[int, float, str, Tuple[int, int, int]]]) -> Optional[Union[int, float, Tuple[int, int, int]]]:
    if isinstance(color, str):
        colors_tuple = COLORS_RGB_DICT.get(color)
        return colors_tuple
    return color


def vflip(image: NDArray):
    return cv2.flip(src=image, flipCode=0)


def hflip(image: NDArray):
    return cv2.flip(src=image, flipCode=1)


def dflip(image: NDArray):
    return cv2.flip(src=image, flipCode=-1)


def transform(image: NDArray,
              angle: float,
              scale: float,
              inter: str = 'linear',
              border: str = 'constant',
              value: Optional[Union[int, float, str, Tuple[int, int, int]]] = None) -> NDArray:
    if not isinstance(value, (str, tuple)):
        warnings.warn(
            'Parameter `value` better use as a string, representing the color\
             or tuple with 3 values, representng RGB channels values'
        )
    inter_mapped = _inter_flag_match(inter)
    border_mapped = _border_flag_match(border)
    valid_value = _color_value_check(value)
    if border != cv2.BORDER_CONSTANT and valid_value:
        warnings.warn('Parameter `value` is not used when `border` is not constant')
    rot_matrix = cv2.getRotationMatrix2D(
        center=(image.shape[1] / 2, image.shape[0] / 2),
        angle=angle,
        scale=scale
    )
    result = cv2.warpAffine(
        src=image,
        M=rot_matrix,
        dsize=image.shape[1::-1],
        flags=inter_mapped,
        borderMode=border_mapped,
        borderValue=valid_value
    )
    return result


def rotate(img, angle, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    return transform(img, angle, 1, inter=inter, border=border, value=value)


def scale(img, factor, inter=cv2.INTER_LINEAR, border=cv2.BORDER_CONSTANT, value=None):
    return transform(img, 0, factor, inter=inter, border=border, value=value)

@type_decorator
def shift(img, x, y, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    x, y = _relative_handle(img, x, y, rel=rel)
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    border, value = _border_value_check(border, value)
    return cv2.warpAffine(img, transMat, dimensions, borderMode=border, borderValue=value)


def xshift(img, x, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    h, w = img.shape[:2]
    x = round(x * w if _relative_check(x, rel=rel) else x)
    return translate(img, x, 0, border=border, value=value)


def yshift(img, y, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    h, w = img.shape[:2]
    y = round(y * h if _relative_check(y, rel=rel) else y)
    return translate(img, 0, y, border=border, value=value)


@type_decorator
def resize(img, width, height, inter=cv2.INTER_LINEAR, rel=None):
    if isinstance(inter, str):
        inter = _inter_flag_match(inter)
    width, height = _relative_handle(img, width, height, rel=rel)
    if width == 0 or height == 0:
        if not rel:
            warnings.warn('Try to set `rel` to True')
        raise ValueError('Width or height have zero size')
    return cv2.resize(img, (width, height), interpolation=inter)


@type_decorator
def crop(img, x0, y0, x1, y1, mode='xyxy', rel=None):
    """
    Returns copied crop of the image
    """
    x0, y0, x1, y1 = _handle_rect_coords(img, x0, y0, x1, y1, mode=mode, rel=rel)

    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    x0 = max(x0, 0)
    y0 = max(y0, 0)

    if y1 == y0 or x1 == x0:
        if not rel:
            warnings.warn('zero-size array. Try to set `rel` to True')
    return img[y0:y1, x0:x1].copy()


@type_decorator
def pad(img, y0, y1, x0, x1, border=cv2.BORDER_CONSTANT, value=None, rel=None):
    border, value = _border_value_check(border, value)
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, rel=rel)
    return cv2.copyMakeBorder(img, y0, y1, x0, x1, borderType=border, value=value)


translate = shift
xtranslate = xshift
ytranslate = yshift
rotate90 = partial(rotate, angle=90)
rotate180 = partial(rotate, angle=180)
rotate270 = partial(rotate, angle=270)
copyMakeBorder = pad
