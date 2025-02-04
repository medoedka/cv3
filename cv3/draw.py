import warnings
from typing import List, Tuple, Union

import cv2
import numpy as np
from nptyping import NDArray

from . import opt
from ._utils import (
    type_decorator,
    _relative_check,
    _relative_handle,
    _process_color,
    _process_thickness,
    COLORS_RGB_DICT
)

__all__ = [
    'rectangle',
    'polylines',
    'fill_poly',
    'circle',
    'point',
    'points',
    'line',
    'hline',
    'vline',
    'text',
    'putText',
    'rectangles',
    'COLORS'
]

COLORS = list(COLORS_RGB_DICT)


def _line_type_flag_match(flag: str) -> int:
    line_types_mapping = {
        'filled': cv2.FILLED,
        'line_4': cv2.LINE_4,
        'line_8': cv2.LINE_8,
        'line_aa': cv2.LINE_AA
    }
    if flag not in line_types_mapping.keys() or not isinstance(flag, str):
        raise ValueError(f'Parameter `line` has to be a string from: {list(line_types_mapping.keys())}')
    return line_types_mapping[flag]


def _font_flag_match(flag):
    fonts_mapping = {
        'simplex': cv2.FONT_HERSHEY_SIMPLEX,
        'plain': cv2.FONT_HERSHEY_PLAIN,
        'duplex': cv2.FONT_HERSHEY_DUPLEX,
        'complex': cv2.FONT_HERSHEY_COMPLEX,
        'triplex': cv2.FONT_HERSHEY_TRIPLEX,
        'complex_small': cv2.FONT_HERSHEY_COMPLEX_SMALL,
        'script_simplex': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        'script_complex': cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        'italic': cv2.FONT_ITALIC
    }
    if flag not in fonts_mapping.keys() or not isinstance(flag, str):
        raise ValueError(f'Parameter `font` has to be a string from: {list(fonts_mapping.keys())}')
    return fonts_mapping[flag]


def _draw_decorator(func):
    @type_decorator
    def wrapper(img, *args, color=None, line_type=cv2.LINE_8, copy=False, **kwargs):
        if copy:
            img = img.copy()

        if isinstance(line_type, str):
            line_type = _line_type_flag_match(line_type)

        kwargs['t'] = round(kwargs.get('t', opt.THICKNESS))

        return func(img, *args, color=color, line_type=line_type, **kwargs)

    return wrapper


def rectangle(
        image: NDArray,
        vertex1: Tuple[int, int],
        vertex2: Tuple[int, int],
        color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
        thickness: int = 1,
        line_type: str = 'line_8',
        fill: bool = False) -> NDArray:
    color = _process_color(color)
    thickness = _process_thickness(thickness)
    line_type_processed = _line_type_flag_match(line_type)
    if fill:
        thickness = -1
    cv2.rectangle(
        img=image,
        pt1=vertex1,
        pt2=vertex2,
        color=color,
        thickness=thickness,
        lineType=line_type_processed)
    return image


def _handle_poly_pts(img, pts, rel=None):
    pts = np.array(pts).reshape(-1)
    pts = _relative_handle(img, *pts, rel=rel)
    pts = np.int32(pts).reshape(-1, 1, 2)
    return pts


@_draw_decorator
def polylines(img, pts, is_closed=False, rel=None, **kwargs):
    """
    :param img:
    :param pts: np.array or List[List] ot Tuple[Tuple]
    :param is_closed: bool
    :return:
    """
    pts = _handle_poly_pts(img, pts, rel=rel)
    cv2.polylines(img, [pts], is_closed, kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
    return img

@_draw_decorator
def fill_poly(img, pts, rel=None, **kwargs):
    """
    :param img:
    :param pts: np.array or List[List] ot Tuple[Tuple]
    :return:
    """
    pts = _handle_poly_pts(img, pts, rel=rel)
    cv2.fillPoly(img, [pts], kwargs['color'])
    return img


@_draw_decorator
def circle(img, x0, y0, r, rel=None, **kwargs):
    x0, y0 = _relative_handle(img, x0, y0, rel=rel)
    r = round(r)
    cv2.circle(img, (x0, y0), r, kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
    return img


def point(img, x0, y0, r=None, rel=None, **kwargs):
    if 't' in kwargs:
        kwargs.pop('t')
        warnings.warn('Parameter `t` is not used')
    if r is None:
        r = opt.PT_RADIUS
    return circle(img, x0, y0, r, t=-1, rel=rel, **kwargs)


@_draw_decorator
def line(img, x0, y0, x1, y1, rel=None, **kwargs):
    x0, y0, x1, y1 = _relative_handle(img, x0, y0, x1, y1, rel=rel)
    cv2.line(img, (x0, y0), (x1, y1), kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
    return img


@_draw_decorator
def hline(img, y, rel=None, **kwargs):
    h, w = img.shape[:2]
    y = round(y * h if _relative_check(y, rel=rel) else y)
    cv2.line(img, (0, y), (w, y), kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
    return img


@_draw_decorator
def vline(img, x, rel=None, **kwargs):
    h, w = img.shape[:2]
    x = round(x * w if _relative_check(x, rel=rel) else x)
    cv2.line(img, (x, 0), (x, h), kwargs['color'], kwargs['t'], lineType=kwargs['line_type'])
    return img


@_draw_decorator
def text(img, text, x=0.5, y=0.5, font=None, scale=None, flip=False, rel=None, **kwargs):
    if font is None:
        font = opt.FONT
    elif isinstance(font, str):
        font = _font_flag_match(font)
    scale = scale or opt.SCALE
    x, y = _relative_handle(img, x, y, rel=rel)
    cv2.putText(
        img,
        str(text),
        (x, y),
        fontFace=font,
        fontScale=scale,
        color=kwargs['color'],
        thickness=kwargs['t'],
        lineType=kwargs['line_type'],
        bottomLeftOrigin=flip
    )
    return img


@type_decorator
def rectangles(img: np.array, rects: List[List], **kwargs) -> np.array:
    for rect in rects:
        img = rectangle(img, *rect, **kwargs)
    return img


@type_decorator
def points(img: np.array, pts: List[List], **kwargs) -> np.array:
    for pt in pts:
        img = point(img, *pt, **kwargs)
    return img


putText = text
