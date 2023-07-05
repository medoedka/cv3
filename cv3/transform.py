import warnings
from typing import Tuple, Optional, Union, TypeAlias

import cv2
import numpy as np
from nptyping import NDArray

from ._utils import COLORS_RGB_DICT
from ._utils import type_decorator, _relative_check, _relative_handle, _handle_rect_coords

__all__ = [
    'vertical_flip',
    'horizontal_flip',
    'diagonal_flip',
    'transform',
    'rotate_90_left',
    'rotate_90_right',
    'rotate_180',
    'scale',
    'shift',
    'xshift',
    'xtranslate',
    'yshift',
    'ytranslate',
    'resize',
    'crop',
    'pad',
    'copyMakeBorder'
]

InterpolationType: TypeAlias = int
PaddingType: TypeAlias = int


def _inter_flag_match(flag: str) -> InterpolationType:
    """
    Matches string interpolation flag to cv2 interpolation attribute.

    Parameters
    ----------
        flag : str
            Flag to match with the existing cv2 interpolation attribute.

    Returns
    -------
        InterpolationType
            cv2 interpolation attribute.
    """
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


def _border_flag_match(flag: str) -> PaddingType:
    """
    Matches string border type flag to cv2 border attribute.

    Parameters
    ----------
        flag : str
            Flag to match with the existing cv2 border attribute.

    Returns
    -------
        PaddingType
            cv2 border attribute.

    Raises
    ------
        ValueError
            If `border` does not contains in the mapping dict.
    """
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


def vertical_flip(image: NDArray) -> NDArray:
    """
    Flip the image along x-axis.

    Parameters
    ----------
        image : numpy.ndarray
            Image to flip.

    Returns
    -------
        Flipped image.
    """
    return cv2.flip(src=image, flipCode=0)


def horizontal_flip(image: NDArray) -> NDArray:
    """
    Flip the image along y-axis.

    Parameters
    ----------
        image : numpy.ndarray
            Image to flip.

    Returns
    -------
        Flipped image.
    """
    return cv2.flip(src=image, flipCode=1)


def diagonal_flip(image: NDArray) -> NDArray:
    """
    Flip the image along its diagonal.

    Parameters
    ----------
        image : numpy.ndarray
            Image to flip.

    Returns
    -------
        Flipped image.
    """
    return cv2.flip(src=image, flipCode=-1)


def transform(image: NDArray,
              angle: Union[float, int],
              scale: Union[float, int] = 0.0,
              inter: str = 'linear',
              border: str = 'constant',
              value: Optional[Union[int, float, str, Tuple[int, int, int]]] = None) -> NDArray:
    """
    Transform original image by rotating and/or scaling,
    as well as using interpolation techniques and border fillings.

    Parameters
    ----------
        image : NDArray
            Image to transform as numpy.ndarray.

        angle : Union[float, int]
            Angle of rotation

        scale : Union[float, int]
            Parameter of scaling the image. Can be interpreted as zoom.

        inter : str
            Interpolation technique to use.
            Available at the moment: 'nearest', 'linear', 'area', 'cubic', 'lanczos4'.

        border : str
            Type of border filling.
            Available at the moment: 'constant', 'replicate', 'reflect', 'wrap', 'default'.

        value : Optional[Union[int, float, str, Tuple[int, int, int]]]
            Can be treated as a color for the
            Default value is None.
    """
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


def rotate_90_left(image: NDArray) -> NDArray:
    """
    Rotate the image on 90 degrees counterclockwise (left).

    Parameters
    ----------
        image : numpy.ndarray
            Image to rotate.

    Returns
    -------
        numpy.ndarray
            Rotated image.
    """
    return cv2.rotate(src=image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_90_right(image: NDArray) -> NDArray:
    """
    Rotate the image on 90 degrees clockwise (right).

    Parameters
    ----------
        image : numpy.ndarray
            Image to rotate.

    Returns
    -------
        numpy.ndarray
            Rotated image.
    """
    return cv2.rotate(src=image, rotateCode=cv2.ROTATE_90_CLOCKWISE)


def rotate_180(image: NDArray) -> NDArray:
    """
    Rotate the image on 180 degrees.

    Parameters
    ----------
        image : numpy.ndarray
            Image to rotate.

    Returns
    -------
        numpy.ndarray
            Rotated image.
    """
    return cv2.rotate(src=image, rotateCode=cv2.ROTATE_180)


def rotate(image: NDArray,
           angle: Union[float, int],
           inter: str = 'linear',
           border: str = 'constant',
           value: Optional[Union[int, float, str, Tuple[int, int, int]]] = None) -> NDArray:
    """
    Rotate the image on arbitrary degrees clockwise (right).

    Parameters
    ----------
        image: numpy.ndarray
            Image to rotate.

        angle : Union[float, int]
            Angle of rotation

        inter : str
            Interpolation technique to use.
            Available at the moment: 'nearest', 'linear', 'area', 'cubic', 'lanczos4'.

        border : str
            Type of border filling.
            Available at the moment: 'constant', 'replicate', 'reflect', 'wrap', 'default'.

        value : Optional[Union[int, float, str, Tuple[int, int, int]]]
            Can be treated as a color for the
            Default value is None.

    Returns
    -------
        numpy.ndarray
            Rotated image.
    """
    return transform(image=image, angle=angle, scale=1, inter=inter, border=border, value=value)


def scale(image: NDArray,
          scale: Union[float, int],
          inter: str = 'linear',
          border: str = 'constant',
          value: Optional[Union[int, float, str, Tuple[int, int, int]]] = None) -> NDArray:
    """
    Scale the image on arbitrary factor.

    Parameters
    ----------
        image: numpy.ndarray
            Image to rotate.

        scale : Union[float, int]
            Parameter of scaling the image. Can be interpreted as zoom.

        inter : str
            Interpolation technique to use.
            Available at the moment: 'nearest', 'linear', 'area', 'cubic', 'lanczos4'.

        border : str
            Type of border filling.
            Available at the moment: 'constant', 'replicate', 'reflect', 'wrap', 'default'.

        value : Optional[Union[int, float, str, Tuple[int, int, int]]]
            Can be treated as a color for the
            Default value is None.

    Returns
    -------
        numpy.ndarray
            Scaled image.
    """
    return transform(image=image, angle=0, scale=scale, inter=inter, border=border, value=value)


def shift(image: NDArray,
          x: Union[float, int],
          y: Union[float, int],
          border: str = 'constant',
          value: Optional[Union[int, float, str, Tuple[int, int, int]]] = None) -> NDArray:
    """
    Shift the image along x and y axes. You can treat `x` and `y` parameters
    as new coordinates of the image relative to the cuurent ones.
    As a reminder: image coordinates starts at (0,0) point which is upper left corner.

    Parameters
    ----------
        image: numpy.ndarray
            Image to shift.

        x : Union[float, int]
            Parameter of shifting x coordinate of the image.

        y : Union[float, int]
            Parameter of shifting y coordinate of the image.

        border : str
            Type of border filling.
            Available at the moment: 'constant', 'replicate', 'reflect', 'wrap', 'default'.

        value : Optional[Union[int, float, str, Tuple[int, int, int]]]
            Can be treated as a color for the
            Default value is None.

    Returns
    -------
        numpy.ndarray
            Scaled image.
    """
    translation_matrix = np.array(
        [[1, 0, x],
         [0, 1, y]]
    ).astype(np.float32)
    dimensions = (image.shape[1], image.shape[0])
    border_mapped = _border_flag_match(border)
    valid_value = _color_value_check(value)
    translated_image = cv2.warpAffine(
        src=image,
        M=translation_matrix,
        dsize=dimensions,
        borderMode=border_mapped,
        borderValue=valid_value
    )
    return translated_image


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


xtranslate = xshift
ytranslate = yshift
copyMakeBorder = pad
