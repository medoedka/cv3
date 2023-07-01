import cv2
from nptyping import Int, NDArray, Shape


__all__ = [
    'rgb2bgr',
    'bgr2rgb',
    'rgb2hsv',
    'bgr2hsv'
]


def rgb2bgr(image: NDArray) -> NDArray:
    """
    Transform image from RGB to BGR colorspace.

    Parameters
    ----------
        image : numpy.ndarray
            Image to transform in RGB space.

    Returns
    -------
        numpy.ndarray
            Transformed to BGR image.

    Raises
    ------
        ValueError
            If image is not numpy.ndarray instance
            or shape does not corresponds to ['W', 'H', 3].
    """
    if not isinstance(image, NDArray[Shape["*, *, 3"], Int]):
        raise ValueError("Image have to be RGB-like numpy.ndarray with shape ['W', 'H', 3] and integer datatype")
    return cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)


def bgr2rgb(image: NDArray) -> NDArray:
    """
    Transform image from BGR to RGB colorspace.

    Parameters
    ----------
        image : numpy.ndarray
            Image to transform in BGR space.

    Returns
    -------
        numpy.ndarray
            Transformed to RGB image.

    Raises
    ------
        ValueError
            If image is not numpy.ndarray instance
            or shape does not corresponds to ['W', 'H', 3].
    """
    if not isinstance(image, NDArray[Shape["*, *, 3"], Int]):
        raise ValueError("Image have to be BGR-like numpy.ndarray with shape ['W', 'H', 3] and integer datatype")
    return cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)


def rgb2hsv(image: NDArray) -> NDArray:
    """
    Transform image from RGB to HSV colorspace.

    Parameters
    ----------
        image : numpy.ndarray
            Image to transform in RGB space.

    Returns
    -------
        numpy.ndarray
            Transformed to HSV image.

    Raises
    ------
        ValueError
            If image is not numpy.ndarray instance
            or shape does not corresponds to ['W', 'H', 3].
    """
    if not isinstance(image, NDArray[Shape["*, *, 3"], Int]):
        raise ValueError("Image have to be RGB-like numpy.ndarray with shape ['W', 'H', 3] and integer datatype")
    return cv2.cvtColor(src=image, code=cv2.COLOR_RGB2HSV)


def bgr2hsv(image: NDArray) -> NDArray:
    """
    Transform image from BGR to HSV colorspace.

    Parameters
    ----------
        image : numpy.ndarray
            Image to transform in BGR space.

    Returns
    -------
        numpy.ndarray
            Transformed to HSV image.

    Raises
    ------
        ValueError
            If image is not numpy.ndarray instance
            or shape does not corresponds to ['W', 'H', 3].
    """
    if not isinstance(image, NDArray[Shape["*, *, 3"], Int]):
        raise ValueError("Image have to be BGR-like numpy.ndarray with shape ['W', 'H', 3] and integer datatype")
    return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
