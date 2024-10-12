import cv2


__all__ = ['NEAREST_INTER', 'LINEAR_INTER', 'CUBIC_INTER', 'resize_image']


"""
Constants
"""
NEAREST_INTER = cv2.INTER_NEAREST
LINEAR_INTER = cv2.INTER_LINEAR
CUBIC_INTER = cv2.INTER_CUBIC


def resize_image(image, new_size, interpolation=LINEAR_INTER):
    """
    Resize the image with the specified scale and interpolation method.

    Args:
        image:          The image to be resized.
        new_size:       The new size of the image.
        interpolation:  The interpolation method.

    Returns:
        The resized image.
    """
    return cv2.resize(image, new_size, interpolation=interpolation)
