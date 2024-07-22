"""
    Copyright (c) 2024 YouYoung Icewired Du

    The copyright of this code belongs to YouYoung Icewired Du. Any use of this code implies that you agree to abide by the terms in the accompanying LICENSE file.
    Any use not explicitly authorized by the LICENSE is prohibited.

    If you have any questions or comments, please send an email to duyouyang957@gmail.com to contact us.

    This code is released under the Apache 2.0 License. The full text of the license can be found in the accompanying LICENSE file.
    This code is provided "as is" without any express or implied warranties. Under no circumstances shall YouYoung Icewired Du be held liable for any claims, damages, or other liabilities arising from the use of this code.

    --------------------------------------------------------------------------------------------------------------------

    IMAGE IO MODULE

    This module is aiming at provide uniform interfaces for image data IO operations and format conversions.
    With this module, the image information can be easily converted between different formats, such as numpy, torch, cv::Mat, PIL, etc.
    For the file formats, such as PNG, JPEG, GIF, EXR, etc., the module also provides the interfaces to read and write the image data.
    All the file types and runtime data types are unified.

        e.g.:
            If you want to load a PNG image file into a torch tensor, you can simply call:
                ```python
                    tensor = Convert('path/to/image.png', from_type=PNG_FILE, to_type=TORCH)
                ```
            Or precisely version:
                ```python
                    intermediate = From('path/to/image.png', data_type=PNG_FILE)
                    tensor = To(intermediate, data_type=TORCH)
                ```

            If you want to save a torch tensor to a PNG image file, you can simply call:
                ```python
                    Convert(tensor, from_type=TORCH, to_type=PNG_FILE, save_path='path/to/image.png', save_mode='RGB')
                ```
            Or precisely version:
                ```python
                    intermediate = From(tensor, data_type=TORCH)
                    To(intermediate, data_type=PNG_FILE, save_path='path/to/image.png', save_mode='RGB')
                ```

            If you want to convert a numpy image data to a cv::Mat, you can simply call:
                ```python
                    mat = Convert(numpy_image, from_type=NUMPY, to_type=CV_MAT)
                ```
            Or precisely version:
                ```python
                    intermediate = From(numpy_image, data_type=NUMPY)
                    mat = To(intermediate, data_type=CV_MAT)
                ```

    To achieve this easy and uniform image data IO operations, you only need to add:
        ```python
            from imageIO import *
        ```
    at the beginning of your code.



    EXTENSION:

    If you want to make this module compatible with more image data types of your interest,
     you just need to implement a corresponding builder class inherited from the ImageDataBuilder class.
    Then, call the RegisterBuilders(builder1, builder2, ...) at the end of your .py file to add the builder to the image factory.

    The builder class should implement the following methods:
        - CanBuild(data) -> bool: Check if the builder can build the data.
        - GetTag() -> str: Get the tag of the builder.
        - BuildIntermediate(data) -> ImageIntermediate: Build the image intermediate from the data.
            For current version, the image intermediate is a numpy array in [BxCxHxW] in float32.
        - BuildData(intermediate: ImageIntermediate, **kwargs) -> Build the data from the image intermediate.
            Since the 'B' maybe not 1, the batch dimension should be considered in the implementation.
            Returning a list of data is also supported.
"""

from .builders import NUMPY, TORCH, CV_MAT, PIL_IMAGE, EXR_FILE, PNG_FILE, JPEG_FILE, GIF_FILE, NUMPY_FILE, PLT_FIG
from .image_factory import ImageFactory
from .image_intermediate import ImageIntermediate
from .builders import ImageDataBuilder


__all__ = [
    'NUMPY', 'TORCH', 'CV_MAT', 'PIL_IMAGE',
    'EXR_FILE', 'PNG_FILE', 'JPEG_FILE', 'GIF_FILE', 'NUMPY_FILE', 'PLT_FIG',
    'From', 'To', 'Convert', 'RegisterBuilders',
    'ImageDataBuilder'
]


_image_factory = ImageFactory()


def From(data, data_type=None) -> ImageIntermediate:
    """
    Convert the data from the specified type to the image intermediate format.

    Args:
        data:       The data to be converted.
        data_type:  The type of the data.

    Returns:
        The data in the image intermediate format.
    """
    # Infer the data type if not specified
    return _image_factory.CreateIntermediate(data, data_type)


def To(intermediate,
       data_type: str,
       **kwargs):
    """
    Convert the image intermediate format to the specified type.

    If the data type is a file type, the [save_path] should be specified.
        For PNG_FILE, you can also specify the [save_mode], like 'I' or 'RGB'.
        For GIF_FILE, you can also specify the [duration] and [loop].

    Args:
        intermediate:   The data in the image intermediate format.
        data_type:  The type of the data to be converted to.

    Returns:
        The data in the specified type.
    """
    return _image_factory.CreateData(intermediate, data_type, **kwargs)


def Convert(data, *, from_type, to_type, **kwargs):
    """
    Convert the data from the specified type to the specified type.

    If the data type is a file type, the [save_path] should be specified, or it will be saved to the current directory.
    For PNG_FILE, you can also specify the [save_mode], like 'I' or 'RGB'.
    For GIF_FILE, you can also specify the [duration] and [loop].

    Args:
        data:       The data to be converted.
        from_type:  The type of the data.
        to_type:    The type of the data to be converted to.

    Returns:
        The data in the specified type.
    """
    intermediate = From(data, from_type)
    return To(intermediate, to_type, **kwargs)


def RegisterBuilders(*new_builders):
    """
    Register a single builder to the factory.

    Args:
        new_builders:   The new builders to be registered.
    """
    _image_factory.RegisteredBuilders(list(new_builders))
