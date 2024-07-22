"""
    Copyright (c) 2024 YouYoung Icewired Du

    The copyright of this code belongs to YouYoung Icewired Du. Any use of this code implies that you agree to abide by the terms in the accompanying LICENSE file.
    Any use not explicitly authorized by the LICENSE is prohibited.

    If you have any questions or comments, please send an email to duyouyang957@gmail.com to contact us.

    This code is released under the Apache 2.0 License. The full text of the license can be found in the accompanying LICENSE file.
    This code is provided "as is" without any express or implied warranties. Under no circumstances shall YouYoung Icewired Du be held liable for any claims, damages, or other liabilities arising from the use of this code.

    --------------------------------------------------------------------------------------------------------------------

    The interface of the image intermediate builder.

"""


from abc import ABC, abstractmethod
from ..image_intermediate import ImageIntermediate


__all__ = [
    'ImageDataBuilder'
]


class ImageDataBuilder(ABC):
    """ The interface of the image intermediate builder. """

    @abstractmethod
    def CanBuild(self, data) -> bool:
        """ Check whether the data can be built by the image intermediate builder. """
        ...

    @abstractmethod
    def GetTag(self) -> str:
        """ Get the tag of the image intermediate builder. """
        ...

    @abstractmethod
    def BuildIntermediate(self, data) -> ImageIntermediate:
        """ Build the image intermediate from the data of interest. """
        ...

    @abstractmethod
    def BuildData(self, intermediate: ImageIntermediate,  **kwargs):
        """ Build the data from the image intermediate. """
        ...