"""
    Copyright 2024 YouYoung Icewired Du

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

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