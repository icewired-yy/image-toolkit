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

    This file defines image intermediate format.

"""
import numpy as np


__all__ = [
    'ImageIntermediate'
]


class ImageIntermediate:
    """
    The class to represent the image intermediate format.
    The image intermediate format is a numpy array with the shape of [B, C, H, W].
    """

    def __init__(self,
                 data: np.ndarray,):
        """"""
        self.data = data
        ...

    def GetData(self) -> np.ndarray:
        """ Get the data in the intermediate format. """
        return self.data

    def GetNumOfImages(self) -> int:
        """ Get the number of images in the intermediate format. """
        return self.data.shape[0]

    def GetNumOfChannels(self) -> int:
        """ Get the number of channels in the intermediate format. """
        return self.data.shape[1]

    def GetResolution(self) -> tuple[int, int]:
        """ Get the resolution of the image. """
        return self.data.shape[2], self.data.shape[3]

    def Union(self, other: 'ImageIntermediate') -> 'ImageIntermediate':
        """ Union the two image intermediate data. """
        # Check the resolution
        # If the resolution is not matched, we need to resize the image.
        # By default, we use the bi-linear interpolation and consider the larger resolution to be the target resolution.
        # We use the 'Area' of image to determine whether the resolution is larger or smaller.
        self_resolution = self.GetResolution()
        other_resolution = other.GetResolution()

        if self_resolution[0] != other_resolution[0] or self_resolution[1] != other_resolution[1]:
            # check that whether the transpose can solve the problem.
            if self_resolution[0] == other_resolution[1] and self_resolution[1] == other_resolution[0]:
                # Transpose the image
                other.data = other.data.transpose((0, 1, 3, 2))
            else:
                # Need to consider: should we resize the image by us, or let the user resize the image by themselves?
                # ---------------------------------------------------------------------------------------------------
                # Resize the image
                # We need to resize the image to the larger resolution.
                # We use the 'Area' method to resize the image.
                # if self_resolution[0] * self_resolution[1] > other_resolution[0] * other_resolution[1]:
                #     other.data = torch.nn.functional.interpolate(other.data, size=(self_resolution[0], self_resolution[1]), mode='bicubic')
                # else:
                #     self.data = torch.nn.functional.interpolate(self.data, size=(other_resolution[0], other_resolution[1]), mode='bicubic')
                raise ValueError(f"The resolution of the two images is not matched. "
                                 f"Get one with the resolution of {self_resolution}, "
                                 f"and the other with the resolution of {other_resolution}.")

        # Union the two images
        return ImageIntermediate(np.concatenate([self.data, other.data], axis=0))
    ...
