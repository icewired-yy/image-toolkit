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

    The implementation of the image builders based on the array data.

"""


import numpy as np
from typing import Final
from .builder_interface import ImageDataBuilder
from ..image_intermediate import ImageIntermediate


"""
    The tags for the image builders.
"""
NUMPY_RT: Final[str] = 'numpy'                 # B x H x W x C
TORCH_RT: Final[str] = 'torch'                 # B x C x H x W or C x H x W
CV_MAT_RT: Final[str] = 'cvMat'                # B x H x W x C
PIL_IMAGE_RT: Final[str] = 'PIL'               # PIL image or list of PIL images
PLT_FIG_RT: Final[str] = 'plt_fig'             # PLT fig or list of PLT figs


__all__ = [
    'NumpyImageDataBuilder', 'TorchImageDataBuilder',
    'MATImageBuilder', 'PILImageDataBuilder',
    'PLTFigDataBuilder',
    'NUMPY_RT', 'TORCH_RT', 'CV_MAT_RT', 'PIL_IMAGE_RT',
    'PLT_FIG_RT'
]


class NumpyImageDataBuilder(ImageDataBuilder):
    def CanBuild(self, data) -> bool:
        return isinstance(data, np.ndarray)

    def GetTag(self) -> str:
        return NUMPY_RT

    def BuildIntermediate(self, data) -> ImageIntermediate:
        """
            The input default shape Numpy image data should be [B,H,W,C], [H,W,C] or [H,W].

        :param data:   The numpy image data.
        :return:       The image intermediate.
        """
        # To B x C x H x W
        len_dimensions = len(data.shape)
        if len_dimensions == 2:
            data = np.expand_dims(data, axis=-1)
            data = data.transpose((2, 0, 1))
            data = np.expand_dims(data, axis=0)
        elif len_dimensions == 3:
            data = data.transpose((2, 0, 1))
            data = np.expand_dims(data, axis=0)
        elif len_dimensions == 4:
            data = data.transpose((0, 3, 1, 2))
        else:
            raise ValueError(f"The shape of the input data {data.shape} is invalid.")
        return ImageIntermediate(data)

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        """
            B x C x H x W -> B x H x W x C.

        :param intermediate:   The image intermediate.
        :return:               The numpy image data.
        """
        return intermediate.GetData().transpose(0, 2, 3, 1)


class TorchImageDataBuilder(ImageDataBuilder):
    def __init__(self):
        super(TorchImageDataBuilder, self).__init__()
        self._prepared = True
        try:
            import torch
        except ImportError:
            self._prepared = False

    def CanBuild(self, data) -> bool:
        if not self._prepared:
            return False
        import torch
        return isinstance(data, torch.Tensor)

    def GetTag(self) -> str:
        return TORCH_RT

    def BuildIntermediate(self, data) -> ImageIntermediate:
        """
            The input default shape Torch image data should be [H W], [C H W], [B C H W].

        :param data:   The torch image data.
        :return:       The image intermediate.
        """
        if not self._prepared:
            raise EnvironmentError(
                """
                Torch is not installed. Please install torch to use the TorchImageDataBuilder.
                
                You can install torch by running:
                pip install torch
                
                Or go to the official website to find a proper version:
                https://pytorch.org
                """)

        if len(data.shape) == 2:
            data = data.unsqueeze(0)
            data = data.unsqueeze(0)
        elif len(data.shape) == 3:
            data = data.unsqueeze(0)
        return ImageIntermediate(data.detach().cpu().numpy())

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        """
            B x C x H x W

        :param intermediate:   The image intermediate.
        :return:               The torch image data.
        """
        if not self._prepared:
            raise EnvironmentError(
                "Torch is not installed. Please install torch to use the TorchImageDataBuilder")

        import torch
        return torch.tensor(intermediate.GetData(), dtype=torch.float32)


class MATImageBuilder(ImageDataBuilder):
    def __init__(self):
        super(MATImageBuilder, self).__init__()
        self._prepared = True
        try:
            import cv2
        except ImportError:
            self._prepared = False

    def CanBuild(self, data) -> bool:
        if not self._prepared:
            return False
        import cv2
        return isinstance(data, cv2.Mat)

    def GetTag(self) -> str:
        return CV_MAT_RT

    def BuildIntermediate(self, data) -> ImageIntermediate:
        """
            The input default shape CV image data should be [H W C] or [H W].

        :param data:   The cv image data.
        :return:       The image intermediate.
        """
        if not self._prepared:
            raise EnvironmentError(
                """
                OpenCV is not installed. Please install OpenCV and torch to use the MATImageBuilder
                
                You can install OpenCV by running:
                pip install opencv-python
                """)

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        data = data.transpose((2, 0, 1))
        data = np.expand_dims(data, axis=0)
        return ImageIntermediate(data)

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        """
            B x C x H x W -> B x H x W x C.

        :param intermediate:   The image intermediate.
        :return:               The cv image data.
        """
        if not self._prepared:
            raise EnvironmentError(
                "OpenCV is not installed. Please install OpenCV and torch to use the MATImageBuilder")

        return intermediate.GetData().transpose(0, 2, 3, 1)


class PILImageDataBuilder(ImageDataBuilder):
    def __init__(self):
        super(PILImageDataBuilder, self).__init__()
        self._prepared = True
        try:
            import PIL.Image
        except ImportError:
            self._prepared = False

    def CanBuild(self, data) -> bool:
        if not self._prepared:
            return False
        import PIL.Image
        return isinstance(data, PIL.Image.Image)

    def GetTag(self) -> str:
        return PIL_IMAGE_RT

    def BuildIntermediate(self, data) -> ImageIntermediate:
        """
            The input default shape PIL image data should be [H W C] or [H W].

        :param data:   The PIL image data.
        :return:       The image intermediate.
        """
        if not self._prepared:
            raise EnvironmentError(
                """
                PIL is not installed. Please install PIL and torch to use the PILImageDataBuilder
                
                You can install PIL by running:
                pip install pillow
                """)

        data = np.array(data)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        data = data.transpose((2, 0, 1))
        data = np.expand_dims(data, axis=0)
        return ImageIntermediate(data)

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        """
            If B is not 1, then return a list of PIL images, otherwise return a single PIL image.

        :param intermediate:   The image intermediate.
        :return:               The PIL image data.
        """
        if not self._prepared:
            raise EnvironmentError(
                "PIL is not installed. Please install PIL and torch to use the PILImageDataBuilder")

        import PIL.Image
        if intermediate.GetNumOfImages() == 1:
            # Get the value range of the intermediate data
            # If the value is not from 0 to 255, then normalize it to 0-255
            data = intermediate.GetData().transpose(0, 2, 3, 1).squeeze(0)
            if data.dtype in [np.float32, np.float64, np.float16]:
                data = np.clip(data, 0, 1)
                data = data * 255
            elif data.dtype in [np.uint8, np.uint16, np.int32, np.int64, np.int16, np.int8]:
                data = np.clip(data, 0, 255)
            else:
                raise ValueError(f"The data type {data.dtype} is not supported.")
            # If only one channel, then squeeze it
            if data.shape[-1] == 1:
                data = np.squeeze(data, axis=-1)
            return PIL.Image.fromarray(data.astype(np.uint8))
        else:
            data = intermediate.GetData().transpose(0, 2, 3, 1)
            images = []
            for i in range(data.shape[0]):
                image = data[i]
                if image.dtype in [np.float32, np.float64, np.float16]:
                    image = np.clip(image, 0, 1)
                    image = image * 255
                elif image.dtype in [np.uint8, np.uint16, np.int32, np.int64, np.int16, np.int8]:
                    image = np.clip(image, 0, 255)
                else:
                    raise ValueError(f"The data type {image.dtype} is not supported.")

                if image.shape[-1] == 1:
                    image = np.squeeze(image, axis=-1)
                images.append(PIL.Image.fromarray(image.astype(np.uint8)))
            return images


class PLTFigDataBuilder(ImageDataBuilder):
    def __init__(self):
        super(PLTFigDataBuilder, self).__init__()
        self._prepared = True
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self._prepared = False

    def CanBuild(self, data) -> bool:
        if not self._prepared:
            return False
        import matplotlib.pyplot as plt
        return isinstance(data, plt.Figure)

    def GetTag(self) -> str:
        return PLT_FIG_RT

    def BuildIntermediate(self, data) -> ImageIntermediate:
        """
            The input data should be a PLT fig.

        :param data:   The PLT fig.
        :return:       The image intermediate.
        """
        if not self._prepared:
            raise EnvironmentError(
                """
                Matplotlib is not installed. Please install Matplotlib and torch to use the PLTFigDataBuilder
                
                You can install Matplotlib by running:
                pip install matplotlib
                """)

        # Get the canvas of the figure
        canvas = data.canvas
        canvas.draw()
        width, height = data.get_size_inches() * data.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return ImageIntermediate(image)

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        """
            If B is not 1, then return a list of PLT figs, otherwise return a single PLT fig.

        :param intermediate:   The image intermediate.
        :return:               The PLT fig.
        """
        if not self._prepared:
            raise EnvironmentError(
                "Matplotlib is not installed. Please install Matplotlib and torch to use the PLTFigDataBuilder")

        import matplotlib.pyplot as plt
        if intermediate.GetNumOfImages() == 1:
            data = intermediate.GetData().transpose(0, 2, 3, 1)[0]
            fig, ax = plt.subplots()
            ax.imshow(data)
            ax.axis('off')
            return fig
        else:
            data = intermediate.GetData().transpose(0, 2, 3, 1)
            num_images = data.shape[0]
            figs = []
            for i in range(num_images):
                image = data[i]
                fig, ax = plt.subplots()
                ax.imshow(image)
                ax.axis('off')
                figs.append(fig)
            return figs
