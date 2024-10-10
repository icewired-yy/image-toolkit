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

    The implementation of the image builders based on the file data.

"""


import os
import time
import numpy as np
from typing import Final
from .builder_interface import ImageDataBuilder
from ..image_intermediate import ImageIntermediate


"""
    The tags for the image builders.
"""
EXR_FILE: Final[str] = 'exr_file'
PNG_FILE: Final[str] = 'png_file'
JPEG_FILE: Final[str] = 'jpeg_file'
GIF_FILE: Final[str] = 'gif_file'
NUMPY_FILE: Final[str] = 'numpy_file'


__all__ = [
    'EXRImageFileBuilder', 'PNGImageFileBuilder',
    'JPEGImageFileBuilder', 'GIFImageFileBuilder',
    'NumpyFileBuilder',
    'EXR_FILE', 'PNG_FILE', 'JPEG_FILE', 'GIF_FILE', 'NUMPY_FILE'
]


class EXRImageFileBuilder(ImageDataBuilder):
    def __init__(self):
        super(EXRImageFileBuilder, self).__init__()
        self._prepared = True
        try:
            import OpenEXR
            import Imath
        except ImportError:
            self._prepared = False

    def CanBuild(self, datapath) -> bool:
        if not self._prepared:
            return False

        if isinstance(datapath, str):
            datapath = os.path.join(datapath)
            file_extension = os.path.basename(datapath).split('.')[-1].lower()
            return file_extension == 'exr'
        return False

    def GetTag(self) -> str:
        return EXR_FILE

    def BuildIntermediate(self, datapath) -> ImageIntermediate:
        if not self._prepared:
            raise ImportError("""
            The OpenEXR package is not installed. Please install OpenEXR to use the EXR_FILE.
            
            You can install OpenEXR by running the following command:
            pip install OpenEXR
            """)

        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(datapath)
        exr_file_header = exr_file.header()
        dw = exr_file_header['dataWindow']
        channels = exr_file_header['channels']

        num_channels = len(channels)
        if num_channels == 1:
            image_channels = ['Z']
        elif num_channels == 3:
            image_channels = ['R', 'G', 'B']
        elif num_channels == 4:
            image_channels = ['R', 'G', 'B', 'A']
        else:
            raise ValueError(f"Unsupported channel number: {num_channels}")

        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        image_data_per_channel_list = []
        for channel_name in image_channels:
            channel = exr_file.channel(channel_name, FLOAT)
            img_data = np.frombuffer(channel, dtype=np.float32)
            image_data_per_channel_list.append(img_data.reshape(size[1], size[0]).copy())

        image_data = np.stack(image_data_per_channel_list, axis=0)[np.newaxis, ...]
        return ImageIntermediate(image_data)

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        """
            If the number of channels is 1, we consider this as a depth map, and save in 'Z' channel.
            If the number of channels is 3, we consider this as an RGB image, and save in 'R', 'G', 'B' channels.
            If the number of channels is 4, we consider this as an RGBA image, and save in 'R', 'G', 'B', 'A' channels.

            There should be a 'save_path' in the kwargs. If not provided, we will use the original path to save the image.
            If the B > 1, we will save all the images, and modify the file name with the index adding behind, e.g., 'xxx_0.exr'.
        :param intermediate:    ImageIntermediate
        :return:                None
        """
        if not self._prepared:
            raise ImportError("The OpenEXR package is not installed. Please install OpenEXR to use the EXR_FILE.")

        import OpenEXR
        import Imath

        image_data = intermediate.GetData()
        image_channels = intermediate.GetNumOfChannels()
        num_images = intermediate.GetNumOfImages()
        save_path = kwargs.get('save_path', None)
        if save_path is None:
            # Use the time stamp as the save name, and current path as the save path
            save_path = os.path.join("", "image_" + str(time.time()) + ".exr")

        if image_channels == 1:
            channel_name_list = ['Z']
        elif image_channels == 3:
            channel_name_list = ['R', 'G', 'B']
        elif image_channels == 4:
            channel_name_list = ['R', 'G', 'B', 'A']
        else:
            raise ValueError(f"Unsupported channel number: {image_data.shape[1]}")

        if num_images == 1:
            image_data = image_data.squeeze(0)
            height, width = image_data.shape[-2], image_data.shape[-1]
            header = OpenEXR.Header(width, height)

            # Clear the channels
            # This operation is not necessary for program correctness, but it is necessary for the visualization
            # Since if we have RGB channel by default setting and if we do not remove it, we cannot visualize the Z channel.
            header['channels'] = {}
            for channel_name in channel_name_list:
                header['channels'][channel_name] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

            pixels = {}
            for i, image_channel_data in enumerate(image_data):
                pixels[channel_name_list[i]] = np.zeros((height, width), dtype=np.float32)
                pixels[channel_name_list[i]][:] = image_channel_data

            exr_file = OpenEXR.OutputFile(save_path, header)
            exr_file.writePixels(pixels)
            exr_file.close()
        else:
            for i in range(num_images):
                image_data_i = image_data[i]
                height, width = image_data_i.shape[-2], image_data_i.shape[-1]
                header = OpenEXR.Header(width, height)
                for channel_name in channel_name_list:
                    header['channels'][channel_name] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

                pixels = {}
                for j, image_channel_data in enumerate(image_data_i):
                    pixels[channel_name_list[j]] = image_channel_data

                exr_file = OpenEXR.OutputFile(save_path.replace('.exr', f'_{i}.exr'), header)
                exr_file.writePixels(pixels)
                exr_file.close()


class PNGImageFileBuilder(ImageDataBuilder):
    def __init__(self):
        super(PNGImageFileBuilder, self).__init__()
        self._prepared = True
        try:
            import PIL.Image as Image
        except ImportError:
            self._prepared = False

    def CanBuild(self, datapath) -> bool:
        if not self._prepared:
            return False

        if isinstance(datapath, str):
            datapath = os.path.join(datapath)
            file_extension = os.path.basename(datapath).split('.')[-1].lower()
            return file_extension == 'png'
        return False

    def GetTag(self) -> str:
        return PNG_FILE

    def BuildIntermediate(self, datapath) -> ImageIntermediate:
        """ """
        if not self._prepared:
            raise ImportError("""
            The PIL package is not installed. Please install PIL to use the PNG_FILE.
            
            You can install PIL by running the following command:
            pip install Pillow
            """)

        import PIL.Image as Image

        image = Image.open(datapath)
        image_mode = image.mode

        if image_mode == 'I':
            image_data = np.array(image)
            image_data = np.expand_dims(image_data, axis=0)
        elif image_mode == 'L':
            image_data = np.array(image)
            image_data = np.expand_dims(image_data, axis=0)
        elif image_mode == 'RGB':
            image_data = np.array(image)
            image_data = image_data.transpose((2, 0, 1))
        elif image_mode == 'RGBA':
            image_data = np.array(image)
            image_data = image_data.transpose((2, 0, 1))
        elif image_mode == 'F':
            image_data = np.array(image)
            image_data = image_data.transpose((2, 0, 1))
        else:
            raise ValueError(f"Unsupported image mode: {image_mode}")

        image_data = image_data[np.newaxis, ...]

        return ImageIntermediate(image_data)

    @staticmethod
    def _ImageDataCasting(img, max_val=255):
        data_type = img.dtype
        if data_type in [np.float32, np.float64, np.float16]:
            img = np.clip(img, 0, 1)
            img = img * max_val
        elif data_type in [np.uint8, np.uint16, np.uint32, np.int32, np.int64]:
            pass
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        return img

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        """
            If the B > 1, we will save all the images, and modify the file name with the index adding behind, e.g., 'xxx_0.png'.
        :param intermediate:    ImageIntermediate
        :return:                None
        """
        if not self._prepared:
            raise ImportError("The PIL package is not installed. Please install PIL to use the PNG_FILE.")
        import PIL.Image as Image

        image_data = intermediate.GetData()
        num_images = intermediate.GetNumOfImages()
        save_path = kwargs.get('save_path', None)
        save_mode = kwargs.get('save_mode', None)
        if save_path is None:
            # Use the time stamp as the save name, and current path as the save path
            save_path = os.path.join("", "image_" + str(time.time()) + ".png")

        # If the save mode is not provided, we will infer it from the image data
        if save_mode is None:
            if image_data.shape[1] == 1:
                save_mode = 'L'
            elif image_data.shape[1] == 3:
                save_mode = 'RGB'
            elif image_data.shape[1] == 4:
                save_mode = 'RGBA'
            else:
                raise ValueError(f"Unsupported channel number: {image_data.shape[1]}")

        if num_images == 1:
            image_data = image_data.squeeze(0)
            image_data = image_data.transpose(1, 2, 0)

            if save_mode == 'L':
                image_data = self._ImageDataCasting(image_data, 255)
                image_data = image_data.astype(np.uint8)[..., 0]
                image_data = Image.fromarray(image_data, mode='L')
                image_data.save(save_path)
            elif save_mode == 'RGB':
                image_data = self._ImageDataCasting(image_data, 255)
                image_data = image_data.astype(np.uint8)
                if image_data.shape[-1] == 1:
                    image_data = np.concatenate([image_data, image_data, image_data], axis=-1)
                image_data = Image.fromarray(image_data, mode='RGB')
                image_data.save(save_path)
            elif save_mode == 'RGBA':
                image_data = self._ImageDataCasting(image_data, 255)
                image_data = image_data.astype(np.uint8)
                if image_data.shape[-1] == 1:
                    image_data = np.concatenate([image_data, image_data, image_data, image_data], axis=-1)
                image_data = Image.fromarray(image_data, mode='RGBA')
                image_data.save(save_path)
            elif save_mode == 'I':
                image_data = self._ImageDataCasting(image_data, 65535)
                image_data = image_data.astype(np.int32)[..., 0]
                image_data = Image.fromarray(image_data, mode='I')
                image_data.save(save_path)
            else:
                raise ValueError(f"Unsupported save mode: {save_mode}")
        else:
            for i in range(num_images):
                image_data_i = image_data[i]
                image_data_i = image_data_i.transpose(1, 2, 0)

                if save_mode == 'L':
                    image_data = self._ImageDataCasting(image_data)
                    image_data_i = image_data_i.astype(np.uint8)[..., 0]
                    image_data_i = Image.fromarray(image_data_i, mode='L')
                    image_data_i.save(save_path.replace('.png', f'_{i}.png'))
                elif save_mode == 'RGB':
                    image_data = self._ImageDataCasting(image_data)
                    image_data_i = image_data_i.astype(np.uint8)
                    if image_data_i.shape[-1] == 1:
                        image_data_i = np.concatenate([image_data_i, image_data_i, image_data_i], axis=-1)
                    image_data_i = Image.fromarray(image_data_i, mode='RGB')
                    image_data_i.save(save_path.replace('.png', f'_{i}.png'))
                elif save_mode == 'RGBA':
                    image_data = self._ImageDataCasting(image_data)
                    image_data_i = image_data_i.astype(np.uint8)
                    if image_data_i.shape[-1] == 1:
                        image_data_i = np.concatenate([image_data_i, image_data_i, image_data_i, image_data_i], axis=-1)
                    image_data_i = Image.fromarray(image_data_i, mode='RGBA')
                    image_data_i.save(save_path.replace('.png', f'_{i}.png'))
                elif save_mode == 'I':
                    image_data = self._ImageDataCasting(image_data, 65535)
                    image_data_i = image_data_i.astype(np.int32)[..., 0]
                    image_data_i = Image.fromarray(image_data_i, mode='I')
                    image_data_i.save(save_path.replace('.png', f'_{i}.png'))
                else:
                    raise ValueError(f"Unsupported save mode: {save_mode}")


class JPEGImageFileBuilder(ImageDataBuilder):
    def __init__(self):
        super(JPEGImageFileBuilder, self).__init__()
        self._prepared = True
        try:
            import PIL.Image as Image
        except ImportError:
            self._prepared = False

    def CanBuild(self, datapath) -> bool:
        if not self._prepared:
            return False

        if isinstance(datapath, str):
            datapath = os.path.join(datapath)
            file_extension = os.path.basename(datapath).split('.')[-1].lower()
            return file_extension == 'jpeg' or file_extension == 'jpg'
        return False

    def GetTag(self) -> str:
        return JPEG_FILE

    def BuildIntermediate(self, datapath) -> ImageIntermediate:
        """ Consider JPG only has 8 uint image. """
        if not self._prepared:
            raise ImportError("""
            The PIL package is not installed. Please install PIL to use the JPEG_FILE.
            
            You can install PIL by running the following command:
            pip install Pillow
            """)
        import PIL.Image as Image

        image = Image.open(datapath)
        image_data = np.array(image)
        image_data = image_data.transpose((2, 0, 1))
        image_data = image_data[np.newaxis, ...]
        return ImageIntermediate(image_data)

    @staticmethod
    def _ImageDataCasting(img, max_val=255):
        data_type = img.dtype
        if data_type in [np.float32, np.float64, np.float16]:
            img = np.clip(img, 0, 1)
            img = img * max_val
        elif data_type in [np.uint8, np.uint16, np.uint32, np.int32, np.int64]:
            pass
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        return img

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        """
            If the B > 1, we will save all the images, and modify the file name with the index adding behind, e.g., 'xxx_0.jpeg'.
        :param intermediate:    ImageIntermediate
        :return:                None
        """
        if not self._prepared:
            raise ImportError("The PIL package is not installed. Please install PIL to use the JPEG_FILE.")
        import PIL.Image as Image

        image_data = intermediate.GetData()
        num_images = intermediate.GetNumOfImages()
        save_path = kwargs.get('save_path', None)
        if save_path is None:
            # Use the time stamp as the save name, and current path as the save path
            save_path = os.path.join("", "image_" + str(time.time()) + ".jpg")

        if num_images == 1:
            image_data = image_data.squeeze(0)
            image_data = image_data.transpose(1, 2, 0)
            image_data = self._ImageDataCasting(image_data)
            image_data = image_data.astype(np.uint8)
            if image_data.shape[-1] == 1:
                image_data = np.concatenate([image_data, image_data, image_data], axis=-1)
            image_data = Image.fromarray(image_data)
            image_data.save(save_path)
        else:
            for i in range(num_images):
                image_data_i = image_data[i]
                image_data_i = image_data_i.transpose(1, 2, 0)
                image_data = self._ImageDataCasting(image_data)
                image_data_i = image_data_i.astype(np.uint8)
                if image_data_i.shape[-1] == 1:
                    image_data_i = np.concatenate([image_data_i, image_data_i, image_data_i], axis=-1)
                Image.fromarray(image_data_i).save(save_path.replace('.jpg', f'_{i}.jpg'))


class GIFImageFileBuilder(ImageDataBuilder):
    def __init__(self):
        super(GIFImageFileBuilder, self).__init__()
        self._prepared = True
        try:
            import PIL.Image as Image
        except ImportError:
            self._prepared = False

    def CanBuild(self, data) -> bool:
        if not self._prepared:
            return False

        if isinstance(data, str):
            data = os.path.join(data)
            file_extension = os.path.basename(data).split('.')[-1].lower()
            return file_extension == 'gif'
        return False

    def GetTag(self) -> str:
        return GIF_FILE

    @staticmethod
    def _ImageDataCasting(img, max_val=255):
        data_type = img.dtype
        if data_type in [np.float32, np.float64, np.float16]:
            img = np.clip(img, 0, 1)
            img = img * max_val
        elif data_type in [np.uint8, np.uint16, np.uint32, np.int32, np.int64]:
            pass
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        return img

    def BuildIntermediate(self, datapath) -> ImageIntermediate:
        """ Consider multi frames as batch. """
        if not self._prepared:
            raise ImportError("""
            The PIL package is not installed. Please install PIL to use the GIF_FILE.
            
            You can install PIL by running the following command:
            pip install Pillow
            """)
        import PIL.Image as Image

        gif = Image.open(datapath)

        gif_frames = []
        frame_count = 0

        while True:
            try:
                gif.seek(frame_count)
                gif_frame = np.array(gif)
                gif_frame = gif_frame.transpose((2, 0, 1))
                gif_frames.append(gif_frame)
                frame_count += 1
            except EOFError:
                break

        gif_data = np.stack(gif_frames, axis=0)

        return ImageIntermediate(gif_data)

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        """ Batch will be considered as multi frames. """
        if not self._prepared:
            raise ImportError("The PIL package is not installed. Please install PIL to use the GIF_FILE.")
        import PIL.Image as Image

        gif_data = intermediate.GetData().transpose(0, 2, 3, 1)
        save_path = kwargs.get('save_path', None)
        duration = kwargs.get('duration', 200)
        loop = kwargs.get('loop', 0)
        if save_path is None:
            # Use the time stamp as the save name, and current path as the save path
            save_path = os.path.join("", "image_" + str(time.time()) + ".gif")

        images = []
        for i in range(gif_data.shape[0]):
            image = gif_data[i]
            image = self._ImageDataCasting(image)
            image = image.astype(np.uint8)
            if image.shape[-1] == 1:
                image = np.concatenate([image, image, image], axis=-1)
            images.append(Image.fromarray(image))

        images[0].save(save_path,
                       save_all=True,
                       append_images=images[1:],
                       duration=duration,
                       loop=loop)


class NumpyFileBuilder(ImageDataBuilder):
    def CanBuild(self, data) -> bool:
        if isinstance(data, str):
            data = os.path.join(data)
            file_extension = os.path.basename(data).split('.')[-1].lower()
            return file_extension == 'npy'
        return False

    def GetTag(self) -> str:
        return NUMPY_FILE

    def BuildIntermediate(self, datapath) -> ImageIntermediate:
        image_data = np.load(datapath)
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, axis=0)
        elif len(image_data.shape) == 3:
            image_data = image_data.transpose((2, 0, 1))
        image_data = image_data[np.newaxis, ...]
        return ImageIntermediate(image_data)

    def BuildData(self, intermediate: ImageIntermediate, **kwargs):
        image_data = intermediate.GetData()
        save_path = kwargs.get('save_path', None)
        if save_path is None:
            # Use the time stamp as the save name, and current path as the save path
            save_path = os.path.join("", "image_" + str(time.time()) + ".npy")

        np.save(save_path, image_data.squeeze(0).transpose(1, 2, 0))
