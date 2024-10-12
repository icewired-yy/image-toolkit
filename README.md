# FaYE-image
A image toolkit designed to convert the image file from `any supported type` to `another one` with **only one line code**.

current supported file types:
- .png
- .jpg / .jpeg
- .exr
- .gif
- .npy

current supported runtime data types:
- numpy.ndarray
- torch.Tensor
- cv2.Mat
- PIL.Image.Image
- matplotlib.pyplot.Figure

## News
- **2021.10.12**: [FaYE 0.2.0](update_record/v0.2.0.md) is released, bringing a totally new version of FaYE-image, with more operations and easier interfaces.

## Requirement
### Necessary packages
- numpy
### Optional packages
- matplotlib
- PIL
- opencv-python
- torch
- OpenEXR

## Usage
**Installation**:
```bash
pip install faye-image
```

To achieve this easy and uniform image data IO operations, you only need to add:
```python
from faye_image import *
```
at the beginning of your code.

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

## Extension
If you want to make this module compatible with more image data types of your interest,
     you just need to implement a corresponding builder class inherited from the `ImageDataBuilder` class.

Then, call the `RegisterBuilders(builder1, builder2, ...)` at the end of your .py file to add the builder to the image factory.


The builder class should implement the following methods:
- `CanBuild(data) -> bool`: Check if the builder can build the data.
- `GetTag() -> str`: Get the tag of the builder.
- `BuildIntermediate(data) -> ImageIntermediate`: Build the image intermediate from the data.
    For current version, the image intermediate is a numpy array in `[BxCxHxW]` in `float32`.
- `BuildData(intermediate: ImageIntermediate, **kwargs)`: Build the data from the image intermediate.
    Since the `B` maybe not 1, the batch dimension should be considered in the implementation.
    Returning a list of data is also supported.