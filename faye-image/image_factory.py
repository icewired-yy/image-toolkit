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

    The image factory module is used to create the image intermediate format from the data and create the data from the intermediate format.

"""

from .builders import *
from .image_intermediate import *
import copy
from typing import Union


__all__ = [
    'ImageFactory',
    # 'ADD_BUILDER', 'ADD_BUILDERS'
]


"""
    The registered builders for the image intermediate format.
    
    If any new builder is implemented, please add an instance of the builder to this list.
"""
REGISTERED_BUILDERS = [
    NumpyImageDataBuilder(),
    TorchImageDataBuilder(),
    MATImageBuilder(),
    PILImageDataBuilder(),
    EXRImageFileBuilder(),
    PNGImageFileBuilder(),
    JPEGImageFileBuilder(),
    GIFImageFileBuilder(),
    NumpyFileBuilder(),
    PLTFigDataBuilder(),
]


class ImageFactory:
    """"""
    _instance: 'ImageFactory' = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ImageFactory, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        """ Load the registered factory classes from the FACTORY_CLASSES list. """
        self.builder_list = []
        self.builder_tag_class_mapper = {}

        self._init_from_registered_builders_list(REGISTERED_BUILDERS)

    def _init_from_registered_builders_list(self, registered_builders: list[ImageDataBuilder]):
        """ Initialize the factory from the registered builders. """
        for builder in registered_builders:
            builder = copy.deepcopy(builder)
            self.builder_list.append(builder)
            self.builder_tag_class_mapper[builder.GetTag()] = builder

    def RegisterBuilder(self, builder: ImageDataBuilder):
        """ Register a single builder to the factory. """
        self.RegisteredBuilders([builder, ])

    def RegisteredBuilders(self, registered_builders: list[ImageDataBuilder]):
        """ Update the registered builders. """
        for builder in registered_builders:
            if not isinstance(builder, ImageDataBuilder):
                raise ValueError(f"Unsupported builder type: {type(builder)}")

            builder = copy.deepcopy(builder)
            if builder.GetTag() not in self.builder_tag_class_mapper.keys():
                self.builder_list.append(builder)
                self.builder_tag_class_mapper[builder.GetTag()] = builder
            else:
                raise UserWarning(f"Builder with tag {builder.GetTag()} already exists.")

    def CreateIntermediate(self, data, source_data_type: Union[str, None]) -> ImageIntermediate:
        """ Create the builder from the data type. """
        # Handle the NONE data type
        if source_data_type is None:
            if isinstance(data, list):
                current_intermediate = None
                for d in data:
                    for builder in self.builder_list:
                        if builder.CanBuild(d):
                            if current_intermediate is None:
                                current_intermediate = builder.BuildIntermediate(d)
                            else:
                                current_intermediate = current_intermediate.Union(builder.BuildIntermediate(d))
                        else:
                            raise UserWarning(f"Unsupported data type: {type(d)}")
                return current_intermediate
            else:
                for builder in self.builder_list:
                    if builder.CanBuild(data):
                        return builder.BuildIntermediate(data)
        else:
            if isinstance(data, list):
                # Handle the list data type
                current_intermediate = None
                for d in data:
                    if source_data_type in self.builder_tag_class_mapper.keys():
                        if current_intermediate is None:
                            current_intermediate = self.builder_tag_class_mapper[source_data_type].BuildIntermediate(d)
                        else:
                            current_intermediate = current_intermediate.Union(self.builder_tag_class_mapper[source_data_type].BuildIntermediate(d))
                return current_intermediate
            if source_data_type in self.builder_tag_class_mapper.keys():
                return self.builder_tag_class_mapper[source_data_type].BuildIntermediate(data)

        raise ValueError(f"Unsupported data type: {source_data_type}")

    def CreateData(self,
                   intermediate: ImageIntermediate,
                   target_data_type: str,
                   **kwargs):
        """ Create the data from the intermediate format. """
        if target_data_type in self.builder_tag_class_mapper.keys():
            return self.builder_tag_class_mapper[target_data_type].BuildData(intermediate, **kwargs)

        raise ValueError(f"Unsupported data type: {target_data_type}")


# def ADD_BUILDER(builder: ImageDataBuilder):
#     """ Add the builder to the registered builders list. """
#     REGISTERED_BUILDERS.append(builder)
#     ImageFactory().AddRegisteredBuilders([builder,])
#
#
# def ADD_BUILDERS(builders: list[ImageDataBuilder]):
#     """ Add the builders to the registered builders list. """
#     REGISTERED_BUILDERS.extend(builders)
#     ImageFactory().AddRegisteredBuilders(builders)
