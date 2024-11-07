import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import json
from enum import Enum


__all__ = ['ParameterType', 'Visualizer']


class ParameterType(Enum):
    NUMERIC = 1
    IMAGE_PATH = 2

class Parameter:
    def __init__(self, name, param_type, min_value=None, max_value=None, current_value=None):
        self.name = name
        self.param_type = param_type
        self.min_value = min_value
        self.max_value = max_value
        if current_value is not None:
            self.current_value = current_value
        elif param_type == ParameterType.NUMERIC and min_value is not None and max_value is not None:
            self.current_value = (min_value + max_value) / 2
        else:
            self.current_value = None

class Context:
    def __init__(self):
        self.shared_data = {}

    def __getitem__(self, item):
        return self.shared_data[item]

class ParameterWidget:
    def __init__(self, parent, parameter: Parameter, on_change_callback, context: Context):
        self.parent = parent
        self.parameter = parameter
        self.on_change_callback = on_change_callback
        self.context = context
        self.create_widget()

    def create_widget(self):
        pass

    def get_value(self):
        return self.parameter.current_value

class NumericParameterWidget(ParameterWidget):
    def __init__(self, parent, parameter: Parameter, on_change_callback, context: Context):
        super().__init__(parent, parameter, on_change_callback, context)
        self.value_label = None
        self.scale = None
        self.var = None
        self.label = None

    def create_widget(self):
        self.label = ttk.Label(self.parent, text=self.parameter.name)
        self.label.pack(anchor=tk.W, pady=(10, 0))

        self.var = tk.DoubleVar()
        self.var.set(self.parameter.current_value)
        self.scale = ttk.Scale(
            self.parent,
            from_=self.parameter.min_value,
            to=self.parameter.max_value,
            orient=tk.HORIZONTAL,
            variable=self.var,
            command=self.on_value_change
        )
        self.scale.pack(fill=tk.X)

        self.value_label = ttk.Label(self.parent, text=f"{self.var.get():.2f}")
        self.value_label.pack(anchor=tk.W, pady=(5, 0))

    def on_value_change(self, *args):
        self.parameter.current_value = self.var.get()
        self.value_label.config(text=f"{self.var.get():.2f}")
        self.on_change_callback()

class ImagePathParameterWidget(ParameterWidget):
    def __init__(self, parent, parameter: Parameter, on_change_callback, context: Context):
        super().__init__(parent, parameter, on_change_callback, context)
        self.button = None
        self.entry = None
        self.path_var = None
        self.label = None

    def create_widget(self):
        self.label = ttk.Label(self.parent, text=self.parameter.name)
        self.label.pack(anchor=tk.W, pady=(10, 0))

        self.path_var = tk.StringVar()
        self.path_var.set(self.parameter.current_value if self.parameter.current_value else "")
        self.entry = ttk.Entry(self.parent, textvariable=self.path_var, state='readonly')
        self.entry.pack(fill=tk.X)

        self.button = ttk.Button(self.parent, text="browse", command=self.browse_image)
        self.button.pack(anchor=tk.E, pady=(5, 0))

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="select image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("BMP Files", "*.bmp"),
                ("GIF Files", "*.gif"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.path_var.set(file_path)
            self.parameter.current_value = file_path
            self.on_change_callback()

    def on_value_change(self, *args):
        self.parameter.current_value = self.path_var.get()
        self.on_change_callback()

class Visualizer:
    """
    An interactive visualizer for the image processing algorithms.

    All you need to do is to provide:
    1. The modifiable parameters.
    2. The callback function that generates the image.
    3. (Optional) The shared data for the callback function.
    and then run the visualizer.

    The callback function should have the following signature:

        def callback_func_name(params, context):
            # Generate the image with the specified parameters.
            return image # Image in PIL format.

    The params is a container that contains the current values of the parameters.
    The context is a shared data dictionary that to store any data that may be used in the callback function.

    To get the specific parameter value, you can use the following code:
    ```python
    def generate_image(params, context):
        # Get the specific parameter value.
        param_value = params.get("param_name", default_value)

        # Generate the image with the specified parameters.
        return image
    ```

    To get the shared data, you can use the following code:
    ```python
    def generate_image(params, context):
        # Get the shared data.
        shared_data = context['shared_data_name']

        # Generate the image with the specified parameters.
        return image
    ```

    e.g.
    ```python
    def generate_image(params, context):
        r = params.get("r", 0)
        g = params.get("g", 0)
        b = params.get("b", 0)

        image = Image.new("RGB", (1024, 1024), color=(r, g, b))
        return image

    vis = Visualizer(
        width=1024,
        height=1024,
        callback=generate_image
    )

    vis.setParameter("r", ParameterType.NUMERIC, 0, 255)
    vis.setParameter("g", ParameterType.NUMERIC, 0, 255)
    vis.setParameter("b", ParameterType.NUMERIC, 0, 255)
    vis.run()

    Args:
        width:      The width of the canvas.
        height:     The height of the canvas.
        callback:   The callback function to generate the image.
        **kwargs:   The shared data for the callback function.
    """

    def __init__(self, width=1024, height=1024, *, callback=None, **kwargs):
        self.callback = callback
        self.width = width
        self.height = height
        self.parameters = []
        self.param_widgets = []
        self.context = Context()
        self.context.shared_data.update(kwargs)

        self.root = tk.Tk()
        self.root.title("Faye visualization")
        # self.update_image()

    def setResolution(self, width, height=None):
        if height is None:
            height = width

        self.width = width
        self.height = height

    def setCallback(self, callback):
        self.callback = callback

    def setParameter(self, name, param_type: ParameterType, min_val, max_val, default=None):
        if default is None:
            default = min_val
        self.parameters.append(Parameter(
            name,
            param_type,
            min_val,
            max_val,
            default
        ))

    def load_parameters(self, json_path):
        try:
            with open(json_path, 'r') as f:
                params_data = json.load(f)
                for name, param_info in params_data.items():
                    param_type_str = param_info.get("type", "numeric").lower()
                    if param_type_str == "numeric":
                        param_type = ParameterType.NUMERIC
                        min_val = param_info.get("min")
                        max_val = param_info.get("max")
                        current_val = param_info.get("current")
                        parameter = Parameter(name, param_type, min_val, max_val, current_val)
                    elif param_type_str == "image_path":
                        param_type = ParameterType.IMAGE_PATH
                        current_val = param_info.get("current")
                        parameter = Parameter(name, param_type, current_value=current_val)
                    else:
                        raise ValueError(f"Unknown data type: {param_type_str}")
                    self.parameters.append(parameter)
        except Exception as e:
            messagebox.showerror("Fatal", f"Cannot load the parameter file: {e}")
            self.root.destroy()

    def setup_widgets(self):
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = ttk.Frame(self.root, padding="10")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.right_frame, bg="white", width=self.width, height=self.height)
        self.canvas.pack(fill=tk.NONE, expand=False)

        for index, parameter in enumerate(self.parameters):
            if parameter.param_type == ParameterType.NUMERIC:
                widget = NumericParameterWidget(self.left_frame, parameter, self.update_image, self.context)
            elif parameter.param_type == ParameterType.IMAGE_PATH:
                widget = ImagePathParameterWidget(self.left_frame, parameter, self.update_image, self.context)
            else:
                continue
            self.param_widgets.append(widget)

            if index < len(self.parameters) - 1:
                separator = ttk.Separator(self.left_frame, orient='horizontal')
                separator.pack(fill=tk.X, pady=5)

        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def on_canvas_resize(self, event):
        self.update_image()

    def update_image(self):
        current_params = {param.name: param.current_value for param in self.parameters}
        try:
            img = self.callback(current_params, self.context)
            if img is None or not isinstance(img, Image.Image):
                raise ValueError("The returns of the callback func is not a valid PIL image.")
            img = img.resize((self.width, self.height), Image.LANCZOS)
        except Exception as e:
            messagebox.showwarning("WARNING", f"Error in generating image: {e}")
            img = Image.new("RGB", (self.width, self.height), color="white")

        self.display_image(img)

    def display_image(self, img):
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.width / 2,
            self.height / 2,
            image=self.img_tk,
            anchor=tk.CENTER
        )

    def run(self):
        self.check()
        self.setup_widgets()
        self.root.mainloop()

    def check(self):
        if self.callback is None:
            raise NotImplementedError("[Faye Visualizer] The callback has not been provided.")
