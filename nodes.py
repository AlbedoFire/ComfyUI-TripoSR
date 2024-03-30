import numpy as np
import rembg
import torch
from PIL import Image

from .tsr.system import TSR
from .tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

class RemoveBackGround:

    @classmethod
    def INPUT_TYPE(cls):
        return {"required": {
            "background_radio": ("FLOAT", ())
        }}
    RETURN_TYPE = ("IMAGE",)
    RETURN_NAME = ("processed image")
    FUNCTION = "remove_background"
    CATEGORY = "TripoSR/remove_background"

    def preprocess(self,input_image, foreground_ratio):
        def fill_background(image):
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
            return image

        rembg_session = rembg.new_session()
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)

        return image



# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "RemoveBackGround": RemoveBackGround
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveBackGround": "Remove Background"
}
