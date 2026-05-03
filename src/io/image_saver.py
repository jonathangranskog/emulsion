import torchvision
import os
import datetime

import torch


class ImageSaver:
    @staticmethod
    def save_image(image: torch.Tensor):
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/{time_str}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchvision.utils.save_image(image, output_path)
