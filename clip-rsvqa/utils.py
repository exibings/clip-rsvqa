import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomResizedCrop,
)
from transformers import CLIPFeatureExtractor
from PIL import Image
import numpy

feature_extractor = CLIPFeatureExtractor.from_pretrained("flax-community/clip-rsicd-v2")


def patchImage(img_path: str) -> list:
    """
    Patches the image and returns the patches and original image

    Args:
        img_path (str): Path to the image to be patched

    Returns:
        list: list with the 4 patches generated from the image and the original image - [top_left, top_right, bottom_left, bottom_right, full_image]
    """
    img = Image.open(img_path)
    return [img.crop((0, 0, img.width//2, img.height//2)), img.crop((img.width//2, 0, img.width, img.height//2)),
            img.crop((0, img.height//2, img.width//2, img.height)), img.crop((img.width//2, img.height//2, img.width, img.height)), img]



class ImageProcessing(torch.nn.Module):
    def __init__(self, augment_images):
        super().__init__()
        self.augment_images = augment_images
        self.image_augmentation = torch.nn.Sequential(
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(degrees=(0, 10)),
            RandomResizedCrop([224,], scale=(0.9, 1.1), ratio=(0.9, 1.3333333333333333)),
        )
    
    def forward(self, img: Image) -> torch.Tensor:        
        with torch.no_grad():
            if self.augment_images:
                return numpy.squeeze(feature_extractor(to_pil_image(self.image_augmentation(pil_to_tensor(img))), return_tensors="np", resample=Image.Resampling.BILINEAR).pixel_values)
            else:
                return numpy.squeeze(feature_extractor(img, return_tensors="np", resample=Image.Resampling.BILINEAR).pixel_values)
