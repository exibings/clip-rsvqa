import torch
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor
from torchvision.transforms import (
    # added for image augmentation
    RandomCrop,
    ColorJitter,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomResizedCrop,
    RandAugment,
    # /added for image augmentation
    CenterCrop, 
    ConvertImageDtype, 
    Normalize, 
    Resize
)
from PIL import Image

class ImageProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            #image augmentation
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(degrees=(0, 10)),
            RandomResizedCrop([224,], scale=(0.9, 1.1), ratio=(0.9, 1.3333333333333333)),
            #/image augmentation
            Resize([224,], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ConvertImageDtype(torch.float),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        )

    def forward(self, img_as_tensor: Image) -> torch.Tensor:        
        with torch.no_grad():
            return self.transforms(pil_to_tensor(img_as_tensor))