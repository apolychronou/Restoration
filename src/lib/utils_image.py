from PIL import Image
import torch
class ModCrop:
    """
    Custom transformation to crop image dimensions to be divisible by a given scale.
    """
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        """
        Args:
            img: PIL Image or PyTorch Tensor (CHW or HWC format).
        Returns:
            Cropped image.
        """

        if not isinstance(img, torch.Tensor):
            raise ValueError("ModCrop expects a single image, not a list or tuple.")

        # Get image dimensions
        if img.ndimension() == 2:  # Grayscale (HW)
            h, w = img.shape
        elif img.ndimension() == 3:  # Color Image (CHW)
            _, h, w = img.shape
        else:
            raise ValueError(f"Invalid image dimensions: {img.ndimension()}.")

        # Crop dimensions to be divisible by scale
        h_r, w_r = h % self.scale, w % self.scale
        img_cropped = img[..., :h - h_r, :w - w_r]

        return img_cropped