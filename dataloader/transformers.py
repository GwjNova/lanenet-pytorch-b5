import cv2
from torchvision import transforms


class Rescale():
    """Rescale the image in a sample to a given size.

    Args:
        output_size (width, height) (tuple): Desired output size (width, height). Output is
            matched to output_size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # sample = resize(sample, self.output_size)
        sample = cv2.resize(sample, dsize=self.output_size, interpolation=cv2.INTER_NEAREST)

        return sample


def get_transforms(resize_height, resize_width):
    train_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue =0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定义验证数据集的数据转换
    val_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    target_transform = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    data_transforms = {
        'train': train_transform,
        'val': val_transform,
    }

    return data_transforms, target_transform
