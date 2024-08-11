import numpy as np
import torch


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class Eval_Score():
    # IoU and F1(Dice)

    def __init__(self, y_pred, y_true, threshold=0.5):
        input_flatten = np.int32(y_pred.flatten() > threshold)
        target_flatten = np.int32(y_true.flatten() > threshold)
        self.intersection = np.sum(input_flatten * target_flatten)
        self.sum = np.sum(target_flatten) + np.sum(input_flatten)
        self.union = self.sum - self.intersection

    def Dice(self, eps=1):
        return np.clip(((2. * self.intersection) / (self.sum + eps)), 1e-5, 0.99999)

    def IoU(self):
        return self.intersection / self.union


def validate_model(model, val_loader, device):
    """在验证集上验证模型"""
    iou_accumulator,  dice = 0, 0
    with torch.no_grad():
        for inputs, binarys, instances in val_loader:
            inputs = inputs.to(device)
            binarys = binarys.to(device)
            outputs = model(inputs)
            y_pred = torch.squeeze(outputs["binary_seg_pred"].cpu()).numpy()
            y_true = torch.squeeze(binarys.cpu()).numpy()
            Score = Eval_Score(y_pred, y_true)
            iou_accumulator += Score.IoU()
            dice += Score.Dice()
    res_validate = {
        'iou':iou_accumulator / len(val_loader.dataset),
        'f1': dice / len(val_loader.dataset)
    }
    return res_validate
