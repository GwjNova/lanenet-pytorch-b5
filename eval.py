import os
import torch
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torchvision import transforms
from model.eval_function import validate_model
from utils.cli_helper import parse_args_eavl

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluation(height, width, dataset, bs, model_type, model, device, exist):
    resize_height = height
    resize_width = width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    dataset_file = os.path.join(dataset, 'val.txt')
    Eval_Dataset = TusimpleSet(dataset_file, transform=data_transform, target_transform=target_transforms)
    eval_dataloader = DataLoader(Eval_Dataset, batch_size=bs, shuffle=True)

    if exist:
        model_path = model
        model = LaneNet(arch=model_type)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return   validate_model(model, eval_dataloader, device)


def handel_eval():
    args = parse_args_eavl()
    res = evaluation(args.height, args.width, args.dataset, args.bs, args.model_type, args.model, device,True)

    print("IoU:%s"%res['iou'])
    print("F1:%s"%res['f1'])

if __name__ == "__main__":
    handel_eval()