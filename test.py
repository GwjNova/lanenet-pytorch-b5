import os

import torch
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
from utils.cli_helper import parse_args_test
import numpy as np
from PIL import Image
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img

def test():
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args_test()
    img_path = args.image
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    input = Image.open(img_path)
    input = input.resize((resize_width, resize_height))
    input = np.array(input)

    instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255

    cv2.imwrite(os.path.join('test_output', 'input.jpg'), input)
    cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'), instance_pred.transpose((1, 2, 0)))
    cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), binary_pred)


if __name__ == "__main__":
    test()