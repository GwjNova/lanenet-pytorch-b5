import argparse

def parse_args_train():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train LaneNet model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="./data/dataset_B", help="Path to dataset directory")  # 选择训练数据集的位置
    parser.add_argument("--height", type=int, default=256, help="Height of the input images")
    parser.add_argument("--width", type=int, default=512, help="Width of the input images")
    parser.add_argument("--bs", type=int, default=18, help="Batch size")
    parser.add_argument("--model_type", type=str, default="UNet", choices=["ENet", "UNet", "DeepLabv3+"], help="Type of the model architecture")  # 根据需求选择合适的模型 ENet or UNet or DeepLabv3+
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--loss_type", type=str, default="FocalLoss", choices=["CrossEntropyLoss", "FocalLoss"], help="Type of loss function")  # 根据任务需求选择合适的损失函数CrossEntropyLoss or FocalLoss
    parser.add_argument("--save", type=str, default="./saved_models", help="Path to save trained models")
    parser.add_argument("--resume", type=bool, default=True, help="Whether to resume training from a checkpoint")
    parser.add_argument("--checkpoint", type=str, default="saved_models/best_UNet_B_model.pth", help="Path to the model checkpoint")
    parser.add_argument("--pin_memory", type=int, default=None, help="Whether to use pin_memory for DataLoader")
    parser.add_argument("--shutdown", type=bool, default=False, help="Will it automatically shut down")  # 训练完成是否自动关机
    return parser.parse_args()

def parse_args_eavl():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="./data/dataset_A", help="Dataset path")
    parser.add_argument("--model_type", help="Model type", default="DeepLabv3+", choices=["ENet", "UNet", "DeepLabv3+"])
    parser.add_argument("--model", help="Model path", default=r'saved_models/best_DeepLabv3Pro_A_model.pth')
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--bs", type=int, default=1, help="Batch size")
    return parser.parse_args()

def parse_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=r"D:\python\data\dataset_B\test\image\170927_063834153_Camera_5.jpg", help="Img path")
    parser.add_argument("--model_type", help="Model type", default='UNet')
    parser.add_argument("--model", help="Model path", default='saved_models/best_UNet_B_model.pth')
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--save", help="Directory to save output", default="test_output")
    return parser.parse_args()

def parse_args_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="dataset path")
    parser.add_argument("--classes", help="dataset class", choices=["train","val"])
    return parser.parse_args()
