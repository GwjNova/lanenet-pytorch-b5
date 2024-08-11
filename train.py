# -*- coding: utf-8 -*-
# author: gwj
# date: 2024-05-04

import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing

from dataloader.transformers import get_transforms
from eval import evaluation
from model.lanenet.train_lanenet import train_model, get_device, load_model
from dataloader.data_loaders import TusimpleSet
from model.lanenet.LaneNet import LaneNet
from utils.cli_helper import parse_args_train

DEVICE = get_device()


def train():
    """主训练函数"""
    args = parse_args_train()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args.save, current_time)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # 日志信息配置
    logging.basicConfig(level=logging.INFO, filename=f'./logs/{current_time}_{args.dataset[-1]}_training.logs',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # tensorboard初始化
    writer = SummaryWriter()

    logging.info("Training starts...")

    # 加载训练集和验证集、测试集
    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')

    data_transforms, target_transforms = get_transforms(args.height, args.width)
    train_dataset = TusimpleSet(train_dataset_file, transform=data_transforms['train'],
                                target_transform=target_transforms)
    val_dataset = TusimpleSet(val_dataset_file, transform=data_transforms['val'],
                              target_transform=target_transforms)

    num_workers = multiprocessing.cpu_count()
    if args.pin_memory is None:
        pin_memory = torch.cuda.is_available()
    else:
        pin_memory = args.pin_memory

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=num_workers,
                            pin_memory=pin_memory)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}
    model = LaneNet(arch=args.model_type)

    logging.info("Use {} as backbone".format(args.model_type))

    if args.resume and args.checkpoint is not None:
        model = load_model(model, args.checkpoint)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=0.0001)  # 添加 weight_decay 参数
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.01)  # 设置 ReduceLROnPlateau 调度器
    logging.info("loss function  is {}".format(args.loss_type))
    epochs = args.epochs
    logging.info(
        f"{epochs} epochs {len(train_loader.dataset)} training samples {len(val_loader.dataset)} val samples\n")
    best_iou = 0.0
    best_model_state = None
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs},lr:{optimizer.param_groups[0]['lr']}, bs:{args.bs}")
        logging.info('-' * 10)
        model.train()
        model, log, time = train_model(model, optimizer, scheduler, dataloaders=dataloaders,
                                       dataset_sizes=dataset_sizes, device=DEVICE, loss_type=args.loss_type,
                                       num_epochs=1)
        test_iou = evaluation(args.height, args.width, args.dataset, 1, args.model_type, model, DEVICE, False)['iou']
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time // 60, time % 60))
        logging.info(f'IoU/test: {test_iou * 100}%,epoch: {epoch + 1}')
        print(f'IoU/test: {test_iou},epoch: {epoch + 1}')
        writer.add_scalar('IoU/test', test_iou, epoch)
        logging.info(f"Loss/train_epoch:{log['training_loss'][0]},epoch: {epoch + 1}")
        logging.info(f"Loss/val_epoch:{log['val_loss'][0]},epoch: {epoch + 1}")
        writer.add_scalar('Loss/train_epoch', log['training_loss'][0], epoch)
        writer.add_scalar('Loss/val_epoch', log['val_loss'][0], epoch)

        if test_iou > best_iou:
            best_iou = test_iou
            best_model_state = model.state_dict().copy()
            model_save_filename = os.path.join(save_path, f'{current_time}_{epoch + 1}_model.pth')
            torch.save(model.state_dict(), model_save_filename)
            logging.info("Model is saved: {}".format(model_save_filename))
        logging.info("\n")
        print("val_loss:", log['val_loss'][0])
        scheduler.step(log['val_loss'][0])
    best_model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(best_model_state, best_model_save_filename)
    logging.info("Best model is saved: {}".format(best_model_save_filename))
    # 是否在代码完成后自动关机
    if args.shutdown:
        os.system('shutdown -s -t 0')


if __name__ == '__main__':
    train()
