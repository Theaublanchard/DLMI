from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim

import unet
from augmentations import ct_transform, aug_transform
from datasetsDLMI import DLMI_Train

def get_arguments():
    parser = argparse.ArgumentParser(description="Train a UNet model", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/datasets", required=True,
                        help='Path to the datasets')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--save-freq", type=int, default=10,
                        help='Save a checkpoint every [save-freq] epochs')

    # Optim
    parser.add_argument("--epochs", type=int, default=20,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=16,
                        help='Batch size')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='Momentum')
    parser.add_argument("--weight-reg", type=float, default=None,
                        help="Gradient regularization weight")

    # Running
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser


def main(args):


    gpu = torch.device(args.device)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(args)
    print(args, file=stats_file)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)


    train_path = os.path.join(args.data_dir, "train","train")
    dataset_train = DLMI_Train(train_path, ct_transform, aug_transform)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_path = os.path.join(args.data_dir, "validation")
    dataset_val = DLMI_Train(val_path, ct_transform, None)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = unet.UNet(12, 1).cuda(gpu)
    optimizer = optim.SGD(model.parameters(),
            lr=args.base_lr, 
            momentum=args.momentum,
            weight_decay=args.wd)
    if args.weight_reg is None:
        criterion = nn.L1Loss()
    else:
        criterion = lambda x, y: nn.L1Loss()(x, y) + Gradient_loss(args.weight_reg)(x, y)

    if (args.exp_dir / "model.pth").is_file():
        print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    if (args.exp_dir / "model_val.pth").is_file():
        print("Found validation checkpoint")
        best_val_loss = ckpt["loss_val"]
    else:
        best_val_loss = np.inf
    

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):

        epoch_loss_train, lr = train_one_epoch(model, epoch,optimizer, criterion, loader_train,gpu)
        epoch_loss_val = validate_one_epoch(model, epoch, criterion, loader_val, gpu)

        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            loss_train = epoch_loss_train,
            loss_val = epoch_loss_val,
        )
        torch.save(state, args.exp_dir / "model.pth")

        if (epoch + 1) % args.save_freq == 0:
            torch.save(state, args.exp_dir / f"model_{epoch}.pth")

        if epoch_loss_val < best_val_loss:
            print("Saving best validation checkpoint at epoch", epoch + 1, "with loss", epoch_loss_val,file=stats_file)
            best_val_loss = epoch_loss_val
            torch.save(state, args.exp_dir / "model_val.pth")

        print(json.dumps({
            "epoch": epoch,
            "loss_train": epoch_loss_train,
            "loss_val": epoch_loss_val,
            "lr": lr,
            "time": time.time() - start_time,
        }), 
        file=stats_file)


def adjust_learning_rate(args, optimizer, loader, step,gpu):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 128
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_one_epoch(model, epoch,optimizer, criterion, loader, gpu ):
    model.train()

    loss_epoch = 0
    batches_seen = 0

    pbar = tqdm(enumerate(loader, start=epoch * len(loader)), total=len(loader), desc=f"Epoch {epoch}")
    
    for step,(x, dose,_) in pbar:
        x = x.cuda(gpu, non_blocking=True).float()
        dose = dose.cuda(gpu, non_blocking=True).float()

        lr = adjust_learning_rate(args, optimizer, loader, step,gpu)
        
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, dose)
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        batches_seen += 1

        pbar.set_postfix_str(f"Loss_batch {loss.item():.4f} - Loss_train_epoch {loss_epoch / batches_seen:.4f} - LR {lr:.4f}")

    return loss_epoch/batches_seen, lr


def validate_one_epoch(model, epoch, criterion, loader, gpu):
    model.eval()

    loss_epoch = 0
    batches_seen = 0

    pbar = tqdm(enumerate(loader, start=epoch * len(loader)), total=len(loader), desc=f"Epoch {epoch}")
    with torch.no_grad():
        for step,(x, dose,_) in pbar:
            x = x.cuda(gpu, non_blocking=True).float()
            dose = dose.cuda(gpu, non_blocking=True).float()

            output = model(x)
            loss = criterion(output, dose)

            loss_epoch += loss.item()
            batches_seen += 1

            pbar.set_postfix_str(f"Loss {loss.item():.4f} - Loss_val_epoch {loss_epoch / batches_seen:.4f}")

    return loss_epoch/batches_seen


class Gradient_loss(nn.Module):
    def __init__(self,weigth_reg) -> None:
        '''
        Computes the gradient of an image
        '''
        super().__init__()
        self.conv_hor = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.conv_ver = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        a = torch.tensor([[1.,2.,1.],
                            [0,0,0],
                            [-1.,-2.,-1.]], requires_grad=False)
        b = torch.tensor([[1.,0,-1.],
                            [2.,0,-2.],
                            [1.,0,-1.]], requires_grad=False)
        a = a.unsqueeze(0).unsqueeze(0).requires_grad_(False)
        b = b.unsqueeze(0).unsqueeze(0)
        self.conv_hor.weight = nn.Parameter(a).requires_grad_(False)
        self.conv_ver.weight = nn.Parameter(b).requires_grad_(False)
        self.weigth_reg = weigth_reg

    def forward(self, x):
        conved_hor = self.conv_hor(x)
        conved_ver = self.conv_ver(x)
        grad = torch.sqrt(torch.square(conved_hor) + torch.square(conved_ver))
        # return self.weigth_reg * torch.sum(grad)
        return self.weigth_reg * torch.mean(grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('UNet training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
