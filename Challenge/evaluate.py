from pathlib import Path
import argparse
import os
import sys
import numpy as np
import shutil

from augmentations import ct_transform, post_process
from datasetsDLMI import DLMI_Test, DLMI_Train
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import pandas as pd
from unet import UNet


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a UNet model", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="./data", required=True,
                        help='Path to the datasets')
    
    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where the model is stored and the results will be stored')
    parser.add_argument("--model-name", type=str, default="model.pth",
                        help='Name of the model to be evaluated')
    parser.add_argument("--folder-name", type=str, default="submission",
                        help='Name of the submission folder')
    
    # Model
    parser.add_argument("--channels_list", type=str, default="64,128,256,512,1024", required=False,
                        help='List of the number of channels for each block. Must be of length 5.')
    parser.add_argument("--num_classes", type=int, default=1, required=False,
                        help='Number of classes for the pixel_wise regression task')
    parser.add_argument("--n_channels", type=int, default=12, required=False,
                        help='Number of channels in the input image')
    parser.add_argument("--post-process", type=bool, default=False, required=False,
                        help='Whether to apply a post-processing step to the predictions during evaluation')

    
    # Running
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size for evaluation')
    
    return parser

def main(args):
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(args, file=stats_file)
    print(" ".join(sys.argv), file=stats_file)
    print(args)
    gpu = torch.device(args.device)

    folder_path = os.path.join(args.exp_dir, args.folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    test_path = os.path.join(args.data_dir, 'test','test')
    test_dataset = DLMI_Test(test_path, ct_transform=ct_transform)
    test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                shuffle=False)
    
    model_path = os.path.join(args.exp_dir, args.model_name)
    ckpt = torch.load(model_path)
    channels_list = [int(x) for x in args.channels_list.split(",")]
    model = UNet(args.n_channels, args.num_classes, channels_list,args.post_process)    
    model.load_state_dict(ckpt['model'])
    model.to(gpu)
  
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating")

        for i, (x, possible_dose_mask,sample_idx) in pbar:
            x = x.to(gpu).float()
            possible_dose_mask = possible_dose_mask.to(gpu)

            output = model(x)

            output = post_process(output, possible_dose_mask)
            output = output.cpu().numpy()
            for j in range(output.shape[0]):
                np.save(os.path.join(folder_path, f"sample_{sample_idx[j]}.npy"), output[j])

    print("Zipping submission folder...", end="")
    shutil.make_archive(folder_path, 'zip', folder_path)
    print("Done")
    #Delete folder
    shutil.rmtree(folder_path)
        
if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)