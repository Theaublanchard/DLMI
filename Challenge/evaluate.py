from pathlib import Path
import argparse
import os

from augmentations import ct_transform
from datasetsDMLI import DLMI_Test, DLMI_Train
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
    parser.add_argument("--file-name", type=str, default="submission.csv",
                        help='Name of the submission file')
    
    
    # Running
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size for evaluation')
    
    return parser

def main(args):
    print(args)
    gpu = torch.device(args.device)
    
    # test_path = os.path.join(args.data_dir, 'test')
    # test_dataset = DLMI_Test(test_path, transform=ct_transform)
    test_path = os.path.join(args.data_dir, "validation")
    test_dataset = DLMI_Train(test_path, ct_transform=ct_transform)
    test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                shuffle=False)
    
    model_path = os.path.join(args.exp_dir, args.model_name)
    ckpt = torch.load(model_path)
    model = UNet(12, 1)
    model.load_state_dict(ckpt['model'])
    model.to(gpu)
  
    dict_list = []
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating")
        mae_total = 0
        batches_seen = 0
        for i, (x, y,sample_idx) in pbar:
            x = x.to(gpu).float()
            y = y.to(gpu).float()

            output = model(x)

            #Compute Mae on the batch
            mae = torch.mean(torch.abs(output - y), dim=(1, 2, 3))
            mae = mae.cpu().numpy()
            sample_idx = sample_idx.numpy()
            for i in range(len(sample_idx)):
                dict_list.append({"Id": sample_idx[i], "Expected": mae[i]})
            batches_seen += 1
            mae_total += mae.mean()

            pbar.set_postfix_str(f"MAE: {mae_total/batches_seen:.4f}, MAE batch: {mae.mean():.4f}")

    df = pd.DataFrame(dict_list)
    df.to_csv(os.path.join(args.exp_dir, args.file_name), index=False)

            
        
if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)