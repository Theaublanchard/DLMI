from torch.utils.data import Dataset
import os
import numpy as np
import torch
import tqdm

from augmentations import GeomTransform

class DLMI_Train(Dataset):
    '''Dataset for the DLMI challenge
    
    Parameters
    ----------
    path : str
        Path to the dataset
    ct_transform : torchvision.transforms
        Transform to apply to the CT image
    dose_transform : torchvision.transforms
        Transform to apply to the dose image
    aug_transform : list of transformations
        List of augmentations to apply to all the images
    max_samples : int
        Maximum number of samples to use, if None, use all the samples. Used for debugging purposes.

    Returns
    -------
    x : torch.Tensor
        (12,128,128) tensor containing the CT, possible_dose_mask and structure_masks
    sample_idx : int
        Index of the sample
    y : torch.Tensor 
        (1,128,128) tensor containing the dose
    '''
    def __init__(self, path,ct_transform = None, aug_transform = [],max_samples: int = None):
        self.path = path
        if max_samples is not None:
            self.samples = os.listdir(path)[:max_samples]
        else:
            self.samples = os.listdir(path)
        self.ct_transform = ct_transform
        
        assert isinstance(aug_transform, list) or aug_transform is None , 'aug_transform must be a list'

        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.path, self.samples[idx])
        ct_path = os.path.join(sample_path, 'ct.npy')
        dose_path = os.path.join(sample_path, 'dose.npy')
        possible_dose_mask_path = os.path.join(sample_path, 'possible_dose_mask.npy')
        structure_masks_path = os.path.join(sample_path, 'structure_masks.npy')

        ct = torch.from_numpy(np.load(ct_path)).unsqueeze(0)
        dose = torch.from_numpy(np.load(dose_path)).unsqueeze(0)
        possible_dose_mask = torch.from_numpy(np.load(possible_dose_mask_path)).unsqueeze(0)
        structure_masks = torch.from_numpy(np.load(structure_masks_path))
        
        if self.ct_transform:
            ct = self.ct_transform(ct)

        if self.aug_transform:
            for aug in self.aug_transform:
                if isinstance(aug, GeomTransform):
                    aug.update_params()
                    possible_dose_mask = aug(possible_dose_mask)
                    structure_masks = aug(structure_masks)
                ct = aug(ct)
                dose = aug(dose)

        #Stack the images
        x = torch.cat((ct, possible_dose_mask, structure_masks), dim=0)

        #Keep only the last digits, ie get rid of the 'sample_' part
        sample_idx = int(self.samples[idx].split('_')[-1])
        return x, dose, sample_idx
    


class DLMI_Test(Dataset):
    '''Dataset for the DLMI challenge

    Parameters
    ----------
    path : str
        Path to the dataset
    ct_transform : torchvision.transforms
        Transform to apply to the CT image

    Returns
    -------
    x : torch.Tensor
        (12,128,128) tensor containing the CT, possible_dose_mask and structure_masks
    idx : int
        Index of the sample
    '''

    def __init__(self, path,ct_transform = None):
        self.path = path
        self.samples = os.listdir(path)
        self.ct_transform = ct_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.path, self.samples[idx])
        ct_path = os.path.join(sample_path, 'ct.npy')
        possible_dose_mask_path = os.path.join(sample_path, 'possible_dose_mask.npy')
        structure_masks_path = os.path.join(sample_path, 'structure_masks.npy')

        ct = torch.from_numpy(np.load(ct_path)).unsqueeze(0)
        possible_dose_mask = torch.from_numpy(np.load(possible_dose_mask_path)).unsqueeze(0)
        structure_masks = torch.from_numpy(np.load(structure_masks_path))
        
        if self.ct_transform:
            ct = self.ct_transform(ct)

        #Stack the images
        x = torch.cat((ct, possible_dose_mask, structure_masks), dim=0)
        sample_idx = int(self.samples[idx].split('_')[-1])
        return x, possible_dose_mask,sample_idx
    

def compute_mean_std(loader):
    mean_ct = 0.
    std_ct = 0.

    mean_dose = 0.
    std_dose = 0.
    nb_samples = 0.

    for (ct, _, _), dose in tqdm(loader):
        batch_samples = ct.size(0)

        ct = ct.view(batch_samples, ct.size(1), -1)
        dose = dose.view(batch_samples, dose.size(1), -1)

        mean_ct += ct.mean((1,2)).sum(0)
        std_ct += ct.std((1,2)).sum(0)

        mean_dose += dose.mean((1,2)).sum(0)
        std_dose += dose.std((1,2)).sum(0)

        nb_samples += batch_samples

    mean_ct /= nb_samples
    std_ct /= nb_samples

    mean_dose /= nb_samples
    std_dose /= nb_samples

    print(f"{mean_ct=}, {std_ct=}, {mean_dose=}, {std_dose=}")
    return mean_ct, std_ct, mean_dose, std_dose