import numpy as np
from torchvision import transforms

class RandomRotationOneShot():
    def __init__(self, angle, p=0.5):	
        self.angle = angle
        self.p = p
        self.rotate = None
        self.rotate_angle = None

    def __call__(self, x):
        if self.rotate:
            x = transforms.functional.rotate(x, self.rotate_angle)
        return x

    def update_params(self):
        self.rotate = np.random.uniform() < self.p
        self.rotate_angle = np.random.uniform(-self.angle, self.angle)

class RandomHorizontalFlipOneShot():
    def __init__(self, p=0.5):
        self.p = p
        self.flip = None

    def __call__(self, x):
        if self.flip:
            x = transforms.functional.hflip(x)
        return x

    def update_params(self):
        self.flip = np.random.uniform() < self.p

class RandomVerticalFlipOneShot():
    def __init__(self, p=0.5):
        self.p = p
        self.flip = None

    def __call__(self, x):
        if self.flip:
            x = transforms.functional.vflip(x)
        return x

    def update_params(self):
        self.flip = np.random.uniform() < self.p


mean_ct = 97.59464646559495
std_ct = 298.82489850204036 
mean_dose = 2.945620140701685 
std_dose =  10.749361438151379

ct_transform = transforms.Compose([transforms.Normalize(mean_ct, std_ct)])
# dose_transform = transforms.Compose([transforms.Normalize(mean_dose, std_dose)])
# aug_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
#                                     transforms.RandomVerticalFlip(p=0.5),
#                                     transforms.RandomRotation(90, fill=(0,)),
#                                     ])
aug_transform = [RandomHorizontalFlipOneShot(p=0.5),
                 RandomVerticalFlipOneShot(p=0.5),
                 RandomRotationOneShot(90, p=0.5),]

test_transform = transforms.Compose([transforms.Normalize(mean_ct, std_ct)])