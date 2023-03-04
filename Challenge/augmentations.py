import numpy as np
from torchvision import transforms

class GeomTransform():
    ''' Base class for geometric transformations that need to be applied to all
        images, masks included.'''
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x
    
    def update_params(self):
        pass


class RandomRotationOneShot(GeomTransform):
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


class RandomHorizontalFlipOneShot(GeomTransform):
    def __init__(self, p=0.5):
        self.p = p
        self.flip = None

    def __call__(self, x):
        if self.flip:
            x = transforms.functional.hflip(x)
        return x

    def update_params(self):
        self.flip = np.random.uniform() < self.p


class RandomVerticalFlipOneShot(GeomTransform):
    def __init__(self, p=0.5):
        self.p = p
        self.flip = None

    def __call__(self, x):
        if self.flip:
            x = transforms.functional.vflip(x)
        return x

    def update_params(self):
        self.flip = np.random.uniform() < self.p


class RandomGaussianBlur(GeomTransform):
    def __init__(self, kernel_size = 5, sigma = None, p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        self.blur = None

    def __call__(self, x):
        if self.blur:
            x = transforms.functional.gaussian_blur(x, self.kernel_size, self.sigma)
        return x

    def update_params(self):
        self.blur = np.random.uniform() < self.p


mean_ct = 97.59464646559495
std_ct = 298.82489850204036 
mean_dose = 2.945620140701685 
std_dose =  10.749361438151379

ct_transform = transforms.Compose([transforms.Normalize(mean_ct, std_ct)])
aug_transform = [RandomHorizontalFlipOneShot(p=0.5),
                 RandomVerticalFlipOneShot(p=0.5),
                 RandomRotationOneShot(90, p=0.5),
                 RandomGaussianBlur(p=0.5,sigma=0.5)]

test_transform = transforms.Compose([transforms.Normalize(mean_ct, std_ct)])

def post_process(possible_dose_mask, dose):
    ''' 
    Post processing of the dose prediction. Given the possible dose mask and the dose prediction, we intersect the two

    Parameters
    ----------
    possible_dose_mask : torch.Tensor
        The possible dose mask. Shape (batch_size, 1, 128, 128)
    dose : torch.Tensor
        The dose prediction. Shape (batch_size, 1, 128, 128)
    '''

    return possible_dose_mask * dose
    