"""SSIM metric that can be run on sequences of images

Adapted from [1] to match more closely to [2]

[1] https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html to match more closely
[2] https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

Example of equivalence:

```
from skimage.metrics import structural_similarity
import numpy as np
from sat_pred.ssim import SSIM3D
import torch
torch.manual_seed(1)

# Create some sample data
n_samples = 1   # only 1 sample so compatible with skimage function
n_channels = 5
n_timesteps = 1 # only 1 time step so compatible with skimage function

x_dim = 400
y_dim = 700

y = torch.rand((n_samples, n_channels, n_timesteps, x_dim, y_dim))
y_hat = torch.rand((n_samples, n_channels, n_timesteps, x_dim, y_dim))


# Compute SSIM map with this class and squeeze out the extra dimensions
ssim_map1 = SSIM3D()(y_hat, y).numpy().squeeze((0,2))

# Compute SSIM map with skimage
_, ssim_map2 = structural_similarity(
    y_hat[0, :, 0].numpy(), # remove extra dimensions and convert to numpy
    y[0, :, 0].numpy(), # remove extra dimensions and convert to numpy
    channel_axis=0, 
    # The settings below are required to match the two calculations
    data_range=1,
    gaussian_weights=True,
    full=True, 
    sigma=1.5,
    use_sample_covariance=False,
)


# The skimage version of SSIM uses reflection padding when applying the gaussian kernel
# whilst our version uses zero padding. We expect the two versions to be the same 
def trim_border(x, num_pixels):
    return x[..., num_pixels:x.shape[-2]-num_pixels, num_pixels:x.shape[-1]-num_pixels]

# If we don't trim the border ~96% of the SSIM values are the same
np.isclose(
    trim_border(ssim_map1, num_pixels=0),
    trim_border(ssim_map2, num_pixels=0),
    atol=1e-05
).mean()
# >> 0.9610714285714286

# If we do trim the border all of the SSIM values are the same

# In both calculations we have used a window size of 11, so a padding size of 11//2 = 5
np.isclose(
    trim_border(ssim_map1, num_pixels=5),
    trim_border(ssim_map2, num_pixels=5),
    atol=1e-05
).mean()
# >> 1.0

```
"""


import torch
from torch import nn
import torch.nn.functional as F


def gaussian(kernel_size: int, sigma: float) -> torch.Tensor:
    ksize_half = (kernel_size - 1) * 0.5
    kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
    return (gauss / gauss.sum())


def create_gaussian_kernel(kernel_size: int | list[int], sigma: float | list[float]) -> torch.Tensor:
    
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
        
    if isinstance(sigma, float):
        sigma = [sigma, sigma]
        
    kernel_x = gaussian(kernel_size[0], sigma[0]).unsqueeze(dim=1)
    kernel_y = gaussian(kernel_size[1], sigma[1]).unsqueeze(dim=0)

    return torch.matmul(kernel_x, kernel_y)  # (kernel_size, 1) * (1, kernel_size)


class SSIM3D(nn.Module):
    def __init__(
        self, 
        kernel_size: int | list[int] = 11, 
        sigma: float | list[float] = 1.5, 
        k1: float = 0.01,
        k2: float = 0.03, 
        data_range: float = 1,
    ):
        super(SSIM3D, self).__init__()
        assert data_range > 0
        assert k1 > 0
        assert k2 > 0
        
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        elif isinstance(kernel_size, Sequence):
            self.kernel_size = kernel_size

        if isinstance(sigma, float):
            self.sigma = [sigma, sigma]
        elif isinstance(sigma, Sequence):
            self.sigma = sigma
        
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.kernel = nn.Parameter(
            data=create_gaussian_kernel(kernel_size=self.kernel_size, sigma=self.sigma),
            requires_grad=False
        )        
        self.pad = [0,] + [(k - 1) // 2 for k in self.kernel_size]
        
        self._nb_channel = None
    
    def forward(self, x, y) -> torch.Tensor:

        batch_size = x.size(0)
        
        if self._nb_channel is None:
            self._nb_channel = x.size(1)
            self.kernel = nn.Parameter(
                self.kernel.expand(self._nb_channel, 1, 1, -1, -1),
                requires_grad=False,
            )

        kernal_inputs = torch.cat([x, y, x**2, y**2, x*y])
        kernel_outputs = F.conv3d(kernal_inputs, self.kernel, padding=self.pad, groups=self._nb_channel)
        del kernal_inputs
    
        ux, uy, uxx, uyy, uxy = [kernel_outputs[i*batch_size:(i+1)*batch_size] for i in range(5)]        
                
        vx = (uxx - ux * ux)
        vy = (uyy - uy * uy)
        vxy = (uxy - ux * uy)
        
        a1 = 2 * ux * uy + self.c1
        a2 = 2 * vxy +  self.c2
        b1 = ux**2 + uy**2 + self.c1
        b2 = vx + vy +  self.c2

        return (a1 * a2) / (b1 * b2)