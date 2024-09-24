"""SSIM metric that can be run on sequences of images

Adapted from: https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html
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
        k2: float = 0.02, 
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
    
    def forward(self, y_pred, y) -> None:

        batch_size = y_pred.size(0)
        
        if self._nb_channel is None:
            nb_channel = y_pred.size(1)
            self.nb_channel = nb_channel
            self.kernel = nn.Parameter(
                self.kernel.expand(nb_channel, 1, 1, -1, -1),
                requires_grad=False,
            )

        kernal_inputs = torch.cat([y_pred, y, y_pred**2, y**2, y_pred*y])
        kernel_outputs = F.conv3d(kernal_inputs, self.kernel, padding=self.pad, groups=self.nb_channel)
        del kernal_inputs
    
        kernel_output_list = [kernel_outputs[i*batch_size:(i+1)*batch_size] for i in range(5)]
        del kernel_outputs

        mu_pred_sq = kernel_output_list[0].pow(2)
        mu_target_sq = kernel_output_list[1].pow(2)
        mu_pred_target = kernel_output_list[0] * kernel_output_list[1]

        sigma_pred_sq = kernel_output_list[2] - mu_pred_sq
        sigma_target_sq = kernel_output_list[3] - mu_target_sq
        sigma_pred_target = kernel_output_list[4] - mu_pred_target

        del kernel_output_list

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_map = (a1 * a2) / (b1 * b2)
        return ssim_map