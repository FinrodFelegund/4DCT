import math

import torch

import bilateralfilter4dlut_gpu_lib as lib

def pad_circulat_t(x: torch.Tensor, pad_t: int):
    if pad_t < 1:
        return x
    
    T = x.shape[2]
    if pad_t > T:
        idx = torch.arange(-pad_t, T + pad_t, device=x.device) % T
        return x.index_select(2, idx).contiguous()
    return torch.cat([x[:, :, -pad_t:], x, x[:, :, :pad_t]], dim=2).contiguous()

def unpad_circular_t(x: torch.Tensor, pad_t: int):
    if pad_t < 1:
        return x

    return x[:, :, pad_t:-pad_t]

def gaussian_1d(sigma, window_size, dtype=torch.float64):
    d = torch.arange(0, window_size, dtype=torch.float64) - (window_size // 2)
    return torch.exp(-(d ** 2) / (2.0 * float(sigma) ** 2)).to(dtype)

def gaussian_4d(sigma_t, sigma_x, sigma_y, sigma_z, window_sizes, dtype=torch.float32):
    wt, wx, wy, wz = window_sizes
    gt = gaussian_1d(sigma_t, wt, dtype)
    gx = gaussian_1d(sigma_x, wx, dtype)
    gy = gaussian_1d(sigma_y, wy, dtype)
    gz = gaussian_1d(sigma_z, wz, dtype)

    return (gt[:, None, None, None] * gx[None, :, None, None] * gy[None, None, :, None] * gz[None, None, None, :]).contiguous()

def gaussian_lut_2d(color_sigma, bins, dtype=torch.float64):
    """F[p, q] = exp(-(a_p - a_q)^2 / (2 sigma_r)), a_p = p / (bins - 1)
    Reproduces the gaussian range kernel initially.
    """

    a = torch.linspace(0.0, 1.0, bins, dtype=torch.float64)
    d = a[:, None] - a[None, :]
    return torch.exp(-(d ** 2) / (2.0 * float(color_sigma) ** 2)).contiguous()

def default_window_sizes(sigma_x, sigma_y, sigma_z, window_size_t=9):
    def w(s):
        return max(int(math.ceil(5.0 * float(s))) | 1, 5)

    assert window_size_t % 2 == 1, 'Window Size must be odd'
    return (window_size_t, w(sigma_x), w(sigma_y), w(sigma_z))

def inverse_softplus(y: torch.Tensor, eps: float =1e-6):
    y = y.clamp_min(eps)
    return y + torch.log(-torch.expm1(-y))


class BilateralFilterFunction4DLutGPU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_img: torch.Tensor, kernel: torch.Tensor, lut: torch.Tensor):
        assert input_img.dim() == 6, "Input must be shape [B, C, T, X, Y, Z]"
        assert input_img.shape[1] == 1, "Currently only one channel images are supported"
        assert kernel.dim() == 4, "Kernel must be shape [Wt, Wx, Wy, Wz]"
        assert lut.dim() == 2 and lut.shape[0] == lut.shape[1], "Lookup table must be a square tensor"

        input_img = input_img.contiguous()
        kernel = kernel.contiguous()
        lut = lut.contiguous()

        out, weights, dO_dx_ki = lib.forward_4dlut_gpu(input_img, kernel, lut)

        ctx.save_for_backward(input_img, kernel, lut, out, weights, dO_dx_ki)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_img, kernel, lut, out, weights, dO_dx_ki = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_input, grad_kernel, grad_lut = lib.backward_4dlut_gpu(
            grad_output, input_img, out, weights, dO_dx_ki, kernel, lut)
        
        return grad_input, grad_kernel, grad_lut

class BilateralFilter4DLUT(torch.nn.Module):
    def __init__(self,
                 sigma_t, sigma_x, sigma_y, sigma_z, color_sigma,
                 lut_bins=32,
                 window_sizes=None,
                 window_size_t=9,
                 symmetric_lut=True,
                 use_gpu=True):

        super().__init__()

        if window_sizes is None:
            window_sizes = default_window_sizes(sigma_x, sigma_y, sigma_z, window_size_t=window_size_t)

        for w in window_sizes:
            assert w % 2 == 1, "window size must be odd"

        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.lut_bins = int(lut_bins)
        self.symmetric_lut = bool(symmetric_lut)

        self.kernel = torch.nn.Parameter(
            gaussian_4d(sigma_t, sigma_x, sigma_y, sigma_z, self.window_sizes)
        )

        lut0 = gaussian_lut_2d(color_sigma, self.lut_bins)
        self.raw_lut = torch.nn.Parameter(inverse_softplus(lut0))

        self.pad_t = self.window_sizes[0] // 2
        self.use_gpu = use_gpu

    def activated_lut(self):
        lut = torch.nn.functional.softplus(self.raw_lut)
        if self.symmetric_lut:
            lut = (lut + lut.t()) / 2

        return lut / lut.diagonal().mean()

    def lut_smoothness_penalty(self):
        lut = self.activated_lut()
        d_row = lut[2:, :] - 2 * lut[1:-1, :] + lut[:-2, :]
        d_col = lut[:, 2:] - 2 * lut[:, 1:-1] + lut[:, :-2]

        return (d_row ** 2).mean() + (d_col ** 2).mean()

    def parameter_groups(self, lr_kernel=1e-3, lr_lut=1e-3):
        return [
            {"params": [self.kernel], "lr": lr_kernel, "weight_decay": 0.0},
            {"params": [self.raw_lut], "lr": lr_lut, "weight_decay": 0.0},
        ]

    def forward(self, input_tensor: torch.Tensor):
        assert input_tensor.dim() == 6, "Input must be shape [B, C, T, X, Y, Z]"
        assert input_tensor.shape[1] == 1, "Currently only one channel images are supported"

        x = input_tensor.permute(0, 1, 2, 4, 5, 3).contiguous()
        x = pad_circulat_t(x, self.pad_t)

        kernel = self.kernel.to(dtype=x.dtype)
        lut = self.activated_lut().to(x.dtype)

        if self.use_gpu:
            output = BilateralFilterFunction4DLutGPU.apply(x, kernel, lut)
        else:
            raise NotImplementedError("CPU is not supported")

        output = unpad_circular_t(output, self.pad_t)
        return output.permute(0, 1, 2, 5, 3, 4).contiguous()