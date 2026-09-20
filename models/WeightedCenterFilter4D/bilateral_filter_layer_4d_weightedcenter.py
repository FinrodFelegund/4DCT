"""
Trainable bilateral filter layer.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""
import torch

import bilateralfilter4dweightedcenter_gpu_lib as bilateral_filter_gpu_lib


def pad_circular_t(x: torch.Tensor, pad_t: int = 4):
    T = x.shape[2]
    if pad_t < 1:
        return x
    
    if pad_t > T:
        idx = torch.arange(-pad_t, T + pad_t, device=x.device) % T
        return x.index_select(2, idx).contiguous()
    return torch.cat([x[:, :, T - pad_t:], x, x[:, :, :pad_t]], dim=2).contiguous()

def unpad_circular_t(x: torch.Tensor, pad_t: int = 4):
    if pad_t < 1:
        return x
    return x[:, :, pad_t:-pad_t]


class BilateralFilterFunction4dsptGPUWeightedCenter(torch.autograd.Function):
    """
    3D Differentiable bilateral filter to remove noise while preserving edges. CUDA accelerated layer.
    See:
        Paris, S. (2007). A gentle introduction to bilateral filtering and its applications: https://dl.acm.org/doi/pdf/10.1145/1281500.1281604
    Args:
        input_img: input tensor: [B, C, X, Y, Z]
        sigma_x: standard deviation of the spatial blur in x direction.
        sigma_y: standard deviation of the spatial blur in y direction.
        sigma_z: standard deviation of the spatial blur in z direction.
        color_sigma: standard deviation of the range kernel.
    Returns:
        output (torch.Tensor): Filtered tensor.
    """

    @staticmethod
    def forward(ctx, input_img, sigma_x, sigma_y, sigma_z, sigma_t, color_sigma, center_alpha):
        assert len(input_img.shape) == 6, "Input shape of 3d bilateral filter layer must equal [B, C, T, X, Y, Z]."
        assert input_img.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Use c++ implementation for better performance.
        outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z, d0_dsig_t, d0_dcenter_alpha = bilateral_filter_gpu_lib.forward_4dcenterweight_gpu(input_img, sigma_x, sigma_y, sigma_z, sigma_t, color_sigma, center_alpha)

        ctx.save_for_backward(input_img,
                              sigma_x,
                              sigma_y,
                              sigma_z,
                              sigma_t,
                              color_sigma,
                              center_alpha,
                              outputTensor,
                              outputWeightsTensor,
                              dO_dx_ki,
                              dO_dsig_r,
                              dO_dsig_x,
                              dO_dsig_y,
                              dO_dsig_z,
                              d0_dsig_t,
                              d0_dcenter_alpha)  # save for backward

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_sig_x = None
        grad_sig_y = None
        grad_sig_z = None
        grad_color_sigma = None

        grad_output = grad_output.contiguous()
        input_img = ctx.saved_tensors[0]  # input image
        sigma_x = ctx.saved_tensors[1]
        sigma_y = ctx.saved_tensors[2]
        sigma_z = ctx.saved_tensors[3]
        sigma_t = ctx.saved_tensors[4]
        color_sigma = ctx.saved_tensors[5]
        center_alpha = ctx.saved_tensors[6]
        outputTensor = ctx.saved_tensors[7]  # filtered image
        outputWeightsTensor = ctx.saved_tensors[8]  # weights
        dO_dx_ki = ctx.saved_tensors[9]  # derivative of output with respect to input, while k==i
        dO_dsig_r = ctx.saved_tensors[10]  # derivative of output with respect to range sigma
        dO_dsig_x = ctx.saved_tensors[11]  # derivative of output with respect to sigma x
        dO_dsig_y = ctx.saved_tensors[12]  # derivative of output with respect to sigma y
        dO_dsig_z = ctx.saved_tensors[13]  # derivative of output with respect to sigma z
        d0_dsig_t = ctx.saved_tensors[14] # derivative of output with respect to sigma t
        d0_dcenter_alpha = ctx.saved_tensors[15]

        # calculate gradient with respect to the sigmas
        grad_color_sigma = torch.sum(grad_output * dO_dsig_r)
        grad_sig_x = torch.sum(grad_output * dO_dsig_x)
        grad_sig_y = torch.sum(grad_output * dO_dsig_y)
        grad_sig_z = torch.sum(grad_output * dO_dsig_z)
        grad_sig_t = torch.sum(grad_output * d0_dsig_t)
        grad_center_alpha = torch.sum(grad_output * d0_dcenter_alpha)

        grad_output_tensor = bilateral_filter_gpu_lib.backward_4dcenterweight_gpu(grad_output,
                                                                     input_img,
                                                                     outputTensor,
                                                                     outputWeightsTensor,
                                                                     dO_dx_ki,
                                                                     sigma_x,
                                                                     sigma_y,
                                                                     sigma_z,
                                                                     sigma_t,
                                                                     color_sigma
                                                                     )

        return grad_output_tensor, grad_sig_x, grad_sig_y, grad_sig_z, grad_sig_t, grad_color_sigma, grad_center_alpha


class BilateralFilter4dspt(torch.nn.Module):
    def __init__(self, sigma_x, sigma_y, sigma_z, sigma_t, color_sigma, center_alpha, pad_t, use_gpu=True):
        super(BilateralFilter4dspt, self).__init__()

        self.use_gpu = use_gpu

        # make sigmas trainable parameters
        self.sigma_x = torch.nn.Parameter(torch.tensor(sigma_x))
        self.sigma_y = torch.nn.Parameter(torch.tensor(sigma_y))
        self.sigma_z = torch.nn.Parameter(torch.tensor(sigma_z))
        self.sigma_t = torch.nn.Parameter(torch.tensor(sigma_t))
        self.color_sigma = torch.nn.Parameter(torch.tensor(color_sigma))
        self.center_alpha = torch.nn.Parameter(torch.tensor(center_alpha))
        self.pad_t = pad_t

    def _get_sigmas(self):
        return self.sigma_x, self.sigma_y, self.sigma_z, self.sigma_t, self.color_sigma, self.center_alpha

    def forward(self, input_tensor: torch.Tensor):

        assert len(input_tensor.shape) == 6, "Input shape of 4d bilateral filter layer must equal [B, C, T, X, Y, Z]."
        assert input_tensor.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Choose between CPU processing and CUDA acceleration.
        B, C, T, Z, X, Y = input_tensor.shape
        input_tensor = input_tensor.permute(0, 1, 2, 4, 5, 3).contiguous()
        input_tensor = pad_circular_t(input_tensor, pad_t=self.pad_t)

        if self.use_gpu:
            output = BilateralFilterFunction4dsptGPUWeightedCenter.apply(input_tensor,
                                                      self.sigma_x,
                                                      self.sigma_y,
                                                      self.sigma_z,
                                                      self.sigma_t,
                                                      self.color_sigma,
                                                      self.center_alpha)
        else:
            raise NotImplementedError("CPU is not supported")

        output = unpad_circular_t(output, pad_t=self.pad_t)
        output = output.permute(0, 1, 2, 5, 3, 4).contiguous()
        return output

