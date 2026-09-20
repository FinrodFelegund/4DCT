"""
Trainable bilateral filter layer.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""
import torch
import bilateralfilter4d_gpu_lib as bilateral_filter_gpu_lib

def pad_circular_t(x: torch.Tensor, pad_t: int = 4):
    if pad_t < 1:
        return x

    T = x.shape[2]
    if pad_t > T:
        idx = torch.arange(-pad_t, T + pad_t, device=x.device) % T
        return x.index_select(2, idx).contiguous()
    return torch.cat([x[:, :, -pad_t:], x, x[:, :, :pad_t]], dim=2).contiguous()

def unpad_circular_t(x: torch.Tensor, pad_t: int = 4):
    if pad_t < 1:
        return x
    
    return x[:, :, pad_t:-pad_t]


class BilateralFilterFunction4dGPU(torch.autograd.Function):
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
    def forward(ctx, input_img, sigma_t, color_sigma):
        #assert len(input_img.shape) == 6, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
        assert input_img.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Use c++ implementation for better performance.
        outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_t = bilateral_filter_gpu_lib.forward_4d_gpu(input_img, sigma_t, color_sigma)

        ctx.save_for_backward(input_img,
                              sigma_t,
                              color_sigma,
                              outputTensor,
                              outputWeightsTensor,
                              dO_dx_ki,
                              dO_dsig_r,
                              dO_dsig_t)  # save for backward

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grad_sig_t = None
        grad_color_sigma = None

        input_img = ctx.saved_tensors[0]  # input image
        sigma_t = ctx.saved_tensors[1]
        color_sigma = ctx.saved_tensors[2]
        outputTensor = ctx.saved_tensors[3]  # filtered image
        outputWeightsTensor = ctx.saved_tensors[4]  # weights
        dO_dx_ki = ctx.saved_tensors[5]  # derivative of output with respect to input, while k==i
        dO_dsig_r = ctx.saved_tensors[6]  # derivative of output with respect to range sigma
        dO_dsig_t = ctx.saved_tensors[7]  # derivative of output with respect to sigma t

        # calculate gradient with respect to the sigmas
        grad_color_sigma = torch.sum(grad_output * dO_dsig_r)
        grad_sig_t = torch.sum(grad_output * dO_dsig_t)

        grad_output_tensor = bilateral_filter_gpu_lib.backward_4d_gpu(grad_output,
                                                                     input_img,
                                                                     outputTensor,
                                                                     outputWeightsTensor,
                                                                     dO_dx_ki,
                                                                     sigma_t,
                                                                     color_sigma)

        return grad_output_tensor, grad_sig_t, grad_color_sigma


class BilateralFilter4d(torch.nn.Module):
    def __init__(self, sigma_t, color_sigma, pad_t, use_gpu=True):
        super(BilateralFilter4d, self).__init__()

        self.use_gpu = use_gpu

        # make sigmas trainable parameters
        self.sigma_t = torch.nn.Parameter(torch.tensor(sigma_t))
        self.color_sigma = torch.nn.Parameter(torch.tensor(color_sigma))
        self.pad_t = pad_t

    def _get_sigmas(self):
        return self.sigma_t, self.color_sigma

    def forward(self, input_tensor: torch.Tensor):

        assert len(input_tensor.shape) == 6, "Input shape of 4d bilateral filter layer must equal [B, C, T, X, Y, Z]."
        assert input_tensor.shape[1] == 1, "Currently channel dimensions >1 are not supported."
        input_tensor = pad_circular_t(input_tensor, self.pad_t)
        B, C, T, Z, X, Y = input_tensor.shape
        input_tensor = input_tensor.permute(0,3,4,5,1,2).contiguous().view(B*X*Y*Z, C, T)

        # Choose between CPU processing and CUDA acceleration.
        if self.use_gpu:
            output = BilateralFilterFunction4dGPU.apply(input_tensor,
                                                      self.sigma_t,
                                                      self.color_sigma)
        else:
            raise NotImplementedError('CPU is not supported')

        output = output.view(B, Z, X, Y, C, T).permute(0, 4, 5, 1, 2, 3).contiguous()
        output = unpad_circular_t(output, self.pad_t)
        return output
