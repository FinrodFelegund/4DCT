from bilateral_filter_layer_3d import BilateralFilter3d
import bilateralfilter4d_gpu_lib as bilateral_filter_gpu_lib
from bilateral_filter_layer_4d import BilateralFilter4d
import bilateralfilter4dspt_gpu_lib
from bilateral_filter_layer_spt import BilateralFilter4dspt


import torch

def test_model(model: torch.nn.Module):
    dummy_tensor = torch.randn(size=(4, 1, 10, 32, 32, 32), device='cuda')

    output = model(dummy_tensor)
    print(output.shape)

def forward_4d():
    x = torch.randn(4*32*32*32, 1, 10, device='cuda')
    out, w, dO_dx, dO_dsig_r, dO_dsig_t = bilateral_filter_gpu_lib.forward_4d_gpu(
        x, torch.tensor(0.5), torch.tensor(0.05))
    torch.cuda.synchronize()
    print(out.abs().sum().item(), w.abs().sum().item(), w.min().item())


def forward_spt():
    x = torch.randn(4, 1, 10, 32, 32, 32, device='cuda')
    out, w, dO_dx, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_t = bilateralfilter4dspt_gpu_lib.forward_4dspt_gpu(
        x, torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.05))
    torch.cuda.synchronize()
    print(out.abs().sum().item(), w.abs().sum().item(), w.min().item())

    
if __name__ == '__main__':
    model3d = BilateralFilter3d(0.5, 0.5, 0.5, 0.001, use_gpu=True)
    model4d = BilateralFilter4d(0.5, 0.001, use_gpu=True)
    modelspt = BilateralFilter4dspt(0.5, 0.5, 0.5, 0.5, 0.001, use_gpu=True)

    test_model(model3d)
    test_model(model4d)
    test_model(modelspt)

    forward_4d()
    forward_spt()
