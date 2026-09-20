#include "bilateral.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_4d_gpu", &BilateralFilterCudaForward, "BF forward 4d gpu");
    m.def("backward_4d_gpu", &BilateralFilterCudaBackward, "BF backward 4d gpu");
}
