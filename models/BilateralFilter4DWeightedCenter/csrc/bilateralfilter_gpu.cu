#include "bilateral.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_4dcenterweight_gpu", &BilateralFilterCudaForward, "BF forward joint gpu");
    m.def("backward_4dcenterweight_gpu", &BilateralFilterCudaBackward, "BF backward joint gpu");
}
