#include "bilateral.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_4dspt_gpu", &BilateralFilterCudaForward, "BF forward joint gpu");
    m.def("backward_4dspt_gpu", &BilateralFilterCudaBackward, "BF backward joint gpu");
}
