#include "bilateral.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_4dlearnablespt_gpu", &BilateralFilterCudaForward, "BF forward joint gpu");
    m.def("backward_4dlearnablespt_gpu", &BilateralFilterCudaBackward, "BF backward joint gpu");
}
