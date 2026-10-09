#include "bilateral.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward_gpu", &BilateralFilterCudaForward, "BF forward nd gpu");
    m.def("backward_gpu", &BilateralFilterCudaBackward, "BF backward nd gpu");
}