#include "bilateral.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward_gpu", &BilateralFilterCudaForward, "Trainable sigmas and intensity lookup");
    m.def("backward_gpu", &BilateralFilterCudaBackward, "Trainable sigmas and intensity lookup");
}