#include "bilateral.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward_gpu", &BilateralFilterCudaForward, "Trainable sigmas and intensity feed forward network");
    m.def("backward_gpu", &BilaterlFilterCudaBackward, "Trainable sigmas and intensity feed forward network");
}