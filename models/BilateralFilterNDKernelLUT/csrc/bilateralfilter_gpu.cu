#include "bilateral.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward_4dlut_gpu", &BilateralFilterCudaForward, "Trainable kernel and intensity lut forward");
    m.def("backward_4dlut_gpu", &BilateralFilterCudaBackward, "Trainable kernel and intensity lut backward");
}