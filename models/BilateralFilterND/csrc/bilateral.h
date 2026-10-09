/*
Author: Fabian Wagner
Contact: fabian.wagner@fau.de
*/

#pragma once

#include <torch/extension.h>
#include "../utils/common_utils.h"
#include "../utils/tensor_description.h"
#include <vector>

#define BF_CUDA_MAX_CHANNELS 1
#define BF_CUDA_MAX_SPATIAL_DIMENSION 4

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BilateralFilterCudaForward(torch::Tensor inputTensor,
                                                                                                                                               float sigma_t,
                                                                                                                                               float sigma_x,
                                                                                                                                               float sigma_y,
                                                                                                                                               float sigma_z,
                                                                                                                                               float color_sigma);
   
torch::Tensor BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                          torch::Tensor inputTensor,
                                          torch::Tensor outputTensor,
                                          torch::Tensor outputWeightsTensor,
                                          torch::Tensor dO_dx_ki,
                                          float sigma_t,
                                          float sigma_x,
                                          float sigma_y,
                                          float sigma_z,
                                          float colorSigma);                                                                                                                                               