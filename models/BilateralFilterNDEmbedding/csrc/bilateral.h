#pragma once

#include <torch/extension.h>
#include "../utils/common_utils.h"
#include "../utils/tensor_description.h"
#include "../utils/embedding_utils.cuh"
#include <vector>
#include <algorithm>
#include <iostream>

#define BF_CUDA_MAX_CHANNELS 1
#define BF_CUDA_MAX_SPATIAL_DIMENSIONS 4


std::tuple<torch::Tensor, torch::Tensor>BilateralFilterCudaForward(torch::Tensor inputTensor,
                                                                    float sigma_t,
                                                                    float sigma_x,
                                                                    float sigma_y,
                                                                    float sigma_z,
                                                                    torch::Tensor weights,
                                                                    int hiddenWidth,
                                                                    int nHidden,
                                                                    int pad_t);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                                                                                    torch::Tensor inputTensor,
                                                                                                    torch::Tensor outputTensor,
                                                                                                    torch::Tensor outputWeightsTensor,
                                                                                                    float sigma_t,
                                                                                                    float sigma_x,
                                                                                                    float sigma_y,
                                                                                                    float sigma_z,
                                                                                                    torch::Tensor weights,
                                                                                                    int hiddenWidth,
                                                                                                    int nHidden,
                                                                                                    int pad_t);
                                