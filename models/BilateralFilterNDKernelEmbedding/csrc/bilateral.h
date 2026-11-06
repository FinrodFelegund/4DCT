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
                                                                    torch::Tensor kernel,
                                                                    torch::Tensor embedding,
                                                                    int hiddenWidth,
                                                                    int hiddenLayers,
                                                                    int pad_t,
                                                                    int act);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                                                                                    torch::Tensor inputTensor,
                                                                                                    torch::Tensor outputTensor,
                                                                                                    torch::Tensor outputWeightsTensor,
                                                                                                    torch::Tensor kernel,
                                                                                                    torch::Tensor embedding,
                                                                                                    int hiddenWidth,
                                                                                                    int hiddenlayers,
                                                                                                    int pad_t,
                                                                                                    int act,
                                                                                                    bool computeInputGradient);
                                