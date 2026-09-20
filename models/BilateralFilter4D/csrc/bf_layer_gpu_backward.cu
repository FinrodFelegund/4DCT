/*
Author: Fabian Wagner
Contact: fabian.wagner@fau.de

This file contains modified code, originally published under the following licence:
Copyright 2020 - 2021 MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include "bilateral.h"
#include "../utils/cuda_error_check.h"
#include "../utils/meta_macros.h"
#include <ATen/cuda/CUDAContext.h>

__constant__ int cBatchStrideBack;
__constant__ int cColorStrideBack;
__constant__ int cBatchCountBack;

__constant__ int cSizesBack[3];
__constant__ int cStridesBack[3];


__constant__ int cKernelSizesBack[3];
__constant__ int cHalfWindowSize_arrBack[3];
__constant__ float cGaussianKernel_tBack[256];
__constant__ float cTDistanceSquaredBack[256];
__constant__ float cColorExponentConstantBack;
__constant__ float cSigma_tBack;
__constant__ float cColorSigmaBack;


template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernel4DBackward(scalar_t* gradientInputTensor,
                                                    scalar_t* gradientOutputTensor,
                                                    scalar_t* inputTensor,
                                                    scalar_t* outputTensor,
                                                    scalar_t* outputWeightsTensor,
                                                    scalar_t* dO_dx_ki) {
  long long flatIdx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (flatIdx >= (long long)cBatchCountBack * cColorStrideBack)
    return;

  int homeOffset  = (int)(flatIdx % cColorStrideBack);
  long long batchOffset = (flatIdx / cColorStrideBack) * cBatchStrideBack;

  int homeT = homeOffset / cStridesBack[0];
  int homeIndex[] = {homeT};

  // Zero kernel aggregates.
  scalar_t valueSum = 0;
  scalar_t weightSum = 0;

  for (int kernelT = 0; kernelT < cKernelSizesBack[0]; kernelT++) {
    int neighbourT = max(0, min(homeT + (kernelT - cHalfWindowSize_arrBack[0]), cSizesBack[0] - 1));
    scalar_t gaussianT = cGaussianKernel_tBack[kernelT];

    int neighbourOffset = neighbourT * cStridesBack[0];

    bool flagNotClamped = true;
    int kernelIndex[] = {kernelT};
    int dimensions = 1;  // Must equal the number of spatial dimensions.

    for (int i = 0; i < dimensions; i++) {
        int HalfWindowSizeBack = cHalfWindowSize_arrBack[i];  // Define constant memory as new variable here (!!), otherwise: cudaErrorMisalignedAddress
        int neighbourIndex = homeIndex[i] + kernelIndex[i] - HalfWindowSizeBack;
        int neighbourIndexClamped = min(cSizesBack[i] - 1, max(0, neighbourIndex));
        if (neighbourIndex != neighbourIndexClamped) { flagNotClamped = false; }
    }

    scalar_t colorDistance = 0;
    scalar_t colorDistanceSquared = 0;

#pragma unroll
    for (int c = 0; c < C; c++) {
      scalar_t a = inputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
      scalar_t b = inputTensor[batchOffset + homeOffset + c * cColorStrideBack];  // Be careful: Here it is (X_k - X_i) and not (X_i - X_q)
      scalar_t diff = a - b;
      colorDistance += diff; // Do not take the absolute value here. Be careful with the signs.
      colorDistanceSquared += diff * diff;
    }

    scalar_t spatialWeight = gaussianT;
    scalar_t colorWeight = exp(cColorExponentConstantBack * colorDistanceSquared);
    scalar_t totalWeight = spatialWeight * colorWeight;

    // Aggregating values. Only do this if flagNotClamped: Pixels outside the image are disregarded.
    if (flagNotClamped) {
        scalar_t filter_kernel_back;

#pragma unroll
        for (int c = 0; c < C; c++) {
            // Distinguish cases for k!=i (calculation is done here)
            // and k==i (partial derivatives are precalculated).
            // If statement replaces center element of neighborhood/kernel.
            if (kernelT != cHalfWindowSize_arrBack[0]) {

                filter_kernel_back =
                        -(1 /
                          outputWeightsTensor[batchOffset + neighbourOffset + c * cColorStrideBack]) *
                        outputTensor[batchOffset + neighbourOffset + c * cColorStrideBack] *
                        totalWeight *
                        colorDistance / (cColorSigmaBack * cColorSigmaBack) +
                        (1 /
                          outputWeightsTensor[batchOffset + neighbourOffset + c * cColorStrideBack]) *
                        totalWeight *
                        (1 +
                          inputTensor[batchOffset + homeOffset + c * cColorStrideBack] * colorDistance /
                          (cColorSigmaBack * cColorSigmaBack));  // inputTensorData[homeOffset] !!
            } else {

                filter_kernel_back = dO_dx_ki[batchOffset + homeOffset + c * cColorStrideBack];
            }

            valueSum +=
                    gradientInputTensor[batchOffset + neighbourOffset + c * cColorStrideBack] *
                            filter_kernel_back;

        }

        weightSum += totalWeight;

    }
  }

#pragma unroll
  for (int c = 0; c < C; c++) {
    gradientOutputTensor[batchOffset + homeOffset + c * cColorStrideBack] = valueSum;

  }
}

template <int C, int D>
void BilateralFilterCudaBackwardFunction(torch::Tensor gradientInputTensor,
                                         torch::Tensor gradientOutputTensor,
                                         torch::Tensor inputTensor,
                                         torch::Tensor outputTensor,
                                         torch::Tensor outputWeightsTensor,
                                         torch::Tensor dO_dx_ki,
                                         float sigma_t,
                                         float colorSigma) {
  // Getting tensor description.
  TensorDescription desc = TensorDescription(inputTensor);

  // Pre-calculating gaussian kernel.
  int windowSize_t = 9;
  int halfWindowSize_t = floor(0.5f * windowSize_t);
  int halfWindowSize_arr[] = {halfWindowSize_t};
  float spatialExpConstant_t = -1.0f / (2 * sigma_t * sigma_t);
  float colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

  int* kernelSizes = new int[desc.dimensions];
  kernelSizes[0] = windowSize_t;

  auto* gaussianKernel_t = new float[windowSize_t];
  auto* tDistanceSquared = new float[windowSize_t];


  for (int i = 0; i < windowSize_t; i++) {
      int distance = i - halfWindowSize_t;
      gaussianKernel_t[i] = exp(distance * distance * spatialExpConstant_t);
      tDistanceSquared[i] = distance * distance;
  }


  // Writing constant memory.
  cudaMemcpyToSymbol(cBatchStrideBack, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cColorStrideBack, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSizesBack, desc.sizes, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cStridesBack, desc.strides, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cKernelSizesBack, kernelSizes, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cHalfWindowSize_arrBack, halfWindowSize_arr, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cGaussianKernel_tBack, gaussianKernel_t, sizeof(float) * windowSize_t);
  cudaMemcpyToSymbol(cTDistanceSquaredBack, tDistanceSquared, sizeof(float) * windowSize_t);
  cudaMemcpyToSymbol(cColorExponentConstantBack, &colorExpConstant, sizeof(float));
  cudaMemcpyToSymbol(cSigma_tBack, &sigma_t, sizeof(float));
  cudaMemcpyToSymbol(cColorSigmaBack, &colorSigma, sizeof(float));
  int batchCount = desc.batchCount;
  cudaMemcpyToSymbol(cBatchCountBack, &batchCount, sizeof(int));

  cuda_error_check("Cuda check before kernel call.");

#define BLOCK_SIZE 256
  long long total  = (long long)desc.batchCount * desc.channelStride;
  int blocks = (int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputTensor.scalar_type(), "BilateralFilterCudaKernel4DBackward", ([&] {
            BilateralFilterCudaKernel4DBackward<scalar_t, C>
                <<<dim3(blocks), dim3(BLOCK_SIZE), 0, at::cuda::getCurrentCUDAStream()>>>(
                    gradientInputTensor.data_ptr<scalar_t>(),
                    gradientOutputTensor.data_ptr<scalar_t>(),
                    inputTensor.data_ptr<scalar_t>(),
                    outputTensor.data_ptr<scalar_t>(),
                    outputWeightsTensor.data_ptr<scalar_t>(),
                    dO_dx_ki.data_ptr<scalar_t>());
      }));

  cuda_error_check("Cuda check after kernel call.");
//  delete[] kernel;
  delete[] kernelSizes;
  delete[] gaussianKernel_t;
  delete[] tDistanceSquared;
}

// Function to choose template implementation based on dynamic, channels and dimensions
torch::Tensor BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                          torch::Tensor inputTensor,
                                          torch::Tensor outputTensor,
                                          torch::Tensor outputWeightsTensor,
                                          torch::Tensor dO_dx_ki,
                                          float sigma_t,
                                          float colorSigma) {
  torch::Tensor gradientOutputTensor = torch::zeros_like(gradientInputTensor);
  cuda_error_check("beginning");

#define CASE(c, d) BilateralFilterCudaBackwardFunction<c, d>(gradientInputTensor, gradientOutputTensor, inputTensor, outputTensor, outputWeightsTensor, dO_dx_ki, sigma_t, colorSigma);
  SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, gradientInputTensor.size(1), gradientInputTensor.dim() - 2);

  return gradientOutputTensor;
}
