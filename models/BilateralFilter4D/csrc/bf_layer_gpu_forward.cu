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
#include <ATen/cuda/CUDAContext.h>
#include "bilateral.h"
#include "../utils/cuda_error_check.h"
#include "../utils/meta_macros.h"

__constant__ int cBatchStride;
__constant__ int cColorStride;
__constant__ int cBatchCount;

__constant__ int cSizes[3];
__constant__ int cStrides[3];

__constant__ int cKernelSizes[3];
__constant__ int cHalfWindowSize_arr[3];
__constant__ float cGaussianKernel_t[256];
__constant__ float cTDistanceSquared[256];

__constant__ float cColorExponentConstant;
__constant__ float cSigma_t;
__constant__ float cColorSigma;


template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernel4DForward(scalar_t* input,
                                                   scalar_t* output,
                                                   scalar_t* outputWeightsTensor,
                                                   scalar_t* dO_dx_ki,
                                                   scalar_t* dO_dsig_r,
                                                   scalar_t* dO_dsig_t) {

  long long flatIdx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (flatIdx >= (long long)cBatchCount * cColorStride)
    return;

  int       homeOffset  = (int)(flatIdx % cColorStride);
  long long batchOffset = (flatIdx / cColorStride) * cBatchStride;

  int homeT = homeOffset / cStrides[0];
  int homeIndex[] = {homeT};

  // Zero kernel aggregates.
  scalar_t valueSum = 0;
  scalar_t dw_dx_ki = 0;
  scalar_t dfilter_dx_ki = 0;
  scalar_t colorSum_w = 0;
  scalar_t colorSum_alpha = 0;
  scalar_t tSum_w = 0;
  scalar_t tSum_alpha = 0;
  scalar_t weightSum = 0;

  for (int kernelT = 0; kernelT < cKernelSizes[0]; kernelT++) {
    int neighbourT = max(0, min(homeT + (kernelT - cHalfWindowSize_arr[0]), cSizes[0] - 1));
    scalar_t gaussianT = cGaussianKernel_t[kernelT];


    int neighbourOffset = neighbourT * cStrides[0];

    bool flagNotClamped = true;
    int kernelIndex[] = {kernelT};
    int dimensions = 1;  // Must equal the number of spatial dimensions.

    for (int i = 0; i < dimensions; i++) {
        int HalfWindowSizeBack = cHalfWindowSize_arr[i];  // Define constant memory as new variable here (!!), otherwise: cudaErrorMisalignedAddress
        int neighbourIndex = homeIndex[i] + kernelIndex[i] - HalfWindowSizeBack;
        int neighbourIndexClamped = min(cSizes[i] - 1, max(0, neighbourIndex));
        if (neighbourIndex != neighbourIndexClamped) { flagNotClamped = false; }
    }

    scalar_t colorDistance = 0;
    scalar_t colorDistanceSquared = 0;

#pragma unroll
    for (int c = 0; c < C; c++) {
      scalar_t a = input[batchOffset + homeOffset + c * cColorStride];
      scalar_t b = input[batchOffset + neighbourOffset + c * cColorStride];  // Home - neighbor (!!) in backward the other way around !!
      scalar_t diff = a - b;
      colorDistance += diff; // Do not take the absolute value here. Be careful with the signs.
      colorDistanceSquared += diff * diff;
    }

    scalar_t spatialWeight = gaussianT;
    scalar_t colorWeight = exp(cColorExponentConstant * colorDistanceSquared);
    scalar_t totalWeight = spatialWeight * colorWeight;

    // Aggregating values. Only do this if flagNotClamped: Pixels outside the image are disregarded.
    if (flagNotClamped) {

#pragma unroll
      for (int c = 0; c < C; c++) {
          valueSum += input[batchOffset + neighbourOffset + c * cColorStride] * totalWeight;

          // Derivative of weights with respect to X_i while i=k.
          dw_dx_ki += (-1) * totalWeight * colorDistance / (cColorSigma * cColorSigma);
          // Derivative of convolved image with respect to X_i while i=k.
          dfilter_dx_ki += (-1) * totalWeight * input[batchOffset + neighbourOffset + c * cColorStride] *
                            colorDistance / (cColorSigma *
                                            cColorSigma); // Be careful, the +1 is missing here -> Added before filling dfilter_dx_kiData

          colorSum_w +=
                  totalWeight * colorDistanceSquared /
                  std::abs(cColorSigma * cColorSigma * cColorSigma);
          colorSum_alpha +=
                  totalWeight * input[batchOffset + neighbourOffset + c * cColorStride] *
                  colorDistanceSquared / std::abs(cColorSigma * cColorSigma * cColorSigma);

          tSum_w +=
                  totalWeight * cTDistanceSquared[kernelT] /
                  std::abs(cSigma_t * cSigma_t * cSigma_t);
          tSum_alpha +=
                  totalWeight * input[batchOffset + neighbourOffset + c * cColorStride] *
                  cTDistanceSquared[kernelT] / std::abs(cSigma_t * cSigma_t * cSigma_t);

      }

      weightSum += totalWeight;

        
      }
    }
  

#pragma unroll
  for (int c = 0; c < C; c++) {
//    output[batchOffset + homeOffset + c * cColorStride] /= weightSum;
    output[batchOffset + homeOffset + c * cColorStride] = valueSum / weightSum;

    // Pre-computations for the backward pass:
    outputWeightsTensor[batchOffset + homeOffset + c * cColorStride] = weightSum;
    dO_dx_ki[batchOffset + homeOffset + c * cColorStride] =
          -(1 / weightSum) * (valueSum / weightSum) *
          dw_dx_ki +
          (1 / weightSum) * (dfilter_dx_ki + 1); // +1 for dfilter_dx_ki is added here
    dO_dsig_r[batchOffset + homeOffset + c * cColorStride] =
          -(1 / weightSum) * (valueSum / weightSum) *
          colorSum_w + (1 / weightSum) * colorSum_alpha;
    dO_dsig_t[batchOffset + homeOffset + c * cColorStride] =
          -(1 / weightSum) * (valueSum / weightSum) *
          tSum_w + (1 / weightSum) * tSum_alpha;
  }
}

template <int C, int D>
void BilateralFilterCudaForwardFunction(torch::Tensor inputTensor,
                                        torch::Tensor outputTensor,
                                        torch::Tensor outputWeightsTensor,
                                        torch::Tensor dO_dx_ki,
                                        torch::Tensor dO_dsig_r,
                                        torch::Tensor dO_dsig_t,
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
  cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cColorStride, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSizes, desc.sizes, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cStrides, desc.strides, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cKernelSizes, kernelSizes, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cHalfWindowSize_arr, halfWindowSize_arr, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cGaussianKernel_t, gaussianKernel_t, sizeof(float) * windowSize_t);
  cudaMemcpyToSymbol(cTDistanceSquared, tDistanceSquared, sizeof(float) * windowSize_t);
  cudaMemcpyToSymbol(cColorExponentConstant, &colorExpConstant, sizeof(float));
  cudaMemcpyToSymbol(cSigma_t, &sigma_t, sizeof(float));
  cudaMemcpyToSymbol(cColorSigma, &colorSigma, sizeof(float));
  int batchCount = desc.batchCount;
  cudaMemcpyToSymbol(cBatchCount, &batchCount, sizeof(int));

  cuda_error_check("Cuda check before kernel call.");

#define BLOCK_SIZE 256
  long long total  = (long long)desc.batchCount * desc.channelStride;
  int       blocks = (int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputTensor.scalar_type(), "BilateralFilterCudaKernel4DForward", ([&] {
                BilateralFilterCudaKernel4DForward<scalar_t, C>
                <<<dim3(blocks), dim3(BLOCK_SIZE), 0, at::cuda::getCurrentCUDAStream()>>>(
                    inputTensor.data_ptr<scalar_t>(),
                    outputTensor.data_ptr<scalar_t>(),
                    outputWeightsTensor.data_ptr<scalar_t>(),
                    dO_dx_ki.data_ptr<scalar_t>(),
                    dO_dsig_r.data_ptr<scalar_t>(),
                    dO_dsig_t.data_ptr<scalar_t>());
        }
      ));

  cuda_error_check("Cuda check after kernel call.");
//  delete[] kernel;
  delete[] kernelSizes;
  delete[] gaussianKernel_t;
  delete[] tDistanceSquared;

}

// Function to choose template implementation based on dynamic, channels and dimensions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BilateralFilterCudaForward(torch::Tensor inputTensor,
                                                                                                                                               float sigma_t,
                                                                                                                                               float colorSigma) {
  torch::Tensor outputTensor = torch::zeros_like(inputTensor);
  torch::Tensor outputWeightsTensor = torch::zeros_like(inputTensor);
  torch::Tensor dO_dx_ki = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_t = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_r = torch::zeros_like(inputTensor);
  cuda_error_check("beginning");

#define CASE(c, d) BilateralFilterCudaForwardFunction<c, d>(inputTensor, outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_t, sigma_t, colorSigma);
  SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, inputTensor.size(1), inputTensor.dim() - 2);

  return {outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_t};
}
