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

__constant__ int cBatchStride;
__constant__ int cColorStride;

__constant__ int cSizes[4];
__constant__ int cStrides[4];

__constant__ int cKernelSizes[4];
__constant__ int cHalfWindowSize_arr[4];

__constant__ float cColorExponentConstant;
__constant__ float cColorSigma;

template <typename scalar_t, int C>
__global__ void BilaterFilterCudaKernelLearnableSptForward(
    const scalar_t* input,
    const scalar_t* kernelTensor,
    scalar_t* output,
    scalar_t* outputWeightsTensor,
    scalar_t* d0_dx_ki,
    scalar_t* d0_dsig_r) {
  
  int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
  int batchOffset = blockIdx.y * cBatchStride;

  if(homeOffset >= cColorStride){
    return;
  }

  const int Wt = cKernelSizes[0];
  const int Wx = cKernelSizes[1];
  const int Wy = cKernelSizes[2];
  const int Wz = cKernelSizes[3];

  const int halfT = cHalfWindowSize_arr[0];
  const int halfX = cHalfWindowSize_arr[1];
  const int halfY = cHalfWindowSize_arr[2];
  const int halfZ = cHalfWindowSize_arr[3];

  const int homeT = homeOffset / cStrides[0];
  const int homeX = (homeOffset - homeT * cStrides[0]) / cStrides[1];
  const int homeY = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1]) / cStrides[2];
  const int homeZ = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1] - homeY * cStrides[2]) / cStrides[3];
  const int homeIndex[] = {homeT, homeX, homeY, homeZ};

  const int centerKernelOffset = ((halfT * Wx + halfX) * Wy + halfY) * Wz + halfZ;

  scalar_t valueSum = 0;
  scalar_t weightSum = 0;
  scalar_t dw_dx_ki = 0;
  scalar_t dfilter_dx_ki = 0;
  scalar_t colorSum_w = 0;
  scalar_t colorSum_alpha = 0;

  for(int kernelT = 0; kernelT < Wt; kernelT++){
    int neighbourT = max(0, min(homeT + (kernelT - halfT), cSizes[0] - 1));

    for(int kernelX = 0; kernelX < Wx; kernelX++){
      int neighbourX = max(0, min(homeX + (kernelX - halfX), cSizes[1] - 1));

      for(int kernelY = 0; kernelY < Wy; kernelY++){
        int neighbourY = max(0, min(homeY +(kernelY - halfY), cSizes[2] - 1));

        for(int kernelZ = 0; kernelZ < Wz; kernelZ++){
          int neighbourZ = max(0, min(homeZ + (kernelZ - halfZ), cSizes[3] - 1));

          int neighbourOffset = neighbourT * cStrides[0] + neighbourX * cStrides[1] + neighbourY * cStrides[2] + neighbourZ;

          bool flagNotClamped = true;
          int kernelIndex[] = {kernelT, kernelX, kernelY, kernelZ};
          int dimensions = 4;

          for(int i = 0; i < dimensions; i++){
            int HalfWindowSizeBack = cHalfWindowSize_arr[i];
            int neighbourindex = homeIndex[i] + kernelIndex[i] - HalfWindowSizeBack;
            int neighbourIndexClamped = min(cSizes[i] - 1, max(0, neighbourindex));
            if(neighbourindex != neighbourIndexClamped){
              flagNotClamped = false;
            }
          }

          scalar_t colorDistance = 0;
          scalar_t colorDistanceSquared = 0;

#pragma unroll
          for(int c = 0; c < C; c++){
            scalar_t a = input[batchOffset + homeOffset + c * cColorStride];
            scalar_t b = input[batchOffset + neighbourOffset + c * cColorStride];
            scalar_t diff = a - b;
            colorDistance += diff;
            colorDistanceSquared += diff * diff;
          }

          const int kernelOffset = ((kernelT * Wx + kernelX) * Wy + kernelY) * Wz + kernelZ;
          scalar_t spatialWeight = kernelTensor[kernelOffset];

          scalar_t colorWeight = exp(cColorExponentConstant * colorDistanceSquared);
          scalar_t totalWeight = spatialWeight * colorWeight;

          if(flagNotClamped){
#pragma unroll
            for(int c = 0; c < C; c++){
              scalar_t neighbourValue = input[batchOffset + neighbourOffset + c * cColorStride];

              valueSum += neighbourValue * totalWeight;

              dw_dx_ki += (-1) * totalWeight * colorDistance / (cColorSigma * cColorSigma);
              dfilter_dx_ki += (-1) * totalWeight * neighbourValue * colorDistance / (cColorSigma * cColorSigma);

              colorSum_w += totalWeight * colorDistanceSquared / std::abs(cColorSigma * cColorSigma * cColorSigma);
              colorSum_alpha += totalWeight * neighbourValue * colorDistanceSquared / std::abs(cColorSigma * cColorSigma * cColorSigma);

            }
          
          weightSum += totalWeight;
          }
        }
      }
    }
  }

  const scalar_t centerK = kernelTensor[centerKernelOffset];

#pragma unroll
  for(int c = 0; c < C; c++){
    output[batchOffset + homeOffset + c * cColorStride] = valueSum / weightSum;
    outputWeightsTensor[batchOffset + homeOffset + c * cColorStride] = weightSum;

    d0_dx_ki[batchOffset + homeOffset + c * cColorStride] = -(1 / weightSum) * (valueSum / weightSum) * dw_dx_ki + (1 / weightSum) * (dfilter_dx_ki + centerK);
    d0_dsig_r[batchOffset + homeOffset + c * cColorStride] = -(1 / weightSum) * (valueSum / weightSum) * colorSum_w + (1 / weightSum) * colorSum_alpha;
  }
}

template <int C, int D>
void BilateralFilterCudaForwardFunction(torch::Tensor inputTensor,
                                        torch::Tensor kernelTensor,
                                        torch::Tensor outputTensor,
                                        torch::Tensor outputWeightsTensor,
                                        torch::Tensor d0_dx_ki,
                                        torch::Tensor d0_dsig_r,
                                        float colorSigma) {
                  
  TensorDescription desc = TensorDescription(inputTensor);

  int kernelSizes[4];
  int halfWindowSize_arr[4];
  for(int i = 0; i< 4; i++){
    kernelSizes[i] = (int)kernelTensor.size(i);
    halfWindowSize_arr[i] = kernelSizes[i] / 2;
  }

  float colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

  cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cColorStride, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSizes, desc.sizes, sizeof(int) * 4);
  cudaMemcpyToSymbol(cStrides, desc.strides, sizeof(int) * 4);
  cudaMemcpyToSymbol(cKernelSizes, kernelSizes, sizeof(int) * 4);
  cudaMemcpyToSymbol(cHalfWindowSize_arr, halfWindowSize_arr, sizeof(int) * 4);
  cudaMemcpyToSymbol(cColorExponentConstant, &colorExpConstant, sizeof(float));
  cudaMemcpyToSymbol(cColorSigma, &colorSigma, sizeof(float));

  cuda_error_check("Cuda check before kernel call");

#define BLOCK_SIZE 128
  AT_DISPATCH_FLOATING_TYPES(
    inputTensor.scalar_type(), "BilaterFilterCudaKernelLearnableSptForward", ([&] {
      BilaterFilterCudaKernelLearnableSptForward<scalar_t, C>
          <<<dim3(int(desc.channelStride / BLOCK_SIZE + 1), desc.batchCount),
             dim3(BLOCK_SIZE, 1)>>>(inputTensor.data_ptr<scalar_t>(),
                                    kernelTensor.data_ptr<scalar_t>(),
                                    outputTensor.data_ptr<scalar_t>(),
                                    outputWeightsTensor.data_ptr<scalar_t>(),
                                    d0_dx_ki.data_ptr<scalar_t>(),
                                    d0_dsig_r.data_ptr<scalar_t>());
      }
    ));

  cuda_error_check("Cuda check after kernel call");
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BilateralFilterCudaForward(torch::Tensor inputTensor, torch::Tensor kernelTensor, float colorSigma){
  CHECK_CONTIGUOUS_CUDA(inputTensor);
  CHECK_CONTIGUOUS_CUDA(kernelTensor);
  CHECK_SAME_DTYPE(inputTensor, kernelTensor);
  TORCH_CHECK(inputTensor.dim() == 6, "inputTensor must be [B, C, T, X, Y, Z]");
  TORCH_CHECK(kernelTensor.dim() == 4, "kernelTensor must be [Wt, Wx, Wy, Wz]");
  for (int i = 0; i < 4; i++){
    TORCH_CHECK(kernelTensor.size(i) % 2 == 1, "kernel sizes must be odd");
  } 

  torch::Tensor outputTensor = torch::zeros_like(inputTensor);
  torch::Tensor outputWeightsTensor = torch::zeros_like(inputTensor);
  torch::Tensor d0_dx_ki = torch::zeros_like(inputTensor);
  torch::Tensor d0_dsig_r = torch::zeros_like(inputTensor);

  cuda_error_check("beginning");

#define CASE(c, d)                                                                                                                            \
  BilateralFilterCudaForwardFunction<c, d>(inputTensor, kernelTensor, outputTensor, outputWeightsTensor, d0_dx_ki, d0_dsig_r, colorSigma);
  SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, inputTensor.size(1), inputTensor.dim() - 2)
#undef CASE

  return {outputTensor, outputWeightsTensor, d0_dx_ki, d0_dsig_r};
}

