#include <cuda.h>
#include <cuda_runtime.h>

#include "bilateral.h"
#include "../utils/cuda_error_check.h"
#include "../utils/meta_macros.h"

__constant__ int cBatchStrideBack;
__constant__ int cColorStrideBack;

__constant__ int cSizesBack[4];
__constant__ int cStridesBack[4];

__constant__ int cKernelSizesBack[4];
__constant__ int cHalfWindowSize_arrBack[4];

__constant__ float cColorExponentConstantBack;
__constant__ float cColorSigmaBack;

template <typename scalar_t, int C>
__global__ void BilaterFilterCudaKernelLearnableSptBackward(
  const scalar_t* gradientInputTensor,
  scalar_t* gradientOutputTensor,
  scalar_t* gradKernelTensor,
  const scalar_t* inputTensor,
  const scalar_t* outputTensor,
  const scalar_t* outputWeightsTensor,
  const scalar_t* d0_dx_ki,
  const scalar_t* kernelTensor){

    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStrideBack;

    if(homeOffset >= cColorStrideBack){
      return;
    }
    const bool active = homeOffset < cColorStrideBack;

    int kernelSize = 1;
    for(int i = 0; i < 4; i++){
      kernelSize = kernelSize * cKernelSizesBack[i];
    }
    extern __shared__ char smem_raw[];
    scalar_t* sGradKernel = reinterpret_cast<scalar_t*>(smem_raw);
    for(int i = threadIdx.x; i < kernelSize; i += blockDim.x){
      sGradKernel[i] = (scalar_t)0;
    }
    __syncthreads();

    const int Wt = cKernelSizesBack[0];
    const int Wx = cKernelSizesBack[1];
    const int Wy = cKernelSizesBack[2];
    const int Wz = cKernelSizesBack[3];
    const int halfT = cHalfWindowSize_arrBack[0];
    const int halfX = cHalfWindowSize_arrBack[1];
    const int halfY = cHalfWindowSize_arrBack[2];
    const int halfZ = cHalfWindowSize_arrBack[3];

    const int homeT = homeOffset / cStridesBack[0];
    const int homeX = (homeOffset - homeT * cStridesBack[0]) / cStridesBack[1];
    const int homeY = (homeOffset - homeT * cStridesBack[0] - homeX * cStridesBack[1]) / cStridesBack[2];
    const int homeZ = (homeOffset - homeT * cStridesBack[0] - homeX * cStridesBack[1] - homeY * cStridesBack[2]) / cStridesBack[3];
    const int homeIndex[] = {homeT, homeX, homeY, homeZ};

    scalar_t valueSum = 0;
    if(active){


      for(int kernelT = 0; kernelT < Wt; kernelT++){
        int neighbourT = max(0, min(homeT + (kernelT - halfT), cSizesBack[0] - 1));

        for(int kernelX = 0; kernelX < Wx; kernelX++){
          int neighbourX = max(0, min(homeX + (kernelX - halfX), cSizesBack[1] - 1));

          for(int kernelY = 0; kernelY < Wy; kernelY++){
            int neighbourY = max(0, min(homeY + (kernelY - halfY), cSizesBack[2] - 1));

            for(int kernelZ = 0; kernelZ < Wz; kernelZ++){
              int neighbourZ = max(0, min(homeZ + (kernelZ - halfZ), cSizesBack[3] - 1));

              int neighbourOffset = neighbourT * cStridesBack[0] + neighbourX * cStridesBack[1] + neighbourY * cStridesBack[2] + neighbourZ;

              bool flagNotClamped = true;
              int kernelIndex[] = {kernelT, kernelX, kernelY, kernelZ};
              int dimensions = 4;
              
              for(int i = 0; i < dimensions; i++){
                int HalfWindowSizeBack = cHalfWindowSize_arrBack[i];
                int neighbourIndex = homeIndex[i] + kernelIndex[i] - HalfWindowSizeBack;
                int neighbourIndexClamped = min(cSizesBack[i] - 1, max(0, neighbourIndex));
                if(neighbourIndex != neighbourIndexClamped){
                  flagNotClamped = false;
                }
              }

              scalar_t colorDistance = 0;
              scalar_t colorDistanceSquared = 0;
  #pragma unroll
              for(int c = 0; c < C; c++){
                scalar_t a = inputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                scalar_t b = inputTensor[batchOffset + homeOffset + c * cColorStrideBack];
                scalar_t diff = a - b;
                colorDistance += diff;
                colorDistanceSquared += diff * diff;
              }

              scalar_t colorWeight = exp(cColorExponentConstantBack * colorDistanceSquared);
              const int kernelOffset = ((kernelT * Wx + kernelX) * Wy + kernelY) * Wz + kernelZ;

              const int mirrorT = 2 * halfT - kernelT;
              const int mirrorX = 2 * halfX - kernelX;
              const int mirrorY = 2 * halfY - kernelY;
              const int mirrorZ = 2 * halfZ - kernelZ;
              const int mirroredKernelOffset = ((mirrorT * Wx + mirrorX) * Wy + mirrorY) * Wz + mirrorZ;

              scalar_t totalWeight = kernelTensor[mirroredKernelOffset] * colorWeight;

              if(flagNotClamped){
                scalar_t filter_kernel_back = 0;
  #pragma unroll
                for(int c = 0; c < C; c++){
                  if(kernelT != halfT || kernelX != halfX || kernelY != halfY || kernelZ != halfZ){
                    scalar_t W_i = outputWeightsTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                    scalar_t O_i = outputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                    scalar_t x_k = inputTensor[batchOffset + homeOffset + c * cColorStrideBack];

                    filter_kernel_back = 
                      -(1 / W_i) * O_i * totalWeight * colorDistance / (cColorSigmaBack * cColorSigmaBack) +
                      (1 / W_i) * totalWeight * (1 + x_k * colorDistance / (cColorSigmaBack * cColorSigmaBack));
                  } else {
                    filter_kernel_back = d0_dx_ki[batchOffset + homeOffset + c * cColorStrideBack];
                  }

                  valueSum += gradientInputTensor[batchOffset + neighbourOffset + c * cColorStrideBack] * filter_kernel_back;

                  scalar_t W_m = outputWeightsTensor[batchOffset + homeOffset + c * cColorStrideBack];
                  scalar_t O_m = outputTensor[batchOffset + homeOffset + c * cColorStrideBack];
                  scalar_t x_neighbour = inputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                  scalar_t g_m = gradientInputTensor[batchOffset + homeOffset + c * cColorStrideBack];

                  scalar_t dK_contribution = g_m * (colorWeight / W_m) * (x_neighbour - O_m);
                  atomicAdd(&sGradKernel[kernelOffset], dK_contribution);
                }
              }
            }
          }
        }
      }
    }
    __syncthreads();

    if(active){

      for(int i = threadIdx.x; i < kernelSize; i+= blockDim.x){
        scalar_t v = sGradKernel[i];
        if(v != (scalar_t)0){
          atomicAdd(&gradKernelTensor[i], v);
        }
      }
#pragma unroll
    for(int c = 0; c < C; c++){
      gradientOutputTensor[batchOffset + homeOffset + c * cColorStrideBack] = valueSum;
    }
  }
}


template <int C, int D>
void BilateralFilterCudaBackwardFunction(torch::Tensor gradientInputTensor,
                                       torch::Tensor gradientOutputTensor,
                                       torch::Tensor gradKernelTensor,
                                       torch::Tensor inputTensor,
                                       torch::Tensor outputTensor,
                                       torch::Tensor outputWeightsTensor,
                                       torch::Tensor dO_dx_ki,
                                       torch::Tensor kernelTensor,
                                       float colorSigma){
  
  TensorDescription desc = TensorDescription(inputTensor);

  int kernelSizes[4];
  int kernelSize = 1;
  int halfWindowSize_arr[4];
  for(int i = 0; i < 4; i++){
    kernelSizes[i] = (int)kernelTensor.size(i);
    halfWindowSize_arr[i] = kernelSizes[i] / 2;
    kernelSize = kernelSize * kernelSizes[i];
  }

  float colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

  cudaMemcpyToSymbol(cBatchStrideBack, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cColorStrideBack, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSizesBack, desc.sizes, sizeof(int) * 4);
  cudaMemcpyToSymbol(cStridesBack, desc.strides, sizeof(int) * 4);
  cudaMemcpyToSymbol(cKernelSizesBack, kernelSizes, sizeof(int) * 4);
  cudaMemcpyToSymbol(cHalfWindowSize_arrBack, halfWindowSize_arr, sizeof(int) * 4);
  cudaMemcpyToSymbol(cColorExponentConstantBack, &colorExpConstant, sizeof(float));
  cudaMemcpyToSymbol(cColorSigmaBack, &colorSigma, sizeof(float));

  cuda_error_check("Cuda check before kernel call");

#define BLOCK_SIZE 128

  AT_DISPATCH_FLOATING_TYPES(
    inputTensor.scalar_type(), "BilaterFilterCudaKernelLearnableSptBackward", ([&] {
      size_t shared_memory = (size_t)kernelSize * sizeof(scalar_t);
      TORCH_CHECK(shared_memory <= 32768, "Kernel too large for shared memory accumulation (", shared_memory," bytes). Use less bins or float precision");
      BilaterFilterCudaKernelLearnableSptBackward<scalar_t, C>
        <<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount),
           dim3(BLOCK_SIZE, 1), shared_memory>>>(gradientInputTensor.data_ptr<scalar_t>(),
                                      gradientOutputTensor.data_ptr<scalar_t>(),
                                      gradKernelTensor.data_ptr<scalar_t>(),
                                      inputTensor.data_ptr<scalar_t>(),
                                      outputTensor.data_ptr<scalar_t>(),
                                      outputWeightsTensor.data_ptr<scalar_t>(),
                                      dO_dx_ki.data_ptr<scalar_t>(),
                                      kernelTensor.data_ptr<scalar_t>());
    })
  );
  cuda_error_check("Cuda check after kernel call");
}

std::tuple<torch::Tensor, torch::Tensor> BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                                                     torch::Tensor inputTensor,
                                                                     torch::Tensor outputTensor,
                                                                     torch::Tensor outputWeightsTensor,
                                                                     torch::Tensor dO_dx_ki,
                                                                     torch::Tensor kernelTensor,
                                                                     float colorSigma) {
  
  CHECK_CONTIGUOUS_CUDA(gradientInputTensor);
  CHECK_CONTIGUOUS_CUDA(inputTensor);
  CHECK_CONTIGUOUS_CUDA(kernelTensor);

  torch::Tensor gradientOutputTensor = torch::zeros_like(gradientInputTensor);
  torch::Tensor gradKernelTensor = torch::zeros_like(kernelTensor);

  cuda_error_check("Beginning");

#define CASE(c, d)                                                                            \
  BilateralFilterCudaBackwardFunction<c, d>(gradientInputTensor, gradientOutputTensor,        \
                                            gradKernelTensor, inputTensor, outputTensor,      \
                                            outputWeightsTensor, dO_dx_ki, kernelTensor,      \
                                            colorSigma);
  SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION,
            gradientInputTensor.size(1), gradientInputTensor.dim() - 2);
#undef CASE

  return {gradientOutputTensor, gradKernelTensor};
}