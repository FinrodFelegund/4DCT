/*
Author: Fabian Wagner
Contact: fabian.wagner@fau.de
*/

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

__constant__ float cGaussianKernel_tBack[256];
__constant__ float cGaussianKernel_xBack[256];
__constant__ float cGaussianKernel_yBack[256];
__constant__ float cGaussianKernel_zBack[256];

__constant__ float cTDistanceSquaredBack[256];
__constant__ float cXDistanceSquaredBack[256];
__constant__ float cYDistanceSquaredBack[256];
__constant__ float cZDistanceSquaredBack[256];

__constant__ float cColorExponentConstantBack;

__constant__ float cSigma_tBack;
__constant__ float cSigma_xBack;
__constant__ float cSigma_yBack;
__constant__ float cSigma_zBack;
__constant__ float cColorSigma_Back;

template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernelNDBackward(const scalar_t* __restrict__ gradientInputTensor,
                                                    scalar_t* gradientOutputTensor,
                                                    const scalar_t* __restrict__ inputTensor,
                                                    const scalar_t* __restrict__ outputTensor,
                                                    const scalar_t* __restrict__ outputWeightsTensor,
                                                    const scalar_t* __restrict__ dO_dx_ki){

    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStrideBack;

    if(homeOffset >= cColorStrideBack){
        return;
    }

    int homeT = homeOffset / cStridesBack[0];
    int homeX = (homeOffset - homeT * cStridesBack[0]) / cStridesBack[1];
    int homeY = (homeOffset - homeT * cStridesBack[0] - homeX * cStridesBack[1]) / cStridesBack[2];
    int homeZ = (homeOffset - homeT * cStridesBack[0] - homeX * cStridesBack[1] - homeY * cStridesBack[2]) / cStridesBack[3];
    int homeIndex[] = {homeT, homeX, homeY, homeZ};
    scalar_t valueSum = 0;

    for(int kernelT = 0; kernelT < cKernelSizesBack[0]; kernelT++){
        int neighbourT = max(0, min(homeT + (kernelT - cHalfWindowSize_arrBack[0]), cSizesBack[0] - 1));
        scalar_t gaussianT = cGaussianKernel_tBack[kernelT];

        for(int kernelX = 0; kernelX < cKernelSizesBack[1]; kernelX++){
            int neighbourX = max(0, min(homeX + (kernelX - cHalfWindowSize_arrBack[1]), cSizesBack[1] - 1));
            scalar_t gaussianX = cGaussianKernel_xBack[kernelX];

            for(int kernelY = 0; kernelY < cKernelSizesBack[2]; kernelY++){
                int neighbourY = max(0, min(homeY + (kernelY - cHalfWindowSize_arrBack[2]), cSizesBack[2] - 1));
                scalar_t gaussianY = cGaussianKernel_yBack[kernelY];

                for(int kernelZ = 0; kernelZ < cKernelSizesBack[3]; kernelZ++){
                    int neighbourZ = max(0, min(homeZ + (kernelZ - cHalfWindowSize_arrBack[3]), cSizesBack[3]- 1));
                    scalar_t gaussianZ = cGaussianKernel_zBack[kernelZ];

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

                    const bool isCenter = (kernelT == cHalfWindowSize_arrBack[0]) && (kernelX == cHalfWindowSize_arrBack[1]) && (kernelY == cHalfWindowSize_arrBack[2]) && (kernelZ == cHalfWindowSize_arrBack[3]);
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

                    scalar_t spatialWeight = gaussianT * gaussianX * gaussianY * gaussianZ;
                    scalar_t colorWeight = exp(cColorExponentConstantBack * colorDistanceSquared);
                    scalar_t totalWeight = spatialWeight * colorWeight;
                    
                    if(flagNotClamped){
                 
#pragma unroll
                        for(int c = 0; c < C; c++){
                            if(!isCenter){
                                const scalar_t I_k = inputTensor[batchOffset + homeOffset + c * cColorStrideBack];
                                const scalar_t W_i = outputWeightsTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                                const scalar_t O_i = outputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                                const scalar_t g_i = gradientInputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                                valueSum += g_i * (((totalWeight * (1 + I_k * colorDistance / (cColorSigma_Back * cColorSigma_Back))) - (O_i * totalWeight * colorDistance / (cColorSigma_Back * cColorSigma_Back))) / W_i);
                            } else {
                                const scalar_t g_i = gradientInputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                                valueSum += g_i * dO_dx_ki[batchOffset + homeOffset + c * cColorStrideBack];
                            }
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for(int c = 0; c < C; c++){
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
                                         float sigma_x,
                                         float sigma_y,
                                         float sigma_z,
                                         float colorSigma){


    TensorDescription desc = TensorDescription(inputTensor);

    int windowSize_t = sigma_t ? 9 : 1;
    int windowSize_x = sigma_x ? std::max(((int)ceil(5.0f * sigma_x) | 1), 5) : 1;
    int windowSize_y = sigma_y ? std::max(((int)ceil(5.0f * sigma_y) | 1), 5) : 1;
    int windowSize_z = sigma_z ? std::max(((int)ceil(5.0f * sigma_z) | 1), 5) : 1;

    int halfWindowSize_t = floor(0.5f * windowSize_t);
    int halfWindowSize_x = floor(0.5f * windowSize_x);
    int halfWindowSize_y = floor(0.5f * windowSize_y);
    int halfWindowSize_z = floor(0.5f * windowSize_z);
    int halfWindowSize_arr[] = {halfWindowSize_t, halfWindowSize_x, halfWindowSize_y, halfWindowSize_z};

    float spatialExpConstant_t = sigma_t ?  -1.0f / (2 * sigma_t * sigma_t) : 1;
    float spatialExpConstant_x = sigma_x ?  -1.0f / (2 * sigma_x * sigma_x) : 1;
    float spatialExpConstant_y = sigma_y ?  -1.0f / (2 * sigma_y * sigma_y) : 1;
    float spatialExpConstant_z = sigma_z ?  -1.0f / (2 * sigma_z * sigma_z) : 1;

    float colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

    int* kernelSizes = new int[desc.dimensions];
    kernelSizes[0] = windowSize_t;
    kernelSizes[1] = windowSize_x;
    kernelSizes[2] = windowSize_y;
    kernelSizes[3] = windowSize_z;

    auto* gaussianKernel_t = new float[windowSize_t];
    auto* gaussianKernel_x = new float[windowSize_x];
    auto* gaussianKernel_y = new float[windowSize_y];
    auto* gaussianKernel_z = new float[windowSize_z];

    auto* tDistanceSquared = new float[windowSize_t];
    auto* xDistanceSquared = new float[windowSize_x];
    auto* yDistanceSquared = new float[windowSize_y];
    auto* zDistanceSquared = new float[windowSize_z];

    for(int i = 0; i < windowSize_t; i++){
        int distance = i - halfWindowSize_t;
        gaussianKernel_t[i] = exp(distance * distance * spatialExpConstant_t);
        tDistanceSquared[i] = distance * distance;
    }

    for(int i = 0; i < windowSize_x; i++){
        int distance = i - halfWindowSize_x;
        gaussianKernel_x[i] = exp(distance * distance * spatialExpConstant_x);
        xDistanceSquared[i] = distance * distance;
    }

    for(int i = 0; i < windowSize_y; i++){
        int distance = i - halfWindowSize_y;
        gaussianKernel_y[i] = exp(distance * distance * spatialExpConstant_y);
        yDistanceSquared[i] = distance * distance;
    }

    for(int i = 0; i < windowSize_z; i++){
        int distance = i - halfWindowSize_z;
        gaussianKernel_z[i] = exp(distance * distance * spatialExpConstant_z);
        zDistanceSquared[i] = distance * distance;
    }

  cudaMemcpyToSymbol(cBatchStrideBack, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cColorStrideBack, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSizesBack, desc.sizes, sizeof(int) * 4);
  cudaMemcpyToSymbol(cStridesBack, desc.strides, sizeof(int) * 4);
  cudaMemcpyToSymbol(cKernelSizesBack, kernelSizes, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cHalfWindowSize_arrBack, halfWindowSize_arr, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cGaussianKernel_tBack, gaussianKernel_t, sizeof(float) * windowSize_t);
  cudaMemcpyToSymbol(cGaussianKernel_xBack, gaussianKernel_x, sizeof(float) * windowSize_x);
  cudaMemcpyToSymbol(cGaussianKernel_yBack, gaussianKernel_y, sizeof(float) * windowSize_y);
  cudaMemcpyToSymbol(cGaussianKernel_zBack, gaussianKernel_z, sizeof(float) * windowSize_z);
  cudaMemcpyToSymbol(cTDistanceSquaredBack, tDistanceSquared, sizeof(float) * windowSize_t);
  cudaMemcpyToSymbol(cXDistanceSquaredBack, xDistanceSquared, sizeof(float) * windowSize_x);
  cudaMemcpyToSymbol(cYDistanceSquaredBack, yDistanceSquared, sizeof(float) * windowSize_y);
  cudaMemcpyToSymbol(cZDistanceSquaredBack, zDistanceSquared, sizeof(float) * windowSize_z);
  cudaMemcpyToSymbol(cColorExponentConstantBack, &colorExpConstant, sizeof(float));
  cudaMemcpyToSymbol(cSigma_tBack, &sigma_t, sizeof(float));
  cudaMemcpyToSymbol(cSigma_xBack, &sigma_x, sizeof(float));
  cudaMemcpyToSymbol(cSigma_yBack, &sigma_y, sizeof(float));
  cudaMemcpyToSymbol(cSigma_zBack, &sigma_z, sizeof(float));
  cudaMemcpyToSymbol(cColorSigma_Back, &colorSigma, sizeof(float));

  cuda_error_check("Cuda check before kernel call");

#define BLOCK_SIZE 128
  
  AT_DISPATCH_FLOATING_TYPES(
    inputTensor.scalar_type(), "BilateralFilterCudaKernelNDBackward", ([&] {
        BilateralFilterCudaKernelNDBackward<scalar_t, C>
            <<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount), dim3(BLOCK_SIZE, 1)>>>(
                gradientInputTensor.data_ptr<scalar_t>(),
                gradientOutputTensor.data_ptr<scalar_t>(),
                inputTensor.data_ptr<scalar_t>(),
                outputTensor.data_ptr<scalar_t>(),
                outputWeightsTensor.data_ptr<scalar_t>(),
                dO_dx_ki.data_ptr<scalar_t>());
    })
  );

  cuda_error_check("Cuda check after kernel call");
  delete[] kernelSizes;
  delete[] gaussianKernel_t;
  delete[] gaussianKernel_x;
  delete[] gaussianKernel_y;
  delete[] gaussianKernel_z;
  delete[] tDistanceSquared;
  delete[] xDistanceSquared;
  delete[] yDistanceSquared;
  delete[] zDistanceSquared;
}

torch::Tensor BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                          torch::Tensor inputTensor,
                                          torch::Tensor outputTensor,
                                          torch::Tensor outputWeightsTensor,
                                          torch::Tensor dO_dx_ki,
                                          float sigma_t,
                                          float sigma_x,
                                          float sigma_y,
                                          float sigma_z,
                                          float colorSigma){


    CHECK_CONTIGUOUS_CUDA(gradientInputTensor);
    CHECK_CONTIGUOUS_CUDA(inputTensor);
    CHECK_CONTIGUOUS_CUDA(outputTensor);
    CHECK_CONTIGUOUS_CUDA(outputWeightsTensor);
    CHECK_CONTIGUOUS_CUDA(dO_dx_ki);

    torch::Tensor gradientOutputTensor = torch::zeros_like(gradientInputTensor);
    cuda_error_check("beginning");

#define CASE(c, d) \
    BilateralFilterCudaBackwardFunction<c, d>(gradientInputTensor, gradientOutputTensor, inputTensor, outputTensor, outputWeightsTensor, dO_dx_ki, sigma_t, sigma_x, sigma_y, sigma_z, colorSigma);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, gradientInputTensor.size(1), gradientInputTensor.dim() - 2);
#undef CASE
    
    return gradientOutputTensor;
}