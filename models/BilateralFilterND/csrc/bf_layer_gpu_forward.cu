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

__constant__ float cGaussianKernel_t[256];
__constant__ float cGaussianKernel_x[256];
__constant__ float cGaussianKernel_y[256];
__constant__ float cGaussianKernel_z[256];

__constant__ float cTDistanceSquared[256];
__constant__ float cXDistanceSquared[256];
__constant__ float cYDistanceSquared[256];
__constant__ float cZDistanceSquared[256];

__constant__ float cColorExponentConstant;

__constant__ float cSigma_t;
__constant__ float cSigma_x;
__constant__ float cSigma_y;
__constant__ float cSigma_z;
__constant__ float cColorSigma;

template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernelNDForward(const scalar_t* __restrict__ inputTensor,
                                                   scalar_t* outputTensor,
                                                   scalar_t* outputWeightsTensor,
                                                   scalar_t* dO_dx_ki,
                                                   scalar_t* dO_dsig_t,
                                                   scalar_t* dO_dsig_x,
                                                   scalar_t* dO_dsig_y,
                                                   scalar_t* dO_dsig_z,
                                                   scalar_t* dO_dsig_r){
    
    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStride;

    if(homeOffset >= cColorStride){
        return;
    }

    int homeT = homeOffset / cStrides[0];
    int homeX = (homeOffset - homeT * cStrides[0]) / cStrides[1];
    int homeY = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1]) / cStrides[2];
    int homeZ = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1] - homeY * cStrides[2]) / cStrides[3];
    int homeIndex[] = {homeT, homeX, homeY, homeZ};

    scalar_t valueSum = 0, weightSum = 0;
    scalar_t dfilter_dx_ki = 0, dw_dx_ki = 0;
    scalar_t colorSum_w = 0, colorSum_alpha = 0;
    scalar_t tSum_w = 0, tSum_alpha = 0;
    scalar_t xSum_w = 0, xSum_alpha = 0;
    scalar_t ySum_w = 0, ySum_alpha = 0;
    scalar_t zSum_w = 0, zSum_alpha = 0;

    //axis wiith window 1 has no learnable width; skip its derivative (sigma may be 0 there)
    const bool learnT = cKernelSizes[0] > 1, learnX = cKernelSizes[1] > 1;
    const bool learnY = cKernelSizes[2] > 1, learnZ = cKernelSizes[3] > 1;

    //inv Terms will be zero, if width is not learnable, so it is safe to multiply it below
    const scalar_t invT3 = learnT ? (scalar_t)1 / ((scalar_t)cSigma_t * cSigma_t * cSigma_t) : 0;
    const scalar_t invX3 = learnX ? (scalar_t)1 / ((scalar_t)cSigma_x * cSigma_x * cSigma_x) : 0;
    const scalar_t invY3 = learnY ? (scalar_t)1 / ((scalar_t)cSigma_y * cSigma_y * cSigma_y) : 0;
    const scalar_t invZ3 = learnZ ? (scalar_t)1 / ((scalar_t)cSigma_z * cSigma_z * cSigma_z) : 0;

    for(int kernelT = 0; kernelT < cKernelSizes[0]; kernelT++){
        const int neighbourT = max(0, min(homeT + (kernelT - cHalfWindowSize_arr[0]), cSizes[0] - 1));
        const scalar_t gaussianT = cGaussianKernel_t[kernelT];

        for(int kernelX = 0; kernelX < cKernelSizes[1]; kernelX++){
            const int neighbourX = max(0, min(homeX + (kernelX - cHalfWindowSize_arr[1]), cSizes[1] - 1));
            const scalar_t gaussianX = cGaussianKernel_x[kernelX];

            for(int kernelY = 0; kernelY < cKernelSizes[2]; kernelY++){
                const int neighbourY = max(0, min(homeY + (kernelY - cHalfWindowSize_arr[2]), cSizes[2] - 1));
                const scalar_t gaussianY = cGaussianKernel_y[kernelY];

                for(int kernelZ = 0; kernelZ < cKernelSizes[3]; kernelZ++){
                    const int neighbourZ = max(0, min(homeZ + (kernelZ - cHalfWindowSize_arr[3]), cSizes[3] - 1));
                    const scalar_t gaussianZ = cGaussianKernel_z[kernelZ];

                    int neighbourOffset = neighbourT * cStrides[0] + neighbourX * cStrides[1] + neighbourY * cStrides[2] + neighbourZ;

                    bool flagNotClamped = true;
                    int kernelIndex[] = {kernelT, kernelX, kernelY, kernelZ};
                    int dimensions = 4;
                    for(int i = 0; i < dimensions; i++){
                        int HalfWindowSize = cHalfWindowSize_arr[i];
                        int neighbourIndex = homeIndex[i] + kernelIndex[i] - HalfWindowSize;
                        int neighbourIndexClamped = min(cSizes[i] - 1, max(0, neighbourIndex));
                        if(neighbourIndex != neighbourIndexClamped){
                            flagNotClamped = false;
                        }
                    }

                    scalar_t colorDistance = 0;
                    scalar_t colorDistanceSquared = 0;

#pragma unroll
                    for(int c = 0; c < C; c++){
                        scalar_t a = inputTensor[batchOffset + homeOffset + c * cColorStride];
                        scalar_t b = inputTensor[batchOffset + neighbourOffset + c * cColorStride];
                        scalar_t diff = a - b;
                        colorDistance += diff;
                        colorDistanceSquared += diff * diff;
                    }

                    scalar_t spatialWeight = gaussianT * gaussianX * gaussianY * gaussianZ;
                    scalar_t colorWeight = exp(cColorExponentConstant * colorDistanceSquared);
                    scalar_t totalWeight = spatialWeight * colorWeight;

                    if(flagNotClamped){

#pragma unroll
                        for(int c = 0; c < C; c++){
                            const scalar_t neighbourValue = inputTensor[batchOffset + neighbourOffset + c * cColorStride];

                            valueSum += neighbourValue * totalWeight;
                            weightSum += totalWeight;

                            dw_dx_ki += (-1) * totalWeight * colorDistance / (cColorSigma * cColorSigma);
                            dfilter_dx_ki += (-1) * totalWeight * neighbourValue * colorDistance / (cColorSigma * cColorSigma);

                            colorSum_w += totalWeight *  colorDistanceSquared / std::abs(cColorSigma * cColorSigma * cColorSigma);
                            colorSum_alpha += totalWeight * neighbourValue * colorDistanceSquared / std::abs(cColorSigma * cColorSigma * cColorSigma);

                            const scalar_t wt = totalWeight * cTDistanceSquared[kernelT] * invT3;
                            const scalar_t wx = totalWeight * cXDistanceSquared[kernelX] * invX3;
                            const scalar_t wy = totalWeight * cYDistanceSquared[kernelY] * invY3;
                            const scalar_t wz = totalWeight * cZDistanceSquared[kernelZ] * invZ3;

                            tSum_w += wt, tSum_alpha += wt * neighbourValue;
                            xSum_w += wx, xSum_alpha += wx * neighbourValue;
                            ySum_w += wy, ySum_alpha += wy * neighbourValue;
                            zSum_w += wz, zSum_alpha += wz * neighbourValue;
                        }                        
                    }
                }
            }
        }
    }

#pragma unroll
    for(int c = 0; c < C; c++){
        const scalar_t W = (weightSum == (scalar_t)0) ? (scalar_t)1e-12 : weightSum;
        const scalar_t out = valueSum / W;
        const int idx = batchOffset + homeOffset + c * cColorStride;

        outputTensor[idx] = out;
        outputWeightsTensor[idx] = W;
        dO_dx_ki[idx] = (dfilter_dx_ki + 1 - dw_dx_ki * out) / W;
        dO_dsig_r[idx] = (colorSum_alpha - colorSum_w * out) / W;
        dO_dsig_t[idx] = (tSum_alpha - out * tSum_w) / W;
        dO_dsig_x[idx] = (xSum_alpha - out * xSum_w) / W;
        dO_dsig_y[idx] = (ySum_alpha - out * ySum_w) / W;
        dO_dsig_z[idx] = (zSum_alpha - out * zSum_w) / W;
    }
}

template <int C, int D>
void BilateralFilterCudaForwardFunction(torch::Tensor inputTensor,
                                        torch::Tensor outputTensor,
                                        torch::Tensor outputWeightsTensor,
                                        torch::Tensor dO_dx_ki,
                                        torch::Tensor dO_dsig_t,
                                        torch::Tensor dO_dsig_x,
                                        torch::Tensor dO_dsig_y,
                                        torch::Tensor dO_dsig_z,
                                        torch::Tensor dO_dsig_r,
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
    float spatialExpConstant_t = sigma_t ? -1.0 / (2 * sigma_t * sigma_t) : 1;
    float spatialExpConstant_x = sigma_x ? -1.0 / (2 * sigma_x * sigma_x) : 1;
    float spatialExpConstant_y = sigma_y ? -1.0 / (2 * sigma_y * sigma_y) : 1;
    float spatialExpConstant_z = sigma_z ? -1.0 / (2 * sigma_z * sigma_z) : 1;
    float colorExpConstant = -1.0 / (2 * colorSigma * colorSigma);

    int* kernelSizes = new int[desc.dimensions];
    kernelSizes[0] = windowSize_t;
    kernelSizes[1] = windowSize_x;
    kernelSizes[2] = windowSize_y;
    kernelSizes[3] = windowSize_z;

    auto* gaussianKernel_t = new float[windowSize_t];
    auto* gaussianKernel_x = new float[windowSize_x];
    auto* gaussianKernel_y = new float[windowSize_y];
    auto* gaussianKernel_z = new float[windowSize_z];
    auto tDistanceSquared = new float[windowSize_t];
    auto xDistanceSquared = new float[windowSize_x];
    auto yDistanceSquared = new float[windowSize_y];
    auto zDistanceSquared = new float[windowSize_z];

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

  cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
  cudaMemcpyToSymbol(cColorStride, &desc.channelStride, sizeof(int));
  cudaMemcpyToSymbol(cSizes, desc.sizes, sizeof(int) * 4);
  cudaMemcpyToSymbol(cStrides, desc.strides, sizeof(int) * 4);
  cudaMemcpyToSymbol(cKernelSizes, kernelSizes, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cHalfWindowSize_arr, halfWindowSize_arr, sizeof(int) * desc.dimensions);
  cudaMemcpyToSymbol(cGaussianKernel_t, gaussianKernel_t, sizeof(float) * windowSize_t);
  cudaMemcpyToSymbol(cGaussianKernel_x, gaussianKernel_x, sizeof(float) * windowSize_x);
  cudaMemcpyToSymbol(cGaussianKernel_y, gaussianKernel_y, sizeof(float) * windowSize_y);
  cudaMemcpyToSymbol(cGaussianKernel_z, gaussianKernel_z, sizeof(float) * windowSize_z);
  cudaMemcpyToSymbol(cTDistanceSquared, tDistanceSquared, sizeof(float) * windowSize_t);
  cudaMemcpyToSymbol(cXDistanceSquared, xDistanceSquared, sizeof(float) * windowSize_x);
  cudaMemcpyToSymbol(cYDistanceSquared, yDistanceSquared, sizeof(float) * windowSize_y);
  cudaMemcpyToSymbol(cZDistanceSquared, zDistanceSquared, sizeof(float) * windowSize_z);
  cudaMemcpyToSymbol(cColorExponentConstant, &colorExpConstant, sizeof(float));
  cudaMemcpyToSymbol(cSigma_t, &sigma_t, sizeof(float));
  cudaMemcpyToSymbol(cSigma_x, &sigma_x, sizeof(float));
  cudaMemcpyToSymbol(cSigma_y, &sigma_y, sizeof(float));
  cudaMemcpyToSymbol(cSigma_z, &sigma_z, sizeof(float));
  cudaMemcpyToSymbol(cColorSigma, &colorSigma, sizeof(float));

  cuda_error_check("Cuda check before kernel call");

#define BLOCK_SIZE 128
  
  AT_DISPATCH_FLOATING_TYPES(
    inputTensor.scalar_type(), "BilateralFilterCudaKernelNDForward", ([&] {
        BilateralFilterCudaKernelNDForward<scalar_t, C>
        <<<dim3(int(desc.channelStride / BLOCK_SIZE) + 1, desc.batchCount), dim3(BLOCK_SIZE, 1)>>>(
            inputTensor.data_ptr<scalar_t>(),
            outputTensor.data_ptr<scalar_t>(),
            outputWeightsTensor.data_ptr<scalar_t>(),
            dO_dx_ki.data_ptr<scalar_t>(),
            dO_dsig_t.data_ptr<scalar_t>(),
            dO_dsig_x.data_ptr<scalar_t>(),
            dO_dsig_y.data_ptr<scalar_t>(),
            dO_dsig_z.data_ptr<scalar_t>(),
            dO_dsig_r.data_ptr<scalar_t>());
        
    })
  );

  cuda_error_check("Cuda check afterr kernel call");

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BilateralFilterCudaForward(torch::Tensor inputTensor,
                                                                                                                                               float sigma_t,
                                                                                                                                               float sigma_x,
                                                                                                                                               float sigma_y,
                                                                                                                                               float sigma_z,
                                                                                                                                               float color_sigma){

    CHECK_CONTIGUOUS_CUDA(inputTensor);
    TORCH_CHECK(inputTensor.dim() == 6, "inputTensor must be [B, C, T, H, W, D]");

    torch::Tensor outputTensor = torch::zeros_like(inputTensor);
    torch::Tensor outputWeightsTensor = torch::zeros_like(inputTensor);
    torch::Tensor dO_dx_ki = torch::zeros_like(inputTensor);
    torch::Tensor dO_dsig_t = torch::zeros_like(inputTensor);
    torch::Tensor dO_dsig_x = torch::zeros_like(inputTensor);
    torch::Tensor dO_dsig_y = torch::zeros_like(inputTensor);
    torch::Tensor dO_dsig_z = torch::zeros_like(inputTensor);
    torch::Tensor dO_dsig_r = torch::zeros_like(inputTensor);

    cuda_error_check("beginning");

#define CASE(c, d) \
    BilateralFilterCudaForwardFunction<c, d>(inputTensor, outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_r, sigma_t, sigma_x, sigma_y, sigma_z, color_sigma);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, inputTensor.size(1), inputTensor.dim() - 2);
#undef CASE
   
    return {outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_r};

}
