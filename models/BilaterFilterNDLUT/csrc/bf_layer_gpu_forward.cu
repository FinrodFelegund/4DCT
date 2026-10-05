#include <cuda.h>
#include <cuda_runtime.h>

#include "bilateral.h"
#include "../utils/lut_utils.cuh"
#include "..utils/cuda_error_check.h"
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

__constant__ float cSigma_t;
__constant__ float cSigma_x;
__constant__ float cSigma_y;
__constant__ float cSigma_z;
__constant__ int cLutBins;


template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernelNDLutForward(
        const scalar_t* __restrict__ inputTensor,
        const scalar_t* __restrict__ lutTensor,
        scalar_t* outputTensor,
        scalar_t* outputWeightsTensor,
        scalar_t* dO_dx_ki,
        scalar_t* dO_dsig_t,
        scalar_t* dO_dsig_x,
        scalar_t* dO_dsig_y,
        scalar_t* dO_dsig_z){


    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStride;

    if(homeOffset >= cColorStride){
        return;
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
                                        float sigma_t,
                                        float sigma_x,
                                        float sigma_y,
                                        float sigma_z,
                                        torch::Tensor lutTensor){
                                        
    TensorDescription desc = TensorDescription(inputTensor);

    int windowSize_t = 9 if sigma_t else 1;
    int windowSize_x = std::max(((int)ceil(5.0f * sigma_x) | 1), 5) if sigma_x else 1;
    int windowSize_y = std::max(((int)ceil(5.0f * sigma_y) | 1), 5) if sigma_y else 1;
    int windowSize_z = std::max(((int)ceil(5.0f * sigma_z) | 1), 5) if sigma_z else 1;
    int halfWindowSize_t = floor(0.5f * windowSize_t);
    int halfWindowSize_x = floor(0.5f * windowSize_x);
    int halfWindowSize_y = floor(0.5f * windowSize_y);
    int halfWindowSize_z = floor(0.5f * windowSize_z);
    int halfWindowSize_arr[] = {halfWindowSize_x, halfWindowSize_y, halfWindowSize_z, halfWindowSize_t};
    float spatialExpConstant_t = -1.0 / ( 2 * sigma_t * sigma_t) if sigma_t else 1;
    float spatialExpConstant_x = -1.0 / ( 2 * sigma_x * sigma_x) if sigma_x else 1;
    float spatialExpConstant_y = -1.0 / ( 2 * sigma_y * sigma_y) if sigma_y else 1;
    float spatialExpConstant_z = -1.0 / ( 2 * sigma_z * sigma_z) if sigma_z else 1; 

    int* kernelSizes = new int[desc.dimensions]
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

    int lutBins = (int)lutTensor.size(0);

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
    cudaMemcpyToSymbol(cSigma_t, &sigma_t, sizeof(float));
    cudaMemcpyToSymbol(cSigma_x, &sigma_x, sizeof(float));
    cudaMemcpyToSymbol(cSigma_y, &sigma_y, sizeof(float));
    cudaMemcpyToSymbol(cSigma_z, &sigma_z, sizeof(float));
    cudaMemcpyToSymbol(cLutBins, &lutBins, sizeof(int));

    cuda_error_check("Cuda check before kernel call");

    const int blockSize = 128;
    AT_DISPATCH_FLOATING_TYPES(
        inputTensor.scalar_type(), "BilateralFilterCudaKernelNDLutForward", ([&] {
            BilateralFilterCudaKernelNDLutForward<scalar_t, C>
                <<<dim3(int(desc.channelStride / blockSize) + 1, desc.batchCount),
                dim3(blockSize, 1)>>>(inputTensor.data_ptr<scalar_t>(),
                                        lutTensor.data_ptr<scalar_t>(),
                                        outputTensor.data_ptr<scalar_t>(),
                                        outputWeightsTensor.data_ptr<scalar_t>(),
                                        dO_dx_ki.data_ptr<scalar_t>(),
                                        dO_dsig_t.data_ptr<scalar_t>(),
                                        dO_dsig_x.data_ptr<scalar_x>(),
                                        dO_dsig_y.data_ptr<scalar_y>(),
                                        dO_dsig_z.data_ptr<scalar_z>());
        })
    );
    cuda_error_check("Cuda check after kernel call")
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BilateralFilterCudaForward(torch::Tensor inputTensor,
                                                                                                                                               float sigma_x,
                                                                                                                                               float sigma_y,
                                                                                                                                               float sigma_z,
                                                                                                                                               float sigma_t,
                                                                                                                                               torch::Tensor lutTensor)
                                                                                                                                               {

    
    
    CHECK_CONTIGUOUS_CUDA(inputTensor);
    CHECK_CONTIGUOUS_CUDA(lutTensor);
    CHECK_SAME_DTYPE(inputTensor, lutTensor);
    TORCH_CHECK(inputTensor.dim() == 6, "inputTensor must be [B, C, T, H, W, D]");
    TORCH_CHECK(lutTensor.dim() == 2 && lutTensor.size(0) == lutTensor.size(1), "lutTensor must be square tensor [bins, bins]");
    TORCH_CHECK(lutTensor.size(0) >= 2, "lutTensor needs at least 2 bins");

                                                                                    

    torch::Tensor outputTensor = torch::zeros_like(inputTensor);
    torch::Tensor outputWeightsTensor = torch::zeros_like(inputTensor);
    torch::Tensor dO_dx_ki = torch::zeros_like(inputTensor);
    torch::Tensor dO_dsig_x = nullptr;
    torch::Tensor dO_dsig_y = nullptr;
    torch::Tensor dO_dsig_z = nullptr;
    torch::Tensor dO_dsig_t = nullptr;
    if(sigma_x){
        dO_dsig_x = torch::zeros_like(inputTensor);
    }
        if(sigma_y){
        dO_dsig_y = torch::zeros_like(inputTensor);
    }
        if(sigma_z){
        dO_dsig_z = torch::zeros_like(inputTensor);
    }
        if(sigma_t){
        dO_dsig_t = torch::zeros_like(inputTensor);
    }
    cuda_error_check("beginning");

#define CASE(c, d)
    BilateralFilterCudaForwardFunction<c, d>(inputTensor, outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_t, sigma_x, sigma_y, sigma_z, sigma_t, lutTensor);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, inputTensor.size(1), inputTensor.dim() - 2);
#undef CASE

    return {outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_x, dO_dsig_y, dO_dsig_z, dO_dsig_t};
}