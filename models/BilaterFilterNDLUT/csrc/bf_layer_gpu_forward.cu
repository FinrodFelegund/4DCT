/*

Disclaimer: This will not work if C != 1!!!

*/

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

    int homeT = homeOffset / cStrides[0];
    int homeX = (homeOffset - homeT * cStrides[0]) / cStrides[1];
    int homeY = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1]) / cStrides[2];
    int homeZ = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1] - homeY * cStrides[2]) / cStrides[3];
    int homeIndex[] = {homeT, homeX, homeY, homeZ};

    int bins = cLutBins;
    const scalar_t invDelta = (scalar_t)(bins - 1);


    const scalar_t homeValue = inputTensor[batchOffset + homeOffset];
    int homeBin;
    scalar_t homeFrac;
    bool homeInRange;
    lutLocate<scalar_t>(homeValue, bins, invDelta, homeBin, homeFrac, homeInRange);

    // Zero kernel aggregates.
    scalar_t valueSum = 0;
    scalar_t weightSum = 0;

    scalar_t dw_dx_ki = 0;
    scalar_t dfilter_dx_ki = 0;
    
    scalar_t tSum_w = 0;
    scalar_t tSum_alpha = 0;
    scalar_t xSum_w = 0;
    scalar_t xSum_alpha = 0;
    scalar_t ySum_w = 0;
    scalar_t ySum_alpha = 0;
    scalar_t zSum_w = 0;
    scalar_t zSum_alpha = 0;

    scalar_t centerWeight = 0;
    scalar_t dw_dx_ki = 0;
    scalar_t dfilter_dx_ki = 0;

    for(kernelT = 0; kernelT < cKernelSizes[0]; kernelT++){
        int neighbourT = max(0, min(homeT + (kernelT - cHalfWindowSize_arr[0]), cSizes[0] - 1));
        scalar_t gaussianT = cGaussianKernel_t[kernelT]; 
    

        for (int kernelX = 0; kernelX < cKernelSizes[1]; kernelX++) {
            int neighbourX = max(0, min(homeX + (kernelX - cHalfWindowSize_arr[1]), cSizes[1] - 1));
            scalar_t gaussianX = cGaussianKernel_x[kernelX];

            for (int kernelY = 0; kernelY < cKernelSizes[2]; kernelY++) {
            int neighbourY = max(0, min(homeY + (kernelY - cHalfWindowSize_arr[2]), cSizes[2] - 1));
            scalar_t gaussianY = cGaussianKernel_y[kernelY];

                for (int kernelZ = 0; kernelZ < cKernelSizes[3]; kernelZ++) {
                    int neighbourZ = max(0, min(homeZ + (kernelZ - cHalfWindowSize_arr[3]), cSizes[3] - 1));
                    scalar_t gaussianZ = cGaussianKernel_z[kernelZ];

                    int neighbourOffset = neighbourT * cStrides[0] + neighbourX * cStrides[1] + neighbourY * cStrides[2] + neighbourZ;

                    bool flagNotClamped = true;
                    int kernelIndex[] = {kernelT, kernelX, kernelY, kernelZ};
                    int dimensions = 4;  // Must equal the number of spatial dimensions.

                    for (int i = 0; i < dimensions; i++) {
                        int HalfWindowSizeBack = cHalfWindowSize_arr[i];  // Define constant memory as new variable here (!!), otherwise: cudaErrorMisalignedAddress
                        int neighbourIndex = homeIndex[i] + kernelIndex[i] - HalfWindowSizeBack;
                        int neighbourIndexClamped = min(cSizes[i] - 1, max(0, neighbourIndex));
                        if (neighbourIndex != neighbourIndexClamped){
                            flagNotClamped = false; 
                        }
                    }

                    if(!flagnotClmaped){
                        continue;
                    }

                    const bool isCenter = (kernelT == cHaldWindowSize_arr[0]) && (kernelX == cHaldWindowSize_arr[1]) && (kernelY == cHaldWindowSize_arr[2]) && (kernelZ == cHaldWindowSize_arr[3]);


                    scalar_t spatialWeight = gaussianT * gaussianX * gaussianY * gaussianZ; 

#pragma unroll
                    for(int c = 0; c < C; c++){
                        const scalar_t neighbourValue = inputTensor[batchOffset + neighbourOffset + c * cColorStride];
                        int nbBin;
                        scalar_t nbFrac;
                        bool nbInRange;
                        lutLocate<scalar_t>(neighbourValue, bins, invDelta, nbBin, nbFrac, nbInRange);

                        scalar_t F, dF_da, dF_db;
                        lutSample<scalar_t>(lutTensor, bins, invDelta,
                                            homeBin, homeFrac, homeInRange,
                                            nbBin, nbFrac, nbInRange,
                                            F, dF_da, dF_db);
                        const scalar_t totalWeight = spatialWeight * F;
                        valueSum += neighbourvalue * totalWeight;
                        weightSum += totalWeight;

                        scalar_t dF_dhome = dF_da;
                        if(isCenter){
                            dF_dhome += dF_db;
                            centerWeight = totalWeight;
                        }

                        tSum_w += (cKernelSizes[0] > 1) ? totalWeight * cTDistanceSquared[kernelT] / std::abs(cSigma_t * cSigma_t * cSigma_t) : (scalar_t)0;
                        tSum_alpha += (cKernelSizes[0] > 1) ? totalWeight * neighbourValue * cTDistanceSquared[kernelT] / std::abs(cSigma_t * cSigma_t * cSigma_t) : (scalar_t)0;
                    
                        xSum_w += (cKernelSizes[0] > 1) ? totalWeight * cXDistanceSquared[kernelX] / std::abs(cSigma_x * cSigma_x * cSigma_x) : (scalar_t)0;
                        xSum_alpha += (cKernelSizes[0] > 1) ? totalWeight * neighbourValue * cXDistanceSquared[kernelX] / std::abs(cSigma_x * cSigma_x * cSigma_x) : (scalar_t)0;

                        ySum_w += (cKernelSizes[0] > 1) ? totalWeight * cYDistanceSquared[kernelY] / std::abs(cSigma_y * cSigma_y * cSigma_y);
                        ySum_alpha += (cKernelSizes[0] > 1) ? totalWeight * neighbourValue * cYDistanceSquared[kernelY] / std::abs(cSigma_y * cSigma_y * cSigma_y) : (scalar_t)0;

                        zSum_w += (cKernelSizes[0] > 1) ? totalWeight * cZDistanceSquared[kernelZ] / std::abs(cSigma_z * cSigma_z * cSigma_z);
                        zSum_alpha += (cKernelSizes[0] > 1) ? totalWeight * neighbourValue * cZDistanceSquared[kernelZ] / std::abs(cSigma_z * cSigma_z * cSigma_z) : (scalar_t)0;

                        const scalar_t dw_dxk = spatialWeight * dF_dhome;
                        dw_dx_ki += dw_dxk;
                        dfilter_dx_ki += dw_dxk * neighbourValue;

                       

                    }
                }
            }
        }
    }

#pragma unroll
    for(int c = 0; c < C, c++){
        const scalar_t safeWeightSum = (weightSum == 0) ? (scalar_t)1e-12 : weightSum;
        const scalar_t out = valueSum / safeWeightSum;
        outputTensor[batchOffset + homeOffset + c * cColorStride] = out;
        outputWeightsTensor[batchOffset + homeOffset + c * cColorStride] = safeWeightSum;
        dO_dx_ki[batchOffset + homeOffset + c * cColorStride] = (dfilter_dx_ki + centerWeight - out * dw_dx_ki) / safeWeightSum;
        if(cKernelSizes[0] > 1){
            dO_dsig_t[batchOffset + homeOffset + c * cColorStride] = -(1 / safeWeightSum) * (valueSum / safeWeightSum) * tSum_w + (1 / safeWeightSum) * tSum_alpha; 
        }
        if(cKernelSizes[1] > 1){
            dO_dsig_x[batchOffset + homeOffset + c * cColorStride] = -(1 / safeWeightSum) * (valueSum / safeWeightSum) * xSum_w + (1 / safeWeightSum) * xSum_alpha;
        }
        if(cKernelSizes[2] > 1){
            dO_dsig_y[batchOffset + homeOffset + c * cColorStride] = -(1 / safeWeightSum) * (valueSum / safeWeightSum) * ySum_w + (1 / safeWeightSum) * ySum_alpha;
        }
        if(cKernelSizes[3] > 1){
            dO_dsig_z[batchOffset + homeOffset + c * cColorStride] = -(1 / safeWeightSum) * (valueSum / safeWeightSum) * zSum_w
        }

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
    int halfWindowSize_arr[] = {halfWindowSize_t, halfWindowSize_x, halfWindowSize_y, halfWindowSize_z};
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
                                                                                                                                                              float sigma_t,
                                                                                                                                                              float sigma_x,
                                                                                                                                                              float sigma_y,
                                                                                                                                                              float sigma_z,
                                                                                                                                                              torch::Tensor lutTensor){
                                                                                                                                               

    
    
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
    BilateralFilterCudaForwardFunction<c, d>(inputTensor, outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z, sigma_t, sigma_x, sigma_y, sigma_z, lutTensor);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, inputTensor.size(1), inputTensor.dim() - 2);
#undef CASE

    return {outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_t, dO_dsig_x, dO_dsig_y, dO_dsig_z};
}