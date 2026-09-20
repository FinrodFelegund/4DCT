#include <cuda.h>
#include <cuda_runtime.h>

#include "bilateral.h"
#include "../utils/lut_utils.cuh"
#include "../utils/cuda_error_check.h"
#include "../utils/meta_macros.h"

__constant__ int cBatchStride;
__constant__ int cColorStride;

__constant__ int cSizes[4];
__constant__ int cStrides[4];

__constant__ int cKernelSizes[4];
__constant__ int cHalfWindowSize_arr[4];

__constant__ int cLutBins;

template <typename scalar_t, int C>
__global__ void BilateralFilterCudaKernel4DLutForward(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ kernelTensor,
        const scalar_t* __restrict__ lutTensor,
        scalar_t* output,
        scalar_t* outputWeightsTensor,
        scalar_t* d0_dx_ki){
    
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

    const int bins = cLutBins;
    const scalar_t invDelta = (scalar_t)(bins - 1);

    const int homeT = homeOffset / cStrides[0];
    const int homeX = (homeOffset - homeT * cStrides[0]) / cStrides[1];
    const int homeY = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1]) / cStrides[2];
    const int homeZ =
        (homeOffset - homeT * cStrides[0] - homeX * cStrides[1] - homeY * cStrides[2]) / cStrides[3];
    const int homeIndex[] = {homeT, homeX, homeY, homeZ};

    // The home intensity is fixed for this thread: locate it once.
    // (C == 1 is enforced in Python; the channel loops below are kept for
    // structural parity with the original code.)
    const scalar_t homeValue = input[batchOffset + homeOffset];
    int homeBin;
    scalar_t homeFrac;
    bool homeInRange;
    lutLocate<scalar_t>(homeValue, bins, invDelta, homeBin, homeFrac, homeInRange);

    scalar_t valueSum = 0;    // V_m
    scalar_t weightSum = 0;   // W_m
    scalar_t dW_dxk = 0;      // dW_m / dx_m
    scalar_t dV_dxk = 0;      // dV_m / dx_m, WITHOUT the +w_mm term
    scalar_t centerWeight = 0;  // w_mm

    for (int kernelT = 0; kernelT < Wt; kernelT++) {
    int neighbourT = max(0, min(homeT + (kernelT - halfT), cSizes[0] - 1));

        for (int kernelX = 0; kernelX < Wx; kernelX++) {
            int neighbourX = max(0, min(homeX + (kernelX - halfX), cSizes[1] - 1));

            for (int kernelY = 0; kernelY < Wy; kernelY++) {
            int neighbourY = max(0, min(homeY + (kernelY - halfY), cSizes[2] - 1));

                for (int kernelZ = 0; kernelZ < Wz; kernelZ++) {
                    int neighbourZ = max(0, min(homeZ + (kernelZ - halfZ), cSizes[3] - 1));

                    int neighbourOffset = neighbourT * cStrides[0] + neighbourX * cStrides[1] + neighbourY * cStrides[2] + neighbourZ;

                    bool flagNotClamped = true;
                    int kernelIndex[] = {kernelT, kernelX, kernelY, kernelZ};
                    
                    int dimensions = 4;
                    for(int i = 0; i < dimensions; i++){
                        int HalfWindowSizeBack = cHalfWindowSize_arr[i];
                        int neighbourIndex = homeIndex[i] + kernelIndex[i] - HalfWindowSizeBack;
                        int neighbourIndexClamped = min(cSizes[i] - 1, max(0, neighbourIndex));
                        if(neighbourIndex != neighbourIndexClamped){
                            flagNotClamped = false;
                        }
                    }
                    if(!flagNotClamped){
                        continue;
                    }

                    const bool isCenter = (kernelT == halfT) && (kernelX == halfX) && (kernelY == halfY) && (kernelZ == halfZ);
                    const int kernelOffset = ((kernelT * Wx + kernelX) * Wy + kernelY) * Wz + kernelZ;
                    const scalar_t spatialWeight = kernelTensor[kernelOffset];
#pragma unroll
                    for(int c = 0; c < C; c++){
                        const scalar_t neighbourValue = input[batchOffset + neighbourOffset + c * cColorStride];
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
                        valueSum += neighbourValue * totalWeight;
                        weightSum += totalWeight;

                        scalar_t dF_dhome = dF_da;
                        if(isCenter){
                            dF_dhome += dF_db;
                            centerWeight = totalWeight;
                        }

                        const scalar_t dw_dxk = spatialWeight * dF_dhome;
                        dW_dxk += dw_dxk;
                        dV_dxk += dw_dxk * neighbourValue;
                    }
                }
            }
        }
    }

#pragma unroll
    for(int c = 0; c < C; c++){
        const scalar_t safeWeightSum = (weightSum == 0) ? (scalar_t)1e-12 : weightSum;
        const scalar_t out = valueSum / (safeWeightSum);

        output[batchOffset + homeOffset + c * cColorStride] = out;
        outputWeightsTensor[batchOffset + homeOffset + c * cColorStride] = safeWeightSum;
        d0_dx_ki[batchOffset + homeOffset + c * cColorStride] = (1 / safeWeightSum) * (dV_dxk + centerWeight - out * dW_dxk);
    }
}

template <int C, int D>
void BilateralFilterCudaForwardFunction(torch::Tensor inputTensor,
                                        torch::Tensor kernelTensor,
                                        torch::Tensor lutTensor,
                                        torch::Tensor outputTensor,
                                        torch::Tensor outputWeightsTensor,
                                        torch::Tensor dO_dx_ki){

    TensorDescription desc = TensorDescription(inputTensor);
    int kernelSizes[4];
    int halfWindowSize_arr[4];
    for(int i = 0; i < 4; i++){
        kernelSizes[i] = kernelTensor.size(i);
        halfWindowSize_arr[i] = kernelSizes[i] / 2;
    }

    int lutBins = (int)lutTensor.size(0);

    cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
    cudaMemcpyToSymbol(cColorStride, &desc.channelStride, sizeof(int));
    cudaMemcpyToSymbol(cSizes, desc.sizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cStrides, desc.strides, sizeof(int) * 4);
    cudaMemcpyToSymbol(cKernelSizes, kernelSizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cHalfWindowSize_arr, halfWindowSize_arr, sizeof(int) * 4);
    cudaMemcpyToSymbol(cLutBins, &lutBins, sizeof(int));

    cuda_error_check("Cuda check before kernel call.");

    const int blockSize = 128;

    AT_DISPATCH_FLOATING_TYPES(
        inputTensor.scalar_type(), "BilateralFilterCudaKernel4DLutForward", ([&] {
        BilateralFilterCudaKernel4DLutForward<scalar_t, C>
            <<<dim3(int(desc.channelStride / blockSize) + 1, desc.batchCount),
                dim3(blockSize, 1)>>>(inputTensor.data_ptr<scalar_t>(),
                                        kernelTensor.data_ptr<scalar_t>(),
                                        lutTensor.data_ptr<scalar_t>(),
                                        outputTensor.data_ptr<scalar_t>(),
                                        outputWeightsTensor.data_ptr<scalar_t>(),
                                        dO_dx_ki.data_ptr<scalar_t>());
        }));

    cuda_error_check("Cuda check after kernel call.");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>BilateralFilterCudaForward(torch::Tensor inputTensor,
                                                                                torch::Tensor kernelTensor,
                                                                                torch::Tensor lutTensor){

    CHECK_CONTIGUOUS_CUDA(inputTensor);
    CHECK_CONTIGUOUS_CUDA(kernelTensor);
    CHECK_CONTIGUOUS_CUDA(lutTensor);
    CHECK_SAME_DTYPE(inputTensor, kernelTensor);
    CHECK_SAME_DTYPE(inputTensor, lutTensor);
    TORCH_CHECK(inputTensor.dim() == 6, "inputTensor must be [B, C, T, X, Y, Z].");
    TORCH_CHECK(kernelTensor.dim() == 4, "kernelTensor must be [Wt, Wx, Wy, Wz].");
    TORCH_CHECK(lutTensor.dim() == 2 && lutTensor.size(0) == lutTensor.size(1),
                "lutTensor must be square [Bins, Bins].");
    TORCH_CHECK(lutTensor.size(0) >= 2, "lutTensor needs at least 2 bins.");
    for (int i = 0; i < 4; i++) {
        TORCH_CHECK(kernelTensor.size(i) % 2 == 1, "All kernel window sizes must be odd.");
    }

    torch::Tensor outputTensor = torch::zeros_like(inputTensor);
    torch::Tensor outputWeightsTensor = torch::zeros_like(inputTensor);
    torch::Tensor dO_dx_ki = torch::zeros_like(inputTensor);

    cuda_error_check("Beginning");

#define CASE(c, d)                                                                          \
    BilateralFilterCudaForwardFunction<c, d>(inputTensor, kernelTensor, lutTensor,          \
                                             outputTensor, outputWeightsTensor, dO_dx_ki);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION, inputTensor.size(1), inputTensor.dim() - 2);
#undef CASE

    return {outputTensor, outputWeightsTensor, dO_dx_ki};
}