/*

Disclaimer: Filter will not work for C > 1

*/

#include <cuda.h>
#include <cuda_runtime.h>

#include "bilateral.h"
#include "../utils/cuda_error_check.h"
#include "../utils/meta_macros.h"

#define MAX_WINDOW 64

__constant__ int cBatchStride;
__constant__ int cColorStride;

__constant__ int cSizes[4];
__constant__ int cStrides[4];

__constant__ int cKernelSizes[4];
__constant__ int cHalfWindowSize_arr[4];

__constant__ float cGaussianKernel[4][MAX_WINDOW];

__constant__ int cPadT;

template <typename scalar_t, int H, int L, int ACT>
__global__ void BilateralFilterCudaKernelForward(const scalar_t* __restrict__ inputTensor,
                                                 const scalar_t* __restrict__ weightsTensor,
                                                 scalar_t* outputTensor,
                                                 scalar_t* outputWeightsTensor){
    
    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStride;

    if(homeOffset >= cColorStride){
        return;
    }

    int homeT = homeOffset / cStrides[0];
    const int homeIdx = batchOffset + homeOffset;
    if(homeT < cPadT || homeT >= cSizes[0] - cPadT){
        outputTensor[homeIdx] = inputTensor[homeIdx];
        outputWeightsTensor[homeIdx] = (scalar_t)1;
        return;
    }

    const Embedding<scalar_t, H, L, ACT> network(weightsTensor);

    int homeX = (homeOffset - homeT * cStrides[0]) / cStrides[1];
    int homeY = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1]) / cStrides[2];
    int homeZ = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1] - homeY * cStrides[2]) / cStrides[3];

    const scalar_t homeValue = inputTensor[batchOffset + homeOffset];
    scalar_t homePart[H];
    embeddingHome<scalar_t, H, L, ACT>(network, homeValue, homePart); // this is always the same for each kernel tap, so compute it once to save some time
    scalar_t valueSum = 0, weightSum = 0;


    for(int kernelT = 0; kernelT < cKernelSizes[0]; kernelT++){
        const int neighbourT = homeT + kernelT - cHalfWindowSize_arr[0];
        if(neighbourT < 0 || neighbourT >= cSizes[0]){
            continue;
        }
        const scalar_t gaussianT = cGaussianKernel[0][kernelT];

        for(int kernelX = 0; kernelX < cKernelSizes[1]; kernelX++){
            const int neighbourX = homeX + kernelX - cHalfWindowSize_arr[1];
            if(neighbourX < 0 || neighbourX >= cSizes[1]){
                continue;
            }
            const scalar_t gaussianX = cGaussianKernel[1][kernelX];

            for(int kernelY = 0; kernelY < cKernelSizes[2]; kernelY++){
                const int neighbourY =homeY + kernelY - cHalfWindowSize_arr[2];
                if(neighbourY < 0 || neighbourY >= cSizes[2]){
                    continue;
                }
                const scalar_t gaussianY = cGaussianKernel[2][kernelY];

                for(int kernelZ = 0; kernelZ < cKernelSizes[3]; kernelZ++){
                    const int neighbourZ = homeZ + kernelZ - cHalfWindowSize_arr[3];
                    if(neighbourZ < 0 || neighbourZ >= cSizes[3]){
                        continue;
                    }
                    const scalar_t gaussianZ = cGaussianKernel[3][kernelZ];

                    int neighbourOffset = neighbourT * cStrides[0] + neighbourX * cStrides[1] + neighbourY * cStrides[2] + neighbourZ;
                    
            
                    const scalar_t neighbourValue = inputTensor[batchOffset + neighbourOffset];
                    const scalar_t colorWeight = embeddingForward<scalar_t, H, L, ACT>(network, homePart, neighbourValue);
                    
                    const scalar_t spatialWeight = gaussianT * gaussianX * gaussianY * gaussianZ;
                    
                    const scalar_t totalWeight = spatialWeight * colorWeight;

                    valueSum += neighbourValue * totalWeight;
                    weightSum += totalWeight;



                }
            }
        }
    }
    const scalar_t W = (weightSum == (scalar_t)0) ? (scalar_t)1e-12 : weightSum;
    const scalar_t out = valueSum / W;
    outputTensor[homeIdx] = out;
    outputWeightsTensor[homeIdx] = W;
}

template <int C, int D>
void BilateralFilterForwardFunction(torch::Tensor inputTensor,
                                    torch::Tensor outputTensor,
                                    torch::Tensor outputWeightsTensor,
                                    float sigma_t,
                                    float sigma_x,
                                    float sigma_y,
                                    float sigma_z,
                                    torch::Tensor weights,
                                    int hiddenWidth,
                                    int hiddenLength,
                                    int pad_t,
                                    int act){
    
    TensorDescription desc = TensorDescription(inputTensor);

    int windowSize[4];
    windowSize[0] = sigma_t ? 9 : 1;
    windowSize[1] = sigma_x ? std::max(((int)ceil(5.0f * sigma_x) | 1), 5) : 1;
    windowSize[2] = sigma_y ? std::max(((int)ceil(5.0f * sigma_y) | 1), 5) : 1;
    windowSize[3] = sigma_z ? std::max(((int)ceil(5.0f * sigma_z) | 1), 5) : 1;

    TORCH_CHECK(windowSize[1] <= MAX_WINDOW && windowSize[2] <= MAX_WINDOW && windowSize[3] <= MAX_WINDOW,
                "window size exceeds MAX_WINDOW (", MAX_WINDOW, ")");

    int halfWindowSizes[4];
    halfWindowSizes[0] = floor(0.5f * windowSize[0]);
    halfWindowSizes[1] = floor(0.5f * windowSize[1]);
    halfWindowSizes[2] = floor(0.5f * windowSize[2]);
    halfWindowSizes[3] = floor(0.5f * windowSize[3]);
    
    float spatialExpConstant[4];
    spatialExpConstant[0] = sigma_t ? -1.0 / (2 * sigma_t * sigma_t) : 1;
    spatialExpConstant[1] = sigma_x ? -1.0 / (2 * sigma_x * sigma_x) : 1;
    spatialExpConstant[2] = sigma_y ? -1.0 / (2 * sigma_y * sigma_y) : 1;
    spatialExpConstant[3] = sigma_z ? -1.0 / (2 * sigma_z * sigma_z) : 1;

    int* kernelSizes = new int[desc.dimensions];
    kernelSizes[0] = windowSize[0];
    kernelSizes[1] = windowSize[1];
    kernelSizes[2] = windowSize[2];
    kernelSizes[3] = windowSize[3];

    float gaussianKernel[4][MAX_WINDOW] = {};

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < windowSize[i]; j++){
            int distance = j - halfWindowSizes[i];
            gaussianKernel[i][j] = exp(distance * distance * spatialExpConstant[i]);
        }
    }


    cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
    cudaMemcpyToSymbol(cColorStride, &desc.channelStride, sizeof(int));
    cudaMemcpyToSymbol(cSizes, desc.sizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cStrides, desc.strides, sizeof(int) * 4);
    cudaMemcpyToSymbol(cKernelSizes, kernelSizes, sizeof(int) * desc.dimensions);
    cudaMemcpyToSymbol(cHalfWindowSize_arr, halfWindowSizes, sizeof(int) * desc.dimensions);
    cudaMemcpyToSymbol(cGaussianKernel, gaussianKernel, sizeof(gaussianKernel));
    cudaMemcpyToSymbol(cPadT, &pad_t, sizeof(int));

    cuda_error_check("Cuda check before kernel call");
    
    const int blockSize = 128;
    const dim3 grid((desc.channelStride + blockSize - 1) / blockSize, desc.batchCount);

    AT_DISPATCH_FLOATING_TYPES(
        inputTensor.scalar_type(), "BilateralFilterCudaKernelForward", ([&] {

#define LAUNCH(HW, HL, ACT)                                                             \
            case (HW) * 100 + (HL) + (ACT):                                                     \
                BilateralFilterCudaKernelForward<scalar_t, HW, HL, ACT><<<grid, blockSize>>>(    \
                    inputTensor.data_ptr<scalar_t>(),                                   \
                    weights.data_ptr<scalar_t>(),                                       \
                    outputTensor.data_ptr<scalar_t>(),                                  \
                    outputWeightsTensor.data_ptr<scalar_t>());                          \
                break;
            
            switch(hiddenWidth * 100 + hiddenLength + act){
                LAUNCH(8, 0, 0) LAUNCH(8, 1, 0) LAUNCH(8, 2, 0) LAUNCH(8, 3, 0)
                LAUNCH(16, 0, 0) LAUNCH(16, 1, 0) LAUNCH(16, 2, 0) LAUNCH(16, 3, 0)

                LAUNCH(8, 0, 1) LAUNCH(8, 1, 1) LAUNCH(8, 2, 1) LAUNCH(8, 3, 1)
                LAUNCH(16, 0, 1) LAUNCH(16, 1, 1) LAUNCH(16, 2, 1) LAUNCH(16, 3, 1)

                LAUNCH(8, 0, 2) LAUNCH(8, 1, 2) LAUNCH(8, 2, 2) LAUNCH(8, 3, 2)
                LAUNCH(16, 0, 2) LAUNCH(16, 1, 2) LAUNCH(16, 2, 2) LAUNCH(16, 3, 2)

                LAUNCH(8, 0, 3) LAUNCH(8, 1, 3) LAUNCH(8, 2, 3) LAUNCH(8, 3, 3)
                LAUNCH(16, 0, 3) LAUNCH(16, 1, 3) LAUNCH(16, 2, 3) LAUNCH(16, 3, 3)

                LAUNCH(8, 0, 4) LAUNCH(8, 1, 4) LAUNCH(8, 2, 4) LAUNCH(8, 3, 4)
                LAUNCH(16, 0, 4) LAUNCH(16, 1, 4) LAUNCH(16, 2, 4) LAUNCH(16, 3, 4)

                LAUNCH(8, 0, 5) LAUNCH(8, 1, 5) LAUNCH(8, 2, 5) LAUNCH(8, 3, 5)
                LAUNCH(16, 0, 5) LAUNCH(16, 1, 5) LAUNCH(16, 2, 5) LAUNCH(16, 3, 5)
                default: TORCH_CHECK(false, "unsupported network: width ", hiddenWidth, ", ", hiddenLength, " hidden H x H layers");
            }
#undef LAUNCH
        }));



    cuda_error_check("Cuda check after kernel call");

  delete[] kernelSizes;

}

std::tuple<torch::Tensor, torch::Tensor>BilateralFilterCudaForward(torch::Tensor inputTensor,
                                                                    float sigma_t,
                                                                    float sigma_x,
                                                                    float sigma_y,
                                                                    float sigma_z,
                                                                    torch::Tensor weights,
                                                                    int hiddenWidth,
                                                                    int hiddenLength,
                                                                    int pad_t,
                                                                    int act){


    CHECK_CONTIGUOUS_CUDA(inputTensor);
    CHECK_CONTIGUOUS_CUDA(weights);
    CHECK_SAME_DTYPE(inputTensor, weights);
    TORCH_CHECK(inputTensor.dim() == 6, "inputTensor must be [B, C, T, X, Y, Z].");
    TORCH_CHECK(inputTensor.size(1) == 1, "only one channel images are supported");
    TORCH_CHECK(hiddenLength >= 0, "hiddenLength must be >= 0");
    TORCH_CHECK(weights.dim() == 1 && weights.numel() == 4 * hiddenWidth + 1 + hiddenLength * (hiddenWidth * hiddenWidth + hiddenWidth),
                "weights must be flat network (W_in | b_in | W_hidden | b_hidden | W_out | b_out) for width ",
                hiddenWidth, " and ", hiddenLength, " hidden H * H layers, got ", weights.numel(), " values");

    torch::Tensor outputTensor = torch::zeros_like(inputTensor);
    torch::Tensor outputWeightsTensor = torch::zeros_like(inputTensor);

    cuda_error_check("beginning");

#define CASE(c, d)                                                                       \
    BilateralFilterForwardFunction<c, d>(inputTensor, outputTensor, outputWeightsTensor, \
                                         sigma_t, sigma_x, sigma_y, sigma_z, weights, hiddenWidth, hiddenLength, pad_t, act);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSIONS, inputTensor.size(1), inputTensor.dim() - 2);
#undef CASE

    return {outputTensor, outputWeightsTensor};
}