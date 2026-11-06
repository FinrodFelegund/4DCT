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
__constant__ int cHalfWindowSizes[4];

__constant__ int cPadT;

template<typename scalar_t, int H, int L, int ACT>
__global__ void BilateralFilterCudakernelForward(const scalar_t* __restrict__ inputTensor,
                                                 scalar_t* outputTensor,
                                                 scalar_t* outputWeightsTensor,
                                                 const scalar_t* __restrict__ kernel,
                                                 const scalar_t* __restrict__ embedding,
                                                 int nParamsKernel,
                                                 int nParamsEmbedding){
    
    extern __shared__ unsigned char smemRaw[];
    scalar_t* sParams = reinterpret_cast<scalar_t*>(smemRaw);
    scalar_t* sParamsKernel = sParams;
    scalar_t* sParamsEmbedding = sParams + nParamsKernel;
    for(int i = threadidx.x; i < nParamsKernel; i += blockDim.x){
        sParamsKernel[i] = kernel[i];
    }
    for(int i = 0; i < nParamsEmbedding; i += blockDim.x){
        sParamsEmbedding[i] = embedding[i];
    }
    __syncthreads();

    int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int batchOffset = blockIdx.y * cBatchStride;

    if(homeOffset >= cColorStride){
        return;
    }

    const int homeIdx = batchOffset + homeOffset; 
    int homeT = homeOffset / cStrides[0];
    if(homeT < cPadT || homeT >= cSizes[0] - cPadT){
        outputTensor[homeIdx] = inputTensor[homeIdx];
        outputWeightsTensor[homeIdx] = (scalar_t)1;
        return;
    }

    int homeX = (homeOffset - homeT * cStrides[0]) / cStrides[1];
    int homeY = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1]) / cStrides[2];
    int homeZ = (homeOffset - homeT * cStrides[0] - homeX * cStrides[1] - homeY * cStrides[2]) / cStrides[3];

    const scalar_t homeValue = inputTensor[homeIdx];

    int kernelOffsets[4] = { (scalar_t) 0};
    kernelOffsets[0] = cKernelSizes[1] * cKernelSizes[2] * cKernelSizes[3];
    kernelOffsets[1] = ckernelSizes[2] * cKernelSizes[3];
    kernelOffsets[2] = cKernelSizes[3];
    kernelOffsets[3] = (scalar_t)1;

    scalar_t homepart[H];
    embeddingHome<scalar_t, H, L, ACT>(network, homeValue, homePart);
    scalar_t valueSum = 0, weightSum = 0;

    const Embedding<scalar_t, H, L, ACT> network(sParamsEmebdding);
    
    for(int kernelT = 0; kernelT < ckernelSizes[0]; kernelT++){
        const int neighbourT = homeT + kernelT - calfWindowSizes[0];
        if(neighbourT < 0 || neighbourT >= cSizes[0]){
            continue;
        }

        for(int kernelX = 0; kernelX < ckernelSizes[1]; kernelX++){
            const int neighbourX = homeX + kernelX - cHalfWindowSizes[1];
            if(neighbourX < 0 || neighbourX >= cSizes[1]){
                continue;
            }

            for(int kernelY = 0; kernelY < cHalfWindowSizes[2]; kernelY++){
                const int neighbourY = homeY + kernelY - cHalfWindowSizes[2];
                if(neighbourY < 0 || neighbourY >= cSizes[2]){
                    continue;
                }

                for(int kernelZ = 0; kernelZ < cKernelSizes[3]; kernelZ++){
                    const int neighbourZ = homeZ + kernelZ - cHalfWindowSizes[3];
                    if(neighbourZ < 0 || neighbourZ >= cSizes[3]){
                        continue;
                    }

                    const int neighbourOffset = neighbourT * cStrides[0] + neighbourX * cStrides[1] + neighbourY * cStrides[2] + neighbourZ;
                    const scalar_t neighbourValue = inputTensor[batchOffset + neighbourOffset];
                    const int kernelOffset = kernelT * kernelOffsets[0] + kernelX * kernelOffsets[1] + kernelY * kernelOffsets[2] + kernelZ * kernelOffset[3];

                    scalar_t spatialWeight = sParamsKernel[kernelOffset];
                    scalar_t colorWeight = embeddingForward<scalar_t, H, W, ACT>(network, homePart, homeValue, neighbourValue);
                    scalar_t totalWeight = spatialWeight * colorWeight;

                    valueSum += totalWeight * neighbourValue;
                    weightSum += totalWeight;
                }
            }
        }
    }
    const scalar_t safeWeightSum = (weightSum == (scalar_t)0) ? (scalar_t)1e-12 : weightSum;
    outputTensor[homeIdx] = valueSum / safeWeightSum;
    outputWeightsTensor[homeIdx] = safeWeightSum;


}

template<int C, int D>
void BilateralFilterForwardFunction(torch::Tensor inputTensor, torch::Tensor outputTensor, torch::Tensor outputWeightsTensor,
                                    torch::Tensor kernel, torch::Tensor embedding,
                                    int hiddenWidth, int hiddenLayers, int pad_t, int act){
    int x;
    TensorDescription desc = TensorDescription(inputTensor);
    int kernelSizes[4];
    int halfWindowSizes[4];
    for(int i = 0; i < 4; i++){
        kernelSizes[i] = kernel.size(i);
        halfWindowSizes[i] = kernel.size(i) / 2;
    }

    cudaMemcpyToSymbol(cBatchStride, &desc.batchStride, sizeof(int));
    cudaMemcpyToSymbol(cColorStride, &desc.channelStride, sizeof(int));
    cudaMemcpyToSymbol(cSizes, desc.sizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cStrides, desc.strides, sizeof(int) * 4);
    cudaMemcpyToSymbol(cKernelSizes, kernelSizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cHalfWindowSizes, halfWindowSizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cPadT, &pad_t, sizeof(int));

    cuda_error_check("Cuda check before kernel call");

    const int blockSize = 128;
    const dim3 grid((desc.channelStride + blockSize - 1) / blockSize, desc.batchCount);
    const int nParamsKernel= (int)kernel.numel();
    const int nParamsEmbedding = (int)embedding.numel();

#define LAUNCH(HW, HL, ACT)
    case (HW) * 100 + (HL) * 10 + (ACT):
        BilateralFilterCudaKernelEmbeddingForward<scalar_t, HW, HL, ACT><<<grid, blockSize, smem>>>(
            inputTensor.data_ptr<scalar_t>(),
            outputTensor.data_ptr<scalar_t>(),
            outputWeightsTensor.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            embedding.data_ptr<scalar_t>(),
            nParamsKernel,
            nParamsEmbedding);
        break;

    AT_DISPATCH_FLOATING_TYPES(
        inputTensor.scalar_type(), "BilateralFilterCudakernelForward", ([&] {
            const size_t smem = (size_t)(nParamsKernel + nParamsEmbedding) * sizeof(scalar_t);
            TORCH_CHECK(smem <= 49152, "kernel and network too large for shared memory (", smem, " bytes)");

            switch(hiddenWidth * 100 + hiddenLength * 10 + ACT){
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
        })
    );
#undef LAUNCH

    cuda_error_check("Cuda check after kernel call");

}


std::tuple<torch::Tensor, torch::Tensor>BilateralFilterCudaForward(torch::Tensor inputTensor,
                                                                   torch::Tensor kernel,
                                                                   torch::Tensor embedding,
                                                                   const int hiddenWidth,
                                                                   const int hiddenLayers,
                                                                   const int pad_t,
                                                                   const int act){


    CHECK_CONTIGUOUS_CUDA(inputTensor);
    CHECK_CONTIGUOUS_CUDA(kernel);
    CHECK_CONTIGUOUS_CUDA(embedding);
    CHECK_SAME_DTYPE(inputTensor, kernel);
    CHECK_SAME_DTYPE(inputTensor, embedding);
    TORCH_CHECK(inputTensor.dim() == 6, "inputTensor must be [B, C, T, X, Y, Z].");
    TORCH_CHECK(kernel.dim() == 4, "kernelTensor must be [Wt, Wx, Wy, Wz].");

    for (int i = 0; i < 4; i++) {
        TORCH_CHECK(kernel.size(i) % 2 == 1, "All kernel window sizes must be odd.");
    }
    TORCH_CHECK(hiddenLayers >= 0, "hiddenLength must be >= 0");
    TORCH_CHECK(embedding.dim() == 1 && embedding.numel() == 4 * hiddenWidth + hiddenWidth + hiddenLayers * (hiddenWidth * hiddenWidth + hiddenWidth) + hiddenWidth + 1,
                "weights must be flat network (W_in | b_in | W_hidden | b_hidden | W_out | b_out) for width ",
                hiddenWidth, " and ", hiddenLayers, " hidden H * H layers, got ", embedding.numel(), " values");

    torch::Tensor outputTensor = torch::zeros_like(inputTensor);
    torch::Tensor outputWeightsTensor = torch::zeros_like(inputTensor);

    cuda_error_check("beginning");

#define CASE(c, d)                                                                                          \
    BilateralFilterForwardFunction<c, d>(inputTensor, outputTensor, outputWeightsTensor,                    \
                                         kernel, embedding, hiddenWidth, hiddenLayers, pad_t, act);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSIONS, inputTensor.size(1), inputTensor.dim() - 2);
#undef CASE
    
    return { outputTensor, outputWeightsTensor };
}
                                                                  