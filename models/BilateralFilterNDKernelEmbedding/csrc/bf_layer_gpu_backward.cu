#include <cuda.h>
#include <cuda_runtime.h>

#include "bilateral.h"
#include "../utils/embedding_utils.cuh"
#include "../utils/cuda_error_check.h"
#include "../utils/meta_macros.h"

__constant__ int cBatchStrideBack;
__constant__ int cColorStrideBack;

__constant__ int cSizesBack[4];
__constant__ int cStridesBack[4];

__constant__ int cKernelSizes[4];
__constant__ int cHalfWindowSizesBack[4];

__constant__ int cPadTBack;


template <int C, int D>
void BilateralFilterCudaBackwardFunction(torch::Tensor gradientInputTensor,
                                         torch::Tensor inputTensor,
                                         torch::Tensor outputTensor,
                                         torch::Tensor outputWeightsTensor,
                                         torch::Tensor kernel,
                                         torch::Tensor embedding,
                                         int hiddenWidth,
                                         int hiddenLayers,
                                         int pad_t,
                                         int act,
                                         bool computeInputGradient,
                                         torch::Tensor gradientOutputTensor,
                                         torch::Tensor gradientKernel,
                                         torch::Tensor gradientEmbedding){
    int x;
    TensorDescription desc = TensorDescription(inputTensor);

    int kernelSizes[4];
    int halfWindowSizes[4];
    for(int i = 0; i < 4; i++){
        kernelSizes[i] = kernel.size(i);
        halfWindowSizes[i] = kernel.size(i) / 2;
    }

    cudaMemcpyToSymbol(cBatchStrideBack, &desc.batchStride, sizeof(int));
    cudaMemcpyToSymbol(cColorStrideBack, &desc.channelStride, sizeof(int));
    cudaMemcpyToSymbol(cSizesBack, desc.sizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cStridesBack, desc.strides, sizeof(int) * 4);
    cudaMemcpyToSymbol(cKernelSizesBack, kernelSizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cHalfWindowSizesBack, halfWindowSizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cPadTBack, &pad_t, sizeof(int));

    cuda_error_check("Cuda check before kernel call");

    const int blockSize = 128;
    const dim3 grid(int(desc.channelStride + blockSize - 1) / blockSize, desc.batchCount);
    const int nParamsKernel = kernel.numel();
    const int nParamsEmbedding = embedding.numel();

#define LAUNCH(HW, HL, ACT)
    case (HW) * 100 + (HL) * 10 + ACT:
        BilateralFilterCudaKernelEmbeddingBackward<scalar_t, HW, HL, ACT><<<grid, blockSize, smem>>>(
            gradientInputTensor.data_ptr<scalar_t>(),
            inputTensor.data_ptr<scalar_t>(),
            outputTensor.data_ptr<scalar_t>(),
            outputWeightsTensor.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            embedding.data_ptr<scalar_t>(),
            nParamsKernel,
            nParamsEmbedding,
            computeInputGradient,
            gradientOutputTensor.data_ptr<scalar_t>(),
            gradientKernel.data_ptr<scalar_t>(),
            gradientEmbedding.data_ptr<scalar_t>()); 
        break;

    AT_DISPATCH_FLOATING_TYPES(
        inputTensor.scalar_type(), "BilateralFilterCudaKernelEmbeddingBackward", ([&] {
            const size_t smem = (size_t)2 * (nParamsKernel + nParamsEmbedding) * sizeof(scalar_t);
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                                                                    torch::Tensor inputTensor,
                                                                                    torch::Tensor outputTensor,
                                                                                    torch::Tensor outputWeightsTensor,
                                                                                    torch::Tensor kernel,
                                                                                    torch::Tensor embedding,
                                                                                    int hiddenWidth,
                                                                                    int hiddenLayers,
                                                                                    int pad_t,
                                                                                    int act,
                                                                                    bool computeInputGradient){

    CHECK_CONTIGUOUS_CUDA(gradientInputTensor);
    CHECK_CONTIGUOUS_CUDA(inputTensor);
    CHECK_CONTIGUOUS_CUDA(kernel);
    CHECK_CONTIGUOUS_CUDA(embedding);
    CHECK_CONTIGUOUS_CUDA(outputTensor);
    CHECK_CONTIGUOUS_CUDA(outputWeightsTensor);
                                                                                    
    torch::Tensor gradientOutputTensor = torch::zeros_like(gradientInputTensor);
    torch::Tensor gradientKernel = torch::zeros_like(kernel);
    torch::Tensor gradientEmbedding = torch::zeros_like(embedding);

    cuda_error_check("beginning");

#define CASE(c, d)                                                      \
    BilateralFilterCudaBackwardFunction<c, d>(gradientInputTensor,      \
                                              inputTensor,              \
                                              outputTensor,             \
                                              outputWeightsTensor,      \
                                              kernel,                   \
                                              embedding,                \
                                              hiddenWidth,              \
                                              hiddenLayers,             \
                                              pad_t,                    \
                                              act,                      \
                                              computeInputGradient,     \
                                              gradientOutputTensor,     \
                                              gradientKernel,           \
                                              gradientEmbedding);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSIONS, gradientInputTensor.size(1), gradientInputTensor.dim() - 2);
#undef CASE
    
    return { gradientInputTensor, gradientKernel, gradientEmbedding };
}