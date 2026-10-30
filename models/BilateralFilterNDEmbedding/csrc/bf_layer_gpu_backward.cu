#include <cuda.h>
#include <cuda_runtime.h>

#include "bilateral.h"
#include "../utils/cuda_error_check.h"
#include "../utils/meta_macros.h"

#define MAX_WINDOW 64

__constant__ int cBatchStrideBack;
__constant__ int cColorStrideBack;

__constant__ int cSizesBack[4];
__constant__ int cStridesBack[4];

__constant__ int cKernelSizesBack[4];
__constant__ int cHalfWindowSizesBack[4];

__constant__ float cSigmasBack[4];

__constant__ float cGaussianKernelBack[4][MAX_WINDOW];
__constant__ float cDistanceSquaredBack[4][MAX_WINDOW];

__constant__ int cPadTBack;

template <typename scalar_t, int H, int L, int ACT>
__global__ void BilateralFilterCudaKernelBackward(const scalar_t* __restrict__ gradientInputTensor,
                                                  const scalar_t* __restrict__ inputTensor,
                                                  const scalar_t* __restrict__ outputTensor,
                                                  const scalar_t* __restrict__ outputWeightsTensor,
                                                  const scalar_t* __restrict__ weightsTensor,
                                                  scalar_t* gradientOutputTensor,
                                                  scalar_t* dL_dsigma_t,
                                                  scalar_t* dL_dsigma_x,
                                                  scalar_t* dL_dsigma_y,
                                                  scalar_t* dL_dsigma_z,
                                                  scalar_t* dL_dweights){


    const int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    const int batchOffset = blockIdx.y * cBatchStrideBack;

    const bool active = homeOffset < cColorStrideBack;
  
    if(active){
        int homeT = homeOffset / cStridesBack[0];
        int homeX = (homeOffset - homeT * cStridesBack[0]) / cStridesBack[1];
        int homeY = (homeOffset - homeT * cStridesBack[1] - homeX * cStridesBack[1]) / cStridesBack[2];
        int homeZ = (homeOffset - homeT * cStridesBack[1] - homeX * cStridesBack[1] - homeY * cStridesBack[2]) / cStridesBack[3];
        int homeIndex[] = {homeT, homeX, homeY, homeZ};

        const bool homePadded = homeT < cPadTBack || homeT >= (cSizesBack[0] - cPadTBack);
        const Embedding<scalar_t, H, L, ACT> network(weightsTensor);

        const int home = homeOffset + batchOffset;
        const scalar_t homeValue = inputTensor[home];
        const scalar_t W_k = outputWeightsTensor[home];
        const scalar_t O_k = outputTensor[home];
        const scalar_t g_k = gradientInputTensor[home];

        const int halfT = cHalfWindowSizesBack[0], halfX = cHalfWindowSizesBack[1];
        const int halfY = cHalfWindowSizesBack[2], halfZ = cHalfWindowSizesBack[3];

        const bool learnT = cKernelSizesBack[0] > 0, learnX = cKernelSizesBack[1] > 0;
        const bool learnY = cKernelSizesBack[2] > 0, learnZ = cKernelSizesBack[3] > 0;

        const scalar_t invT3 = learnT ? (scalar_t)1 / ((scalar_t)cSigmasBack[0] * cSigmasBack[0] * cSigmasBack[0]);
        const scalar_t invX3 = learnX ? (scalar_t)1 / ((scalar_t)cSigmasBack[1] * cSigmasBack[1] * cSigmasBack[1]);
        const scalar_t invY3 = learnY ? (scalar_t)1 / ((scalar_t)cSigmasBack[2] * cSigmasBack[2] * cSigmasBack[2]);
        const scalar_t invZ3 = learnZ ? (scalar_t)1 / ((scalar_t)cSigmasBack[3] * cSigmasBack[3] * cSigmasBack[3]);

        scalar_t dw_dx_ki = 0;
        scalar_t tSum_w = 0, xSum_w = 0, ySum_w = 0, zSum_w = 0;
        scalar_t tSum_alpha = 0, xSum_alpha = 0, ySum_alpha = 0, zSum_alpha = 0;

        scalar_t homePart[H];
        embeddingHome<scalar_t, H, L, ACT>(network, homeValue, homePart);

        for(int kernelT = 0; kernelT < cKernelSizesBack[0]; kernelT++){
            const int neighbourT = homeT + kernelT - halfT;
            const bool neighbourTPadded = neighbourT < cPadTBack || neighbourT >= (cSizesBack[0] - cPadTBack);
            if(neighbourT < 0 || neighbourT >= cSizes[0] || homePadded || neighbourTPadded){
                continue;
            }
            const scalar_t gaussianT = cGaussianKernelBack[0][kernelT];

            for(int kernelX = 0; kernelX < cKernelSizesBack[1]; kernelX++){
                const int neighbourX = homeX + kernelX - halfX;
                if(neighbourX < 0 || neighbourX >= cSizesBack[1]){
                    continue;
                }
                const scalar_t gaussianX = cGaussianKernelBack[1][kernelX];

                for(int kernelY = 0; kernelY < cKernelSizesBack[2]; kernelY++){
                    const int neighbourY = homeY + kernelY - halfY;
                    if(neighbourY < 0 || neighbourY >= cSizesBack[2]){
                        continue;
                    }
                    const scalar_t gaussianY = cGaussianKernelBack[2][kernelY];

                    for(int kernelZ = 0; kernelZ < cKernelSizesBack[3]; kernelZ++){
                        const int neighbourZ = homeZ + kernelZ - halfZ;
                        if(neighbourZ < 0 || neighbourZ >= cSizesBack[3]){
                            continue;
                        }
                        const scalar_t gaussianZ = cGaussianKernelBack[3][kernelZ];

                        int neighbourOffset = neighbourT * cStridesBack[0] + neighbourX * cStridesBack[1] + \
                                              neighbourY * cStridesBack[2] + neighbourZ;
                                    
                        const bool isCenter = (kernelT == halfT) && (kernelX == halfX) && (kernelY == halfY) && (kernelZ == halfZ);
                        const scalar_t neighbour = neighbourOffset + batchOffset;
                        const scalar_t neighbourValue = inputTensor[neighbour];
                        const scalar_t W_i = outputWeightsTensor[neighbour];
                        const scalar_t O_i = outputTensor[neighbour];
                        const scalar_t g_i = gradientInputTensor[neighbour];

                        //dL / dsigma
                        const scalar_t wt = W_i * cDistanceSquaredBack[0][kernelT] * invT3;
                        const scalar_t wx = W_i * cDistanceSquaredBack[1][kernelX] * invY3;
                        const scalar_t wy = W_i * cDistanceSquaredBack[2][kernelY] * invX3;
                        const scalar_t wz = W_i * cDistanceSquaredBack[3][kernelZ] * invZ3;

                        tSum_w += wt, tSum_alpha += wt * neighbourValue;
                        xSum_w += wx, xSum_alpha += wx * neighbourValue;
                        ySum_w += wy, ySum_alpha += wy * neighbourValue;
                        zSum_w += wz, zSum_alpha += wz * neighbourValue;

                        //dL / d(weights and biases) of the embedding
                        scalar_t layerRegister[L + 1][H];
                        scalar_t colorWeight = embeddingForwardRegistered<scalar_t, H, L, ACT>(network, homePart, neighbourValue, layerRegister);
                        scalar_t spatialWeight = gaussianT * gaussianX * gaussianY * gaussianZ;
                        //dY_k/dz_out
                        scalar_t dcolorWeight = (1 / W_k) * (neighbourValue - O_k) * spatialWeight * g_k;
                        embeddingBackward<scalar_t, H, L, ACT>(network, dcolorWeight, layerRegister, dL_dweights);


                    }
                }
            }
        }
        dL_dsigma_t[home] = (1 / W) * (tSum_alpha - O_k * tSum_w) * g_k;
        dL_dsigma_x[home] = (1 / W) * (xSum_alpha - O_k * xSum_w) * g_k;
        dL_dsigma_y[home] = (1 / W) * (ySum_alpha - O_k * ySum_w) * g_k;
        dL_dsigma_z[home] = (1 / W) * (zSum_alpha - O_k * zSum_w) * g_k; 
    }

}

template <int C, int D>
void BilateralFilterCudaBackwardFunction(torch::Tensor gradientInputTensor, torch::Tensor inputTensor,
                                         torch::Tensor outputTensor, torch::Tensor outputWeightsTensor,
                                         float sigma_t, float sigma_x, float sigma_y, float sigma_z,
                                         torch::Tensor weights, int hiddenWidth, int hiddenLength, int pad_t, int act,
                                         torch::Tensor gradientOutputTensor, torch::Tensor dL_dsig_t, torch::Tensor dL_dsig_x,
                                         torch::Tensor dL_dsig_y, torch::Tensor dL_dsig_z, torch::Tensor dL_dweights
                                        ){
                                        
    TensorDescription desc = TensorDescription(inputTensor);

    int windowSizes[4];
    windowSizes[0] = sigma_t ? 9 : 1;
    windowSizes[1] = sigma_x ? std::max(((int)ceil(5.0f * sigma_x) | 1), 5) : 1;
    windowSizes[2] = sigma_y ? std::max(((int)ceil(5.0f * sigma_y) | 1), 5) : 1;
    windowSizes[3] = sigma_z ? std::max(((int)ceil(5.0f * sigma_z) | 1), 5) : 1;

    TORCH_CHECK(windowSizes[1] <= MAX_WINDOW && windowSizes[2] <= MAX_WINDOW && windowSizes[3] <= MAX_WINDOW,
                "window size exceeds MAX_WINDOW (", MAX_WINDOW, ")");

    int halfWindowSizes[4];
    halfWindowSizes[0] = floor(0.5f * windowSizes[0]);
    halfWindowSizes[1] = floor(0.5f * windowSizes[1]);
    halfWindowSizes[2] = floor(0.5f * windowSizes[2]);
    halfWindowSizes[3] = floor(0.5f * windowSizes[3]);
    
    float spatialExpConstant[4];
    spatialExpConstant[0] = sigma_t ? -1.0 / (2 * sigma_t * sigma_t) : 1;
    spatialExpConstant[1] = sigma_x ? -1.0 / (2 * sigma_x * sigma_x) : 1;
    spatialExpConstant[2] = sigma_y ? -1.0 / (2 * sigma_y * sigma_y) : 1;
    spatialExpConstant[3] = sigma_z ? -1.0 / (2 * sigma_z * sigma_z) : 1;

    int kernelSizes[4];
    kernelSizes[0] = windowSizes[0];
    kernelSizes[1] = windowSizes[1];
    kernelSizes[2] = windowSizes[2];
    kernelSizes[3] = windowSizes[3];

    float gaussianKernel[4][MAX_WINDOW] = {};
    float distanceSquaredKernel[4][MAX_WINDOW] = {};

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < windowSize[i]; j++){
            int distance = j - halfWindowSizes[i];
            gaussianKernel[i][j] = exp(distance * distance * spatialExpConstant[i]);
            distanceSquaredKernel[i][j] = distance * distance;
        }
    }

    float sigmasBack[4] = {sigma_t, sigma_x, sigma_y, sigma_z};

    cudaMemcpyToSymbol(cBatchStrideBack, &desc.batchStride, sizeof(int));
    cudaMemcpyToSymbol(cColorStrideBack, &desc.channelStride, sizeof(int));
    cudaMemcpyToSymbol(cSizesBack, desc.sizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cStridesBack, desc.strides, sizeof(int) * 4);
    cudaMemcpyToSymbol(cKernelSizesBack, kernelSizes, sizeof(int) * desc.dimensions);
    cudaMemcpyToSymbol(cHalfWindowSizesBack, halfWindowSizes, sizeof(int) * desc.dimensions);
    cudaMemcpyToSymbol(cGaussianKernelBack, gaussianKernel, sizeof(gaussianKernel));
    cudaMemcpyToSymbol(cDistanceSquaredBack, distanceSquaredKernel, sizeof(distanceSquaredKernel));
    cudaMemcpyToSymbol(cSigmasBack, sigmasBack, sizeof(float) * 4);
    cudaMemcpyToSymbol(cPadTBack, &pad_t, sizeof(int));

    cuda_error_check("Cuda check before kernel call");

    const int blockSize = 128;
    const dim3 grid((desc.channelStride + blockSize - 1) / blockSize, desc.batchCount);

    AT_DISPATCH_FLOATING_TYPES(
        inputTensor.scalar_type(), "BilateralFilterCudaKernelBackward", ([&] {

#define LAUNCH(HW, HL, ACT)                                                                      \
            case (HW) * 100 + (HL) + (ACT):                                                         \
                BilateralFilterCudaKernelBackward<scalar_t, HW, HL, ACT><<<grid, blockSize>>>(   \
                    gradientInputTensor.data_ptr<scalar_t>(),                               \
                    inputTensor.data_ptr<scalar_t>(),                                       \
                    outputTensor.data_ptr<scalar_t>(),                                      \
                    outputWeightsTensor.data_ptr<scalar_t>(),                               \
                    weights.data_ptr<scalar_t>(),                                           \
                    gradientOutputTensor.data_ptr<scalar_t>(),                              \
                    dL_dsig_t.data_ptr<scalar_t>(),                                         \
                    dL_dsig_x.data_ptr<scalar_t>(),                                         \
                    dL_dsig_y.data_ptr<scalar_t>(),                                         \
                    dL_dsig_z.data_ptr<scalar_t>(),                                         \
                    dL_dweights.data_ptr<scalar_t>());                                      \
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
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                                                                                                                torch::Tensor inputTensor,
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
  

    CHECK_CONTIGUOUS_CUDA(gradientInputTensor);
    CHECK_CONTIGUOUS_CUDA(inputTensor);
    CHECK_CONTIGUOUS_CUDA(outputTensor);
    CHECK_CONTIGUOUS_CUDA(outputWeightsTensor);
    CHECK_CONTIGUOUS_CUDA(weights);
    CHECK_SAME_DTYPE(inputTensor, weights);
    TORCH_CHECK(inputTensor.dim() == 6, "inputTensor must be [B, C, T, X, Y, Z].");
    TORCH_CHECK(inputTensor.size(1) == 1, "only one channel images are supported");
    TORCH_CHECK(hiddenLength >= 0, "hiddenLength must be >= 0");
    TORCH_CHECK(weights.dim() == 1 && weights.numel() == 4 * hiddenWidth + 1 + hiddenLength * (hiddenWidth * hiddenWidth + hiddenWidth),
                "weights must be flat network (W_in | b_in | W_hidden | b_hidden | W_out | b_out) for width ",
                hiddenWidth, " and ", hiddenLength, " hidden H * H layers, got ", weights.numel(), " values");

    torch::Tensor gradientOutputTensor = torch::zeros_like(gradientInputTensor);
    torch::Tensor dL_dsig_t = torch::zeros_like(inputTensor);
    torch::Tensor dL_dsig_x = torch::zeros_like(inputTensor);
    torch::Tensor dL_dsig_y = torch::zeros_like(inputTensor);
    torch::Tensor dL_dsig_z = torch::zeros_like(inputTensor);
    torch::Tensor dL_dweights = torch::zeros_like(weights);
    cuda_error_check("beginning");

#define CASE(c, d)                                                                                                          \
    BilateralFilterCudaBackwardFunction<c, d>(gradientInputTensor, inputTensor, outputTensor, outputWeightsTensor,          \
                                          sigma_t, sigma_x, sigma_y, sigma_z, weights, hiddenWidth, hiddenLength, pad_t, act,    \
                                          gradientOutputTensor, dL_dsig_t, dL_dsig_x, dL_dsig_y, dL_dsig_z, dL_dweights);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSIONS, inputTensor.size(1), inputTensor.dim() - 2);
#undef CASE

    return {gradientOutputTensor, dL_dsig_t, dL_dsig_x, dL_dsig_y, dL_dsig_z, dL_dweights};
                                
}