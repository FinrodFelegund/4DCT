#include <cuda.h>
#include <cuda_runtime.h>

#include "bilateral.h"
#include "../utils/lut_utils.cuh"
#include "../utils/cuda_error_check.h"
#include "../utils/meta_macros.h"

__constant__ int cBatchStrideBack;
__constant__ int cColorStrideBack;

__constant__ int cSizesBack[4];
__constant__ int cStridesBack[4];

__constant__ int cKernelSizesBack[4];
__constant__ int cHalfWindowSize_arrBack[4];

__constant__ int cLutBinsBack;

template<typename scalar_t, int C>
__global__ void BilateralFilterCudaKernel4DLutBackward(const scalar_t* __restrict__ gradientInputTensor,
                                                     scalar_t* gradientOutputTensor,
                                                     scalar_t* gradKernelTensor,
                                                     scalar_t* gradLutTensor,
                                                     const scalar_t* __restrict__ inputTensor,
                                                     const scalar_t* __restrict__ outputTensor,
                                                     const scalar_t* __restrict__ outputWeightsTensor,
                                                     const scalar_t* __restrict__ d0_dx_ki,
                                                     const scalar_t* __restrict__ kernelTensor,
                                                     const scalar_t* __restrict__ lutTensor){

    const int Wt = cKernelSizesBack[0];
    const int Wx = cKernelSizesBack[1];
    const int Wy = cKernelSizesBack[2];
    const int Wz = cKernelSizesBack[3];    
    const int kernelSize = Wt * Wx * Wy * Wz;
    const int bins = cLutBinsBack;
    const scalar_t invDelta = (scalar_t)(bins - 1);

    extern __shared__ char smem_raw[];
    scalar_t* sGradLut = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* sGradKernel = sGradLut + bins * bins; //pointer to the first byte after the lut kernel


    for(int i = threadIdx.x; i < kernelSize; i += blockDim.x){
        sGradKernel[i] = (scalar_t)0;
    }
    __syncthreads();

    for(int i = threadIdx.x; i < bins * bins; i += blockDim.x){
        sGradLut[i] = (scalar_t)0;
    }
    __syncthreads();

    const int homeOffset = blockIdx.x * blockDim.x + threadIdx.x;
    const int batchOffset = blockIdx.y * cBatchStrideBack;

    const bool active = homeOffset < cColorStrideBack;

    scalar_t valueSum = 0;
    if(active){


        const int halfT = cHalfWindowSize_arrBack[0];
        const int halfX = cHalfWindowSize_arrBack[1];
        const int halfY = cHalfWindowSize_arrBack[2];
        const int halfZ = cHalfWindowSize_arrBack[3];

        const int homeT = homeOffset / cStridesBack[0];
        const int homeX = (homeOffset - homeT * cStridesBack[0]) / cStridesBack[1];
        const int homeY =
            (homeOffset - homeT * cStridesBack[0] - homeX * cStridesBack[1]) / cStridesBack[2];
        const int homeZ = (homeOffset - homeT * cStridesBack[0] - homeX * cStridesBack[1] -
                        homeY * cStridesBack[2]) /
                        cStridesBack[3];
        const int homeIndex[] = {homeT, homeX, homeY, homeZ};

        const scalar_t homeValue = inputTensor[homeOffset + batchOffset];
        int homeBin;
        scalar_t homeFrac;
        bool homeInRange;
        lutLocate<scalar_t>(homeValue, bins, invDelta, homeBin, homeFrac, homeInRange);

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
                        for(int i = 0; i < 4; i++){
                            int HalfWindowSize = cHalfWindowSize_arrBack[i];
                            int neighbourIndex = homeIndex[i] + kernelIndex[i] - HalfWindowSize;
                            int neighbourIndexClamped = min(cSizesBack[i] - 1, max(0, neighbourIndex));
                            if(neighbourIndex != neighbourIndexClamped){
                                flagNotClamped = false;
                            }
                        }

                        if(!flagNotClamped){
                            continue;
                        }

                        const bool isCenter = (kernelT == halfT) && (kernelX == halfX) && (kernelY == halfY) && (kernelZ == halfZ);
                        const int kernelOffset = ((kernelT * Wx + kernelX) * Wy + kernelY) * Wz + kernelZ;
                        const int mirroredKernelOffset = (((2 * halfT - kernelT) * Wx + (2 * halfX - kernelX)) * Wy + (2 * halfY - kernelY)) * Wz + (2 * halfZ - kernelZ);

    #pragma unroll
                        for(int c = 0; c < C; c++){
                            const scalar_t neighbourValue = inputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                            int nBin;
                            scalar_t nFrac;
                            bool nInRange;
                            lutLocate<scalar_t>(neighbourValue, bins, invDelta, nBin, nFrac, nInRange);

                            //F(x_home, x_neighbour)
                            scalar_t F_home, dFh_da, dFh_db;
                            lutSample<scalar_t>(lutTensor, bins, invDelta,
                                                homeBin, homeFrac, homeInRange,
                                                nBin, nFrac, nInRange,
                                                F_home, dFh_da, dFh_db);
                            
                            const scalar_t W_m = outputWeightsTensor[batchOffset + homeOffset + c * cColorStrideBack];
                            const scalar_t O_m = outputTensor[batchOffset + homeOffset + c * cColorStrideBack];
                            const scalar_t g_m = gradientInputTensor[batchOffset + homeOffset + c * cColorStrideBack];

                            // dL/dK_j
                            // dO_m/dK_j = (F(x_m, x_n) / W_m) * (x_n - O_m)
                            atomicAdd(&sGradKernel[kernelOffset], g_m * (F_home * ((neighbourValue - O_m) / W_m)));

                            //dL/dF_pq
                            //dO_m/dF_pq = (K_j * beta_pq / W_m) * (x_n - O_m)
                            const scalar_t spatialWeight = kernelTensor[kernelOffset];
                            int lutIdx[4];
                            scalar_t lutWeights[4];
                            lutBilinearWeights<scalar_t>(bins, homeBin, homeFrac, nBin, nFrac, lutIdx, lutWeights);
                            
                            const scalar_t lutScale = (g_m * (neighbourValue - O_m) / W_m) * spatialWeight;
    #pragma unroll
                            for(int i= 0; i < 4; i++){
                                atomicAdd(&sGradLut[lutIdx[i]], lutScale * lutWeights[i]);
                            }

                            //dL/dx_k
                            if(!isCenter){
                                scalar_t F_adj, dFa_da, dFa_db;
                                lutSample<scalar_t>(lutTensor, bins, invDelta,
                                                    nBin, nFrac, nInRange,
                                                    homeBin, homeFrac, homeInRange,
                                                    F_adj, dFa_da, dFa_db);

                                const scalar_t kernelMirrored = kernelTensor[mirroredKernelOffset];
                                const scalar_t w_ik = kernelMirrored * F_adj;
                                const scalar_t dw_ik_dxk = kernelMirrored * dFa_db;

                                const scalar_t W_i = outputWeightsTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                                const scalar_t O_i = outputTensor[batchOffset + neighbourOffset + c * cColorStrideBack];
                                const scalar_t filter_kernel_back = (1 / W_i) * (w_ik + dw_ik_dxk * (homeValue - O_i));

                                valueSum += gradientInputTensor[batchOffset + neighbourOffset + c * cColorStrideBack] * filter_kernel_back;
                            } else {
                                valueSum += gradientInputTensor[batchOffset + homeOffset + c * cColorStrideBack] * d0_dx_ki[batchOffset + homeOffset + c * cColorStrideBack];
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < bins * bins; i += blockDim.x){
        scalar_t v = sGradLut[i];
        if(v != (scalar_t)0){
            atomicAdd(&gradLutTensor[i], v);
        }
    }

    __syncthreads();
    for(int i = threadIdx.x; i < kernelSize; i += blockDim.x){
        scalar_t v = sGradKernel[i];
        if(v != (scalar_t)0){
            atomicAdd(&gradKernelTensor[i], v);
        }
    }

    if(active){

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
                                       torch::Tensor gradLutTensor,
                                       torch::Tensor inputTensor,
                                       torch::Tensor outputTensor,
                                       torch::Tensor outputWeightsTensor,
                                       torch::Tensor d0_dx_ki,
                                       torch::Tensor kernelTensor,
                                       torch::Tensor lutTensor){

    TensorDescription desc = TensorDescription(inputTensor);

    int kernelSizes[4];
    int halfWindowSize_arr[4];
    int kernelSize = 1;
    for(int i = 0; i < 4; i++){
        kernelSizes[i] = (int)kernelTensor.size(i);
        halfWindowSize_arr[i] = kernelSizes[i] / 2;
        kernelSize = kernelSize * kernelSizes[i];
    }

    int lutBins = (int)lutTensor.size(0);

    cudaMemcpyToSymbol(cBatchStrideBack, &desc.batchStride, sizeof(int));
    cudaMemcpyToSymbol(cColorStrideBack, &desc.channelStride, sizeof(int));
    cudaMemcpyToSymbol(cSizesBack, desc.sizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cStridesBack, desc.strides, sizeof(int) * 4);
    cudaMemcpyToSymbol(cKernelSizesBack, kernelSizes, sizeof(int) * 4);
    cudaMemcpyToSymbol(cHalfWindowSize_arrBack, halfWindowSize_arr, sizeof(int) * 4);
    cudaMemcpyToSymbol(cLutBinsBack, &lutBins, sizeof(int));

    cuda_error_check("Cuda check before kernel call");

    const int blockSize = 128;

    AT_DISPATCH_FLOATING_TYPES(
        inputTensor.scalar_type(), "BilateralFilterCudaKernel4DLutBackward", ([&] {
            size_t shared_memory = (size_t)(lutBins * lutBins + kernelSize) * sizeof(scalar_t);
            TORCH_CHECK(shared_memory <= 49152, "LUT too large for shared memory accumulation (", shared_memory," bytes). Use less bins or float precision");
            BilateralFilterCudaKernel4DLutBackward<scalar_t, C>
                <<<dim3(int(desc.channelStride / blockSize) + 1, desc.batchCount),
                   dim3(blockSize, 1), shared_memory>>>(gradientInputTensor.data_ptr<scalar_t>(),
                                                        gradientOutputTensor.data_ptr<scalar_t>(),
                                                        gradKernelTensor.data_ptr<scalar_t>(),
                                                        gradLutTensor.data_ptr<scalar_t>(),
                                                        inputTensor.data_ptr<scalar_t>(),
                                                        outputTensor.data_ptr<scalar_t>(),
                                                        outputWeightsTensor.data_ptr<scalar_t>(),
                                                        d0_dx_ki.data_ptr<scalar_t>(),
                                                        kernelTensor.data_ptr<scalar_t>(),
                                                        lutTensor.data_ptr<scalar_t>());
        })
    );

    cuda_error_check("Cuda check after kernel call");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>BilateralFilterCudaBackward(torch::Tensor gradientInputTensor,
                                                                                 torch::Tensor inputTensor,
                                                                                 torch::Tensor outputTensor,
                                                                                 torch::Tensor outputWeightsTensor,
                                                                                 torch::Tensor dO_dx_ki,
                                                                                 torch::Tensor kernelTensor,
                                                                                 torch::Tensor lutTensor){

    CHECK_CONTIGUOUS_CUDA(gradientInputTensor);
    CHECK_CONTIGUOUS_CUDA(inputTensor);
    CHECK_CONTIGUOUS_CUDA(kernelTensor);
    CHECK_CONTIGUOUS_CUDA(lutTensor);
    CHECK_CONTIGUOUS_CUDA(outputTensor);
    CHECK_CONTIGUOUS_CUDA(outputWeightsTensor);
    CHECK_CONTIGUOUS_CUDA(dO_dx_ki);

    torch::Tensor gradientOutputTensor = torch::zeros_like(gradientInputTensor);
    torch::Tensor gradKernelTensor = torch::zeros_like(kernelTensor);
    torch::Tensor gradLutTensor = torch::zeros_like(lutTensor);
    
    cuda_error_check("beginning");

#define CASE(c, d)                                                                      \
    BilateralFilterCudaBackwardFunction<c, d>(gradientInputTensor, gradientOutputTensor,  \
                                            gradKernelTensor, gradLutTensor,            \
                                            inputTensor, outputTensor,                  \
                                            outputWeightsTensor, dO_dx_ki,              \
                                            kernelTensor, lutTensor);
    SWITCH_AB(CASE, BF_CUDA_MAX_CHANNELS, BF_CUDA_MAX_SPATIAL_DIMENSION,
              gradientInputTensor.size(1), gradientInputTensor.dim() - 2);
#undef CASE

    return {gradientOutputTensor, gradKernelTensor, gradLutTensor};

}