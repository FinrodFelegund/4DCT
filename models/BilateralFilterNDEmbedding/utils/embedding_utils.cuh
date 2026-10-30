#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>


template <typename scalar_t>
__device__ __forceinline__ scalar_t linearActivation(scalar_t z){
    return z;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t linearActivationd(scalar_t h){
    return (scalar_t)1;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoidActivation(scalar_t z){
    return (scalar_t)1 / (1 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoidActivationd(scalar_t h){
    return h * (1 - h);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t tanhActivation(scalar_t z){
    return tanh(z);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t tanhActivationd(scalar_t h){
    return 1 - h * h;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t reluActivation(scalar_t z){
    return z > (scalar_t)0 ? z : (scalar_t)0; 
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t reluActivationd(scalar_t h){
    return h > (scalar_t)0 ? (scalar_t)1 : (scalar_t)0;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t leakyReluActivation(scalar_t z, const scalar_t negativeSlope){
    return z > (scalar_t)0 ? z : negativeSlope * z;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t leakyReluActivationd(scalar_t h, const scalar_t negativeSlope){
    return h > (scalar_t)0 ? (scalar_t)1 : negativeSlope;
}

template <typename scalar_t, int H, int L, int ACT>
struct Embedding {

    const scalar_t* weightsIn;
    const scalar_t* biasIn;
    const scalar_t* weightsHidden;
    const scalar_t* biasHidden;
    const scalar_t* weightsOut;
    const scalar_t* biasOut;



    __device__ Embedding(const scalar_t* weights){
        //do some pointer aithmetic such that each member pointer points to the start of their correpsonding memory block

        weightsIn = weights; weights += 2 * H;
        biasIn = weights; weights += H;

        weightsHidden = weights; weights += L * H * H;
        biasHidden = weights; weights += L * H;

        weightsOut = weights; weights += H;
        biasOut = weights;

    }

    __device__ __forceinline__ scalar_t activation(const scalar_t z) const {
        if constexpr (ACT == 0) return linearActivation(z);
        else if constexpr (ACT == 1) return sigmoidActivation(z);
        else if constexpr (ACT == 2) return tanhActivation(z);
        else if constexpr (ACT == 3) return reluActivation(z);
        else return leakyReluActivation(z, (scalar_t)0.2);
    }
    

    __device__ __forceinline__ scalar_t derivativeActivation(const scalar_t h) const {
        if constexpr (ACT == 0) return linearActivationd(h);
        else if constexpr (ACT == 1) return sigmoidActivationd(h);
        else if constexpr (ACT == 2) return tanhActivationd(h);
        else if constexpr (ACT == 3) return reluActivationd(h);
        else return leakyReluActivationd(h, (scalar_t)0.2);
    }
};



template <typename scalar_t, int H, int L, int ACT>
__device__ __forceinline__ void embeddingHome(const Embedding<scalar_t, H, L, ACT>& network, scalar_t xHome, scalar_t (&homePart)[H]){
#pragma unroll
    for(int i = 0; i < H; i++){
        homePart[i] = network.weightsIn[2 * i] * xHome + network.biasIn[i];
    }
}

template <typename scalar_t, int H, int L, int ACT>
__device__ __forceinline__ scalar_t embeddingForward(const Embedding<scalar_t, H, L, ACT>& network, const scalar_t (&homePart)[H], scalar_t xNeighbour){
    scalar_t h[H];

#pragma unroll
    for(int i = 0; i < H; i++){
        h[i] = network.activation(homePart[i] + network.weightsIn[2 * i + 1] * xNeighbour); //bias and weights for home voxel were already added
    }

#pragma unroll
    for(int l = 0; l < L; l++){
        const scalar_t* W = network.weightsHidden + l * H * H;
        const scalar_t* b = network.biasHidden + l * H;
        scalar_t next[H];
#pragma unroll
        for(int i = 0; i < H; i++){
            scalar_t z = b[i];
#pragma unroll
            for(int j = 0; j < H; j++){
                z += W[i * H + j] * h[j];
            }
            next[i] = network.activation(z);
        }
#pragma unroll
        for(int i = 0; i < H; i++){
            h[i] = next[i];
        }
    }
    scalar_t out = *(network.biasOut);
#pragma unroll
    for(int j = 0; j < H; j++){
        out += network.weightsOut[j] * h[j];
    }
    return out;
}

template <typename scalar_t, int H, int L, int ACT>
__device__ __forceinline__ scalar_t embeddingForwardRegistered(const Embedding<scalar_t, H, L, ACT>& network, const scalar_t (&homePart)[H], scalar_t xNeighbour, scalar_t (&layerRegister)[L + 1][H]){

#pragma unroll
    for(int i = 0; i < H; i++){
        layerRegister[0][i] = network.activation(homePart[i] + network.weightsIn[2 * i + 1] * xNeighbour);
    }

#pragma unroll
    for(int l = 1; l < L + 1; l++){
        const scalar_t* W = network.weightsHidden + (l - 1) * H * H;
        const scalar_t* b = network.biasHidden + (l - 1) * H;
#pragma unroll
        for(int i = 0; i < H; i++){
            scalar_t z = b[i];
#pragma unroll
            for(int j = 0; j < H; j++){
                z += W[i * H + j] * layerRegister[l - 1][j];
            }
            layerRegister[l][i] = network.activation(z);
        }
    }
    scalar_t out = *(network.biasOut);
#pragma unroll
    for(int j = 0; j < H; j++){
        out += network.weightsOut[j] * layerRegister[L][j];
    }
    return out;
}

//partial derivatives with respect to weights and biases, multiplied by netwok scaling factor
template <typename scalar_t, int H, int L, int ACT>
__device__ __forceinline__ void embeddingBackward(const Embedding<scalar_t, H, L, ACT>& network, const scalar_t (&input)[2], scalar_t (&inputGradient)[2], scalar_t outputGradient, scalar_t (&layerRegister)[L + 1][H], scalar_t *gradientRegister){

    inputGradient[0] = (scalar_t)0;
    inputGradient[1] = (scalar_t)0;

    //we already computed the derivative with respect to the network output
    scalar_t *weightGradients = gradientRegister + 2 * H + H + L * H * H + L * H; //we land at the memeory location of the output weights
    scalar_t *biasGradients = gradientRegister +   2 * H + H + L * H * H + L * H + H; //we land at the memory location of the output bias
    *biasGradients += outputGradient;
#pragma unroll
    for(int j = 0; j < H; j++){
        weightGradients[j] += outputGradient * layerRegister[L][j];
        layerRegister[L][j] = outputGradient * network.weightsOut[j] * network.derivativeActivation(layerRegister[L][j]);
    }

    if(L > 0){
        weightGradients = weightGradients - L * H - H * H; //we land at the memory location of the last hidden layer
        biasGradients = biasGradients - 2 * H; //we land at the memory locatioj of the last hidden bias
    }
#pragma unroll
    for(int l = L - 1; l >= 0; l--){
        scalar_t temp_layer[H] = { (scalar_t)0 };
        const scalar_t* W = network.weightsHidden + l * H * H; //memory location of the network hidden weights
        //const scalar_t* b = network.biasHidden + l * H; //memery location of the netowrk hidden biases
#pragma unroll
        for(int i = 0; i < H; i++){
            biasGradients[i] += layerRegister[l + 1][i]; //gradient with respect to the bias

#pragma unroll
            for(int j = 0; j < H; j++){
                weightGradients[i * H + j] += layerRegister[l][j] * layerRegister[l + 1][i]; //gradient with respect to weight w^l_ij
                temp_layer[j] += W[i * H + j] * layerRegister[l + 1][i];
            }
        }
#pragma unroll
        for(int i = 0; i < H; i++){
            layerRegister[l][i] = temp_layer[i] * network.derivativeActivation(layerRegister[l][i]);
        }
        if(l != 0){
            weightGradients = weightGradients - H * H;
            biasGradients = biasGradients - H;
        }
    }

    if(L > 0){
        weightGradients = weightGradients - 2 * H - H;
        biasGradients = biasGradients - L * H * H - H;
    } else {
        weightGradients = gradientRegister;
        biasGradients = gradientRegister + 2 * H;
    }


#pragma unroll
    for(int i = 0; i < H; i++){
        biasGradients[i] += layerRegister[0][i];

#pragma unroll
        for(int j = 0; j < 2; j++){
            weightGradients[2 * i + j] += input[j] * layerRegister[0][i];
            inputGradient[j] += network.weightsIn[2 * i + j] * layerRegister[0][i];
        }
    }
}

