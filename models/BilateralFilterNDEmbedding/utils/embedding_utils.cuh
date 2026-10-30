#pragma once

#include <cuda.h>
#include <cuda_runtime.h>



template <typename scalar_t>
__device__ __forceinline__ scalar_t linearActivation(scalar_t z){
    return z;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t linearActivationd(scalar_t z){
    return z;

}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoidActivation(scalar_t z){

}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoidActivationd(scalar_t z){
    
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t tanhActivation(scalar_t z){

}

template <typename scalar_t>
__device__ __forceinline__ scalar_t tanhActivationd(scalar_t z){
    
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t reluActivation(scalar_t z){

}

template <typename scalar_t>
__device__ __forceinline__ scalar_t reluActivationd(scalar_t z){

}

template <typename scalar_t>
__device__ __forceinline__ scalar_t leakyReluActivation(scalar_t z){

}

template <typename scalar_t>
__device__ __forceinline__ scalar_t leakyReluActivationd(scalar_t z){

}

template <typename scalar_t, int H, int L>
struct Embedding {

    const scalar_t* weightsIn;
    const scalar_t* biasIn;
    const scalar_t* weightsHidden;
    const scalar_t* biasHidden;
    const scalar_t* weightsOut;
    const scalar_t* biasOut;

    const char* activationFunction;


    __device__ Embedding(const scalar_t* weights, char* activationFunction){
        //do some pointer aithmetic such that each member pointer points to the start of their correpsonding memory block

        weightsIn = weights; weights += 2 * H;
        biasIn = weights; weights += H;

        weightsHidden = weights; weights += L * H * H;
        biasHidden = weights; weights += L * H;

        weightsOut = weights; weights += H;
        biasOut = weights;

        this->activationFunction = activationFunction;
    }

    __device__ scalar_t activation(scalar_t){
        switch(this->activationFunction){
            case "linear":

        }
    }

    __device__ scalar_t derivativeActivation(scalar_t){
        switch(this->activationFunction){
            case "linear":
        }
    }
};



template <typename scalar_t, int H, int L>
__device__ __forceinline__ void embeddingHome(const Embedding<scalar_t, H, L>& network, scalar_t xHome, scalar_t (&homePart)[H]){
#pragma unroll
    for(int i = 0; i < H; i++){
        homePart[i] = network.weightsIn[2 * i] * xHome + network.biasIn[i];
    }
}

template <typename scalar_t, int H, int L>
__device__ __forceinline__ scalar_t embeddingForward(const Embedding<scalar_t, H, L>& network, const scalar_t (&homePart)[H], scalar_t xNeighbour){
    scalar_t h[H];

#pragma unroll
    for(int i = 0; i < H; i++){
        h[i] = activation(homePart[i] + network.weightsIn[2 * i + 1] * xNeighbour); //bias and weights for home voxel were already added
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
            next[i] = activation(z);
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

template <typename scalar_t, int H, int L>
__device__ __forceinline__ scalar_t embeddingForwardRegistered(const Embedding<scalar_t, H, L>& network, const scalar_t (&homePart)[H], scalar_t xNeighbour, scalar_t (&layerRegister)[L + 1][H]){

#pragma unroll
    for(int i = 0; i < H; i++){
        layerRegister[0][i] = activation(homePart[i] + network.weightsIn[2 * i + 1] * xNeighbour);
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
            layerRegister[l][i] = activation(z);
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
template <typename scalar_t, int H, int L>
__device__ __forceinline__ scalar_t* embeddingBackward(const Embedding<scalar_t, H, L>& network, const scalar_t (&input)[2], scalar_t outputGradient, scalar_t (&layerRegister)[L + 1][H], scalar_t *gradientRegister){

    //we already computed the derivative with respect to the network output
    scalar_t *weightGradients = gradientRegister + 2 * H + H + L * H * H + L * H; //we land at the memeory location of the output weights
    scalar_t *biasGradients = gradientRegister + 2 * H + H + L * H * H + L * H + H; //we land at the memory location of the output bias
    *biasGradients += outputGradient;
#pragma unroll
    for(int j = 0; j < H; j++){
        weightGradients[j] += outputGradient * layerRegister[L][j];
        layerRegister[L][j] = outputGradient * network.weightsOut[j];
    }

    weightsGradients = weightGradients - L * H - H * H; //we land at the memory location of the last hidden layer
    biasGradients = biasGradients - 2 * H; //we land at the memory locatioj of the last hidden bias
#pragma unroll
    for(int l = L - 1; l >= 0; l--){
        scalar_t temp_layer[H] = { (scalar_t)0 };
        const scalar_t* W = network.weightsHidden + (l - 1) * H * H; //memory location of the network hidden weights
        const scalar_t* b = network.biasHidden + (l - 1) * H; //memery location of the netowrk hidden biases
#pragma unroll
        for(int i = 0; i < H; i++){
            biasGradients[i] += layerRegister[l + 1][i]; //gradient with respect to the bias

#pragma unroll
            for(int j = 0; j < H; j++){
                weightGradients[i * H + j] = layerRegister[l][j] * layerRegister[l + 1][i]; //gradient with respect to weight w^l_ij
                temp_layer W[j] += W[i * H + j] * layerRegister[l + 1][i];
            }
        }

        for(int i = 0; i < H; i++){
            layerRegister[l][i] = temp_layer[i];
        }
        if(l != 0){
            weightGradients = weightGradients - H * H;
            biasGradients = biasGradients - H;
        }
    }

    weightGradients = weightGradients - 2 * H * H;
    biasGradients = biasGradients - L * H * H - H;

    scalar_t* voxel_gradient = new scalar_t[2](); //make sure to delete this array in the kernel!!!
#pragma unroll
    for(int i = 0; i < H; i++){
        biasGradients[i] += layerRegister[0][i];

#pragma unroll
        for(int j = 0; j < 2; j++){
            weightGradients[i * H + j] = input[j] * layerRegister[0][i];
            voxel_gradient[j] += network.weightsIn[i * H + j] * layerRegister[0][i];
        }
    }

    return voxel_gradient;

}

