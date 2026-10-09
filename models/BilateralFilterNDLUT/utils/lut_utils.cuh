#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ void lutLocate(scalar_t v,
                                          int bins,
                                          scalar_t invDelta,
                                          int& bin,
                                          scalar_t& frac,
                                          bool& inRange){

    /*
        v: intensity we want to look up
        bins: number of bins
        invDelta: spacing of the bins in intensity -> (bins - 1)
        bin: the bin to return
        frac: fraction of the way to bin + 1
        inRange: value was clamped to be inside [0, bins - 1)
    */

    scalar_t u = v * invDelta;
    inRange = (u >= (scalar_t)(0)) && (u <= (scalar_t)(bins - 1));
    u = min(max(u, (scalar_t)0), (scalar_t)(bins - 1));
    int b = (int)floor(u);
    if(b > bins - 2){
        b = bins - 2;
    }
    if(b < 0){
        b = 0;
    }
    bin = b;
    frac = u - (scalar_t)b;
}

template <typename scalar_t>
__device__ __forceinline__ void lutSample(const scalar_t* __restrict__ lut,
                                          int bins,
                                          scalar_t invDelta,
                                          int pa, scalar_t fa, bool aInRange,
                                          int qb, scalar_t fb, bool bInRange,
                                          scalar_t& F,
                                          scalar_t& dF_da,
                                          scalar_t& dF_db){

    /*
        lut: the lookup table
        bins: number of bins
        invDelta: spacing of the bins in intensity -> (bins - 1)
        px, fx, xInRange: bin, fraction and clamp flag for the respective argument
        F: interpolated value of the intensities
        dF_dX: partial derivative for the respective argument
    */
    const scalar_t f00 = lut[pa * bins + qb];
    const scalar_t f01 = lut[pa * bins + (qb + 1)];
    const scalar_t f10 = lut[(pa + 1) * bins + qb];
    const scalar_t f11 = lut[(pa + 1) * bins + (qb + 1)];

    const scalar_t oa = (scalar_t)1 - fa;
    const scalar_t ob = (scalar_t)1 - fb;

    F = oa * ob * f00 + oa * fb * f01 + fa * ob * f10 + fa * fb * f11;

    dF_da = aInRange ? invDelta * (ob * (f10 - f00) + fb * (f11 - f01)) : (scalar_t)0;
    dF_db = bInRange ? invDelta * (oa * (f01 - f00) + fa * (f11 - f10)) : (scalar_t)0;
}

template <typename scalar_t>
__device__ __forceinline__ void lutBilinearWeights(int bins,
                                                   int pa, scalar_t fa,
                                                   int qb, scalar_t fb,
                                                   int (&idx)[4],
                                                   scalar_t (&wgt)[4]){
    
    /*
        bins: number of bins
        pa, fa: bin and fraction of the first argument
        qb, fb: bin and fraction of the second argument
        idx: indexes in the lookup table, needed for backward
        wgt: weights of the respective entries in the lookup table, needed for backward
    */

    const scalar_t oa = (scalar_t)1 - fa;
    const scalar_t ob = (scalar_t)1 - fb;

    idx[0] = pa * bins + qb;
    idx[1] = pa * bins + (qb + 1);
    idx[2] = (pa + 1) * bins + qb;
    idx[3] = (pa + 1) * bins + (qb + 1);

    wgt[0] = oa * ob;
    wgt[1] = oa * fb;
    wgt[2] = fa * ob;
    wgt[3] = fa * fb;

}