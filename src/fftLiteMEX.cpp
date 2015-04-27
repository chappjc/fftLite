// fftLiteMEX.cpp - Test library (MATLAB MEX-file) for the fftLite class.
// by Jonathan Chappelow (chappjc)

#include "stdafx.h"
#include "mex.h"

#include <vector>

#include "fftLite.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if (nrhs != 1 || !mxIsDouble(prhs[0]))
        mexErrMsgTxt("Not double input.");

    const mwSize *dims = mxGetDimensions(prhs[0]);
    size_t ndims = mxGetNumberOfDimensions(prhs[0]);
    
    if (ndims > 2 || (dims[0]>1 && dims[1]>1))
        mexErrMsgTxt("Input must be a vector.");

    const size_t N = mxGetNumberOfElements(prhs[0]);

    double const *vec = mxGetPr(prhs[0]);

    if (nlhs > 3)
        mexErrMsgTxt("Must have two outputs for real and imaginary components.");

    plhs[0] = mxCreateNumericMatrix(dims[0], dims[1], mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(dims[0], dims[1], mxDOUBLE_CLASS, mxREAL);
    double *Fr = mxGetPr(plhs[0]);
    double *Fi = mxGetPr(plhs[1]);

    fftLite xform(N);
    
    int err;
    //for (int i = 0; i < 1; ++i) { // benchmarking
    //    memcpy(Fr, vec, N*sizeof(double));
    //    memset(Fi, 0, N*sizeof(double));
        err = xform.fft(Fr, Fi);
    //}

    // test: optionally compute the inverse of the transformed result ifft(fft(x))
    if (nlhs > 2)
        plhs[2] = mxCreateNumericMatrix(dims[0], dims[1], mxDOUBLE_CLASS, mxREAL);
    else
        return;

    double *out = mxGetPr(plhs[2]);
    memcpy(out, Fr, N*sizeof(double));

    double *tmp = new double[N];
    memcpy(tmp, Fi, N*sizeof(double));

    err = xform.ifft(out, tmp);

    for (int i = 0; i < N; ++i) {
        out[i] /= N;
    }
}

