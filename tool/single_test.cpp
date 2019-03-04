/* Example for testing single layer. */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include <mkl.h>
#include "dnn.hpp"

template <typename Dtype>
void print4d(Dtype *data, int N, int C, int H, int W){
    int n, c, h, w;
    int gap[3] = {W, H*W, C*H*W};
    int pos;

    for(n = 0; n < N; n++){
        for(c = 0; c < C; c++){
            printf("Index[N, C]=[%d, %d]>>>>>>>>\n", n, c);
            for(h = 0; h < H; h++){
                for(w = 0; w < W; w++){
                    pos = n*gap[2]+c*gap[1]+h*gap[0]+w;
                    printf("%g ", data[pos]);
                }
                printf("\n");
            }
        }
    }
}

int main(int argc, char *argv[]){
    srand((unsigned int)time(NULL));

    ACSACnnInitLib<float>(); 

    int N, C, H, W, K;
    int pad_h, pad_w;
    int outHeight, outWidth;

    int algo, bb, mg;

    float *in, *filter, *out;
    int inSize, filterSize, outSize;

    N = 64;
    C = 3;
    H = W = 40;
    K = 64;
    pad_h = 0;
    pad_w = 0;
    outHeight = H+2*pad_h-2;
    outWidth = W+2*pad_w-2;
    algo = 2;
    bb = 0;
    mg = 4;

    inSize = N*C*H*W;
    filterSize = K*C*3*3;
    outSize = N*K*outHeight*outWidth;
    in = (float *)mkl_malloc(inSize*sizeof(float), 64);
    filter = (float *)mkl_malloc(filterSize*sizeof(float), 64);
    out = (float *)mkl_malloc(outSize*sizeof(float), 64);

    for(int i = 0; i < inSize; i++)
        in[i] = rand()%3;
    for(int i = 0; i < filterSize; i++)
        filter[i] = rand()%2;
    memset(out, 0, outSize*sizeof(float));

    ACSATensor4d tensorIn, tensorFilter, tensorOut;
    ACSAConvMessage convMess;
    ACSAWinoMessage winoMess;

    ACSASetTensor4d(tensorIn, N, C, H, W);
    ACSASetTensor4d(tensorFilter, K, C, 3, 3);
    ACSASetTensor4d(tensorOut, N, K, outHeight, outWidth);
    ACSASetConvMessage(convMess, 3, 3, pad_h, pad_w, 1, 1);
    ACSASetWinoMessage(winoMess, ACSA_WINOGRAD_4X3, bb, mg);

    ACSAWinoConvolution_4x3<float>(in, filter, out,
            &tensorIn, &tensorFilter, &tensorOut, &convMess, &winoMess);

    printf(">>>>>>>>>> The data for input <<<<<<<<<<\n");
    print4d(in, N, C, H, W);
    printf(">>>>>>>>>> The data for filter <<<<<<<<<<\n");
    print4d(filter, K, C, 3, 3);
    printf(">>>>>>>>>> The data for output <<<<<<<<<<\n");
    print4d(out, N, K, outHeight, outWidth);

    mkl_free(in);
    mkl_free(filter);
    mkl_free(out);

    ACSACnnFreeLib<float>(); 

    return 0;
}
