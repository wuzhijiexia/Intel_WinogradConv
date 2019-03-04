/* Example for testing convolution performance.*/

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

#define CYCLE_NUM		100

#define F_2X3			2
#define F_3X3			3
#define F_4X3			4
#define F_6X3			6
#define F_HYBRID		0

int counter = 0;

/* Direct manual convolution. */
int myDirectConv(float *in, float *kn, float *out,
        const int N, const int C, const int H, const int W, const int K,
        const int ph, const int pw)
{
    int inpos, knpos, outpos;

    int dimIn[4]  = {N, C, H+2*ph, W+2*pw};
    int dimKn[4]  = {K, C, 3, 3};
    int dimOut[4] = {N, K, H+2*ph-2, W+2*pw-2};

    int ingap[3] = {dimIn[1]*dimIn[2]*dimIn[3], dimIn[2]*dimIn[3], dimIn[3]};
    int kngap[3] = {dimKn[1]*dimKn[2]*dimKn[3], dimKn[2]*dimKn[3], dimKn[3]};
    int outgap[3] = {dimOut[1]*dimOut[2]*dimOut[3], dimOut[2]*dimOut[3], dimOut[3]};

#pragma omp parallel for private(inpos, knpos, outpos)
    for(int inn = 0; inn < dimIn[0]; inn++)
        for(int knn = 0; knn < dimKn[0]; knn++)
            for(int inc = 0; inc < dimIn[1]; inc++){
                for(int outh = 0; outh < dimOut[2]; outh++)
                    for(int outw = 0; outw < dimOut[3]; outw++){
                        outpos = inn*outgap[0] + knn*outgap[1] + outh*outgap[2] + outw;
                        for(int knh = 0; knh < dimKn[2]; knh++)
                            for(int knw = 0; knw < dimKn[3]; knw++){
                                inpos = inn*ingap[0] + inc*ingap[1] + (outh+knh)*ingap[2] + (outw+knw);
                                //knpos = knn*kngap[0] + inc*kngap[1] + 8 - (knh*kngap[2] + knw);
                                knpos = knn*kngap[0] + inc*kngap[1] + knh*kngap[2] + knw;
                                out[outpos] += in[inpos] * kn[knpos];
                            }
                    }
            }

    return 0;
}

/* Verity */
int verity(int counter,
        float *v_in, float *v_filter, float *out,
        const int N, const int C, const int H, const int W, const int K,
        const int ph, const int pw)
{
    int accury = 1;
    int sizeO = (H+2*ph-2) * (W+2*pw-2);

    float* v_out = (float *)mkl_malloc(N*K*sizeO*sizeof(float), 64);
    memset(v_out, 0, N*K*sizeO*sizeof(float));
    myDirectConv(v_in, v_filter, v_out, N, C, H, W, K, ph, pw);

    printf("Conv[%2d]: Tensor(%-3d %-3d %-3d %-3d %-3d) Pad(%-2d %-2d) ##  ",
            counter, N, C, H, W, K, ph, pw);

    for(int i = 0; i < N*sizeO*K; i++){
        if(fabs((out[i] - v_out[i])/v_out[i]) > 1e-4){
            printf("Output Error!!! [Index=%d, data[input]=%g | data[verity]=%g]\n", i, out[i], v_out[i]);
            accury = 0;
            break; 
        }
    }

    if(accury)
        printf("Output  True!!!\n");

    mkl_free(v_out);

    return 0;
}

/* Winograd Covoluton. */
void winograd_conv(const int N, const int C, const int H, const int W, const int K,
        const int ph, const int pw,
        const int algo, const int bb, const int mg,
        long *total_flops, double *total_time, const int verify)
{
    const int outHeight = H + 2*ph - 2; 
    const int outWidth = W + 2*pw - 2;
    const int sizeI = H*W; 
    const int sizeF = 3*3; 
    const int sizeO = outHeight*outWidth; 

    /* Malloc the space for data. */
    float* in, *filter, *out; 
    in = (float *)mkl_malloc(N*C*sizeI*sizeof(float), 64);
    assert(in != NULL); 
    filter = (float *)mkl_malloc(K*C*sizeF*sizeof(float), 64);
    assert(filter != NULL); 
    out = (float *)mkl_malloc(N*K*sizeO*sizeof(float), 64);
    memset(out, 0, N*K*sizeO*sizeof(float));
    assert(out != NULL); 

    float *v_in;
    const int v_sizeI = (H+2*ph) * (W+2*pw);
    v_in = (float *)mkl_malloc(N*C*v_sizeI*sizeof(float), 64);
    assert(v_in != NULL);
    memset(v_in, 0, N*C*v_sizeI*sizeof(float));

    //initialize in in parallel
#pragma omp parallel for
    for(int i = 0; i < N*C; i++)
        for(int j = 0; j < sizeI; j++){
            int rows, cols, index;
            rows = j/W;
            cols = j%W;
            index = i*v_sizeI + (rows+ph)*(H+2*pw) + (cols+pw);
            v_in[index] = in[i*sizeI+j] = (float)(j%7) + 0.1;
        }

#pragma omp parallel for
    for(int i = 0; i < K*C*sizeF; i++)
        filter[i] = (float)(i%5); 

    double wino_timer, sgemm_time, stime, etime;
    long nflops;
    double gflops;

    ACSATensor4d tensorIn, tensorFilter, tensorOut;
    ACSAConvMessage convMess;
    ACSAWinoMessage winoMess;

    ACSASetTensor4d(tensorIn, N, C, H, W);
    ACSASetTensor4d(tensorFilter, K, C, 3, 3);
    ACSASetTensor4d(tensorOut, N, K, outHeight, outWidth);
    ACSASetConvMessage(convMess, 3, 3, ph, pw, 1, 1); 

    /* Preheat for winograd convoluton. */
    switch(algo)
    {
        case F_2X3:
            ACSASetWinoMessage(winoMess, ACSA_WINOGRAD_2X3, bb, mg);
            ACSAWinoConvolution_2x3<float>(in, filter, out,
                    &tensorIn, &tensorFilter, &tensorOut, &convMess, &winoMess);
            break;
        case F_3X3:
            ACSASetWinoMessage(winoMess, ACSA_WINOGRAD_3X3, bb, mg);
            ACSAWinoConvolution_3x3<float>(in, filter, out,
                    &tensorIn, &tensorFilter, &tensorOut, &convMess, &winoMess);
            break;
        case F_4X3:
            ACSASetWinoMessage(winoMess, ACSA_WINOGRAD_4X3, bb, mg);
            ACSAWinoConvolution_4x3<float>(in, filter, out,
                    &tensorIn, &tensorFilter, &tensorOut, &convMess, &winoMess);
            break;
        case F_6X3:
            ACSASetWinoMessage(winoMess, ACSA_WINOGRAD_6X3, bb, mg);
            ACSAWinoConvolution_6x3<float>(in, filter, out,
                    &tensorIn, &tensorFilter, &tensorOut, &convMess, &winoMess);
            break;
        default:
            printf("There is no specified algorithm for winograd!\n");
            break;
    }

    stime = dsecnd();
    for(int i = 0; i < CYCLE_NUM; i++){
        switch(algo)
        {
            case F_2X3:
                ACSAWinoConvolution_2x3<float>(in, filter, out,
                        &tensorIn, &tensorFilter, &tensorOut, &convMess, &winoMess);
                break;
            case F_3X3:
                ACSAWinoConvolution_3x3<float>(in, filter, out,
                        &tensorIn, &tensorFilter, &tensorOut, &convMess, &winoMess);
                break;
            case F_4X3:
                ACSAWinoConvolution_4x3<float>(in, filter, out,
                        &tensorIn, &tensorFilter, &tensorOut, &convMess, &winoMess);
                break;
            case F_6X3:
                ACSAWinoConvolution_6x3<float>(in, filter, out,
                        &tensorIn, &tensorFilter, &tensorOut, &convMess, &winoMess);
                break;
            default:
                printf("There is no specified algorithm for winograd!\n");
                break;
        }
    }
    etime = dsecnd();

    /* Compute time and GFLOPS for single layer and all network. */
    wino_timer = 1.0f*(etime - stime) / CYCLE_NUM; 
    nflops = N*K*C*(H+2*ph-2)*(W+2*pw-2)*3*3*2; 
    gflops = (double) nflops*1.0e-9/wino_timer; 
    *total_flops += nflops; 
    *total_time += wino_timer; 

    /* Test time or test data accuray. */
    if(verify){
        float *v_filter = filter;
        verity(counter, v_in, v_filter, out, N, C, H, W, K, ph, pw);
    }
    else{ 
        printf("CONV[%2d]: GFlops=%7.2f, time=%8.3f ms.\n", counter, gflops, wino_timer*1000); 
    }

    /* Counter for layer num */
    counter = (counter%16)+1;

    /* Free data memory space. */
    mkl_free(in); 
    mkl_free(filter); 
    mkl_free(out); 
    mkl_free(v_in);
}

int main(int argc, char** argv){
    if(argc < 3){
        printf("Enter batch_size verity/noverity!!!\n"); 
        exit(-1); 
    }

    int i, j; 
    int batch = atoi(argv[1]); 
    int verify = atoi(argv[2]); 

    /* VGG19 Conv Layer */
    const int layer_num = 16;
    const int HW = 288;
    const int C_arr[16] = {3, 64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512}; 
    const int K_arr[16] = {64, 64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512}; 
#if 0
    int padFlag = 1;
    const int H_arr[16] = {HW+0, HW+0, HW/2+0, HW/2+0, HW/4+0, HW/4+0, HW/4+0, HW/4+0, HW/8+0, HW/8+0, HW/8+0, HW/8+0, HW/16+0, HW/16+0, HW/16+0, HW/16+0}; 
    const int W_arr[16] = {HW+0, HW+0, HW/2+0, HW/2+0, HW/4+0, HW/4+0, HW/4+0, HW/4+0, HW/8+0, HW/8+0, HW/8+0, HW/8+0, HW/16+0, HW/16+0, HW/16+0, HW/16+0}; 
    const int pad_h_arr[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const int pad_w_arr[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#else
    int padFlag = 0;
    const int H_arr[16] = {HW+3, HW+3, HW/2+3, HW/2+3, HW/4+3, HW/4+3, HW/4+3, HW/4+3, HW/8+3, HW/8+3, HW/8+3, HW/8+3, HW/16+3, HW/16+3, HW/16+3, HW/16+3}; 
    const int W_arr[16] = {HW+3, HW+3, HW/2+3, HW/2+3, HW/4+3, HW/4+3, HW/4+3, HW/4+3, HW/8+3, HW/8+3, HW/8+3, HW/8+3, HW/16+3, HW/16+3, HW/16+3, HW/16+3}; 
    const int pad_h_arr[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int pad_w_arr[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif

    const int mg_arr[16] = {2, 1, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8};
    const int bb_arr[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int algo_arr[16]  = {4, 4, 4, 4, 3, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3};

    double total_time; 
    long total_flops;
    int N, C, H, W, K, ph, pw;
    int algo, bb, mg;

    /* Prepare the environment for Winograd.
     * 1. init
     * */ 
    ACSACnnInitLib<float>(); 

    /* Compute Convoltuion Using F(2,3) Winograd */
    total_time = 0.0f; 
    total_flops = 0;

    if(verify && padFlag)
        printf(">>>  Verify Winograd Conv Correctness [PAD]\n");
    else if(verify && !padFlag)
        printf(">>>  Verify Winograd Conv Correctness [NO PAD]\n");
    else if(!verify && padFlag)
        printf(">>>  Test Winograd CONV Performance [PAD]\n");
    else
        printf(">>>  Test Winograd CONV Performance [NO PAD]\n");
    //printf("Please choose again for verify/noverify and pad/nopad!\n");

    for(int t = 0; t < layer_num; t++){
        N = batch;
        C = C_arr[t];
        H = H_arr[t];
        W = W_arr[t];
        K = K_arr[t];
        ph = pad_h_arr[t];
        pw = pad_w_arr[t];

        algo = algo_arr[t];
        bb = bb_arr[t];
        mg = mg_arr[t];

#if 1
        /* Use the best merge value. */
        winograd_conv(N, C, H, W, K, ph, pw, 
                algo, bb, mg,
                &total_flops, &total_time, verify);
#else
        /* Use the assigned merge value. */
        winograd_conv(N, C, H, W, K, ph, pw, 
                atoi(argv[argc-2]), bb, atoi(argv[argc-1]),
                &total_flops, &total_time, verify);
#endif
    }
    if(!verify){
        printf("Winograd:GFLOPS=%.2f ||| Time=%.4f ms.\n",
                (double)total_flops*1.0e-9/total_time, total_time*1000); 
    }
    printf("\n *******************************************************************\n\n"); 

    ACSACnnFreeLib<float>(); 

    return 0; 
}
