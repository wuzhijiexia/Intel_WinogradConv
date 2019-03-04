/* Please Note:
 * 1. The data is 4D.
 * 2. Conv Default:
 *      filter size is 3x3
 *      stride = 1, pad = 0/1
 * 3. Pool Default:
 *      kernel size is 2x2
 *      stride = 2, pad = 0
 **/

#ifndef _DNN_HPP_
#define _DNN_HPP_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <typeinfo>
#include <assert.h>
#include <omp.h>
#include <mkl.h>

#include "dnnDescriptor.hpp"

// The below parameters are required to generate the scratch pad memory
// It is required to reserve enough memory to store data for all the sizes you will be working on
// For example : by default, the below parameters are set for the VGG-16 Type D network 
#define MAX_BATCH           64
#define MAX_IMAGE_CHANNELS  64
#define MAX_IROWS           226
#define MAX_FILTER_CHANNELS 512
#define MAX_FILTERS         512

#define ACSA_MESSAGE(s) printf("%s : %s, %d >>> %s\n", \
        __FILE__, __FUNCTION__, __LINE__, s);

#define ACSA_CHECK(judge) do { \
    if(!judge){ \
        printf("%s : %s, %d\n", __FILE__, __FUNCTION__, __LINE__); \
        exit(0); \
    } \
}while(0)

/* ISTRIDE is the max batch*ntile*channel.
 * FSTRIDE is the max C*K for filter.
 * OSTRIDE is the max batch*ntile*nfilter.
 **/
const long MAX_TILES = (MAX_IROWS-2)*(MAX_IROWS-2)*0.25; 
#if 1
/* The max stride for bridge data. */
const long ISTRIDE = (MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13)*1.5; 
const long FSTRIDE = (MAX_FILTER_CHANNELS+1)*(MAX_FILTERS+1); 
const long OSTRIDE = ISTRIDE; 
#else
extern long ISTRIDE; 
extern long FSTRIDE;
extern long OSTRIDE;
#endif

/* Pointer for bridge data. */
extern void* winoIn;    
extern void* winoFilter;    
extern void* winoOut;    

#if 0
/* Decide wether to use batch block strategy. */
void decide_batch_block(const int lyn,
        const int N, const int *pC, const int *pH, const int *pW, const int * pK,
        int *pbb2x3, int *pbb3x3);
#endif

/* Kernel API for Convolution */
ACSAStatus ACSASetTensor4d(ACSATensor4d &tensor, int n, int c, int h, int w);
ACSAStatus ACSASetConvMessage(ACSAConvMessage &convMess,
        int kernel_h, int kernel_w,
        int pad_h, int pad_w, int stride_h, int stride_w);
ACSAStatus ACSASetPoolMessage(ACSAPoolMessage &poolMess,
        int kernel_h, int kernel_w,
        int pad_h, int pad_w, int stride_h, int stride_w);
ACSAStatus ACSASetWinoMessage(ACSAWinoMessage &winoMess,
        ACSAWinogradAlgo algo, int bb, int mg);

template<typename Dtype>
ACSAStatus ACSAWinoConvolutionFwd(const Dtype *in, const Dtype *filter, Dtype *out,
        ACSATensor4d* tensorIn, ACSATensor4d* tensorFilter, ACSATensor4d* tensorOut,
        ACSAConvMessage* convMess, ACSAWinoMessage *winoMess);
		  
template<typename Dtype>
ACSAStatus ACSAWinoConvolution_2x3(const Dtype *in, const Dtype *filter, Dtype *out,
        ACSATensor4d* tensorIn, ACSATensor4d* tensorFilter, ACSATensor4d* tensorOut,
        ACSAConvMessage* convMess, ACSAWinoMessage *winoMess);

template<typename Dtype>
ACSAStatus ACSAWinoConvolution_3x3(const Dtype *in, const Dtype *filter, Dtype *out,
        ACSATensor4d* tensorIn, ACSATensor4d* tensorFilter, ACSATensor4d* tensorOut,
        ACSAConvMessage* convMess, ACSAWinoMessage *winoMess);

template<typename Dtype>
ACSAStatus ACSAWinoConvolution_4x3(const Dtype *in, const Dtype *filter, Dtype *out,
        ACSATensor4d* tensorIn, ACSATensor4d* tensorFilter, ACSATensor4d* tensorOut,
        ACSAConvMessage* convMess, ACSAWinoMessage *winoMess);

template<typename Dtype>
ACSAStatus ACSAWinoConvolution_6x3(const Dtype *in, const Dtype *filter, Dtype *out,
        ACSATensor4d* tensorIn, ACSATensor4d* tensorFilter, ACSATensor4d* tensorOut,
        ACSAConvMessage* convMess, ACSAWinoMessage *winoMess);

/* kernel API for activation. */
template<typename Dtype>
ACSAStatus ACSAReLUInplaceFwd(Dtype *in_out, ACSATensor4d* tensor);
template<typename Dtype>
ACSAStatus ACSAReLUOutplaceFwd(const Dtype *in, Dtype *out, ACSATensor4d* tensor);

/* kernel API for pooling. */
template<typename Dtype>
ACSAStatus ACSAMaxPoolingFwd(const Dtype *in, Dtype *out, ACSATensor4d* tensor,
        ACSAPoolMessage* poolMess);
template<typename Dtype>
ACSAStatus ACSAAvePoolingFwd(const Dtype *in, Dtype *out, ACSATensor4d* tensor,
        ACSAPoolMessage* poolMess);

/* kernel API for other. */
template<typename Dtype>
ACSAStatus ACSAGetInputTile(Dtype *tmp, const Dtype *data,
        int r_tile, int c_tile, int r_init, int c_init, int c_num,
        int r_up, int r_down, int c_left, int c_right);
template<typename Dtype>
ACSAStatus ACSAGetFinalOutput(Dtype *data, const Dtype *tmp,
        int r_out, int c_out, int r_init, int c_init, int c_num,
        int r_up, int r_down, int c_left, int c_right);

#if 0
/* Compute stride for bridge data. */
inline ACSAStatus no4k_aligned(long *);
ACSAStatus compute_max_stride(const int, const int, const int *, const int *, const int *, const int *);
#endif

#if 0
/* Decide to size of merge. */
ACSAStatus decide_merge(...);
#endif

/* Init and Clean the environment of winograd. */
template<typename Dtype>
ACSAStatus ACSACnnInitLib();

template<typename Dtype>
ACSAStatus ACSACnnFreeLib();

#endif
