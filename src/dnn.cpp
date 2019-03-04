/* Public contents for kernel implementation about winograd algorithm.
 * */

#include <dnn.hpp>

/* Pointer for bridge data. */
void* winoIn;    
void* winoFilter;    
void* winoOut;    

#if 0
/* The default of stride for bridge data. */
long ISTRIDE = (MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
long FSTRIDE = (MAX_FILTER_CHANNELS+1)*(MAX_FILTERS+1); 
long OSTRIDE = ISTRIDE;
#endif

/* Create Tensor4d */
ACSAStatus ACSASetTensor4d(ACSATensor4d &tensor,
        int n, int c, int h, int w)
{
    tensor.n_ = n;
    tensor.c_ = c;
    tensor.h_ = h;
    tensor.w_ = w;

    return ACSASUCCESS;
}

/* Create convolution message. */
ACSAStatus ACSASetConvMessage(ACSAConvMessage &convMess,
        int kernel_h, int kernel_w, 
        int pad_h, int pad_w, int stride_h, int stride_w)
{
    convMess.kernel_h_ = kernel_h;
    convMess.kernel_w_ = kernel_w;
    convMess.pad_h_ = pad_h;
    convMess.pad_w_ = pad_w;
    convMess.stride_h_ = stride_h;
    convMess.stride_w_ = stride_w;

    return ACSASUCCESS;
}

/* Create winograd message. */
ACSAStatus ACSASetWinoMessage(ACSAWinoMessage &winoMess,
        ACSAWinogradAlgo algo, int bb, int mg)
{
    winoMess.algo_ = algo;
    winoMess.batch_block_ = bb;
    winoMess.merge_ = mg;

    return ACSASUCCESS;
}

/* Set pooling message. */
ACSAStatus ACSASetPoolMessage(ACSAPoolMessage &poolMess,
        int kernel_h, int kernel_w,
        int pad_h, int pad_w, int stride_h, int stride_w)
{
    poolMess.kernel_h_ = kernel_h;
    poolMess.kernel_w_ = kernel_w;
    poolMess.pad_h_ = pad_h;
    poolMess.pad_w_ = pad_w;
    poolMess.stride_h_ = stride_h;
    poolMess.stride_w_ = stride_w;

    return ACSASUCCESS;
}

/* Get input tile with zero-padding. */
template <typename Dtype>
ACSAStatus ACSAGetInputTile(Dtype *tmp, const Dtype *data,
        int r_tile, int c_tile, int r_init, int c_init, int c_num,
        int r_up, int r_down, int c_left, int c_right)
{
    int i, j;
    int tmpIndex, iIndex, jIndex;

    memset(tmp, 0, r_tile*c_tile*sizeof(Dtype));
    for(i = r_up; i < (r_tile-r_down); i++){
        iIndex = i - r_up;
        for(j = c_left; j < (c_tile-c_right); j++){
            jIndex = j - c_left;
            tmpIndex = i*c_tile + j;
            tmp[tmpIndex] = data[(r_init+iIndex)*c_num + (c_init+jIndex)];
        }
    }

    return ACSASUCCESS;
}

/* Get final output with virtual value. */
template <typename Dtype>
ACSAStatus ACSAGetFinalOutput(Dtype *data, const Dtype *tmp,
        int r_out, int c_out, int r_init, int c_init, int c_num,
        int r_up, int r_down, int c_left, int c_right)
{
    int i, j;

    for(i = r_up; i < (r_out-r_down); i++){
        for(j = c_left; j < (c_out-c_right); j++){
            data[(r_init+i)*c_num + (c_init+j)] = tmp[i*c_out + j];
        }
    }

    return ACSASUCCESS;
}

/* Init to prepare the environment of winograd. */
template<typename Dtype>
ACSAStatus ACSACnnInitLib()
{
    int ret;
	winoIn = mkl_malloc(16*ISTRIDE*sizeof(Dtype), 64);
    assert(winoIn != NULL); 
	winoFilter = mkl_malloc(64*FSTRIDE*sizeof(Dtype), 64);
    assert(winoFilter != NULL); 
	winoOut = mkl_malloc(16*OSTRIDE*sizeof(Dtype), 64);
    assert(winoOut != NULL); 

    return ACSASUCCESS;
}

/* Clean the environment of winograd. */
template<typename Dtype>
ACSAStatus ACSACnnFreeLib()
{
    if(winoIn != NULL)
        mkl_free(winoIn);
        
    if(winoFilter != NULL)
        mkl_free(winoFilter);    
    
    if(winoOut != NULL)
        mkl_free(winoOut);    
    
    return ACSASUCCESS;
}

/* Fix Winograd Alogrithm */
template<typename Dtype>
ACSAStatus ACSAWinoConvolutionFwd(const Dtype *in, const Dtype *filter, Dtype *out,
        ACSATensor4d* tensorIn, ACSATensor4d* tensorFilter, ACSATensor4d* tensorOut,
        ACSAConvMessage* convMess, ACSAWinoMessage *winoMess)
{
    ACSAWinogradAlgo algo = winoMess->algo_;

    switch(algo)
    {
        case ACSA_WINOGRAD_2X3:
            ACSAWinoConvolution_2x3(in, filter, out,
                    tensorIn, tensorFilter, tensorOut, convMess, winoMess);
            break;
        case ACSA_WINOGRAD_3X3:
            ACSAWinoConvolution_3x3(in, filter, out,
                    tensorIn, tensorFilter, tensorOut, convMess, winoMess);
            break;
        case ACSA_WINOGRAD_4X3:
            ACSAWinoConvolution_4x3(in, filter, out,
                    tensorIn, tensorFilter, tensorOut, convMess, winoMess);
            break;
        case ACSA_WINOGRAD_6X3:
            ACSAWinoConvolution_6x3(in, filter, out,
                    tensorIn, tensorFilter, tensorOut, convMess, winoMess);
            break;
        default:
            ACSA_MESSAGE("ERROR: This winograd algorithm is nonexistent!\n");
            break;
    }

    return ACSASUCCESS;
}

/* Instantiate Template */
template ACSAStatus ACSAGetInputTile<float>(float *, const float *,
        int, int, int, int, int,
        int, int, int, int);
template ACSAStatus ACSAGetInputTile<double>(double *, const double *,
        int, int, int, int, int,
        int, int, int, int);

template ACSAStatus ACSAGetFinalOutput<float>(float *, const float *,
        int, int, int, int, int,
        int, int, int, int);
template ACSAStatus ACSAGetFinalOutput<double>(double *, const double *,
        int, int, int, int, int,
        int, int, int, int);

template ACSAStatus ACSACnnInitLib<float>();
template ACSAStatus ACSACnnInitLib<double>();

template ACSAStatus ACSACnnFreeLib<float>();
template ACSAStatus ACSACnnFreeLib<double>();

template ACSAStatus ACSAWinoConvolutionFwd<float>(const float*, const float*, float*,
        ACSATensor4d*, ACSATensor4d*, ACSATensor4d*,
        ACSAConvMessage*, ACSAWinoMessage*);
template ACSAStatus ACSAWinoConvolutionFwd<double>(const double*, const double*, double*,
        ACSATensor4d*, ACSATensor4d*, ACSATensor4d*,
        ACSAConvMessage*, ACSAWinoMessage*);
