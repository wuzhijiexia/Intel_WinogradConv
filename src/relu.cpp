/* ReLu activation 
 * */

#include "dnn.hpp"

template<typename Dtype>
ACSAStatus ACSAReLUInplaceFwd(Dtype *in_out, ACSATensor4d* tensor)
{
    int N, C, H, W;
    int size;
    N = tensor->n_;
    C = tensor->c_;
    H = tensor->h_;
    W = tensor->w_;
    size = N*C*H*W;

    #pragma omp parallel for
    for(int i = 0; i < size; i++)
	in_out[i] = std::max(in_out[i], Dtype(0));

    return ACSASUCCESS;
}

template<typename Dtype>
ACSAStatus ACSAReLUOutplaceFwd(const Dtype *in, Dtype *out, ACSATensor4d* tensor)
{
    int N, C, H, W;
    int size;
    N = tensor->n_;
    C = tensor->c_;
    H = tensor->h_;
    W = tensor->w_;
    size = N*C*H*W;

    #pragma omp parallel for
    for(int i = 0; i < size; i++)
	out[i] = std::max(in[i], Dtype(0));

    return ACSASUCCESS;
}

template ACSAStatus ACSAReLUInplaceFwd<float>(float *in_out, ACSATensor4d* tensor);
template ACSAStatus ACSAReLUInplaceFwd<double>(double *in_out, ACSATensor4d* tensor);
template ACSAStatus ACSAReLUOutplaceFwd<float>(const float *in, float *out, ACSATensor4d* tensor);
template ACSAStatus ACSAReLUOutplaceFwd<double>(const double *in, double *out, ACSATensor4d* tensor);
