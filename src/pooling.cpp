/* Max Pooling: pool_h/pool_w = stride_h/stride_w = 2x2
 * */

#include "dnn.hpp"

template <typename Dtype>
ACSAStatus ACSAMaxPoolingFwd(const Dtype *in, Dtype *out, ACSATensor4d* tensor,
        ACSAPoolMessage* poolMess)
{
    int N, C, H, W;
    N = tensor->n_;
    C = tensor->c_;
    H = tensor->h_;
    W = tensor->w_;

	#pragma omp parallel for
	for(int k = 0; k < N*C; k++){
	    const Dtype *bin = in + k*H*W;
	    Dtype *bout = out + k*(H/2)*(W/2);
	    Dtype num0, num1;
		int counter = 0;

		for(int i = 0; i < H; i += 2)
		    for(int j = 0; j < W; j += 2){
		    	num0 = std::max(bin[i*W+j], bin[i*W+j+1]);
		    	num1 = std::max(bin[(i+1)*W+j], bin[(i+1)*W+j+1]);
		    	bout[counter] = std::max(num0, num1);
                        counter++;
		    }
	}

    return ACSASUCCESS;
}

template <typename Dtype>
ACSAStatus ACSAAvePoolingFwd(const Dtype *in, Dtype *out, ACSATensor4d* tensor,
        ACSAPoolMessage* poolMess)
{
    int N, C, H, W;
    N = tensor->n_;
    C = tensor->c_;
    H = tensor->h_;
    W = tensor->w_;

    #pragma omp parallel for
    for(int k = 0; k < N*C; k++){
        const Dtype *bin = in + k*H*W;
        Dtype *bout = out + k*(H/2)*(W/2);
        int counter = 0;

        for(int i = 0; i < H; i++)
            for(int j = 0; j < W; j++){
                bout[counter] = (bin[i*W+j] + bin[i*W+(j+1)] +
                bin[(i+1)*W+j] + bin[(i+1)*W+(j+1)])/4;
                counter++;
            }
    }

    return ACSASUCCESS;
}

template ACSAStatus ACSAMaxPoolingFwd<float>(const float*, float*, ACSATensor4d*,
        ACSAPoolMessage*);
template ACSAStatus ACSAAvePoolingFwd<float>(const float*, float*, ACSATensor4d*,
        ACSAPoolMessage*);

template ACSAStatus ACSAMaxPoolingFwd<double>(const double*, double*, ACSATensor4d*,
        ACSAPoolMessage*);
template ACSAStatus ACSAAvePoolingFwd<double>(const double*, double*, ACSATensor4d*,
        ACSAPoolMessage*);
