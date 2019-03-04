/* Example for testing cnn performance. */
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mkl.h>
#include "dnn.hpp"

#define F2X3            2
#define F3X3            3
#define F4X3            4

#define MAX             66
#define AVG             67

#define IN_PLACE        99
#define OUT_PLACE       100

/* class layer */
template <typename Dtype>
class layer {
    public:
        ACSALayerType type_;
        char *name_;
        layer<Dtype> *prv_layer_;

        virtual ACSATensor4d* get_output_tensor() = 0;
        virtual Dtype* get_output_pdata() = 0;
        virtual ACSAStatus set_message() = 0;
        virtual ACSAStatus forward() = 0;
        virtual ACSAStatus print_data() =0;
};

/* class inputLayer */
template <typename Dtype>
class inputLayer : public layer<Dtype> {
    public:
        ACSATensor4d t_in_;
        ACSATensor4d t_out_;

        Dtype *in_, *out_;
        layer<Dtype> *prv_layer_;

        inputLayer(int n, int c, int h, int w, char *name = "no_name");
        ~inputLayer();
        ACSATensor4d* get_output_tensor(){ return &t_out_; }
        Dtype* get_output_pdata(){ return out_; }
        ACSAStatus set_message();
        ACSAStatus forward(){ return ACSASUCCESS; }
        ACSAStatus print_data();
};

    template <typename Dtype>
inputLayer<Dtype>::inputLayer(int n, int c, int h, int w, char *name)
{
    ACSASetTensor4d(t_in_, n, c, h, w);
    this->name_ = name;
    this->type_ = INPUT;
}

    template <typename Dtype>
ACSAStatus inputLayer<Dtype>::set_message()
{
    int size;
    int N, C, H, W;
    N = t_in_.n_;
    C = t_in_.c_;
    H = t_in_.h_;
    W = t_in_.w_;

    ACSASetTensor4d(t_out_, N, C, H, W);

    size = N*C*H*W;
    in_ = (Dtype *)mkl_malloc(size*sizeof(Dtype), 64);
    out_ = in_;

    // random data for input
    for(int i = 0; i < size; i++)
        in_[i] = rand()%3;//1.0*(rand()%255)/255;

#if 0
    printf("%10s: in-add=%X, out-add=%X\n", this->name_, in_, out_);
#endif

    return ACSASUCCESS;
}

    template <typename Dtype>
inputLayer<Dtype>::~inputLayer()
{
    if(out_ != NULL)
        mkl_free(out_);
}

    template <typename Dtype>
ACSAStatus inputLayer<Dtype>::print_data()
{
    int N, C, H, W;
    int pos;

    N = t_in_.n_;
    C = t_in_.c_;
    H = t_in_.h_;
    W = t_in_.w_;

    printf("===== %10s =====\n", this->name_);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < C; j++){
            printf(">>> [%d, %d] <<<\n", i, j);
            for(int h = 0; h < H; h++){
                for(int w = 0; w < W; w++){
                    pos = (i*C+j)*H*W + h*W + w;
                    printf("%g ", in_[pos]);
                }
                printf("\n");
            }
        }
    }

    return ACSASUCCESS;
}

/* class convLayer */
template <typename Dtype>
class convLayer : public layer<Dtype> {
    public:
        ACSATensor4d t_in_;
        ACSATensor4d t_filter_;
        ACSATensor4d t_out_;
        ACSAConvMessage conv_mess_;
        ACSAWinoMessage wino_mess_;
        int nfilters_;

        Dtype *in_, *filter_, *out_;

        convLayer(int k, int algo, int bb, int mg, char *name = "no_name");
        ~convLayer();
        ACSATensor4d* get_output_tensor() { return &t_out_; }
        Dtype* get_output_pdata() { return out_; }
        ACSAStatus set_message();
        ACSAStatus forward();
        ACSAStatus print_data();
};

    template <typename Dtype>
convLayer<Dtype>::convLayer(int k, int algo, int bb, int mg, char *name)
{
    ACSASetConvMessage(conv_mess_, 3, 3, 1, 1, 1, 1);
    nfilters_ = k;

    switch(algo)
    {
        case F2X3:
            ACSASetWinoMessage(wino_mess_, ACSA_WINOGRAD_2X3, bb, mg);
            break;                    
        case F3X3:                    
            ACSASetWinoMessage(wino_mess_, ACSA_WINOGRAD_3X3, bb, mg);
            break;                    
        case F4X3:                    
            ACSASetWinoMessage(wino_mess_, ACSA_WINOGRAD_4X3, bb, mg);
            break;
        default:
            ACSA_CHECK(0);
            break;
    }

    this->name_ = name;
    this->type_ = CONVOLUTION;
}

    template <typename Dtype>
convLayer<Dtype>::~convLayer()
{
    if(filter_ != NULL)
        mkl_free(filter_);
    if(out_ != NULL)
        mkl_free(out_);
}

    template <typename Dtype>
ACSAStatus convLayer<Dtype>::set_message()
{
    int N, C, H, W, K;
    int inSize, filterSize, outSize;
    int pad_h, pad_w;

    ACSATensor4d *prv_tensor = this->prv_layer_->get_output_tensor();
    N = prv_tensor->n_;
    C = prv_tensor->c_;
    H = prv_tensor->h_;
    W = prv_tensor->w_;
    K = nfilters_;
    pad_h = conv_mess_.pad_h_;
    pad_w = conv_mess_.pad_w_;

    ACSASetTensor4d(t_in_, N, C, H, W);
    ACSASetTensor4d(t_filter_, K, C, 3, 3);
    ACSASetTensor4d(t_out_, N, K, H+2*pad_h-2, W+2*pad_w-2);

    inSize = N*C*H*W;
    filterSize = K*C*3*3;
    outSize = N*K*t_out_.h_*t_out_.w_;

    in_ = this->prv_layer_->get_output_pdata();
    filter_ = (Dtype *)mkl_malloc(filterSize*sizeof(Dtype), 64);
    out_ = (Dtype *)mkl_malloc(outSize*sizeof(Dtype), 64);

    // random data for weight
    for(int i = 0; i < filterSize; i++)
        filter_[i] = rand()%3-1;//1.0*(rand()%10)/4000;

#if 0
    printf("%10s: in-add=%X, out-add=%X\n", this->name_, in_, out_);
#endif

    return ACSASUCCESS;
}

    template <typename Dtype>
ACSAStatus convLayer<Dtype>::forward()
{
    switch(wino_mess_.algo_)
    {
        case ACSA_WINOGRAD_2X3:
            ACSAWinoConvolution_2x3<Dtype>(in_, filter_, out_,
                    &t_in_, &t_filter_, &t_out_, &conv_mess_, &wino_mess_);
            break;
        case ACSA_WINOGRAD_3X3:
            ACSAWinoConvolution_3x3<Dtype>(in_, filter_, out_,
                    &t_in_, &t_filter_, &t_out_, &conv_mess_, &wino_mess_);
            break;
        case ACSA_WINOGRAD_4X3:
            ACSAWinoConvolution_4x3<Dtype>(in_, filter_, out_,
                    &t_in_, &t_filter_, &t_out_, &conv_mess_, &wino_mess_);
            break;
        default:
            ACSA_CHECK(0);
            break;
    }

    return ACSASUCCESS;
}

    template <typename Dtype>
ACSAStatus convLayer<Dtype>::print_data()
{
    int N, C, H, W, K;
    int pos;

    N = t_out_.n_;
    C = t_out_.c_;
    H = t_out_.h_;
    W = t_out_.w_;
    K = nfilters_;

    printf("===== %10s =====\n", this->name_);

    printf(">>> filter <<<\n");
    for(int i = 0; i < K; i++){
        for(int j = 0; j < C; j++){
            printf(">>> [%d, %d] <<<\n", i, j);
            for(int h = 0; h < 3; h++){
                for(int w = 0; w < 3; w++){
                    pos = (i*C+j)*9 + h*3 + w;
                    printf("%g ", filter_[pos]);
                }
                printf("\n");
            }
        }
    }

    printf(">>> output <<<\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < C; j++){
            printf(">>> [%d, %d] <<<\n", i, j);
            for(int h = 0; h < H; h++){
                for(int w = 0; w < W; w++){
                    pos = (i*C+j)*H*W + h*W + w;
                    printf("%g ", out_[pos]);
                }
                printf("\n");
            }
        }
    }

    return ACSASUCCESS;
}

/* class relu */
template <typename Dtype>
class reluLayer : public layer<Dtype>{
    public:
        ACSATensor4d t_in_;
        ACSATensor4d t_out_;

        Dtype *in_, *out_;
        int place_type_;

        reluLayer(int flag, char *name = "no_name");
        ~reluLayer();
        ACSATensor4d* get_output_tensor(){ return &t_out_; }
        Dtype* get_output_pdata() { return out_; }
        ACSAStatus set_message();
        ACSAStatus forward();
        ACSAStatus print_data();
};

    template <typename Dtype>
reluLayer<Dtype>::reluLayer(int flag, char *name)
{
    place_type_ = flag;

    this->name_ = name;
    this->type_ = RELU;
}

    template <typename Dtype>
reluLayer<Dtype>::~reluLayer()
{
    if(out_ != NULL)
        mkl_free(out_);
}

    template <typename Dtype>
ACSAStatus reluLayer<Dtype>::set_message()
{
    ACSATensor4d *prv_tensor = this->prv_layer_->get_output_tensor();
    t_in_ = t_out_ = *prv_tensor;

    in_  = this->prv_layer_->get_output_pdata();

    switch(place_type_)
    {
        case IN_PLACE:
            out_ = in_;
            break;
        case OUT_PLACE:
            int size = t_out_.n_ * t_out_.c_ * t_out_.h_ * t_out_.w_;
            out_ = (Dtype *)mkl_malloc(size*sizeof(Dtype), 64);
            break;
        default:
            ACSA_CHECK(0);
            break;
    }

#if 0
    printf("%10s: in-add=%X, out-add=%X\n", this->name_, in_, out_);
#endif

    return ACSASUCCESS;
}

    template <typename Dtype>
ACSAStatus reluLayer<Dtype>::forward()
{
    ACSA_CHECK(((place_type_ == IN_PLACE) || (place_type_ == OUT_PLACE)));

    switch(place_type_)
    {
        case IN_PLACE:
            ACSAReLUInplaceFwd<Dtype>(in_, &t_in_);
            break;
        case OUT_PLACE:
            ACSAReLUOutplaceFwd<Dtype>(in_, out_, &t_in_);
            break;
        default:
            ACSA_CHECK(0);
            break;
    }

    return ACSASUCCESS;
}

    template <typename Dtype>
ACSAStatus reluLayer<Dtype>::print_data()
{
    int N, C, H, W;
    int pos;

    N = t_out_.n_;
    C = t_out_.c_;
    H = t_out_.h_;
    W = t_out_.w_;

    printf("===== %10s =====\n", this->name_);
    printf(">>> output <<<\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < C; j++){
            printf(">>> [%d, %d] <<<\n", i, j);
            for(int h = 0; h < H; h++){
                for(int w = 0; w < W; w++){
                    pos = (i*C+j)*H*W + h*W + w;
                    printf("%g ", out_[pos]);
                }
                printf("\n");
            }
        }
    }

    return ACSASUCCESS;
}

/* class pool */
template <typename Dtype>
class poolLayer : public layer<Dtype> {
    public:
        ACSATensor4d t_in_;
        ACSATensor4d t_out_;
        ACSAPoolMessage pool_mess_;

        Dtype *in_, *out_;
        int algo_;

        poolLayer(int algo, char *name = "no_name");
        ~poolLayer();
        ACSATensor4d* get_output_tensor(){ return &t_out_; }
        Dtype* get_output_pdata() { return out_; }
        ACSAStatus set_message();
        ACSAStatus forward();
        ACSAStatus print_data();
};

    template <typename Dtype>
poolLayer<Dtype>::poolLayer(int algo, char *name)
{
    ACSASetPoolMessage(pool_mess_, 2, 2, 0, 0, 2, 2);
    algo_ = algo;

    this->name_ = name;
    this->type_ = POOLING;
}

    template <typename Dtype>
poolLayer<Dtype>::~poolLayer()
{
    if(out_ != NULL)
        mkl_free(out_);
}

    template <typename Dtype>
ACSAStatus poolLayer<Dtype>::set_message()
{
    int sizeOut;
    int N, C, H, W;

    ACSATensor4d *prv_tensor = this->prv_layer_->get_output_tensor();
    t_in_ = *prv_tensor;

    N = t_in_.n_;
    C = t_in_.c_;
    H = t_in_.h_ / pool_mess_.kernel_h_;
    W = t_in_.w_ / pool_mess_.kernel_w_;

    ACSASetTensor4d(t_out_, N, C, H, W);

    sizeOut = N*C*H*W;
    in_ = this->prv_layer_->get_output_pdata();
    out_ = (Dtype *)mkl_malloc(sizeOut*sizeof(Dtype), 64);

#if 0
    printf("%10s: in-add=%X, out-add=%X\n", this->name_, in_, out_);
#endif

    return ACSASUCCESS;
}

    template <typename Dtype>
ACSAStatus poolLayer<Dtype>::forward()
{
    int N, C, H, W;
    int kn_h, kn_w, std_h, std_w;
    N = t_in_.n_;
    C = t_in_.c_;
    H = t_in_.h_;
    W = t_in_.w_;
    kn_h = pool_mess_.kernel_h_;
    kn_w = pool_mess_.kernel_w_;
    std_h = pool_mess_.stride_h_;
    std_w = pool_mess_.stride_w_;

    ACSA_CHECK(((kn_h == 2) && (kn_w == 2)));
    ACSA_CHECK(((std_h == 2) && (std_w == 2)));
    ACSA_CHECK(((H % kn_h == 0) && (W % kn_w == 0)));

    switch(algo_)
    {
        case MAX:
            ACSAMaxPoolingFwd<Dtype>(in_, out_, &t_in_, &pool_mess_);
            break;
        case AVG:
            ACSAAvePoolingFwd<Dtype>(in_, out_, &t_in_, &pool_mess_);
            break;
        default:
            ACSA_CHECK(0);
            break;
    }

    return ACSASUCCESS;
}

    template <typename Dtype>
ACSAStatus poolLayer<Dtype>::print_data()
{
    int N, C, H, W;
    int pos;

    N = t_out_.n_;
    C = t_out_.c_;
    H = t_out_.h_;
    W = t_out_.w_;

    printf("===== %10s =====\n", this->name_);
    printf(">>> output <<<\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < C; j++){
            printf(">>> [%d, %d] <<<\n", i, j);
            for(int h = 0; h < H; h++){
                for(int w = 0; w < W; w++){
                    pos = (i*C+j)*H*W + h*W + w;
                    printf("%g ", out_[pos]);
                }
                printf("\n");
            }
        }
    }
    return ACSASUCCESS;
}

/* model for network */
template <typename Dtype>
class model {
    public:
        std::vector<layer<Dtype> *> graph_;

        model();
        ~model();        
        ACSAStatus add(layer<Dtype> *new_layer);
        ACSAStatus run();
};

    template <typename Dtype>
model<Dtype>::model()
{
    // Prapared environment for acsa cnn
    ACSACnnInitLib<Dtype>();
}

    template <typename Dtype>
model<Dtype>::~model()
{
    // Free environment for acsa cnn
    ACSACnnFreeLib<Dtype>();
}

    template <typename Dtype>
ACSAStatus model<Dtype>::add(layer<Dtype> *new_layer)
{
    ACSA_CHECK(new_layer != NULL);

    graph_.push_back(new_layer);
    if(graph_.size() == 1){
        new_layer->prv_layer_ = NULL;
    }
    else {
        new_layer->prv_layer_ = graph_[graph_.size()-2];
    }

    new_layer->set_message();

    return ACSASUCCESS;
}

    template <typename Dtype>
ACSAStatus model<Dtype>::run()
{
    for(int i = 0; i < graph_.size(); i++){
        graph_[i]->forward();
    }

    return ACSASUCCESS;
}

/* instantiation */
template class layer<float>;
template class inputLayer<float>;
template class convLayer<float>;
template class reluLayer<float>;
template class poolLayer<float>;
template class model<float>;

template class layer<double>;
template class inputLayer<double>;
template class convLayer<double>;
template class reluLayer<double>;
template class poolLayer<double>;
template class model<double>;

int main(int argc, char *argv[]){
    ACSA_CHECK((argc > 1));

    srand((unsigned int)time(NULL));

    /* VGG19 Conv Layer */
    int cycleNum = atoi(argv[argc-1]);
    const int HW = 288;
    const int K_arr[16] = {64, 64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512}; 
    const int mg_arr[16] = {2, 1, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8};
    const int bb_arr[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int algo_arr[16]  = {4, 4, 4, 4, 3, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3};

    model<float> *md = new model<float>();

#if 1
    md->add(new inputLayer<float>(64, 3, HW, HW, "input"));
    md->add(new convLayer<float>(K_arr[ 0], algo_arr[ 0], bb_arr[ 0], mg_arr[ 0], "conv1_1"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu1_1"));                                  
    md->add(new convLayer<float>(K_arr[ 1], algo_arr[ 1], bb_arr[ 1], mg_arr[ 1], "conv1_2"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu1_2"));                                  
    md->add(new poolLayer<float>(MAX, "pool1"));

    md->add(new convLayer<float>(K_arr[ 2], algo_arr[ 2], bb_arr[ 2], mg_arr[ 2], "conv2_1"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu2_1"));                                  
    md->add(new convLayer<float>(K_arr[ 3], algo_arr[ 3], bb_arr[ 3], mg_arr[ 3], "conv2_2"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu2_2"));                                  
    md->add(new poolLayer<float>(MAX, "pool2"));                                    

    md->add(new convLayer<float>(K_arr[ 4], algo_arr[ 4], bb_arr[ 4], mg_arr[ 4], "conv3_1"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu3_1"));                                  
    md->add(new convLayer<float>(K_arr[ 5], algo_arr[ 5], bb_arr[ 5], mg_arr[ 5], "conv3_2"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu3_2"));                                  
    md->add(new convLayer<float>(K_arr[ 6], algo_arr[ 6], bb_arr[ 6], mg_arr[ 6], "conv3_3"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu3_3"));                                  
    md->add(new convLayer<float>(K_arr[ 7], algo_arr[ 7], bb_arr[ 7], mg_arr[ 7], "conv3_4"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu3_4"));                                  
    md->add(new poolLayer<float>(MAX, "pool3"));                                    

    md->add(new convLayer<float>(K_arr[ 8], algo_arr[ 8], bb_arr[ 8], mg_arr[ 8], "conv4_1"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu4_1"));                                  
    md->add(new convLayer<float>(K_arr[ 9], algo_arr[ 9], bb_arr[ 9], mg_arr[ 9], "conv4_2"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu4_2"));                                  
    md->add(new convLayer<float>(K_arr[10], algo_arr[10], bb_arr[10], mg_arr[10], "conv4_3"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu4_3"));                                  
    md->add(new convLayer<float>(K_arr[11], algo_arr[11], bb_arr[11], mg_arr[11], "conv4_4"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu4_4"));                                  
    md->add(new poolLayer<float>(MAX, "pool4"));                                    

    md->add(new convLayer<float>(K_arr[12], algo_arr[12], bb_arr[12], mg_arr[12], "conv5_1"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu5_1"));                                  
    md->add(new convLayer<float>(K_arr[13], algo_arr[13], bb_arr[13], mg_arr[13], "conv5_2"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu5_2"));                                  
    md->add(new convLayer<float>(K_arr[14], algo_arr[14], bb_arr[14], mg_arr[14], "conv5_3"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu5_3"));                                  
    md->add(new convLayer<float>(K_arr[15], algo_arr[15], bb_arr[15], mg_arr[15], "conv5_4"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu5_4"));
    md->add(new poolLayer<float>(MAX, "pool5"));
#else
    md->add(new inputLayer<float>(2, 1, 6, 6, "input"));
    md->add(new convLayer<float>(1, 2, 0, 1, "conv"));
    md->add(new reluLayer<float>(OUT_PLACE, "relu"));
    md->add(new poolLayer<float>(MAX, "pool"));
#endif
    // run and time
    for(int i = 0; i < md->graph_.size(); i++){
        (md->graph_[i])->forward();
        //(md->graph_[i])->print_data();
    }

    double stime, etime;
    double lytime[50];
    memset(lytime, 0, 50*sizeof(double));
    double alltime = 0.0;

    for(int c = 0; c < cycleNum; c++)
        for(int i = 0; i < md->graph_.size(); i++){
            stime = dsecnd();
            (md->graph_[i])->forward();
            etime = dsecnd();
            lytime[i] += (etime - stime);
        }

    printf("### Run time for the parts of VGG19(%d iteration) ###\n", cycleNum);
    for(int i = 0; i < md->graph_.size(); i++){
        lytime[i] = lytime[i]/cycleNum*1000;
        alltime += lytime[i];
        printf("%10s layer run time: %g ms.\n", md->graph_[i]->name_, lytime[i]);
    }
    printf("All time: %g ms.\n", alltime);

    return 0;
}
