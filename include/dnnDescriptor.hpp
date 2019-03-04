/* All Descriptor for ACSA CNN. */

#ifndef _DNN_DESCRIPTOR_HPP_
#define _DNN_DESCRIPTOR_HPP_

enum ACSAStatus {
    ACSASUCCESS,
    ACSAFAIL
};

enum ACSAWinogradAlgo {
    ACSA_WINOGRAD_2X3,
    ACSA_WINOGRAD_3X3,
    ACSA_WINOGRAD_4X3,
    ACSA_WINOGRAD_6X3
};

enum ACSALayerType {
    INPUT,
    CONVOLUTION,
    RELU,
    POOLING
};

struct ACSATensor4d {
    int n_;
    int c_;
    int h_;
    int w_;
};

struct ACSAConvMessage {
    int kernel_h_;
    int kernel_w_;
    int pad_h_;
    int pad_w_;
    int stride_h_;
    int stride_w_;
};

struct ACSAWinoMessage {
    ACSAWinogradAlgo algo_;
    int batch_block_;
    int merge_;
};

struct ACSAPoolMessage {
    int kernel_h_;
    int kernel_w_;
    int pad_h_;
    int pad_w_;
    int stride_h_;
    int stride_w_;
};

struct ACSATailMessage {
    int tail_h_;
    int tail_w_;
};
#endif
