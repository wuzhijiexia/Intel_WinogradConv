/* F(2x2,3x3) implementations for winograd. */

#include <immintrin.h>
#include "dnn.hpp"

#define ZERO_LENGTH(tail) (2-tail)%2

static const long ISTRIDE2X3 = ISTRIDE;
static const long FSTRIDE2X3 = FSTRIDE;
static const long OSTRIDE2X3 = OSTRIDE;

/* AT-G-BT for F(2,3).
 * The dimensions for AT/G/BT.
 * dim-AT: F_M, TILE
 * dim-G : TILE , F_R
 * dim-BT: TILE , TILE
 */
static const float AT[8] = {
    1, 1,  1,  0,
    0, 1, -1, -1
};

static const float G[12] = {
    1,     0,      0,
    1.0/2,  1.0/2, 1.0/2,
    1.0/2, -1.0/2, 1.0/2,
    0,      0,     1
};

static const float BT[16] = {
    2,  0, -1,  0,
    0,  1,  1,  0,
    0, -1,  1,  0,
    0,  1,  0, -1
};

// Twice transform for input data to get tiles
#define TRANS_BT_FST(A, B, C) \
{ \
    C[ 0] = A[ 0]*B[ 0] + A[ 1]*B[ 4] + A[ 2]*B[ 8] + A[ 3]*B[12]; \
    C[ 1] = A[ 0]*B[ 1] + A[ 1]*B[ 5] + A[ 2]*B[ 9] + A[ 3]*B[13]; \
    C[ 2] = A[ 0]*B[ 2] + A[ 1]*B[ 6] + A[ 2]*B[10] + A[ 3]*B[14]; \
    C[ 3] = A[ 0]*B[ 3] + A[ 1]*B[ 7] + A[ 2]*B[11] + A[ 3]*B[15]; \
    C[ 4] = A[ 4]*B[ 0] + A[ 5]*B[ 4] + A[ 6]*B[ 8] + A[ 7]*B[12]; \
    C[ 5] = A[ 4]*B[ 1] + A[ 5]*B[ 5] + A[ 6]*B[ 9] + A[ 7]*B[13]; \
    C[ 6] = A[ 4]*B[ 2] + A[ 5]*B[ 6] + A[ 6]*B[10] + A[ 7]*B[14]; \
    C[ 7] = A[ 4]*B[ 3] + A[ 5]*B[ 7] + A[ 6]*B[11] + A[ 7]*B[15]; \
    C[ 8] = A[ 8]*B[ 0] + A[ 9]*B[ 4] + A[10]*B[ 8] + A[11]*B[12]; \
    C[ 9] = A[ 8]*B[ 1] + A[ 9]*B[ 5] + A[10]*B[ 9] + A[11]*B[13]; \
    C[10] = A[ 8]*B[ 2] + A[ 9]*B[ 6] + A[10]*B[10] + A[11]*B[14]; \
    C[11] = A[ 8]*B[ 3] + A[ 9]*B[ 7] + A[10]*B[11] + A[11]*B[15]; \
    C[12] = A[12]*B[ 0] + A[13]*B[ 4] + A[14]*B[ 8] + A[15]*B[12]; \
    C[13] = A[12]*B[ 1] + A[13]*B[ 5] + A[14]*B[ 9] + A[15]*B[13]; \
    C[14] = A[12]*B[ 2] + A[13]*B[ 6] + A[14]*B[10] + A[15]*B[14]; \
    C[15] = A[12]*B[ 3] + A[13]*B[ 7] + A[14]*B[11] + A[15]*B[15]; \
}

#define TRANS_BT_SED(A, B, C, TC, ST) \
{ \
    C[TC +  0*ST] = A[ 0]*B[ 0] + A[ 1]*B[ 1] + A[ 2]*B[ 2] + A[ 3]*B[ 3]; \
    C[TC +  1*ST] = A[ 0]*B[ 4] + A[ 1]*B[ 5] + A[ 2]*B[ 6] + A[ 3]*B[ 7]; \
    C[TC +  2*ST] = A[ 0]*B[ 8] + A[ 1]*B[ 9] + A[ 2]*B[10] + A[ 3]*B[11]; \
    C[TC +  3*ST] = A[ 0]*B[12] + A[ 1]*B[13] + A[ 2]*B[14] + A[ 3]*B[15]; \
    C[TC +  4*ST] = A[ 4]*B[ 0] + A[ 5]*B[ 1] + A[ 6]*B[ 2] + A[ 7]*B[ 3]; \
    C[TC +  5*ST] = A[ 4]*B[ 4] + A[ 5]*B[ 5] + A[ 6]*B[ 6] + A[ 7]*B[ 7]; \
    C[TC +  6*ST] = A[ 4]*B[ 8] + A[ 5]*B[ 9] + A[ 6]*B[10] + A[ 7]*B[11]; \
    C[TC +  7*ST] = A[ 4]*B[12] + A[ 5]*B[13] + A[ 6]*B[14] + A[ 7]*B[15]; \
    C[TC +  8*ST] = A[ 8]*B[ 0] + A[ 9]*B[ 1] + A[10]*B[ 2] + A[11]*B[ 3]; \
    C[TC +  9*ST] = A[ 8]*B[ 4] + A[ 9]*B[ 5] + A[10]*B[ 6] + A[11]*B[ 7]; \
    C[TC + 10*ST] = A[ 8]*B[ 8] + A[ 9]*B[ 9] + A[10]*B[10] + A[11]*B[11]; \
    C[TC + 11*ST] = A[ 8]*B[12] + A[ 9]*B[13] + A[10]*B[14] + A[11]*B[15]; \
    C[TC + 12*ST] = A[12]*B[ 0] + A[13]*B[ 1] + A[14]*B[ 2] + A[15]*B[ 3]; \
    C[TC + 13*ST] = A[12]*B[ 4] + A[13]*B[ 5] + A[14]*B[ 6] + A[15]*B[ 7]; \
    C[TC + 14*ST] = A[12]*B[ 8] + A[13]*B[ 9] + A[14]*B[10] + A[15]*B[11]; \
    C[TC + 15*ST] = A[12]*B[12] + A[13]*B[13] + A[14]*B[14] + A[15]*B[15]; \
}

// Get output tile & First transform
#define GET_OUTPUT_TILE(TMP, DATA, TC, ST) \
{ \
    TMP[0 ] = DATA[TC+ 0 *ST]; \
    TMP[1 ] = DATA[TC+ 1 *ST]; \
    TMP[2 ] = DATA[TC+ 2 *ST]; \
    TMP[3 ] = DATA[TC+ 3 *ST]; \
    TMP[4 ] = DATA[TC+ 4 *ST]; \
    TMP[5 ] = DATA[TC+ 5 *ST]; \
    TMP[6 ] = DATA[TC+ 6 *ST]; \
    TMP[7 ] = DATA[TC+ 7 *ST]; \
    TMP[8 ] = DATA[TC+ 8 *ST]; \
    TMP[9 ] = DATA[TC+ 9 *ST]; \
    TMP[10] = DATA[TC+ 10*ST]; \
    TMP[11] = DATA[TC+ 11*ST]; \
    TMP[12] = DATA[TC+ 12*ST]; \
    TMP[13] = DATA[TC+ 13*ST]; \
    TMP[14] = DATA[TC+ 14*ST]; \
    TMP[15] = DATA[TC+ 15*ST]; \
}

#define TRANS_AT_FST(A, B, C) \
{ \
    C[ 0] = A[ 0]*B[ 0] + A[ 1]*B[ 4] + A[ 2]*B[ 8] + A[ 3]*B[12]; \
    C[ 1] = A[ 0]*B[ 1] + A[ 1]*B[ 5] + A[ 2]*B[ 9] + A[ 3]*B[13]; \
    C[ 2] = A[ 0]*B[ 2] + A[ 1]*B[ 6] + A[ 2]*B[10] + A[ 3]*B[14]; \
    C[ 3] = A[ 0]*B[ 3] + A[ 1]*B[ 7] + A[ 2]*B[11] + A[ 3]*B[15]; \
    C[ 4] = A[ 4]*B[ 0] + A[ 5]*B[ 4] + A[ 6]*B[ 8] + A[ 7]*B[12]; \
    C[ 5] = A[ 4]*B[ 1] + A[ 5]*B[ 5] + A[ 6]*B[ 9] + A[ 7]*B[13]; \
    C[ 6] = A[ 4]*B[ 2] + A[ 5]*B[ 6] + A[ 6]*B[10] + A[ 7]*B[14]; \
    C[ 7] = A[ 4]*B[ 3] + A[ 5]*B[ 7] + A[ 6]*B[11] + A[ 7]*B[15]; \
}

#define TRANS_AT_SED(A, B, C) \
{ \
    C[0] = A[ 0]*B[ 0] + A[ 1]*B[ 1] + A[ 2]*B[ 2] + A[ 3]*B[ 3]; \
    C[1] = A[ 0]*B[ 4] + A[ 1]*B[ 5] + A[ 2]*B[ 6] + A[ 3]*B[ 7]; \
    C[2] = A[ 4]*B[ 0] + A[ 5]*B[ 1] + A[ 6]*B[ 2] + A[ 7]*B[ 3]; \
    C[3] = A[ 4]*B[ 4] + A[ 5]*B[ 5] + A[ 6]*B[ 6] + A[ 7]*B[ 7]; \
}

/* Don't use pad: 
 * compute the bridge data for in, and transform to form matrix A.
 * */
    template<typename Dtype>
static void inByTransform_nopad(const Dtype *in, Dtype *dataDst,
        const int N, const int C, const int rows, const int cols,
        ACSATailMessage *tailMess, const int ntiles, const int mg2x3)
{   
    int d1, d2;
    int sizeI = rows*cols;

    int rowSeg1, rowSeg2;
    int colSeg1, colSeg2;
    rowSeg1 = (rows-2)/2*2;
    rowSeg2 = rows-2 - rowSeg1;
    colSeg1 = (cols-2)/2*2;
    colSeg2 = cols-2 - colSeg1;

    ACSA_CHECK((rowSeg2 == tailMess->tail_h_));
    ACSA_CHECK((colSeg2 == tailMess->tail_w_));

#pragma omp parallel for private(d1) 
    for(d1 = 0; d1 < N*C; d1++){
        int i, j, k; 
        Dtype tmp[16] __attribute__((aligned(64)));
        Dtype bridge[16] __attribute__((aligned(64)));

        const int t1 = d1/(C*mg2x3);
        const int t2 = (d1%(C*mg2x3))/mg2x3;
        const int t3 = d1%mg2x3;

        // Merge value influence the sequence of in-data.
        const Dtype *data = in + (t1*mg2x3*C + t3*C + t2)*sizeI;
        int tileCount = d1*ntiles;

        for(i = 0; i < rowSeg1; i += 2){
#pragma simd
            // Process no tail
            for(j = 0; j < colSeg1; j += 2){
                tmp[0 ] = data[(i+0)*cols + (j+0)]; 
                tmp[1 ] = data[(i+0)*cols + (j+1)]; 
                tmp[2 ] = data[(i+0)*cols + (j+2)]; 
                tmp[3 ] = data[(i+0)*cols + (j+3)]; 
                tmp[4 ] = data[(i+1)*cols + (j+0)]; 
                tmp[5 ] = data[(i+1)*cols + (j+1)]; 
                tmp[6 ] = data[(i+1)*cols + (j+2)]; 
                tmp[7 ] = data[(i+1)*cols + (j+3)]; 
                tmp[8 ] = data[(i+2)*cols + (j+0)]; 
                tmp[9 ] = data[(i+2)*cols + (j+1)]; 
                tmp[10] = data[(i+2)*cols + (j+2)]; 
                tmp[11] = data[(i+2)*cols + (j+3)]; 
                tmp[12] = data[(i+3)*cols + (j+0)]; 
                tmp[13] = data[(i+3)*cols + (j+1)]; 
                tmp[14] = data[(i+3)*cols + (j+2)]; 
                tmp[15] = data[(i+3)*cols + (j+3)]; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
                tileCount++; 
            }

            // Process col tail
            if(colSeg2 != 0){
                ACSAGetInputTile(tmp, data, 4, 4, i, j, cols, 0, 0, 0, 2-colSeg2);

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
                tileCount++; 
            }
        }

        // Process row tail
        if(ZERO_LENGTH(rowSeg2) == 1){
#pragma simd
            for(j = 0; j < colSeg1; j += 2){
                tmp[0 ] = data[(rowSeg1+0)*cols + (j+0)]; 
                tmp[1 ] = data[(rowSeg1+0)*cols + (j+1)]; 
                tmp[2 ] = data[(rowSeg1+0)*cols + (j+2)]; 
                tmp[3 ] = data[(rowSeg1+0)*cols + (j+3)]; 
                tmp[4 ] = data[(rowSeg1+1)*cols + (j+0)]; 
                tmp[5 ] = data[(rowSeg1+1)*cols + (j+1)]; 
                tmp[6 ] = data[(rowSeg1+1)*cols + (j+2)]; 
                tmp[7 ] = data[(rowSeg1+1)*cols + (j+3)]; 
                tmp[8 ] = data[(rowSeg1+2)*cols + (j+0)]; 
                tmp[9 ] = data[(rowSeg1+2)*cols + (j+1)]; 
                tmp[10] = data[(rowSeg1+2)*cols + (j+2)]; 
                tmp[11] = data[(rowSeg1+2)*cols + (j+3)]; 
                tmp[12] = (Dtype)0.0; 
                tmp[13] = (Dtype)0.0; 
                tmp[14] = (Dtype)0.0; 
                tmp[15] = (Dtype)0.0;

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
                tileCount++;
            }
        }

        // Process row&col tail
        if((rowSeg2 != 0) && (colSeg2 != 0)){
            ACSAGetInputTile(tmp, data, 4, 4, rowSeg1, colSeg1, cols, 0, ZERO_LENGTH(rowSeg2), 0, ZERO_LENGTH(colSeg2));

            // The tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
            tileCount++;
        }
    }
}

/* Use pad for big-scale in-data: 
 * compute the bridge data for in, and transform to form matrix A.
 * */
    template<typename Dtype>
static void inByTransform_padBigScale(const Dtype *in, Dtype *dataDst,
        const int N, const int C, const int rows, const int cols,
        const int pad_h, const int pad_w,
        ACSATailMessage *tailMess, const int ntiles, const int mg2x3)
{   
    int d1, d2;
    int rows_pad = rows + 2*pad_h;
    int cols_pad = cols + 2*pad_w;
    int sizeI = rows*cols;

    int row_nTiles, col_nTiles;
    int tail_h = tailMess->tail_h_;
    int tail_w = tailMess->tail_w_;

    if(tail_h == 0)
        row_nTiles = (rows_pad-2)/2;
    else
        row_nTiles = (rows_pad-2)/2 + 1;

    if(tail_w == 0)
        col_nTiles = (cols_pad-2)/2;
    else
        col_nTiles = (cols_pad-2)/2 +1;

#pragma omp parallel for private(d1) 
    for(d1 = 0; d1 < N*C; d1++){
        int i, j; 
        Dtype tmp[16] __attribute__((aligned(64)));
        Dtype bridge[16] __attribute__((aligned(64)));

        const int t1 = d1/(C*mg2x3);
        const int t2 = (d1%(C*mg2x3))/mg2x3;
        const int t3 = d1%mg2x3;

        // merge value influence the sequence of in data.
        const Dtype *data = in + (t1*mg2x3*C + t3*C + t2)*sizeI;
        int tileCount = d1*ntiles;
        int baseTileCount = tileCount;

        /* The real data part. */
        const int si = 1;
        const int sj = 1;
        for(i = si; i < (rows-3); i += 2){
            tileCount = baseTileCount + ((i+pad_h)/2)*col_nTiles + 1;
#pragma simd
            // Center for first type region
            for(j = sj; j < (cols-3); j += 2){
                tmp[0 ] = data[(i+0)*cols + (j+0)]; 
                tmp[1 ] = data[(i+0)*cols + (j+1)]; 
                tmp[2 ] = data[(i+0)*cols + (j+2)]; 
                tmp[3 ] = data[(i+0)*cols + (j+3)]; 

                tmp[4 ] = data[(i+1)*cols + (j+0)]; 
                tmp[5 ] = data[(i+1)*cols + (j+1)]; 
                tmp[6 ] = data[(i+1)*cols + (j+2)]; 
                tmp[7 ] = data[(i+1)*cols + (j+3)]; 

                tmp[8 ] = data[(i+2)*cols + (j+0)]; 
                tmp[9 ] = data[(i+2)*cols + (j+1)]; 
                tmp[10] = data[(i+2)*cols + (j+2)]; 
                tmp[11] = data[(i+2)*cols + (j+3)]; 

                tmp[12] = data[(i+3)*cols + (j+0)]; 
                tmp[13] = data[(i+3)*cols + (j+1)]; 
                tmp[14] = data[(i+3)*cols + (j+2)]; 
                tmp[15] = data[(i+3)*cols + (j+3)]; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
                tileCount++; 
            }
        }

        /* The vitural data part for pad=1. */
        tmp[0] = tmp[1] = tmp[2] = tmp[3] = (Dtype)0.0;
        tileCount = baseTileCount + 1;
#pragma simd
        // Top for second region
        for(j = sj; j < (cols-3); j += 2){
            tmp[4 ] = data[0*cols + (j+0)];
            tmp[5 ] = data[0*cols + (j+1)];
            tmp[6 ] = data[0*cols + (j+2)];
            tmp[7 ] = data[0*cols + (j+3)];

            tmp[8 ] = data[1*cols + (j+0)];
            tmp[9 ] = data[1*cols + (j+1)];
            tmp[10] = data[1*cols + (j+2)];
            tmp[11] = data[1*cols + (j+3)];

            tmp[12] = data[2*cols + (j+0)];
            tmp[13] = data[2*cols + (j+1)];
            tmp[14] = data[2*cols + (j+2)];
            tmp[15] = data[2*cols + (j+3)];

            // Tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
            tileCount++;
        }

        // Bottom for second region
        tmp[12] = tmp[13] = tmp[14] = tmp[15] = (Dtype)0.0;
        tileCount = baseTileCount + (row_nTiles - 1)*col_nTiles + 1;
        if((2-tail_h) == 1){
            tmp[ 8] = tmp[ 9] = tmp[10] = tmp[11] = (Dtype)0.0;

#pragma simd
            for(j = sj; j < (cols-3); j += 2){
                tmp[0 ] = data[(rows-2)*cols + (j+0)];
                tmp[1 ] = data[(rows-2)*cols + (j+1)];
                tmp[2 ] = data[(rows-2)*cols + (j+2)];
                tmp[3 ] = data[(rows-2)*cols + (j+3)];

                tmp[4 ] = data[(rows-1)*cols + (j+0)];
                tmp[5 ] = data[(rows-1)*cols + (j+1)];
                tmp[6 ] = data[(rows-1)*cols + (j+2)];
                tmp[7 ] = data[(rows-1)*cols + (j+3)];

                // Tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
                tileCount++;
            }
        }
        else {
#pragma simd
            for(j = sj; j < (cols-3); j += 2){
                tmp[0 ] = data[(rows-3)*cols + (j+0)];
                tmp[1 ] = data[(rows-3)*cols + (j+1)];
                tmp[2 ] = data[(rows-3)*cols + (j+2)];
                tmp[3 ] = data[(rows-3)*cols + (j+3)];

                tmp[4 ] = data[(rows-2)*cols + (j+0)];
                tmp[5 ] = data[(rows-2)*cols + (j+1)];
                tmp[6 ] = data[(rows-2)*cols + (j+2)];
                tmp[7 ] = data[(rows-2)*cols + (j+3)];

                tmp[8 ] = data[(rows-1)*cols + (j+0)];
                tmp[9 ] = data[(rows-1)*cols + (j+1)];
                tmp[10] = data[(rows-1)*cols + (j+2)];
                tmp[11] = data[(rows-1)*cols + (j+3)];

                // Tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
                tileCount++;
            }
        }

        // Left for second region
        tileCount = baseTileCount + col_nTiles;
        for(i = si; i < (rows-3); i += 2){
            ACSAGetInputTile(tmp, data, 4, 4, i, 0, cols, 0, 0, 1, 0);

            // Tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
            tileCount += col_nTiles;
        }

        // Right for second region
        tileCount = baseTileCount + 2*col_nTiles -1;
        for(i = si; i < (rows-3); i += 2){
            ACSAGetInputTile(tmp, data, 4, 4, i, cols-3+ZERO_LENGTH(tail_w), cols, 0, 0, 0, ZERO_LENGTH(tail_w)+1);

            // Tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
            tileCount += col_nTiles;
        }

        // Top Left Corner for third region        
        tileCount = baseTileCount;
        ACSAGetInputTile(tmp, data, 4, 4, 0, 0, cols, 1, 0, 1, 0);
        TRANS_BT_FST(BT, tmp, bridge);
        TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);

        // Top Right Corner for third region
        tileCount = baseTileCount + col_nTiles -1;
        ACSAGetInputTile(tmp, data, 4, 4, 0, cols-3+ZERO_LENGTH(tail_w), cols, 1, 0, 0, ZERO_LENGTH(tail_w)+1);
        TRANS_BT_FST(BT, tmp, bridge);
        TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);

        // Bottom Left Corner for third region
        tileCount = baseTileCount + (row_nTiles - 1)*(col_nTiles);
        ACSAGetInputTile(tmp, data, 4, 4, rows-3+ZERO_LENGTH(tail_h), 0, cols, 0, ZERO_LENGTH(tail_h)+1, 1, 0);
        TRANS_BT_FST(BT, tmp, bridge);
        TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);

        // Bottom Right Corner for third region
        tileCount = baseTileCount + 
            row_nTiles*col_nTiles - 1;
        ACSAGetInputTile(tmp, data, 4, 4, rows-3+ZERO_LENGTH(tail_h), cols-3+ZERO_LENGTH(tail_w), cols, 0, ZERO_LENGTH(tail_h)+1, 0, ZERO_LENGTH(tail_w)+1);
        TRANS_BT_FST(BT, tmp, bridge);
        TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
    }
}

/* Use pad for small-scale in-data: 
 * compute the bridge data for in, and transform to form matrix A.
 * */
    template<typename Dtype>
static void inByTransform_padSmallScale(const Dtype *in, Dtype *in_pad, Dtype *dataDst,
        const int N, const int C, const int rows, const int cols,
        const int pad_h, const int pad_w,
        ACSATailMessage *tailMess, const int ntiles, const int mg2x3)
{   
    int d1, d2;
    int rows_pad = rows + 2*pad_h;
    int cols_pad = cols + 2*pad_w;
    int sizeI = rows*cols;

    int rowSeg1, rowSeg2;
    int colSeg1, colSeg2;
    rowSeg1 = (rows_pad-2)/2*2;
    rowSeg2 = rows_pad-2 - rowSeg1;
    colSeg1 = (cols_pad-2)/2*2;
    colSeg2 = cols_pad-2 - colSeg1;

    ACSA_CHECK((rowSeg2 == tailMess->tail_h_));
    ACSA_CHECK((colSeg2 == tailMess->tail_w_));

#pragma omp parallel for private(d1) 
    for(d1 = 0; d1 < N*C; d1++){
        int i, j, k; 
        Dtype tmp[16] __attribute__((aligned(64)));
        Dtype bridge[16] __attribute__((aligned(64)));

        const int t1 = d1/(C*mg2x3);
        const int t2 = (d1%(C*mg2x3))/mg2x3;
        const int t3 = d1%mg2x3;

        // merge value influence the sequence of in data.
        const Dtype *data = in + (t1*mg2x3*C + t3*C + t2)*sizeI;
        int tileCount = d1*ntiles;

        // Pad the in-data which will be processed.
        int index_threads = omp_get_thread_num();
        Dtype *data_pad = in_pad + index_threads*(rows_pad)*(cols_pad);

        for(i = 0; i < rows; i++)
#pragma simd
            for(j = 0; j < cols; j++)
                data_pad[(i+1)*(cols_pad) + (j+1)] = data[i*cols+j];

        for(i = 0; i < rowSeg1; i += 2){
#pragma simd
            // Process no tail
            for(j = 0; j < colSeg1; j += 2){
                tmp[0 ] = data_pad[(i+0)*(cols_pad) + (j+0)]; 
                tmp[1 ] = data_pad[(i+0)*(cols_pad) + (j+1)]; 
                tmp[2 ] = data_pad[(i+0)*(cols_pad) + (j+2)]; 
                tmp[3 ] = data_pad[(i+0)*(cols_pad) + (j+3)]; 

                tmp[4 ] = data_pad[(i+1)*(cols_pad) + (j+0)]; 
                tmp[5 ] = data_pad[(i+1)*(cols_pad) + (j+1)]; 
                tmp[6 ] = data_pad[(i+1)*(cols_pad) + (j+2)]; 
                tmp[7 ] = data_pad[(i+1)*(cols_pad) + (j+3)]; 

                tmp[8 ] = data_pad[(i+2)*(cols_pad) + (j+0)]; 
                tmp[9 ] = data_pad[(i+2)*(cols_pad) + (j+1)]; 
                tmp[10] = data_pad[(i+2)*(cols_pad) + (j+2)]; 
                tmp[11] = data_pad[(i+2)*(cols_pad) + (j+3)]; 

                tmp[12] = data_pad[(i+3)*(cols_pad) + (j+0)]; 
                tmp[13] = data_pad[(i+3)*(cols_pad) + (j+1)]; 
                tmp[14] = data_pad[(i+3)*(cols_pad) + (j+2)]; 
                tmp[15] = data_pad[(i+3)*(cols_pad) + (j+3)]; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
                tileCount++; 
            }

            // Process col tail
            if(colSeg2 != 0){
                ACSAGetInputTile(tmp, data_pad, 4, 4, i, j, cols_pad, 0, 0, 0, ZERO_LENGTH(colSeg2));

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
                tileCount++; 
            }
        }

        // Process row tail
        if(ZERO_LENGTH(rowSeg2) == 1){
#pragma simd
            for(j = 0; j < colSeg1; j += 2){
                tmp[0 ] = data_pad[(rowSeg1+0)*(cols_pad) + (j+0)]; 
                tmp[1 ] = data_pad[(rowSeg1+0)*(cols_pad) + (j+1)]; 
                tmp[2 ] = data_pad[(rowSeg1+0)*(cols_pad) + (j+2)]; 
                tmp[3 ] = data_pad[(rowSeg1+0)*(cols_pad) + (j+3)]; 

                tmp[4 ] = data_pad[(rowSeg1+1)*(cols_pad) + (j+0)]; 
                tmp[5 ] = data_pad[(rowSeg1+1)*(cols_pad) + (j+1)]; 
                tmp[6 ] = data_pad[(rowSeg1+1)*(cols_pad) + (j+2)]; 
                tmp[7 ] = data_pad[(rowSeg1+1)*(cols_pad) + (j+3)]; 

                tmp[8 ] = data_pad[(rowSeg1+2)*(cols_pad) + (j+0)]; 
                tmp[9 ] = data_pad[(rowSeg1+2)*(cols_pad) + (j+1)]; 
                tmp[10] = data_pad[(rowSeg1+2)*(cols_pad) + (j+2)]; 
                tmp[11] = data_pad[(rowSeg1+2)*(cols_pad) + (j+3)]; 

                tmp[12] = (Dtype)0.0; 
                tmp[13] = (Dtype)0.0; 
                tmp[14] = (Dtype)0.0; 
                tmp[15] = (Dtype)0.0; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
                tileCount++; 
            }
        }

        // Process row&col tail
        if((rowSeg2 != 0) && (colSeg2 != 0)){
            ACSAGetInputTile(tmp, data_pad, 4, 4, rowSeg1, colSeg1, cols_pad, 0, ZERO_LENGTH(rowSeg2), 0, ZERO_LENGTH(colSeg2));

            // The tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE2X3);
            tileCount++; 
        }
    }
}

/* Compute the bridge data for filter, and transform to form matrix B. */
    template<typename Dtype>
static void filterByTransform(const Dtype *filter, Dtype *dataDst,
        const int C, const int K)
{

    int d1, d2, d3; 

#pragma omp parallel for collapse(2) private(d1, d2, d3)
    //#pragma prefetch out:1:1
    //#pragma simd
    for(d1 = 0; d1 < K; d1++){
        for(d2 = 0; d2 < C; d2++){
            const Dtype *tmp = filter+d2*3*3 + d1*3*3*C; 
            Dtype ddt[16] __attribute__((aligned(64))); 
            Dtype bridge[12] __attribute__((aligned(64))); 

            // First transform filter data by G
            bridge[ 0] = G[ 0]*tmp[ 0] + G[ 1]*tmp[ 3] + G[ 2]*tmp[ 6];
            bridge[ 1] = G[ 0]*tmp[ 1] + G[ 1]*tmp[ 4] + G[ 2]*tmp[ 7];
            bridge[ 2] = G[ 0]*tmp[ 2] + G[ 1]*tmp[ 5] + G[ 2]*tmp[ 8];
            bridge[ 3] = G[ 3]*tmp[ 0] + G[ 4]*tmp[ 3] + G[ 5]*tmp[ 6];
            bridge[ 4] = G[ 3]*tmp[ 1] + G[ 4]*tmp[ 4] + G[ 5]*tmp[ 7];
            bridge[ 5] = G[ 3]*tmp[ 2] + G[ 4]*tmp[ 5] + G[ 5]*tmp[ 8];
            bridge[ 6] = G[ 6]*tmp[ 0] + G[ 7]*tmp[ 3] + G[ 8]*tmp[ 6];
            bridge[ 7] = G[ 6]*tmp[ 1] + G[ 7]*tmp[ 4] + G[ 8]*tmp[ 7];
            bridge[ 8] = G[ 6]*tmp[ 2] + G[ 7]*tmp[ 5] + G[ 8]*tmp[ 8];
            bridge[ 9] = G[ 9]*tmp[ 0] + G[10]*tmp[ 3] + G[11]*tmp[ 6];
            bridge[10] = G[ 9]*tmp[ 1] + G[10]*tmp[ 4] + G[11]*tmp[ 7];
            bridge[11] = G[ 9]*tmp[ 2] + G[10]*tmp[ 5] + G[11]*tmp[ 8];

            // Second transform filter data by G
            dataDst[ 0*FSTRIDE2X3+d1*C+d2] = bridge[ 0]*G[ 0] + bridge[ 1]*G[ 1] + bridge[ 2]*G[ 2];
            dataDst[ 1*FSTRIDE2X3+d1*C+d2] = bridge[ 0]*G[ 3] + bridge[ 1]*G[ 4] + bridge[ 2]*G[ 5];
            dataDst[ 2*FSTRIDE2X3+d1*C+d2] = bridge[ 0]*G[ 6] + bridge[ 1]*G[ 7] + bridge[ 2]*G[ 8];
            dataDst[ 3*FSTRIDE2X3+d1*C+d2] = bridge[ 0]*G[ 9] + bridge[ 1]*G[10] + bridge[ 2]*G[11];
            dataDst[ 4*FSTRIDE2X3+d1*C+d2] = bridge[ 3]*G[ 0] + bridge[ 4]*G[ 1] + bridge[ 5]*G[ 2];
            dataDst[ 5*FSTRIDE2X3+d1*C+d2] = bridge[ 3]*G[ 3] + bridge[ 4]*G[ 4] + bridge[ 5]*G[ 5];
            dataDst[ 6*FSTRIDE2X3+d1*C+d2] = bridge[ 3]*G[ 6] + bridge[ 4]*G[ 7] + bridge[ 5]*G[ 8];
            dataDst[ 7*FSTRIDE2X3+d1*C+d2] = bridge[ 3]*G[ 9] + bridge[ 4]*G[10] + bridge[ 5]*G[11];
            dataDst[ 8*FSTRIDE2X3+d1*C+d2] = bridge[ 6]*G[ 0] + bridge[ 7]*G[ 1] + bridge[ 8]*G[ 2];
            dataDst[ 9*FSTRIDE2X3+d1*C+d2] = bridge[ 6]*G[ 3] + bridge[ 7]*G[ 4] + bridge[ 8]*G[ 5];
            dataDst[10*FSTRIDE2X3+d1*C+d2] = bridge[ 6]*G[ 6] + bridge[ 7]*G[ 7] + bridge[ 8]*G[ 8];
            dataDst[11*FSTRIDE2X3+d1*C+d2] = bridge[ 6]*G[ 9] + bridge[ 7]*G[10] + bridge[ 8]*G[11];
            dataDst[12*FSTRIDE2X3+d1*C+d2] = bridge[ 9]*G[ 0] + bridge[10]*G[ 1] + bridge[11]*G[ 2];
            dataDst[13*FSTRIDE2X3+d1*C+d2] = bridge[ 9]*G[ 3] + bridge[10]*G[ 4] + bridge[11]*G[ 5];
            dataDst[14*FSTRIDE2X3+d1*C+d2] = bridge[ 9]*G[ 6] + bridge[10]*G[ 7] + bridge[11]*G[ 8];
            dataDst[15*FSTRIDE2X3+d1*C+d2] = bridge[ 9]*G[ 9] + bridge[10]*G[10] + bridge[11]*G[11];
        }
    }
}

/* Kernel compute for bridge data by sgemm.
 * Number of sgemm calls is 16*BATCH. 
 * */ 
    template<typename Dtype>
static void matrix_compute(const Dtype *in, const int irows, const int icols,
        const Dtype *filter, const int frows, const int fcols,
        Dtype *out,
        const int batch)
{

    /* In  - matrix A
     * Filter - matrix B
     * Output - matrix C
     * */
    int d1, d2; 
    const char trans ='n'; 
    const float alpha_f = 1.0; 
    const float beta_f =  0.0; 
    const double alpha_d = 1.0; 
    const double beta_d =  0.0; 
    const int ldi = irows;
    const int ldf = frows;
    const int ldo = irows;

#pragma omp parallel for collapse(2) private(d1, d2)
    for(d1 = 0; d1 < 16; d1++){
        for(d2 = 0; d2 < batch; d2++){
            const Dtype* pin = in+d1*ISTRIDE2X3+d2*irows*icols; 
            const Dtype* pft = filter+d1*FSTRIDE2X3; 
            Dtype* pot = out+d1*OSTRIDE2X3+d2*irows*fcols; 
            if(typeid(Dtype) == typeid(float))
                sgemm(&trans, &trans, &irows, &fcols, &icols, &alpha_f, 
                        (const float *)pin, &ldi, (const float *)pft, &ldf, &beta_f, (float *)pot, &ldo); 
            else if(typeid(Dtype) == typeid(double))
                dgemm(&trans, &trans, &irows, &fcols, &icols, &alpha_d, 
                        (const double *)pin, &ldi, (const double *)pft, &ldf, &beta_d, (double *)pot, &ldo); 
            else
                printf("Currently, only support float/double data!!!\n");
        }
    }
} 

/* Compute the bridge data for out, and transform to form matrix C. */
    template<typename Dtype>
static void outByTransform(Dtype *out, const Dtype *dataSrc,
        const int N, const int K, const int rows, const int cols,
        ACSATailMessage *tailMess, const int ntiles, const int mg2x3)
{
    int d1; 
    int sizeO = rows * cols;

    int rowSeg1, rowSeg2;
    int colSeg1, colSeg2;
    rowSeg1 = rows/2*2;
    rowSeg2 = rows - rowSeg1;
    colSeg1 = cols/2*2;
    colSeg2 = cols - colSeg1;

    ACSA_CHECK((rowSeg2 == tailMess->tail_h_));
    ACSA_CHECK((colSeg2 == tailMess->tail_w_)); 

#pragma omp parallel for private(d1)
    for(d1 = 0; d1 < N*K; d1++){
        int i, j;    
        Dtype tmp[16] __attribute__((aligned(64)));
        Dtype bridge[8] __attribute__((aligned(64))); 
        Dtype middle[4] __attribute__((aligned(64))); 

        const int t1 = d1/(K*mg2x3);
        const int t2 = (d1%(K*mg2x3))/mg2x3;
        const int t3 = d1%mg2x3;

        Dtype *dataDst = out + (t1*mg2x3*K + t3*K + t2)*sizeO;
        int tileCount = d1*ntiles;

        for(i = 0; i < rowSeg1; i += 2){
#pragma simd
            // Process no tail
            for(j = 0; j < colSeg1; j += 2){
                GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE2X3);
                // First inverse transfrom for output data by AT
                TRANS_AT_FST(AT, tmp, bridge);
                // Second inverse transfrom for output data by AT
                dataDst[(i+0)*cols + (j+0)] = bridge[ 0]*AT[ 0] + bridge[ 1]*AT[ 1] + bridge[ 2]*AT[ 2] + bridge[ 3]*AT[ 3];
                dataDst[(i+0)*cols + (j+1)] = bridge[ 0]*AT[ 4] + bridge[ 1]*AT[ 5] + bridge[ 2]*AT[ 6] + bridge[ 3]*AT[ 7];
                dataDst[(i+1)*cols + (j+0)] = bridge[ 4]*AT[ 0] + bridge[ 5]*AT[ 1] + bridge[ 6]*AT[ 2] + bridge[ 7]*AT[ 3];
                dataDst[(i+1)*cols + (j+1)] = bridge[ 4]*AT[ 4] + bridge[ 5]*AT[ 5] + bridge[ 6]*AT[ 6] + bridge[ 7]*AT[ 7];
                tileCount++; 
            }

            // Process col tail
            if(colSeg2 != 0){
                GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE2X3);
                // First inverse transfrom for output data by AT
                TRANS_AT_FST(AT, tmp, bridge);
                // Second inverse transfrom for output data by AT
                TRANS_AT_SED(bridge, AT, middle);

                ACSAGetFinalOutput(dataDst, middle, 2, 2, i, j, cols, 0, 0, 0, ZERO_LENGTH(colSeg2));
                tileCount++; 
            }
        }

        // Process row tail
        if(rowSeg2 == 1){
#pragma simd
            for(j = 0; j < colSeg1; j += 2){
                GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE2X3);
                // First inverse transfrom for output data by AT
                TRANS_AT_FST(AT, tmp, bridge);
                // Second inverse transfrom for output data by AT
                dataDst[(rowSeg1+0)*cols + (j+0)] = bridge[ 0]*AT[ 0] + bridge[ 1]*AT[ 1] + bridge[ 2]*AT[ 2] + bridge[ 3]*AT[ 3];
                dataDst[(rowSeg1+0)*cols + (j+1)] = bridge[ 0]*AT[ 4] + bridge[ 1]*AT[ 5] + bridge[ 2]*AT[ 6] + bridge[ 3]*AT[ 7];
                tileCount++; 
            }
        }

        // Process row&col tail
        if((rowSeg2 != 0) && (colSeg2 != 0)){
            GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE2X3);
            // First inverse transfrom for output data by AT
            TRANS_AT_FST(AT, tmp, bridge);
            // Second inverse transfrom for output data by AT
            TRANS_AT_SED(bridge, AT, middle);

            ACSAGetFinalOutput(dataDst, middle, 2, 2, rowSeg1, colSeg1, cols, 0, ZERO_LENGTH(rowSeg2), 0, ZERO_LENGTH(colSeg2));
            tileCount++; 
        }
    }
}

/* Process the data tail. */
static void tailPreProcess(ACSATensor4d *tensorOut, ACSATailMessage &tailMess, int &ntiles)
{
    int out_h = tensorOut->h_;
    int out_w = tensorOut->w_;
    int tail_h = out_h % 2;
    int tail_w = out_w % 2;

    tailMess.tail_h_ = tail_h;
    tailMess.tail_w_ = tail_w;

    if(tail_h == 0)
        ntiles = out_h/2;
    else
        ntiles = out_h/2 + 1;

    if(tail_w == 0)
        ntiles *= out_w/2;
    else
        ntiles *= out_w/2 +1;
}

/* API for winograd F(2,3). */
    template<typename Dtype>
ACSAStatus ACSAWinoConvolution_2x3(const Dtype *in, const Dtype *filter, Dtype *out,
        ACSATensor4d* tensorIn, ACSATensor4d* tensorFilter, ACSATensor4d* tensorOut,
        ACSAConvMessage* convMess, ACSAWinoMessage *winoMess)
{
    const int N = tensorIn->n_;
    const int C = tensorIn->c_;
    const int H = tensorIn->h_;
    const int W = tensorIn->w_;
    const int K = tensorFilter->n_;
    const int pad_h = convMess->pad_h_;
    const int pad_w = convMess->pad_w_;
    const int bb2x3 = winoMess->batch_block_;
    const int mg2x3 = winoMess->merge_;
    const int outHeight = tensorOut->h_; 
    const int outWidth = tensorOut->w_; 
    //const int ntiles = (outHeight)*0.5*(outWidth)*0.5; 
    int ntiles;
    ACSATailMessage tailMess;

    tailPreProcess(tensorOut, tailMess, ntiles);

    const Dtype *b_in;
    Dtype *b_out;

    int b_bts = bb2x3;
    if(b_bts == 0)
        b_bts = N;

    Dtype *wino_in = (Dtype *)winoIn;
    Dtype *wino_filter = (Dtype *)winoFilter; 
    Dtype *wino_out = (Dtype *)winoOut;

    // Check 
    ACSA_CHECK((N%b_bts == 0));
    ACSA_CHECK(((pad_h < 2) || (pad_w < 2)));

    /* Get the needed threads number. */
    int num_threads;
#pragma omp parallel
#pragma omp master
    {
        num_threads = omp_get_num_threads();
    }

    filterByTransform(filter, wino_filter, C, K);
    for(int i = 0; i < N; i += b_bts){
        b_in = in + i*C*H*W;
        b_out = out + i*K*outHeight*outWidth;
        if(pad_h == 0 && pad_w == 0)
            inByTransform_nopad(b_in, wino_in, b_bts, C, H, W, &tailMess, ntiles, mg2x3);
        else if(H*W > 1225)
            inByTransform_padBigScale(b_in, wino_in, b_bts, C, H, W, pad_h, pad_w, &tailMess, ntiles, mg2x3);
        else{
            Dtype *in_pad = (Dtype *)mkl_malloc(num_threads*(H+2*pad_h)*(W+2*pad_w)*sizeof(Dtype), 64);
            memset(in_pad, 0, num_threads*(H+2*pad_h)*(W+2*pad_w)*sizeof(Dtype));
            inByTransform_padSmallScale(b_in, in_pad, wino_in, N, C, H, W, pad_h, pad_w, &tailMess, ntiles, mg2x3);
            mkl_free(in_pad);
        }
        matrix_compute(wino_in, mg2x3*ntiles, C, wino_filter, C, K, wino_out, b_bts/mg2x3);
        outByTransform(b_out, wino_out, b_bts, K, outHeight, outWidth, &tailMess, ntiles, mg2x3);
    }

    return ACSASUCCESS;
}

/* Instantiate Template */
template void inByTransform_nopad<float>(const float *, float *,
        const int, const int, const int, const int,
        ACSATailMessage *, const int, const int);
template void inByTransform_padBigScale<float>(const float *, float *,
        const int, const int, const int, const int,
        const int, const int,
        ACSATailMessage *, const int, const int);
template void inByTransform_padSmallScale<float>(const float *, float *, float *,
        const int, const int, const int, const int,
        const int, const int,
        ACSATailMessage *, const int, const int);
template void filterByTransform<float>(const float *, float *,
        const int, const int);
template void matrix_compute<float>(const float *, const int, const int,
        const float *, const int, const int,
        float *, const int);
template void outByTransform<float>(float *, const float *,
        const int, const int, const int, const int,
        ACSATailMessage *, const int, const int);
template ACSAStatus ACSAWinoConvolution_2x3<float>(const float *, const float *, float *,
        ACSATensor4d*, ACSATensor4d*, ACSATensor4d*,
        ACSAConvMessage*, ACSAWinoMessage*);

template void inByTransform_nopad<double>(const double *, double *,
        const int, const int, const int, const int,
        ACSATailMessage *, const int, const int);
template void inByTransform_padBigScale<double>(const double *, double *,
        const int, const int, const int, const int,
        const int, const int,
        ACSATailMessage *, const int, const int);
template void inByTransform_padSmallScale<double>(const double *, double *,  double *,
        const int, const int, const int, const int,
        const int, const int,
        ACSATailMessage *, const int, const int);
template void filterByTransform<double>(const double *, double *,
        const int, const int);
template void matrix_compute<double>(const double *, const int, const int,
        const double *, const int, const int,
        double *, const int);
template void outByTransform<double>(double *, const double *,
        const int, const int, const int, const int,
        ACSATailMessage *, const int, const int);
template ACSAStatus ACSAWinoConvolution_2x3<double>(const double *, const double *, double *,
        ACSATensor4d*, ACSATensor4d*, ACSATensor4d*,
        ACSAConvMessage*, ACSAWinoMessage*);
