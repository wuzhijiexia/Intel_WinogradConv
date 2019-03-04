/* F(4x4,3x3) implementations for winograd. */

#include <immintrin.h>
#include "dnn.hpp"

#define ZERO_LENGTH(tail) (4-tail)%4

const long ISTRIDE4X3 = ISTRIDE/36*16;
const long FSTRIDE4X3 = FSTRIDE;
const long OSTRIDE4X3 = OSTRIDE/36*16;

/* AT-G-BT for F(4,3).
 * The dimensions for AT/G/BT.
 * dim-AT: F_M, TILE
 * dim-G : TILE , F_R
 * dim-BT: TILE , TILE
 */
const float AT[24] = {
    1, 1,  1, 1,  1, 0,
    0, 1, -1, 2, -2, 0,
    0, 1,  1, 4,  4, 0,
    0, 1, -1, 8, -8, 1
};

const float G[18] = {
    1.0/4,       0,      0,
    -1.0/6,  -1.0/6, -1.0/6,
    -1.0/6,   1.0/6, -1.0/6,
    1.0/24,  1.0/12,  1.0/6,
    1.0/24, -1.0/12,  1.0/6,
    0,       0,      1
};

const float BT[36] = {
    4,  0, -5,  0, 1, 0,
    0, -4, -4,  1, 1, 0,
    0,  4, -4, -1, 1, 0,
    0, -2, -1,  2, 1, 0,
    0,  2, -1, -2, 1, 0,
    0,  4,  0, -5, 0, 1
};

// Twice transform for input data to get tiles
#define TRANS_BT_FST(A, B, C) \
{ \
    C[ 0] = A[ 0]*B[ 0] + A[ 1]*B[ 6] + A[ 2]*B[12] + A[ 3]*B[18] + A[ 4]*B[24] + A[ 5]*B[30]; \
    C[ 1] = A[ 0]*B[ 1] + A[ 1]*B[ 7] + A[ 2]*B[13] + A[ 3]*B[19] + A[ 4]*B[25] + A[ 5]*B[31]; \
    C[ 2] = A[ 0]*B[ 2] + A[ 1]*B[ 8] + A[ 2]*B[14] + A[ 3]*B[20] + A[ 4]*B[26] + A[ 5]*B[32]; \
    C[ 3] = A[ 0]*B[ 3] + A[ 1]*B[ 9] + A[ 2]*B[15] + A[ 3]*B[21] + A[ 4]*B[27] + A[ 5]*B[33]; \
    C[ 4] = A[ 0]*B[ 4] + A[ 1]*B[10] + A[ 2]*B[16] + A[ 3]*B[22] + A[ 4]*B[28] + A[ 5]*B[34]; \
    C[ 5] = A[ 0]*B[ 5] + A[ 1]*B[11] + A[ 2]*B[17] + A[ 3]*B[23] + A[ 4]*B[29] + A[ 5]*B[35]; \
    C[ 6] = A[ 6]*B[ 0] + A[ 7]*B[ 6] + A[ 8]*B[12] + A[ 9]*B[18] + A[10]*B[24] + A[11]*B[30]; \
    C[ 7] = A[ 6]*B[ 1] + A[ 7]*B[ 7] + A[ 8]*B[13] + A[ 9]*B[19] + A[10]*B[25] + A[11]*B[31]; \
    C[ 8] = A[ 6]*B[ 2] + A[ 7]*B[ 8] + A[ 8]*B[14] + A[ 9]*B[20] + A[10]*B[26] + A[11]*B[32]; \
    C[ 9] = A[ 6]*B[ 3] + A[ 7]*B[ 9] + A[ 8]*B[15] + A[ 9]*B[21] + A[10]*B[27] + A[11]*B[33]; \
    C[10] = A[ 6]*B[ 4] + A[ 7]*B[10] + A[ 8]*B[16] + A[ 9]*B[22] + A[10]*B[28] + A[11]*B[34]; \
    C[11] = A[ 6]*B[ 5] + A[ 7]*B[11] + A[ 8]*B[17] + A[ 9]*B[23] + A[10]*B[29] + A[11]*B[35]; \
    C[12] = A[12]*B[ 0] + A[13]*B[ 6] + A[14]*B[12] + A[15]*B[18] + A[16]*B[24] + A[17]*B[30]; \
    C[13] = A[12]*B[ 1] + A[13]*B[ 7] + A[14]*B[13] + A[15]*B[19] + A[16]*B[25] + A[17]*B[31]; \
    C[14] = A[12]*B[ 2] + A[13]*B[ 8] + A[14]*B[14] + A[15]*B[20] + A[16]*B[26] + A[17]*B[32]; \
    C[15] = A[12]*B[ 3] + A[13]*B[ 9] + A[14]*B[15] + A[15]*B[21] + A[16]*B[27] + A[17]*B[33]; \
    C[16] = A[12]*B[ 4] + A[13]*B[10] + A[14]*B[16] + A[15]*B[22] + A[16]*B[28] + A[17]*B[34]; \
    C[17] = A[12]*B[ 5] + A[13]*B[11] + A[14]*B[17] + A[15]*B[23] + A[16]*B[29] + A[17]*B[35]; \
    C[18] = A[18]*B[ 0] + A[19]*B[ 6] + A[20]*B[12] + A[21]*B[18] + A[22]*B[24] + A[23]*B[30]; \
    C[19] = A[18]*B[ 1] + A[19]*B[ 7] + A[20]*B[13] + A[21]*B[19] + A[22]*B[25] + A[23]*B[31]; \
    C[20] = A[18]*B[ 2] + A[19]*B[ 8] + A[20]*B[14] + A[21]*B[20] + A[22]*B[26] + A[23]*B[32]; \
    C[21] = A[18]*B[ 3] + A[19]*B[ 9] + A[20]*B[15] + A[21]*B[21] + A[22]*B[27] + A[23]*B[33]; \
    C[22] = A[18]*B[ 4] + A[19]*B[10] + A[20]*B[16] + A[21]*B[22] + A[22]*B[28] + A[23]*B[34]; \
    C[23] = A[18]*B[ 5] + A[19]*B[11] + A[20]*B[17] + A[21]*B[23] + A[22]*B[29] + A[23]*B[35]; \
    C[24] = A[24]*B[ 0] + A[25]*B[ 6] + A[26]*B[12] + A[27]*B[18] + A[28]*B[24] + A[29]*B[30]; \
    C[25] = A[24]*B[ 1] + A[25]*B[ 7] + A[26]*B[13] + A[27]*B[19] + A[28]*B[25] + A[29]*B[31]; \
    C[26] = A[24]*B[ 2] + A[25]*B[ 8] + A[26]*B[14] + A[27]*B[20] + A[28]*B[26] + A[29]*B[32]; \
    C[27] = A[24]*B[ 3] + A[25]*B[ 9] + A[26]*B[15] + A[27]*B[21] + A[28]*B[27] + A[29]*B[33]; \
    C[28] = A[24]*B[ 4] + A[25]*B[10] + A[26]*B[16] + A[27]*B[22] + A[28]*B[28] + A[29]*B[34]; \
    C[29] = A[24]*B[ 5] + A[25]*B[11] + A[26]*B[17] + A[27]*B[23] + A[28]*B[29] + A[29]*B[35]; \
    C[30] = A[30]*B[ 0] + A[31]*B[ 6] + A[32]*B[12] + A[33]*B[18] + A[34]*B[24] + A[35]*B[30]; \
    C[31] = A[30]*B[ 1] + A[31]*B[ 7] + A[32]*B[13] + A[33]*B[19] + A[34]*B[25] + A[35]*B[31]; \
    C[32] = A[30]*B[ 2] + A[31]*B[ 8] + A[32]*B[14] + A[33]*B[20] + A[34]*B[26] + A[35]*B[32]; \
    C[33] = A[30]*B[ 3] + A[31]*B[ 9] + A[32]*B[15] + A[33]*B[21] + A[34]*B[27] + A[35]*B[33]; \
    C[34] = A[30]*B[ 4] + A[31]*B[10] + A[32]*B[16] + A[33]*B[22] + A[34]*B[28] + A[35]*B[34]; \
    C[35] = A[30]*B[ 5] + A[31]*B[11] + A[32]*B[17] + A[33]*B[23] + A[34]*B[29] + A[35]*B[35]; \
}

#define TRANS_BT_SED(A, B, C, TC, ST) \
{ \
    C[TC +  0*ST] = A[ 0]*B[ 0] + A[ 1]*B[ 1] + A[ 2]*B[ 2] + A[ 3]*B[ 3] + A[ 4]*B[ 4] + A[ 5]*B[ 5]; \
    C[TC +  1*ST] = A[ 0]*B[ 6] + A[ 1]*B[ 7] + A[ 2]*B[ 8] + A[ 3]*B[ 9] + A[ 4]*B[10] + A[ 5]*B[11]; \
    C[TC +  2*ST] = A[ 0]*B[12] + A[ 1]*B[13] + A[ 2]*B[14] + A[ 3]*B[15] + A[ 4]*B[16] + A[ 5]*B[17]; \
    C[TC +  3*ST] = A[ 0]*B[18] + A[ 1]*B[19] + A[ 2]*B[20] + A[ 3]*B[21] + A[ 4]*B[22] + A[ 5]*B[23]; \
    C[TC +  4*ST] = A[ 0]*B[24] + A[ 1]*B[25] + A[ 2]*B[26] + A[ 3]*B[27] + A[ 4]*B[28] + A[ 5]*B[29]; \
    C[TC +  5*ST] = A[ 0]*B[30] + A[ 1]*B[31] + A[ 2]*B[32] + A[ 3]*B[33] + A[ 4]*B[34] + A[ 5]*B[35]; \
    C[TC +  6*ST] = A[ 6]*B[ 0] + A[ 7]*B[ 1] + A[ 8]*B[ 2] + A[ 9]*B[ 3] + A[10]*B[ 4] + A[11]*B[ 5]; \
    C[TC +  7*ST] = A[ 6]*B[ 6] + A[ 7]*B[ 7] + A[ 8]*B[ 8] + A[ 9]*B[ 9] + A[10]*B[10] + A[11]*B[11]; \
    C[TC +  8*ST] = A[ 6]*B[12] + A[ 7]*B[13] + A[ 8]*B[14] + A[ 9]*B[15] + A[10]*B[16] + A[11]*B[17]; \
    C[TC +  9*ST] = A[ 6]*B[18] + A[ 7]*B[19] + A[ 8]*B[20] + A[ 9]*B[21] + A[10]*B[22] + A[11]*B[23]; \
    C[TC + 10*ST] = A[ 6]*B[24] + A[ 7]*B[25] + A[ 8]*B[26] + A[ 9]*B[27] + A[10]*B[28] + A[11]*B[29]; \
    C[TC + 11*ST] = A[ 6]*B[30] + A[ 7]*B[31] + A[ 8]*B[32] + A[ 9]*B[33] + A[10]*B[34] + A[11]*B[35]; \
    C[TC + 12*ST] = A[12]*B[ 0] + A[13]*B[ 1] + A[14]*B[ 2] + A[15]*B[ 3] + A[16]*B[ 4] + A[17]*B[ 5]; \
    C[TC + 13*ST] = A[12]*B[ 6] + A[13]*B[ 7] + A[14]*B[ 8] + A[15]*B[ 9] + A[16]*B[10] + A[17]*B[11]; \
    C[TC + 14*ST] = A[12]*B[12] + A[13]*B[13] + A[14]*B[14] + A[15]*B[15] + A[16]*B[16] + A[17]*B[17]; \
    C[TC + 15*ST] = A[12]*B[18] + A[13]*B[19] + A[14]*B[20] + A[15]*B[21] + A[16]*B[22] + A[17]*B[23]; \
    C[TC + 16*ST] = A[12]*B[24] + A[13]*B[25] + A[14]*B[26] + A[15]*B[27] + A[16]*B[28] + A[17]*B[29]; \
    C[TC + 17*ST] = A[12]*B[30] + A[13]*B[31] + A[14]*B[32] + A[15]*B[33] + A[16]*B[34] + A[17]*B[35]; \
    C[TC + 18*ST] = A[18]*B[ 0] + A[19]*B[ 1] + A[20]*B[ 2] + A[21]*B[ 3] + A[22]*B[ 4] + A[23]*B[ 5]; \
    C[TC + 19*ST] = A[18]*B[ 6] + A[19]*B[ 7] + A[20]*B[ 8] + A[21]*B[ 9] + A[22]*B[10] + A[23]*B[11]; \
    C[TC + 20*ST] = A[18]*B[12] + A[19]*B[13] + A[20]*B[14] + A[21]*B[15] + A[22]*B[16] + A[23]*B[17]; \
    C[TC + 21*ST] = A[18]*B[18] + A[19]*B[19] + A[20]*B[20] + A[21]*B[21] + A[22]*B[22] + A[23]*B[23]; \
    C[TC + 22*ST] = A[18]*B[24] + A[19]*B[25] + A[20]*B[26] + A[21]*B[27] + A[22]*B[28] + A[23]*B[29]; \
    C[TC + 23*ST] = A[18]*B[30] + A[19]*B[31] + A[20]*B[32] + A[21]*B[33] + A[22]*B[34] + A[23]*B[35]; \
    C[TC + 24*ST] = A[24]*B[ 0] + A[25]*B[ 1] + A[26]*B[ 2] + A[27]*B[ 3] + A[28]*B[ 4] + A[29]*B[ 5]; \
    C[TC + 25*ST] = A[24]*B[ 6] + A[25]*B[ 7] + A[26]*B[ 8] + A[27]*B[ 9] + A[28]*B[10] + A[29]*B[11]; \
    C[TC + 26*ST] = A[24]*B[12] + A[25]*B[13] + A[26]*B[14] + A[27]*B[15] + A[28]*B[16] + A[29]*B[17]; \
    C[TC + 27*ST] = A[24]*B[18] + A[25]*B[19] + A[26]*B[20] + A[27]*B[21] + A[28]*B[22] + A[29]*B[23]; \
    C[TC + 28*ST] = A[24]*B[24] + A[25]*B[25] + A[26]*B[26] + A[27]*B[27] + A[28]*B[28] + A[29]*B[29]; \
    C[TC + 29*ST] = A[24]*B[30] + A[25]*B[31] + A[26]*B[32] + A[27]*B[33] + A[28]*B[34] + A[29]*B[35]; \
    C[TC + 30*ST] = A[30]*B[ 0] + A[31]*B[ 1] + A[32]*B[ 2] + A[33]*B[ 3] + A[34]*B[ 4] + A[35]*B[ 5]; \
    C[TC + 31*ST] = A[30]*B[ 6] + A[31]*B[ 7] + A[32]*B[ 8] + A[33]*B[ 9] + A[34]*B[10] + A[35]*B[11]; \
    C[TC + 32*ST] = A[30]*B[12] + A[31]*B[13] + A[32]*B[14] + A[33]*B[15] + A[34]*B[16] + A[35]*B[17]; \
    C[TC + 33*ST] = A[30]*B[18] + A[31]*B[19] + A[32]*B[20] + A[33]*B[21] + A[34]*B[22] + A[35]*B[23]; \
    C[TC + 34*ST] = A[30]*B[24] + A[31]*B[25] + A[32]*B[26] + A[33]*B[27] + A[34]*B[28] + A[35]*B[29]; \
    C[TC + 35*ST] = A[30]*B[30] + A[31]*B[31] + A[32]*B[32] + A[33]*B[33] + A[34]*B[34] + A[35]*B[35]; \
}

// Get output tile & Twice transform
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
    TMP[16] = DATA[TC+ 16*ST]; \
    TMP[17] = DATA[TC+ 17*ST]; \
    TMP[18] = DATA[TC+ 18*ST]; \
    TMP[19] = DATA[TC+ 19*ST]; \
    TMP[20] = DATA[TC+ 20*ST]; \
    TMP[21] = DATA[TC+ 21*ST]; \
    TMP[22] = DATA[TC+ 22*ST]; \
    TMP[23] = DATA[TC+ 23*ST]; \
    TMP[24] = DATA[TC+ 24*ST]; \
    TMP[25] = DATA[TC+ 25*ST]; \
    TMP[26] = DATA[TC+ 26*ST]; \
    TMP[27] = DATA[TC+ 27*ST]; \
    TMP[28] = DATA[TC+ 28*ST]; \
    TMP[29] = DATA[TC+ 29*ST]; \
    TMP[30] = DATA[TC+ 30*ST]; \
    TMP[31] = DATA[TC+ 31*ST]; \
    TMP[32] = DATA[TC+ 32*ST]; \
    TMP[33] = DATA[TC+ 33*ST]; \
    TMP[34] = DATA[TC+ 34*ST]; \
    TMP[35] = DATA[TC+ 35*ST]; \
}

#define TRANS_AT_FST(A, B, C) \
{ \
    C[ 0] = A[ 0]*B[ 0] + A[ 1]*B[ 6] + A[ 2]*B[12] + A[ 3]*B[18] + A[ 4]*B[24] + A[ 5]*B[30]; \
    C[ 1] = A[ 0]*B[ 1] + A[ 1]*B[ 7] + A[ 2]*B[13] + A[ 3]*B[19] + A[ 4]*B[25] + A[ 5]*B[31]; \
    C[ 2] = A[ 0]*B[ 2] + A[ 1]*B[ 8] + A[ 2]*B[14] + A[ 3]*B[20] + A[ 4]*B[26] + A[ 5]*B[32]; \
    C[ 3] = A[ 0]*B[ 3] + A[ 1]*B[ 9] + A[ 2]*B[15] + A[ 3]*B[21] + A[ 4]*B[27] + A[ 5]*B[33]; \
    C[ 4] = A[ 0]*B[ 4] + A[ 1]*B[10] + A[ 2]*B[16] + A[ 3]*B[22] + A[ 4]*B[28] + A[ 5]*B[34]; \
    C[ 5] = A[ 0]*B[ 5] + A[ 1]*B[11] + A[ 2]*B[17] + A[ 3]*B[23] + A[ 4]*B[29] + A[ 5]*B[35]; \
    C[ 6] = A[ 6]*B[ 0] + A[ 7]*B[ 6] + A[ 8]*B[12] + A[ 9]*B[18] + A[10]*B[24] + A[11]*B[30]; \
    C[ 7] = A[ 6]*B[ 1] + A[ 7]*B[ 7] + A[ 8]*B[13] + A[ 9]*B[19] + A[10]*B[25] + A[11]*B[31]; \
    C[ 8] = A[ 6]*B[ 2] + A[ 7]*B[ 8] + A[ 8]*B[14] + A[ 9]*B[20] + A[10]*B[26] + A[11]*B[32]; \
    C[ 9] = A[ 6]*B[ 3] + A[ 7]*B[ 9] + A[ 8]*B[15] + A[ 9]*B[21] + A[10]*B[27] + A[11]*B[33]; \
    C[10] = A[ 6]*B[ 4] + A[ 7]*B[10] + A[ 8]*B[16] + A[ 9]*B[22] + A[10]*B[28] + A[11]*B[34]; \
    C[11] = A[ 6]*B[ 5] + A[ 7]*B[11] + A[ 8]*B[17] + A[ 9]*B[23] + A[10]*B[29] + A[11]*B[35]; \
    C[12] = A[12]*B[ 0] + A[13]*B[ 6] + A[14]*B[12] + A[15]*B[18] + A[16]*B[24] + A[17]*B[30]; \
    C[13] = A[12]*B[ 1] + A[13]*B[ 7] + A[14]*B[13] + A[15]*B[19] + A[16]*B[25] + A[17]*B[31]; \
    C[14] = A[12]*B[ 2] + A[13]*B[ 8] + A[14]*B[14] + A[15]*B[20] + A[16]*B[26] + A[17]*B[32]; \
    C[15] = A[12]*B[ 3] + A[13]*B[ 9] + A[14]*B[15] + A[15]*B[21] + A[16]*B[27] + A[17]*B[33]; \
    C[16] = A[12]*B[ 4] + A[13]*B[10] + A[14]*B[16] + A[15]*B[22] + A[16]*B[28] + A[17]*B[34]; \
    C[17] = A[12]*B[ 5] + A[13]*B[11] + A[14]*B[17] + A[15]*B[23] + A[16]*B[29] + A[17]*B[35]; \
    C[18] = A[18]*B[ 0] + A[19]*B[ 6] + A[20]*B[12] + A[21]*B[18] + A[22]*B[24] + A[23]*B[30]; \
    C[19] = A[18]*B[ 1] + A[19]*B[ 7] + A[20]*B[13] + A[21]*B[19] + A[22]*B[25] + A[23]*B[31]; \
    C[20] = A[18]*B[ 2] + A[19]*B[ 8] + A[20]*B[14] + A[21]*B[20] + A[22]*B[26] + A[23]*B[32]; \
    C[21] = A[18]*B[ 3] + A[19]*B[ 9] + A[20]*B[15] + A[21]*B[21] + A[22]*B[27] + A[23]*B[33]; \
    C[22] = A[18]*B[ 4] + A[19]*B[10] + A[20]*B[16] + A[21]*B[22] + A[22]*B[28] + A[23]*B[34]; \
    C[23] = A[18]*B[ 5] + A[19]*B[11] + A[20]*B[17] + A[21]*B[23] + A[22]*B[29] + A[23]*B[35]; \
}

#define TRANS_AT_SED(A, B, C) \
{ \
    C[ 0] = A[ 0]*B[ 0] + A[ 1]*B[ 1] + A[ 2]*B[ 2] + A[ 3]*B[ 3] + A[ 4]*B[ 4] + A[ 5]*B[ 5]; \
    C[ 1] = A[ 0]*B[ 6] + A[ 1]*B[ 7] + A[ 2]*B[ 8] + A[ 3]*B[ 9] + A[ 4]*B[10] + A[ 5]*B[11]; \
    C[ 2] = A[ 0]*B[12] + A[ 1]*B[13] + A[ 2]*B[14] + A[ 3]*B[15] + A[ 4]*B[16] + A[ 5]*B[17]; \
    C[ 3] = A[ 0]*B[18] + A[ 1]*B[19] + A[ 2]*B[20] + A[ 3]*B[21] + A[ 4]*B[22] + A[ 5]*B[23]; \
    C[ 4] = A[ 6]*B[ 0] + A[ 7]*B[ 1] + A[ 8]*B[ 2] + A[ 9]*B[ 3] + A[10]*B[ 4] + A[11]*B[ 5]; \
    C[ 5] = A[ 6]*B[ 6] + A[ 7]*B[ 7] + A[ 8]*B[ 8] + A[ 9]*B[ 9] + A[10]*B[10] + A[11]*B[11]; \
    C[ 6] = A[ 6]*B[12] + A[ 7]*B[13] + A[ 8]*B[14] + A[ 9]*B[15] + A[10]*B[16] + A[11]*B[17]; \
    C[ 7] = A[ 6]*B[18] + A[ 7]*B[19] + A[ 8]*B[20] + A[ 9]*B[21] + A[10]*B[22] + A[11]*B[23]; \
    C[ 8] = A[12]*B[ 0] + A[13]*B[ 1] + A[14]*B[ 2] + A[15]*B[ 3] + A[16]*B[ 4] + A[17]*B[ 5]; \
    C[ 9] = A[12]*B[ 6] + A[13]*B[ 7] + A[14]*B[ 8] + A[15]*B[ 9] + A[16]*B[10] + A[17]*B[11]; \
    C[10] = A[12]*B[12] + A[13]*B[13] + A[14]*B[14] + A[15]*B[15] + A[16]*B[16] + A[17]*B[17]; \
    C[11] = A[12]*B[18] + A[13]*B[19] + A[14]*B[20] + A[15]*B[21] + A[16]*B[22] + A[17]*B[23]; \
    C[12] = A[18]*B[ 0] + A[19]*B[ 1] + A[20]*B[ 2] + A[21]*B[ 3] + A[22]*B[ 4] + A[23]*B[ 5]; \
    C[13] = A[18]*B[ 6] + A[19]*B[ 7] + A[20]*B[ 8] + A[21]*B[ 9] + A[22]*B[10] + A[23]*B[11]; \
    C[14] = A[18]*B[12] + A[19]*B[13] + A[20]*B[14] + A[21]*B[15] + A[22]*B[16] + A[23]*B[17]; \
    C[15] = A[18]*B[18] + A[19]*B[19] + A[20]*B[20] + A[21]*B[21] + A[22]*B[22] + A[23]*B[23]; \
}

/* Don't use pad: 
 * compute the bridge data for in, and transform to form matrix A.
 * */
    template<typename Dtype>
static void inByTransform_nopad(const Dtype *in, Dtype *dataDst,
        const int N, const int C, const int rows, const int cols,
        ACSATailMessage *tailMess, const int ntiles, const int mg4x3)
{   
    int d1, d2;
    int sizeI = rows*cols;

    int rowSeg1, rowSeg2;
    int colSeg1, colSeg2;
    rowSeg1 = (rows-2)/4*4;
    rowSeg2 = rows-2 - rowSeg1;
    colSeg1 = (cols-2)/4*4;
    colSeg2 = cols-2 - colSeg1;

    ACSA_CHECK((rowSeg2 == tailMess->tail_h_));
    ACSA_CHECK((colSeg2 == tailMess->tail_w_));

#pragma omp parallel for private(d1) 
    for(d1 = 0; d1 < N*C; d1++){
        int i, j, k; 
        Dtype tmp[36] __attribute__((aligned(64)));
        Dtype bridge[36] __attribute__((aligned(64)));

        const int t1 = d1/(C*mg4x3);
        const int t2 = (d1%(C*mg4x3))/mg4x3;
        const int t3 = d1%mg4x3;

        // merge value influence the sequence of in data.
        const Dtype *data = in + (t1*mg4x3*C + t3*C + t2)*sizeI;
        int tileCount = d1*ntiles;

        for(i = 0; i < rowSeg1; i += 4){
#pragma simd
            // Process no tail
            for(j = 0; j < colSeg1; j += 4){
                tmp[0 ] = data[(i+0)*cols + (j+0)]; 
                tmp[1 ] = data[(i+0)*cols + (j+1)]; 
                tmp[2 ] = data[(i+0)*cols + (j+2)]; 
                tmp[3 ] = data[(i+0)*cols + (j+3)]; 
                tmp[4 ] = data[(i+0)*cols + (j+4)]; 
                tmp[5 ] = data[(i+0)*cols + (j+5)]; 

                tmp[6 ] = data[(i+1)*cols + (j+0)]; 
                tmp[7 ] = data[(i+1)*cols + (j+1)]; 
                tmp[8 ] = data[(i+1)*cols + (j+2)]; 
                tmp[9 ] = data[(i+1)*cols + (j+3)]; 
                tmp[10] = data[(i+1)*cols + (j+4)]; 
                tmp[11] = data[(i+1)*cols + (j+5)]; 

                tmp[12] = data[(i+2)*cols + (j+0)]; 
                tmp[13] = data[(i+2)*cols + (j+1)]; 
                tmp[14] = data[(i+2)*cols + (j+2)]; 
                tmp[15] = data[(i+2)*cols + (j+3)]; 
                tmp[16] = data[(i+2)*cols + (j+4)]; 
                tmp[17] = data[(i+2)*cols + (j+5)]; 

                tmp[18] = data[(i+3)*cols + (j+0)]; 
                tmp[19] = data[(i+3)*cols + (j+1)]; 
                tmp[20] = data[(i+3)*cols + (j+2)]; 
                tmp[21] = data[(i+3)*cols + (j+3)]; 
                tmp[22] = data[(i+3)*cols + (j+4)]; 
                tmp[23] = data[(i+3)*cols + (j+5)]; 

                tmp[24] = data[(i+4)*cols + (j+0)]; 
                tmp[25] = data[(i+4)*cols + (j+1)]; 
                tmp[26] = data[(i+4)*cols + (j+2)]; 
                tmp[27] = data[(i+4)*cols + (j+3)]; 
                tmp[28] = data[(i+4)*cols + (j+4)]; 
                tmp[29] = data[(i+4)*cols + (j+5)]; 

                tmp[30] = data[(i+5)*cols + (j+0)]; 
                tmp[31] = data[(i+5)*cols + (j+1)]; 
                tmp[32] = data[(i+5)*cols + (j+2)]; 
                tmp[33] = data[(i+5)*cols + (j+3)]; 
                tmp[34] = data[(i+5)*cols + (j+4)]; 
                tmp[35] = data[(i+5)*cols + (j+5)]; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

                tileCount++; 
            }

            // Process col tail
            if(colSeg2 != 0){
                ACSAGetInputTile(tmp, data, 6, 6, i, j, cols, 0, 0, 0, ZERO_LENGTH(colSeg2));

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
                tileCount++; 
            }
        }

        // Process row tail
        if(ZERO_LENGTH(rowSeg2) == 1){
#pragma simd
            for(j = 0; j <colSeg1; j += 4){
                tmp[0 ] = data[(rowSeg1+0)*cols + (j+0)]; 
                tmp[1 ] = data[(rowSeg1+0)*cols + (j+1)]; 
                tmp[2 ] = data[(rowSeg1+0)*cols + (j+2)]; 
                tmp[3 ] = data[(rowSeg1+0)*cols + (j+3)]; 
                tmp[4 ] = data[(rowSeg1+0)*cols + (j+4)]; 
                tmp[5 ] = data[(rowSeg1+0)*cols + (j+5)]; 

                tmp[6 ] = data[(rowSeg1+1)*cols + (j+0)]; 
                tmp[7 ] = data[(rowSeg1+1)*cols + (j+1)]; 
                tmp[8 ] = data[(rowSeg1+1)*cols + (j+2)]; 
                tmp[9 ] = data[(rowSeg1+1)*cols + (j+3)]; 
                tmp[10] = data[(rowSeg1+1)*cols + (j+4)]; 
                tmp[11] = data[(rowSeg1+1)*cols + (j+5)]; 

                tmp[12] = data[(rowSeg1+2)*cols + (j+0)]; 
                tmp[13] = data[(rowSeg1+2)*cols + (j+1)]; 
                tmp[14] = data[(rowSeg1+2)*cols + (j+2)]; 
                tmp[15] = data[(rowSeg1+2)*cols + (j+3)]; 
                tmp[16] = data[(rowSeg1+2)*cols + (j+4)]; 
                tmp[17] = data[(rowSeg1+2)*cols + (j+5)]; 

                tmp[18] = data[(rowSeg1+3)*cols + (j+0)]; 
                tmp[19] = data[(rowSeg1+3)*cols + (j+1)]; 
                tmp[20] = data[(rowSeg1+3)*cols + (j+2)]; 
                tmp[21] = data[(rowSeg1+3)*cols + (j+3)]; 
                tmp[22] = data[(rowSeg1+3)*cols + (j+4)]; 
                tmp[23] = data[(rowSeg1+3)*cols + (j+5)]; 

                tmp[24] = data[(rowSeg1+4)*cols + (j+0)]; 
                tmp[25] = data[(rowSeg1+4)*cols + (j+1)]; 
                tmp[26] = data[(rowSeg1+4)*cols + (j+2)]; 
                tmp[27] = data[(rowSeg1+4)*cols + (j+3)]; 
                tmp[28] = data[(rowSeg1+4)*cols + (j+4)]; 
                tmp[29] = data[(rowSeg1+4)*cols + (j+5)]; 

                tmp[30] = (Dtype)0.0; 
                tmp[31] = (Dtype)0.0; 
                tmp[32] = (Dtype)0.0; 
                tmp[33] = (Dtype)0.0; 
                tmp[34] = (Dtype)0.0; 
                tmp[35] = (Dtype)0.0; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

                tileCount++; 
            }
        }
        else if(ZERO_LENGTH(rowSeg2) == 2){
#pragma simd
            for(j = 0; j <colSeg1; j += 4){
                tmp[0 ] = data[(rowSeg1+0)*cols + (j+0)]; 
                tmp[1 ] = data[(rowSeg1+0)*cols + (j+1)]; 
                tmp[2 ] = data[(rowSeg1+0)*cols + (j+2)]; 
                tmp[3 ] = data[(rowSeg1+0)*cols + (j+3)]; 
                tmp[4 ] = data[(rowSeg1+0)*cols + (j+4)]; 
                tmp[5 ] = data[(rowSeg1+0)*cols + (j+5)]; 

                tmp[6 ] = data[(rowSeg1+1)*cols + (j+0)]; 
                tmp[7 ] = data[(rowSeg1+1)*cols + (j+1)]; 
                tmp[8 ] = data[(rowSeg1+1)*cols + (j+2)]; 
                tmp[9 ] = data[(rowSeg1+1)*cols + (j+3)]; 
                tmp[10] = data[(rowSeg1+1)*cols + (j+4)]; 
                tmp[11] = data[(rowSeg1+1)*cols + (j+5)]; 

                tmp[12] = data[(rowSeg1+2)*cols + (j+0)]; 
                tmp[13] = data[(rowSeg1+2)*cols + (j+1)]; 
                tmp[14] = data[(rowSeg1+2)*cols + (j+2)]; 
                tmp[15] = data[(rowSeg1+2)*cols + (j+3)]; 
                tmp[16] = data[(rowSeg1+2)*cols + (j+4)]; 
                tmp[17] = data[(rowSeg1+2)*cols + (j+5)]; 

                tmp[18] = data[(rowSeg1+3)*cols + (j+0)]; 
                tmp[19] = data[(rowSeg1+3)*cols + (j+1)]; 
                tmp[20] = data[(rowSeg1+3)*cols + (j+2)]; 
                tmp[21] = data[(rowSeg1+3)*cols + (j+3)]; 
                tmp[22] = data[(rowSeg1+3)*cols + (j+4)]; 
                tmp[23] = data[(rowSeg1+3)*cols + (j+5)]; 

                tmp[24] = (Dtype)0.0; 
                tmp[25] = (Dtype)0.0; 
                tmp[26] = (Dtype)0.0; 
                tmp[27] = (Dtype)0.0; 
                tmp[28] = (Dtype)0.0; 
                tmp[29] = (Dtype)0.0; 

                tmp[30] = (Dtype)0.0; 
                tmp[31] = (Dtype)0.0; 
                tmp[32] = (Dtype)0.0; 
                tmp[33] = (Dtype)0.0; 
                tmp[34] = (Dtype)0.0; 
                tmp[35] = (Dtype)0.0; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

                tileCount++; 
            }
        }
        else if(ZERO_LENGTH(rowSeg2) == 3){
#pragma simd
            for(j = 0; j <colSeg1; j += 4){
                tmp[0 ] = data[(rowSeg1+0)*cols + (j+0)]; 
                tmp[1 ] = data[(rowSeg1+0)*cols + (j+1)]; 
                tmp[2 ] = data[(rowSeg1+0)*cols + (j+2)]; 
                tmp[3 ] = data[(rowSeg1+0)*cols + (j+3)]; 
                tmp[4 ] = data[(rowSeg1+0)*cols + (j+4)]; 
                tmp[5 ] = data[(rowSeg1+0)*cols + (j+5)]; 

                tmp[6 ] = data[(rowSeg1+1)*cols + (j+0)]; 
                tmp[7 ] = data[(rowSeg1+1)*cols + (j+1)]; 
                tmp[8 ] = data[(rowSeg1+1)*cols + (j+2)]; 
                tmp[9 ] = data[(rowSeg1+1)*cols + (j+3)]; 
                tmp[10] = data[(rowSeg1+1)*cols + (j+4)]; 
                tmp[11] = data[(rowSeg1+1)*cols + (j+5)]; 

                tmp[12] = data[(rowSeg1+2)*cols + (j+0)]; 
                tmp[13] = data[(rowSeg1+2)*cols + (j+1)]; 
                tmp[14] = data[(rowSeg1+2)*cols + (j+2)]; 
                tmp[15] = data[(rowSeg1+2)*cols + (j+3)]; 
                tmp[16] = data[(rowSeg1+2)*cols + (j+4)]; 
                tmp[17] = data[(rowSeg1+2)*cols + (j+5)]; 

                tmp[18] = (Dtype)0.0; 
                tmp[19] = (Dtype)0.0; 
                tmp[20] = (Dtype)0.0; 
                tmp[21] = (Dtype)0.0; 
                tmp[22] = (Dtype)0.0; 
                tmp[23] = (Dtype)0.0; 

                tmp[24] = (Dtype)0.0; 
                tmp[25] = (Dtype)0.0; 
                tmp[26] = (Dtype)0.0; 
                tmp[27] = (Dtype)0.0; 
                tmp[28] = (Dtype)0.0; 
                tmp[29] = (Dtype)0.0; 

                tmp[30] = (Dtype)0.0; 
                tmp[31] = (Dtype)0.0; 
                tmp[32] = (Dtype)0.0; 
                tmp[33] = (Dtype)0.0; 
                tmp[34] = (Dtype)0.0; 
                tmp[35] = (Dtype)0.0; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

                tileCount++; 
            }
        }

        // Process row&col tail
        if((rowSeg2 != 0) && (colSeg2 != 0)){
            ACSAGetInputTile(tmp, data, 6, 6, rowSeg1, colSeg1, cols, 0, ZERO_LENGTH(rowSeg2), 0, ZERO_LENGTH(colSeg2));

            // The tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
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
        ACSATailMessage *tailMess, const int ntiles, const int mg4x3)
{   
    int d1, d2;
    int rows_pad = rows + 2*pad_h;
    int cols_pad = cols + 2*pad_w;
    int sizeI = rows*cols;

    int row_nTiles, col_nTiles;
    int tail_h = tailMess->tail_h_;
    int tail_w = tailMess->tail_w_;

    if(tail_h == 0)
        row_nTiles = (rows_pad-2)/4;
    else
        row_nTiles = (rows_pad-2)/4 + 1;

    if(tail_w == 0)
        col_nTiles = (cols_pad-2)/4;
    else
        col_nTiles = (cols_pad-2)/4 +1;

#pragma omp parallel for private(d1) 
    for(d1 = 0; d1 < N*C; d1++){
        int i, j; 
        Dtype tmp[36] __attribute__((aligned(64)));
        Dtype bridge[36] __attribute__((aligned(64)));

        const int t1 = d1/(C*mg4x3);
        const int t2 = (d1%(C*mg4x3))/mg4x3;
        const int t3 = d1%mg4x3;

        // merge value influence the sequence of in data.
        const Dtype *data = in + (t1*mg4x3*C + t3*C + t2)*sizeI;
        int tileCount = d1*ntiles;
        int baseTileCount = tileCount;

        /* The real data part. */
        int si = 3;
        int sj = 3;
        for(i = si; i < (rows-5); i += 4){
            tileCount = baseTileCount + ((i+pad_h)/4)*col_nTiles + 1;
#pragma simd
            // Center for first type region
            for(j = sj; j < (cols-5); j += 4){
                tmp[0 ] = data[(i+0)*cols + (j+0)]; 
                tmp[1 ] = data[(i+0)*cols + (j+1)]; 
                tmp[2 ] = data[(i+0)*cols + (j+2)]; 
                tmp[3 ] = data[(i+0)*cols + (j+3)]; 
                tmp[4 ] = data[(i+0)*cols + (j+4)]; 
                tmp[5 ] = data[(i+0)*cols + (j+5)]; 

                tmp[6 ] = data[(i+1)*cols + (j+0)]; 
                tmp[7 ] = data[(i+1)*cols + (j+1)]; 
                tmp[8 ] = data[(i+1)*cols + (j+2)]; 
                tmp[9 ] = data[(i+1)*cols + (j+3)]; 
                tmp[10] = data[(i+1)*cols + (j+4)]; 
                tmp[11] = data[(i+1)*cols + (j+5)]; 

                tmp[12] = data[(i+2)*cols + (j+0)]; 
                tmp[13] = data[(i+2)*cols + (j+1)]; 
                tmp[14] = data[(i+2)*cols + (j+2)]; 
                tmp[15] = data[(i+2)*cols + (j+3)]; 
                tmp[16] = data[(i+2)*cols + (j+4)]; 
                tmp[17] = data[(i+2)*cols + (j+5)]; 

                tmp[18] = data[(i+3)*cols + (j+0)]; 
                tmp[19] = data[(i+3)*cols + (j+1)]; 
                tmp[20] = data[(i+3)*cols + (j+2)]; 
                tmp[21] = data[(i+3)*cols + (j+3)]; 
                tmp[22] = data[(i+3)*cols + (j+4)]; 
                tmp[23] = data[(i+3)*cols + (j+5)]; 

                tmp[24] = data[(i+4)*cols + (j+0)]; 
                tmp[25] = data[(i+4)*cols + (j+1)]; 
                tmp[26] = data[(i+4)*cols + (j+2)]; 
                tmp[27] = data[(i+4)*cols + (j+3)]; 
                tmp[28] = data[(i+4)*cols + (j+4)]; 
                tmp[29] = data[(i+4)*cols + (j+5)]; 

                tmp[30] = data[(i+5)*cols + (j+0)]; 
                tmp[31] = data[(i+5)*cols + (j+1)]; 
                tmp[32] = data[(i+5)*cols + (j+2)]; 
                tmp[33] = data[(i+5)*cols + (j+3)]; 
                tmp[34] = data[(i+5)*cols + (j+4)]; 
                tmp[35] = data[(i+5)*cols + (j+5)]; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
                tileCount++; 
            }
        }

        /* The vitural data part for pad=1. */
        tmp[0] = tmp[1] = tmp[2] = tmp[3] = tmp[4] = tmp[5] = (Dtype)0.0;
        tileCount = baseTileCount + 1;
#pragma simd
        // Top for second region
        for(j = sj; j < (cols-5); j += 4){
            tmp[6 ] = data[0*cols + (j+0)];
            tmp[7 ] = data[0*cols + (j+1)];
            tmp[8 ] = data[0*cols + (j+2)];
            tmp[9 ] = data[0*cols + (j+3)];
            tmp[10] = data[0*cols + (j+4)];
            tmp[11] = data[0*cols + (j+5)];

            tmp[12] = data[1*cols + (j+0)];
            tmp[13] = data[1*cols + (j+1)];
            tmp[14] = data[1*cols + (j+2)];
            tmp[15] = data[1*cols + (j+3)];
            tmp[16] = data[1*cols + (j+4)];
            tmp[17] = data[1*cols + (j+5)];

            tmp[18] = data[2*cols + (j+0)];
            tmp[19] = data[2*cols + (j+1)];
            tmp[20] = data[2*cols + (j+2)];
            tmp[21] = data[2*cols + (j+3)];
            tmp[22] = data[2*cols + (j+4)];
            tmp[23] = data[2*cols + (j+5)];

            tmp[24] = data[3*cols + (j+0)];
            tmp[25] = data[3*cols + (j+1)];
            tmp[26] = data[3*cols + (j+2)];
            tmp[27] = data[3*cols + (j+3)];
            tmp[28] = data[3*cols + (j+4)];
            tmp[29] = data[3*cols + (j+5)];

            tmp[30] = data[4*cols + (j+0)];
            tmp[31] = data[4*cols + (j+1)];
            tmp[32] = data[4*cols + (j+2)];
            tmp[33] = data[4*cols + (j+3)];
            tmp[34] = data[4*cols + (j+4)];
            tmp[35] = data[4*cols + (j+5)];

            // Tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
            tileCount++;
        }

        // Bottom for second region
        tmp[30] = tmp[31] = tmp[32] = tmp[33] = tmp[34] = tmp[35] = (Dtype)0.0;
        tileCount = baseTileCount + 
            (row_nTiles - 1)*(col_nTiles) + 1;
        if(ZERO_LENGTH(tail_h) == 1){
            tmp[24] = tmp[25] = tmp[26] = tmp[27] = tmp[28] = tmp[29] = (Dtype)0.0;
#pragma simd
            for(j = sj; j < (cols-5); j += 4){
                tmp[0 ] = data[(rows-4)*cols  + (j+0)];
                tmp[1 ] = data[(rows-4)*cols  + (j+1)];
                tmp[2 ] = data[(rows-4)*cols  + (j+2)];
                tmp[3 ] = data[(rows-4)*cols  + (j+3)];
                tmp[4 ] = data[(rows-4)*cols  + (j+4)];
                tmp[5 ] = data[(rows-4)*cols  + (j+5)];

                tmp[6 ] = data[(rows-3)*cols  + (j+0)];
                tmp[7 ] = data[(rows-3)*cols  + (j+1)];
                tmp[8 ] = data[(rows-3)*cols  + (j+2)];
                tmp[9 ] = data[(rows-3)*cols  + (j+3)];
                tmp[10] = data[(rows-3)*cols  + (j+4)];
                tmp[11] = data[(rows-3)*cols  + (j+5)];

                tmp[12] = data[(rows-2)*cols  + (j+0)];
                tmp[13] = data[(rows-2)*cols  + (j+1)];
                tmp[14] = data[(rows-2)*cols  + (j+2)];
                tmp[15] = data[(rows-2)*cols  + (j+3)];
                tmp[16] = data[(rows-2)*cols  + (j+4)];
                tmp[17] = data[(rows-2)*cols  + (j+5)];

                tmp[18] = data[(rows-1)*cols  + (j+0)];
                tmp[19] = data[(rows-1)*cols  + (j+1)];
                tmp[20] = data[(rows-1)*cols  + (j+2)];
                tmp[21] = data[(rows-1)*cols  + (j+3)];
                tmp[22] = data[(rows-1)*cols  + (j+4)];
                tmp[23] = data[(rows-1)*cols  + (j+5)];

                // Tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
                tileCount++;
            }
        }
        else if(ZERO_LENGTH(tail_h) == 2){
            tmp[24] = tmp[25] = tmp[26] = tmp[27] = tmp[28] = tmp[29] = (Dtype)0.0;
            tmp[18] = tmp[19] = tmp[20] = tmp[21] = tmp[22] = tmp[23] = (Dtype)0.0;
#pragma simd
            for(j = sj; j < (cols-5); j += 4){
                tmp[0 ] = data[(rows-3)*cols  + (j+0)];
                tmp[1 ] = data[(rows-3)*cols  + (j+1)];
                tmp[2 ] = data[(rows-3)*cols  + (j+2)];
                tmp[3 ] = data[(rows-3)*cols  + (j+3)];
                tmp[4 ] = data[(rows-3)*cols  + (j+4)];
                tmp[5 ] = data[(rows-3)*cols  + (j+5)];

                tmp[6 ] = data[(rows-2)*cols  + (j+0)];
                tmp[7 ] = data[(rows-2)*cols  + (j+1)];
                tmp[8 ] = data[(rows-2)*cols  + (j+2)];
                tmp[9 ] = data[(rows-2)*cols  + (j+3)];
                tmp[10] = data[(rows-2)*cols  + (j+4)];
                tmp[11] = data[(rows-2)*cols  + (j+5)];

                tmp[12] = data[(rows-1)*cols  + (j+0)];
                tmp[13] = data[(rows-1)*cols  + (j+1)];
                tmp[14] = data[(rows-1)*cols  + (j+2)];
                tmp[15] = data[(rows-1)*cols  + (j+3)];
                tmp[16] = data[(rows-1)*cols  + (j+4)];
                tmp[17] = data[(rows-1)*cols  + (j+5)];

                // Tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
                tileCount++;
            }
        }
        else if(ZERO_LENGTH(tail_h) == 3){
            tmp[24] = tmp[25] = tmp[26] = tmp[27] = tmp[28] = tmp[29] = (Dtype)0.0;
            tmp[18] = tmp[19] = tmp[20] = tmp[21] = tmp[22] = tmp[23] = (Dtype)0.0;
            tmp[12] = tmp[13] = tmp[14] = tmp[15] = tmp[16] = tmp[17] = (Dtype)0.0;
#pragma simd
            for(j = sj; j < (cols-5); j += 4){
                tmp[0 ] = data[(rows-2)*cols  + (j+0)];
                tmp[1 ] = data[(rows-2)*cols  + (j+1)];
                tmp[2 ] = data[(rows-2)*cols  + (j+2)];
                tmp[3 ] = data[(rows-2)*cols  + (j+3)];
                tmp[4 ] = data[(rows-2)*cols  + (j+4)];
                tmp[5 ] = data[(rows-2)*cols  + (j+5)];

                tmp[6 ] = data[(rows-1)*cols  + (j+0)];
                tmp[7 ] = data[(rows-1)*cols  + (j+1)];
                tmp[8 ] = data[(rows-1)*cols  + (j+2)];
                tmp[9 ] = data[(rows-1)*cols  + (j+3)];
                tmp[10] = data[(rows-1)*cols  + (j+4)];
                tmp[11] = data[(rows-1)*cols  + (j+5)];

                // Tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
                tileCount++;
            }
        }
        else {
#pragma simd
            for(j = sj; j < (cols-5); j += 4){
                tmp[0 ] = data[(rows-5)*cols  + (j+0)];
                tmp[1 ] = data[(rows-5)*cols  + (j+1)];
                tmp[2 ] = data[(rows-5)*cols  + (j+2)];
                tmp[3 ] = data[(rows-5)*cols  + (j+3)];
                tmp[4 ] = data[(rows-5)*cols  + (j+4)];
                tmp[5 ] = data[(rows-5)*cols  + (j+5)];

                tmp[6 ] = data[(rows-4)*cols  + (j+0)];
                tmp[7 ] = data[(rows-4)*cols  + (j+1)];
                tmp[8 ] = data[(rows-4)*cols  + (j+2)];
                tmp[9 ] = data[(rows-4)*cols  + (j+3)];
                tmp[10] = data[(rows-4)*cols  + (j+4)];
                tmp[11] = data[(rows-4)*cols  + (j+5)];

                tmp[12] = data[(rows-3)*cols  + (j+0)];
                tmp[13] = data[(rows-3)*cols  + (j+1)];
                tmp[14] = data[(rows-3)*cols  + (j+2)];
                tmp[15] = data[(rows-3)*cols  + (j+3)];
                tmp[16] = data[(rows-3)*cols  + (j+4)];
                tmp[17] = data[(rows-3)*cols  + (j+5)];

                tmp[18] = data[(rows-2)*cols  + (j+0)];
                tmp[19] = data[(rows-2)*cols  + (j+1)];
                tmp[20] = data[(rows-2)*cols  + (j+2)];
                tmp[21] = data[(rows-2)*cols  + (j+3)];
                tmp[22] = data[(rows-2)*cols  + (j+4)];
                tmp[23] = data[(rows-2)*cols  + (j+5)];

                tmp[24] = data[(rows-1)*cols  + (j+0)];
                tmp[25] = data[(rows-1)*cols  + (j+1)];
                tmp[26] = data[(rows-1)*cols  + (j+2)];
                tmp[27] = data[(rows-1)*cols  + (j+3)];
                tmp[28] = data[(rows-1)*cols  + (j+4)];
                tmp[29] = data[(rows-1)*cols  + (j+5)];

                // Tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
                tileCount++;
            }
        }

        // Left for second region
        tileCount = baseTileCount + col_nTiles;
        for(i = si; i < (rows-5); i += 4){
            ACSAGetInputTile(tmp, data, 6, 6, i, 0, cols, 0, 0, 1, 0);

            // Tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
            tileCount += col_nTiles;
        }

        // Right for second region
        tileCount = baseTileCount + 2*(col_nTiles) -1;
        for(i = si; i < (rows-5); i += 4){
            ACSAGetInputTile(tmp, data, 6, 6, i, cols-5+ZERO_LENGTH(tail_w), cols, 0, 0, 0, ZERO_LENGTH(tail_w)+1);

            // Tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
            tileCount += col_nTiles;
        }

        // Top Left Corner for third region 
        tileCount = baseTileCount;
        ACSAGetInputTile(tmp, data, 6, 6, 0, 0, cols, 1, 0, 1, 0);
        TRANS_BT_FST(BT, tmp, bridge);
        TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

        // Top Right Corner for third region
        tileCount = baseTileCount + col_nTiles -1;
        ACSAGetInputTile(tmp, data, 6, 6, 0, cols-5+ZERO_LENGTH(tail_w), cols, 1, 0, 0, ZERO_LENGTH(tail_w)+1);
        TRANS_BT_FST(BT, tmp, bridge);
        TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

        // Bottom Left Corner for third region
        tileCount = baseTileCount + 
            (row_nTiles - 1)*(col_nTiles);
        ACSAGetInputTile(tmp, data, 6, 6, rows-5+ZERO_LENGTH(tail_h), 0, cols, 0, ZERO_LENGTH(tail_h)+1, 1, 0);
        TRANS_BT_FST(BT, tmp, bridge);
        TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

        // Bottom Right Corner for third region
        tileCount = baseTileCount + 
            (row_nTiles)*(col_nTiles) - 1;
        ACSAGetInputTile(tmp, data, 6, 6, rows-5+ZERO_LENGTH(tail_h), cols-5+ZERO_LENGTH(tail_w), cols, 0, ZERO_LENGTH(tail_h)+1, 0, ZERO_LENGTH(tail_w)+1);
        TRANS_BT_FST(BT, tmp, bridge);
        TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
    }
}

/* Use pad for small-scale in-data: 
 * compute the bridge data for in, and transform to form matrix A.
 * */
    template<typename Dtype>
static void inByTransform_padSmallScale(const Dtype *in, Dtype *in_pad, Dtype *dataDst,
        const int N, const int C, const int rows, const int cols,
        const int pad_h, const int pad_w,
        ACSATailMessage *tailMess, const int ntiles, const int mg4x3)
{   
    int d1, d2;
    int rows_pad = rows + 2*pad_h;
    int cols_pad = cols + 2*pad_w;
    int sizeI = rows*cols;

    int rowSeg1, rowSeg2;
    int colSeg1, colSeg2;
    rowSeg1 = (rows_pad-2)/4*4;
    rowSeg2 = rows_pad-2 - rowSeg1;
    colSeg1 = (cols_pad-2)/4*4;
    colSeg2 = cols_pad-2 - colSeg1;

    ACSA_CHECK((rowSeg2 == tailMess->tail_h_));
    ACSA_CHECK((colSeg2 == tailMess->tail_w_));

#pragma omp parallel for private(d1) 
    for(d1 = 0; d1 < N*C; d1++){
        int i, j, k; 
        Dtype tmp[36] __attribute__((aligned(64)));
        Dtype bridge[36] __attribute__((aligned(64)));

        const int t1 = d1/(C*mg4x3);
        const int t2 = (d1%(C*mg4x3))/mg4x3;
        const int t3 = d1%mg4x3;

        // merge value influence the sequence of in data.
        const Dtype *data = in + (t1*mg4x3*C + t3*C + t2)*sizeI;
        int tileCount = d1*ntiles;

        // Pad the in-data which will be processed.
        int index_threads = omp_get_thread_num();
        Dtype *data_pad = in_pad + index_threads*(rows_pad)*(cols_pad);

        for(i = 0; i < rows; i++)
#pragma simd
            for(j = 0; j < cols; j++)
                data_pad[(i+1)*(cols_pad) + (j+1)] = data[i*cols+j];

        for(i = 0; i < rowSeg1; i += 4){
#pragma simd
            // Process no tail
            for(j = 0; j < colSeg1; j += 4){
                tmp[0 ] = data_pad[(i+0)*cols_pad + (j+0)]; 
                tmp[1 ] = data_pad[(i+0)*cols_pad + (j+1)]; 
                tmp[2 ] = data_pad[(i+0)*cols_pad + (j+2)]; 
                tmp[3 ] = data_pad[(i+0)*cols_pad + (j+3)]; 
                tmp[4 ] = data_pad[(i+0)*cols_pad + (j+4)]; 
                tmp[5 ] = data_pad[(i+0)*cols_pad + (j+5)]; 

                tmp[6 ] = data_pad[(i+1)*cols_pad + (j+0)]; 
                tmp[7 ] = data_pad[(i+1)*cols_pad + (j+1)]; 
                tmp[8 ] = data_pad[(i+1)*cols_pad + (j+2)]; 
                tmp[9 ] = data_pad[(i+1)*cols_pad + (j+3)]; 
                tmp[10] = data_pad[(i+1)*cols_pad + (j+4)]; 
                tmp[11] = data_pad[(i+1)*cols_pad + (j+5)]; 

                tmp[12] = data_pad[(i+2)*cols_pad + (j+0)]; 
                tmp[13] = data_pad[(i+2)*cols_pad + (j+1)]; 
                tmp[14] = data_pad[(i+2)*cols_pad + (j+2)]; 
                tmp[15] = data_pad[(i+2)*cols_pad + (j+3)]; 
                tmp[16] = data_pad[(i+2)*cols_pad + (j+4)]; 
                tmp[17] = data_pad[(i+2)*cols_pad + (j+5)]; 

                tmp[18] = data_pad[(i+3)*cols_pad + (j+0)]; 
                tmp[19] = data_pad[(i+3)*cols_pad + (j+1)]; 
                tmp[20] = data_pad[(i+3)*cols_pad + (j+2)]; 
                tmp[21] = data_pad[(i+3)*cols_pad + (j+3)]; 
                tmp[22] = data_pad[(i+3)*cols_pad + (j+4)]; 
                tmp[23] = data_pad[(i+3)*cols_pad + (j+5)]; 

                tmp[24] = data_pad[(i+4)*cols_pad + (j+0)]; 
                tmp[25] = data_pad[(i+4)*cols_pad + (j+1)]; 
                tmp[26] = data_pad[(i+4)*cols_pad + (j+2)]; 
                tmp[27] = data_pad[(i+4)*cols_pad + (j+3)]; 
                tmp[28] = data_pad[(i+4)*cols_pad + (j+4)]; 
                tmp[29] = data_pad[(i+4)*cols_pad + (j+5)]; 

                tmp[30] = data_pad[(i+5)*cols_pad + (j+0)]; 
                tmp[31] = data_pad[(i+5)*cols_pad + (j+1)]; 
                tmp[32] = data_pad[(i+5)*cols_pad + (j+2)]; 
                tmp[33] = data_pad[(i+5)*cols_pad + (j+3)]; 
                tmp[34] = data_pad[(i+5)*cols_pad + (j+4)]; 
                tmp[35] = data_pad[(i+5)*cols_pad + (j+5)]; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

                tileCount++; 
            }

            // Process col tail
            if(colSeg2 != 0){
                ACSAGetInputTile(tmp, data_pad, 6, 6, i, j, cols_pad, 0, 0, 0,ZERO_LENGTH(colSeg2));

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
                tileCount++; 
            }
        }

        // Process row tail
        if(ZERO_LENGTH(rowSeg2) == 1){
#pragma simd
            for(j = 0; j < colSeg1; j += 4){
                tmp[0 ] = data_pad[(rowSeg1+0)*cols_pad + (j+0)]; 
                tmp[1 ] = data_pad[(rowSeg1+0)*cols_pad + (j+1)]; 
                tmp[2 ] = data_pad[(rowSeg1+0)*cols_pad + (j+2)]; 
                tmp[3 ] = data_pad[(rowSeg1+0)*cols_pad + (j+3)]; 
                tmp[4 ] = data_pad[(rowSeg1+0)*cols_pad + (j+4)]; 
                tmp[5 ] = data_pad[(rowSeg1+0)*cols_pad + (j+5)]; 

                tmp[6 ] = data_pad[(rowSeg1+1)*cols_pad + (j+0)]; 
                tmp[7 ] = data_pad[(rowSeg1+1)*cols_pad + (j+1)]; 
                tmp[8 ] = data_pad[(rowSeg1+1)*cols_pad + (j+2)]; 
                tmp[9 ] = data_pad[(rowSeg1+1)*cols_pad + (j+3)]; 
                tmp[10] = data_pad[(rowSeg1+1)*cols_pad + (j+4)]; 
                tmp[11] = data_pad[(rowSeg1+1)*cols_pad + (j+5)]; 

                tmp[12] = data_pad[(rowSeg1+2)*cols_pad + (j+0)]; 
                tmp[13] = data_pad[(rowSeg1+2)*cols_pad + (j+1)]; 
                tmp[14] = data_pad[(rowSeg1+2)*cols_pad + (j+2)]; 
                tmp[15] = data_pad[(rowSeg1+2)*cols_pad + (j+3)]; 
                tmp[16] = data_pad[(rowSeg1+2)*cols_pad + (j+4)]; 
                tmp[17] = data_pad[(rowSeg1+2)*cols_pad + (j+5)]; 

                tmp[18] = data_pad[(rowSeg1+3)*cols_pad + (j+0)]; 
                tmp[19] = data_pad[(rowSeg1+3)*cols_pad + (j+1)]; 
                tmp[20] = data_pad[(rowSeg1+3)*cols_pad + (j+2)]; 
                tmp[21] = data_pad[(rowSeg1+3)*cols_pad + (j+3)]; 
                tmp[22] = data_pad[(rowSeg1+3)*cols_pad + (j+4)]; 
                tmp[23] = data_pad[(rowSeg1+3)*cols_pad + (j+5)]; 

                tmp[24] = data_pad[(rowSeg1+4)*cols_pad + (j+0)]; 
                tmp[25] = data_pad[(rowSeg1+4)*cols_pad + (j+1)]; 
                tmp[26] = data_pad[(rowSeg1+4)*cols_pad + (j+2)]; 
                tmp[27] = data_pad[(rowSeg1+4)*cols_pad + (j+3)]; 
                tmp[28] = data_pad[(rowSeg1+4)*cols_pad + (j+4)]; 
                tmp[29] = data_pad[(rowSeg1+4)*cols_pad + (j+5)]; 

                tmp[30] = (Dtype)0.0; 
                tmp[31] = (Dtype)0.0; 
                tmp[32] = (Dtype)0.0; 
                tmp[33] = (Dtype)0.0; 
                tmp[34] = (Dtype)0.0; 
                tmp[35] = (Dtype)0.0; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

                tileCount++; 
            }
        }
        else if(ZERO_LENGTH(rowSeg2) == 2){
#pragma simd
            for(j = 0; j < colSeg1; j += 4){
                tmp[0 ] = data_pad[(rowSeg1+0)*cols_pad + (j+0)]; 
                tmp[1 ] = data_pad[(rowSeg1+0)*cols_pad + (j+1)]; 
                tmp[2 ] = data_pad[(rowSeg1+0)*cols_pad + (j+2)]; 
                tmp[3 ] = data_pad[(rowSeg1+0)*cols_pad + (j+3)]; 
                tmp[4 ] = data_pad[(rowSeg1+0)*cols_pad + (j+4)]; 
                tmp[5 ] = data_pad[(rowSeg1+0)*cols_pad + (j+5)]; 

                tmp[6 ] = data_pad[(rowSeg1+1)*cols_pad + (j+0)]; 
                tmp[7 ] = data_pad[(rowSeg1+1)*cols_pad + (j+1)]; 
                tmp[8 ] = data_pad[(rowSeg1+1)*cols_pad + (j+2)]; 
                tmp[9 ] = data_pad[(rowSeg1+1)*cols_pad + (j+3)]; 
                tmp[10] = data_pad[(rowSeg1+1)*cols_pad + (j+4)]; 
                tmp[11] = data_pad[(rowSeg1+1)*cols_pad + (j+5)]; 

                tmp[12] = data_pad[(rowSeg1+2)*cols_pad + (j+0)]; 
                tmp[13] = data_pad[(rowSeg1+2)*cols_pad + (j+1)]; 
                tmp[14] = data_pad[(rowSeg1+2)*cols_pad + (j+2)]; 
                tmp[15] = data_pad[(rowSeg1+2)*cols_pad + (j+3)]; 
                tmp[16] = data_pad[(rowSeg1+2)*cols_pad + (j+4)]; 
                tmp[17] = data_pad[(rowSeg1+2)*cols_pad + (j+5)]; 

                tmp[18] = data_pad[(rowSeg1+3)*cols_pad + (j+0)]; 
                tmp[19] = data_pad[(rowSeg1+3)*cols_pad + (j+1)]; 
                tmp[20] = data_pad[(rowSeg1+3)*cols_pad + (j+2)]; 
                tmp[21] = data_pad[(rowSeg1+3)*cols_pad + (j+3)]; 
                tmp[22] = data_pad[(rowSeg1+3)*cols_pad + (j+4)]; 
                tmp[23] = data_pad[(rowSeg1+3)*cols_pad + (j+5)]; 

                tmp[24] = (Dtype)0.0; 
                tmp[25] = (Dtype)0.0; 
                tmp[26] = (Dtype)0.0; 
                tmp[27] = (Dtype)0.0; 
                tmp[28] = (Dtype)0.0; 
                tmp[29] = (Dtype)0.0; 

                tmp[30] = (Dtype)0.0; 
                tmp[31] = (Dtype)0.0; 
                tmp[32] = (Dtype)0.0; 
                tmp[33] = (Dtype)0.0; 
                tmp[34] = (Dtype)0.0; 
                tmp[35] = (Dtype)0.0; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

                tileCount++; 
            }
        }
        else if(ZERO_LENGTH(rowSeg2) == 3){
#pragma simd
            for(j = 0; j < colSeg1; j += 4){
                tmp[0 ] = data_pad[(rowSeg1+0)*cols_pad + (j+0)]; 
                tmp[1 ] = data_pad[(rowSeg1+0)*cols_pad + (j+1)]; 
                tmp[2 ] = data_pad[(rowSeg1+0)*cols_pad + (j+2)]; 
                tmp[3 ] = data_pad[(rowSeg1+0)*cols_pad + (j+3)]; 
                tmp[4 ] = data_pad[(rowSeg1+0)*cols_pad + (j+4)]; 
                tmp[5 ] = data_pad[(rowSeg1+0)*cols_pad + (j+5)]; 

                tmp[6 ] = data_pad[(rowSeg1+1)*cols_pad + (j+0)]; 
                tmp[7 ] = data_pad[(rowSeg1+1)*cols_pad + (j+1)]; 
                tmp[8 ] = data_pad[(rowSeg1+1)*cols_pad + (j+2)]; 
                tmp[9 ] = data_pad[(rowSeg1+1)*cols_pad + (j+3)]; 
                tmp[10] = data_pad[(rowSeg1+1)*cols_pad + (j+4)]; 
                tmp[11] = data_pad[(rowSeg1+1)*cols_pad + (j+5)]; 

                tmp[12] = data_pad[(rowSeg1+2)*cols_pad + (j+0)]; 
                tmp[13] = data_pad[(rowSeg1+2)*cols_pad + (j+1)]; 
                tmp[14] = data_pad[(rowSeg1+2)*cols_pad + (j+2)]; 
                tmp[15] = data_pad[(rowSeg1+2)*cols_pad + (j+3)]; 
                tmp[16] = data_pad[(rowSeg1+2)*cols_pad + (j+4)]; 
                tmp[17] = data_pad[(rowSeg1+2)*cols_pad + (j+5)]; 

                tmp[18] = (Dtype)0.0; 
                tmp[19] = (Dtype)0.0; 
                tmp[20] = (Dtype)0.0; 
                tmp[21] = (Dtype)0.0; 
                tmp[22] = (Dtype)0.0; 
                tmp[23] = (Dtype)0.0; 

                tmp[24] = (Dtype)0.0; 
                tmp[25] = (Dtype)0.0; 
                tmp[26] = (Dtype)0.0; 
                tmp[27] = (Dtype)0.0; 
                tmp[28] = (Dtype)0.0; 
                tmp[29] = (Dtype)0.0; 

                tmp[30] = (Dtype)0.0; 
                tmp[31] = (Dtype)0.0; 
                tmp[32] = (Dtype)0.0; 
                tmp[33] = (Dtype)0.0; 
                tmp[34] = (Dtype)0.0; 
                tmp[35] = (Dtype)0.0; 

                // The tranformation manually simplified
                TRANS_BT_FST(BT, tmp, bridge);
                TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);

                tileCount++; 
            }
        }

        // Process row&col tail
        if((rowSeg2 != 0) && (colSeg2 != 0)){
            ACSAGetInputTile(tmp, data_pad, 6, 6, rowSeg1, colSeg1, cols_pad, 0, ZERO_LENGTH(rowSeg2), 0, ZERO_LENGTH(colSeg2));

            // The tranformation manually simplified
            TRANS_BT_FST(BT, tmp, bridge);
            TRANS_BT_SED(bridge, BT, dataDst, tileCount, ISTRIDE4X3);
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
            Dtype ddt[36] __attribute__((aligned(64))); 
            Dtype bridge[18] __attribute__((aligned(64))); 

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
            bridge[12] = G[12]*tmp[ 0] + G[13]*tmp[ 3] + G[14]*tmp[ 6];
            bridge[13] = G[12]*tmp[ 1] + G[13]*tmp[ 4] + G[14]*tmp[ 7];
            bridge[14] = G[12]*tmp[ 2] + G[13]*tmp[ 5] + G[14]*tmp[ 8];
            bridge[15] = G[15]*tmp[ 0] + G[16]*tmp[ 3] + G[17]*tmp[ 6];
            bridge[16] = G[15]*tmp[ 1] + G[16]*tmp[ 4] + G[17]*tmp[ 7];
            bridge[17] = G[15]*tmp[ 2] + G[16]*tmp[ 5] + G[17]*tmp[ 8];

            // Second transform filter data by G
            dataDst[ 0*FSTRIDE4X3+d1*C+d2] = bridge[ 0]*G[ 0] + bridge[ 1]*G[ 1] + bridge[ 2]*G[ 2];
            dataDst[ 1*FSTRIDE4X3+d1*C+d2] = bridge[ 0]*G[ 3] + bridge[ 1]*G[ 4] + bridge[ 2]*G[ 5];
            dataDst[ 2*FSTRIDE4X3+d1*C+d2] = bridge[ 0]*G[ 6] + bridge[ 1]*G[ 7] + bridge[ 2]*G[ 8];
            dataDst[ 3*FSTRIDE4X3+d1*C+d2] = bridge[ 0]*G[ 9] + bridge[ 1]*G[10] + bridge[ 2]*G[11];
            dataDst[ 4*FSTRIDE4X3+d1*C+d2] = bridge[ 0]*G[12] + bridge[ 1]*G[13] + bridge[ 2]*G[14];
            dataDst[ 5*FSTRIDE4X3+d1*C+d2] = bridge[ 0]*G[15] + bridge[ 1]*G[16] + bridge[ 2]*G[17];
            dataDst[ 6*FSTRIDE4X3+d1*C+d2] = bridge[ 3]*G[ 0] + bridge[ 4]*G[ 1] + bridge[ 5]*G[ 2];
            dataDst[ 7*FSTRIDE4X3+d1*C+d2] = bridge[ 3]*G[ 3] + bridge[ 4]*G[ 4] + bridge[ 5]*G[ 5];
            dataDst[ 8*FSTRIDE4X3+d1*C+d2] = bridge[ 3]*G[ 6] + bridge[ 4]*G[ 7] + bridge[ 5]*G[ 8];
            dataDst[ 9*FSTRIDE4X3+d1*C+d2] = bridge[ 3]*G[ 9] + bridge[ 4]*G[10] + bridge[ 5]*G[11];
            dataDst[10*FSTRIDE4X3+d1*C+d2] = bridge[ 3]*G[12] + bridge[ 4]*G[13] + bridge[ 5]*G[14];
            dataDst[11*FSTRIDE4X3+d1*C+d2] = bridge[ 3]*G[15] + bridge[ 4]*G[16] + bridge[ 5]*G[17];
            dataDst[12*FSTRIDE4X3+d1*C+d2] = bridge[ 6]*G[ 0] + bridge[ 7]*G[ 1] + bridge[ 8]*G[ 2];
            dataDst[13*FSTRIDE4X3+d1*C+d2] = bridge[ 6]*G[ 3] + bridge[ 7]*G[ 4] + bridge[ 8]*G[ 5];
            dataDst[14*FSTRIDE4X3+d1*C+d2] = bridge[ 6]*G[ 6] + bridge[ 7]*G[ 7] + bridge[ 8]*G[ 8];
            dataDst[15*FSTRIDE4X3+d1*C+d2] = bridge[ 6]*G[ 9] + bridge[ 7]*G[10] + bridge[ 8]*G[11];
            dataDst[16*FSTRIDE4X3+d1*C+d2] = bridge[ 6]*G[12] + bridge[ 7]*G[13] + bridge[ 8]*G[14];
            dataDst[17*FSTRIDE4X3+d1*C+d2] = bridge[ 6]*G[15] + bridge[ 7]*G[16] + bridge[ 8]*G[17];
            dataDst[18*FSTRIDE4X3+d1*C+d2] = bridge[ 9]*G[ 0] + bridge[10]*G[ 1] + bridge[11]*G[ 2];
            dataDst[19*FSTRIDE4X3+d1*C+d2] = bridge[ 9]*G[ 3] + bridge[10]*G[ 4] + bridge[11]*G[ 5];
            dataDst[20*FSTRIDE4X3+d1*C+d2] = bridge[ 9]*G[ 6] + bridge[10]*G[ 7] + bridge[11]*G[ 8];
            dataDst[21*FSTRIDE4X3+d1*C+d2] = bridge[ 9]*G[ 9] + bridge[10]*G[10] + bridge[11]*G[11];
            dataDst[22*FSTRIDE4X3+d1*C+d2] = bridge[ 9]*G[12] + bridge[10]*G[13] + bridge[11]*G[14];
            dataDst[23*FSTRIDE4X3+d1*C+d2] = bridge[ 9]*G[15] + bridge[10]*G[16] + bridge[11]*G[17];
            dataDst[24*FSTRIDE4X3+d1*C+d2] = bridge[12]*G[ 0] + bridge[13]*G[ 1] + bridge[14]*G[ 2];
            dataDst[25*FSTRIDE4X3+d1*C+d2] = bridge[12]*G[ 3] + bridge[13]*G[ 4] + bridge[14]*G[ 5];
            dataDst[26*FSTRIDE4X3+d1*C+d2] = bridge[12]*G[ 6] + bridge[13]*G[ 7] + bridge[14]*G[ 8];
            dataDst[27*FSTRIDE4X3+d1*C+d2] = bridge[12]*G[ 9] + bridge[13]*G[10] + bridge[14]*G[11];
            dataDst[28*FSTRIDE4X3+d1*C+d2] = bridge[12]*G[12] + bridge[13]*G[13] + bridge[14]*G[14];
            dataDst[29*FSTRIDE4X3+d1*C+d2] = bridge[12]*G[15] + bridge[13]*G[16] + bridge[14]*G[17];
            dataDst[30*FSTRIDE4X3+d1*C+d2] = bridge[15]*G[ 0] + bridge[16]*G[ 1] + bridge[17]*G[ 2];
            dataDst[31*FSTRIDE4X3+d1*C+d2] = bridge[15]*G[ 3] + bridge[16]*G[ 4] + bridge[17]*G[ 5];
            dataDst[32*FSTRIDE4X3+d1*C+d2] = bridge[15]*G[ 6] + bridge[16]*G[ 7] + bridge[17]*G[ 8];
            dataDst[33*FSTRIDE4X3+d1*C+d2] = bridge[15]*G[ 9] + bridge[16]*G[10] + bridge[17]*G[11];
            dataDst[34*FSTRIDE4X3+d1*C+d2] = bridge[15]*G[12] + bridge[16]*G[13] + bridge[17]*G[14];
            dataDst[35*FSTRIDE4X3+d1*C+d2] = bridge[15]*G[15] + bridge[16]*G[16] + bridge[17]*G[17];
        }
    }
}

/* Kernel compute for bridge data by sgemm.
 * Number of sgemm calls is 36*BATCH. 
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
    for(d1 = 0; d1 < 36; d1++){
        for(d2 = 0; d2 < batch; d2++){
            const Dtype* pin = in+d1*ISTRIDE4X3+d2*irows*icols; 
            const Dtype* pft = filter+d1*FSTRIDE4X3; 
            Dtype* pot = out+d1*OSTRIDE4X3+d2*irows*fcols; 
            if(typeid(Dtype) == typeid(float))
                sgemm(&trans, &trans, &irows, &fcols, &icols, &alpha_f, 
                        (const float *)pin, &ldi, (const float *)pft, &ldf, &beta_f, (float *)pot, &ldo); 
            else if(typeid(Dtype) == typeid(double))
                dgemm(&trans, &trans, &irows, &fcols, &icols, &alpha_d, 
                        (const double *)pin, &ldi, (const double *)pft, &ldf, &beta_d, (double *)pot, &ldo); 
        }
    }
}

/* Compute the bridge data for out, and transform to form matrix C. */
    template<typename Dtype>
static void outByTransform(Dtype *out, const Dtype *dataSrc,
        const int N, const int K, const int rows, const int cols,
        ACSATailMessage *tailMess, const int ntiles, const int mg4x3)
{
    int d1; 
    int sizeO = rows * cols;

    int rowSeg1, rowSeg2;
    int colSeg1, colSeg2;
    rowSeg1 = rows/4*4;
    rowSeg2 = rows - rowSeg1;
    colSeg1 = cols/4*4;
    colSeg2 = cols - colSeg1;

    ACSA_CHECK((rowSeg2 == tailMess->tail_h_));
    ACSA_CHECK((colSeg2 == tailMess->tail_w_)); 

#pragma omp parallel for private(d1)
    for(d1 = 0; d1 < N*K; d1++){
        int i, j;    
        Dtype tmp[36] __attribute__((aligned(64)));
        Dtype bridge[24] __attribute__((aligned(64)));
        Dtype middle[16] __attribute__((aligned(64))); 

        const int t1 = d1/(K*mg4x3);
        const int t2 = (d1%(K*mg4x3))/mg4x3;
        const int t3 = d1%mg4x3;

        Dtype *dataDst = out + (t1*mg4x3*K + t3*K + t2)*sizeO;
        int tileCount = d1*ntiles; 

        for(i = 0; i < rowSeg1; i += 4){
#pragma simd
            for(j = 0; j < colSeg1; j += 4){
                GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE4X3);
                // First inverse transform for output by AT
                TRANS_AT_FST(AT, tmp, bridge);
                // Second inverse transform for output by AT
                dataDst[(i+0)*cols + (j+0)] = bridge[ 0]*AT[ 0] + bridge[ 1]*AT[ 1] + bridge[ 2]*AT[ 2] + bridge[ 3]*AT[ 3] + bridge[ 4]*AT[ 4] + bridge[ 5]*AT[ 5];
                dataDst[(i+0)*cols + (j+1)] = bridge[ 0]*AT[ 6] + bridge[ 1]*AT[ 7] + bridge[ 2]*AT[ 8] + bridge[ 3]*AT[ 9] + bridge[ 4]*AT[10] + bridge[ 5]*AT[11];
                dataDst[(i+0)*cols + (j+2)] = bridge[ 0]*AT[12] + bridge[ 1]*AT[13] + bridge[ 2]*AT[14] + bridge[ 3]*AT[15] + bridge[ 4]*AT[16] + bridge[ 5]*AT[17];
                dataDst[(i+0)*cols + (j+3)] = bridge[ 0]*AT[18] + bridge[ 1]*AT[19] + bridge[ 2]*AT[20] + bridge[ 3]*AT[21] + bridge[ 4]*AT[22] + bridge[ 5]*AT[23];
                dataDst[(i+1)*cols + (j+0)] = bridge[ 6]*AT[ 0] + bridge[ 7]*AT[ 1] + bridge[ 8]*AT[ 2] + bridge[ 9]*AT[ 3] + bridge[10]*AT[ 4] + bridge[11]*AT[ 5];
                dataDst[(i+1)*cols + (j+1)] = bridge[ 6]*AT[ 6] + bridge[ 7]*AT[ 7] + bridge[ 8]*AT[ 8] + bridge[ 9]*AT[ 9] + bridge[10]*AT[10] + bridge[11]*AT[11];
                dataDst[(i+1)*cols + (j+2)] = bridge[ 6]*AT[12] + bridge[ 7]*AT[13] + bridge[ 8]*AT[14] + bridge[ 9]*AT[15] + bridge[10]*AT[16] + bridge[11]*AT[17];
                dataDst[(i+1)*cols + (j+3)] = bridge[ 6]*AT[18] + bridge[ 7]*AT[19] + bridge[ 8]*AT[20] + bridge[ 9]*AT[21] + bridge[10]*AT[22] + bridge[11]*AT[23];
                dataDst[(i+2)*cols + (j+0)] = bridge[12]*AT[ 0] + bridge[13]*AT[ 1] + bridge[14]*AT[ 2] + bridge[15]*AT[ 3] + bridge[16]*AT[ 4] + bridge[17]*AT[ 5];
                dataDst[(i+2)*cols + (j+1)] = bridge[12]*AT[ 6] + bridge[13]*AT[ 7] + bridge[14]*AT[ 8] + bridge[15]*AT[ 9] + bridge[16]*AT[10] + bridge[17]*AT[11];
                dataDst[(i+2)*cols + (j+2)] = bridge[12]*AT[12] + bridge[13]*AT[13] + bridge[14]*AT[14] + bridge[15]*AT[15] + bridge[16]*AT[16] + bridge[17]*AT[17];
                dataDst[(i+2)*cols + (j+3)] = bridge[12]*AT[18] + bridge[13]*AT[19] + bridge[14]*AT[20] + bridge[15]*AT[21] + bridge[16]*AT[22] + bridge[17]*AT[23];
                dataDst[(i+3)*cols + (j+0)] = bridge[18]*AT[ 0] + bridge[19]*AT[ 1] + bridge[20]*AT[ 2] + bridge[21]*AT[ 3] + bridge[22]*AT[ 4] + bridge[23]*AT[ 5];
                dataDst[(i+3)*cols + (j+1)] = bridge[18]*AT[ 6] + bridge[19]*AT[ 7] + bridge[20]*AT[ 8] + bridge[21]*AT[ 9] + bridge[22]*AT[10] + bridge[23]*AT[11];
                dataDst[(i+3)*cols + (j+2)] = bridge[18]*AT[12] + bridge[19]*AT[13] + bridge[20]*AT[14] + bridge[21]*AT[15] + bridge[22]*AT[16] + bridge[23]*AT[17];
                dataDst[(i+3)*cols + (j+3)] = bridge[18]*AT[18] + bridge[19]*AT[19] + bridge[20]*AT[20] + bridge[21]*AT[21] + bridge[22]*AT[22] + bridge[23]*AT[23];
                tileCount++; 
            }

            // Process col tail
            if(colSeg2 != 0){
                GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE4X3);
                // First inverse transform for output by AT
                TRANS_AT_FST(AT, tmp, bridge);
                // Second inverse transform for output by AT
                TRANS_AT_SED(bridge, AT, middle);

                ACSAGetFinalOutput(dataDst, middle, 4, 4, i, j, cols, 0, 0, 0, ZERO_LENGTH(colSeg2));
                tileCount++; 
            }
        }

        // Process row tail
        if(rowSeg2 == 1){
#pragma simd
            for(j = 0; j < colSeg1; j += 4){
                GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE4X3);
                // First inverse transform for output by AT
                TRANS_AT_FST(AT, tmp, bridge);
                // Second inverse transform for output by AT
                dataDst[(rowSeg1+0)*cols + (j+0)] = bridge[ 0]*AT[ 0] + bridge[ 1]*AT[ 1] + bridge[ 2]*AT[ 2] + bridge[ 3]*AT[ 3] + bridge[ 4]*AT[ 4] + bridge[ 5]*AT[ 5];
                dataDst[(rowSeg1+0)*cols + (j+1)] = bridge[ 0]*AT[ 6] + bridge[ 1]*AT[ 7] + bridge[ 2]*AT[ 8] + bridge[ 3]*AT[ 9] + bridge[ 4]*AT[10] + bridge[ 5]*AT[11];
                dataDst[(rowSeg1+0)*cols + (j+2)] = bridge[ 0]*AT[12] + bridge[ 1]*AT[13] + bridge[ 2]*AT[14] + bridge[ 3]*AT[15] + bridge[ 4]*AT[16] + bridge[ 5]*AT[17];
                dataDst[(rowSeg1+0)*cols + (j+3)] = bridge[ 0]*AT[18] + bridge[ 1]*AT[19] + bridge[ 2]*AT[20] + bridge[ 3]*AT[21] + bridge[ 4]*AT[22] + bridge[ 5]*AT[23];
                tileCount++;
            }
        }
        else if(rowSeg2 == 2){
#pragma simd
            for(j = 0; j < colSeg1; j += 4){
                GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE4X3);
                // First inverse transform for output by AT
                TRANS_AT_FST(AT, tmp, bridge);
                // Second inverse transform for output by AT
                dataDst[(rowSeg1+0)*cols + (j+0)] = bridge[ 0]*AT[ 0] + bridge[ 1]*AT[ 1] + bridge[ 2]*AT[ 2] + bridge[ 3]*AT[ 3] + bridge[ 4]*AT[ 4] + bridge[ 5]*AT[ 5];
                dataDst[(rowSeg1+0)*cols + (j+1)] = bridge[ 0]*AT[ 6] + bridge[ 1]*AT[ 7] + bridge[ 2]*AT[ 8] + bridge[ 3]*AT[ 9] + bridge[ 4]*AT[10] + bridge[ 5]*AT[11];
                dataDst[(rowSeg1+0)*cols + (j+2)] = bridge[ 0]*AT[12] + bridge[ 1]*AT[13] + bridge[ 2]*AT[14] + bridge[ 3]*AT[15] + bridge[ 4]*AT[16] + bridge[ 5]*AT[17];
                dataDst[(rowSeg1+0)*cols + (j+3)] = bridge[ 0]*AT[18] + bridge[ 1]*AT[19] + bridge[ 2]*AT[20] + bridge[ 3]*AT[21] + bridge[ 4]*AT[22] + bridge[ 5]*AT[23];
                dataDst[(rowSeg1+1)*cols + (j+0)] = bridge[ 6]*AT[ 0] + bridge[ 7]*AT[ 1] + bridge[ 8]*AT[ 2] + bridge[ 9]*AT[ 3] + bridge[10]*AT[ 4] + bridge[11]*AT[ 5];
                dataDst[(rowSeg1+1)*cols + (j+1)] = bridge[ 6]*AT[ 6] + bridge[ 7]*AT[ 7] + bridge[ 8]*AT[ 8] + bridge[ 9]*AT[ 9] + bridge[10]*AT[10] + bridge[11]*AT[11];
                dataDst[(rowSeg1+1)*cols + (j+2)] = bridge[ 6]*AT[12] + bridge[ 7]*AT[13] + bridge[ 8]*AT[14] + bridge[ 9]*AT[15] + bridge[10]*AT[16] + bridge[11]*AT[17];
                dataDst[(rowSeg1+1)*cols + (j+3)] = bridge[ 6]*AT[18] + bridge[ 7]*AT[19] + bridge[ 8]*AT[20] + bridge[ 9]*AT[21] + bridge[10]*AT[22] + bridge[11]*AT[23];
                tileCount++;
            }
        }
        else if(rowSeg2 == 3){
#pragma simd
            for(j = 0; j < colSeg1; j += 4){
                GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE4X3);
                // First inverse transform for output by AT
                TRANS_AT_FST(AT, tmp, bridge);
                // Second inverse transform for output by AT
                dataDst[(rowSeg1+0)*cols + (j+0)] = bridge[ 0]*AT[ 0] + bridge[ 1]*AT[ 1] + bridge[ 2]*AT[ 2] + bridge[ 3]*AT[ 3] + bridge[ 4]*AT[ 4] + bridge[ 5]*AT[ 5];
                dataDst[(rowSeg1+0)*cols + (j+1)] = bridge[ 0]*AT[ 6] + bridge[ 1]*AT[ 7] + bridge[ 2]*AT[ 8] + bridge[ 3]*AT[ 9] + bridge[ 4]*AT[10] + bridge[ 5]*AT[11];
                dataDst[(rowSeg1+0)*cols + (j+2)] = bridge[ 0]*AT[12] + bridge[ 1]*AT[13] + bridge[ 2]*AT[14] + bridge[ 3]*AT[15] + bridge[ 4]*AT[16] + bridge[ 5]*AT[17];
                dataDst[(rowSeg1+0)*cols + (j+3)] = bridge[ 0]*AT[18] + bridge[ 1]*AT[19] + bridge[ 2]*AT[20] + bridge[ 3]*AT[21] + bridge[ 4]*AT[22] + bridge[ 5]*AT[23];
                dataDst[(rowSeg1+1)*cols + (j+0)] = bridge[ 6]*AT[ 0] + bridge[ 7]*AT[ 1] + bridge[ 8]*AT[ 2] + bridge[ 9]*AT[ 3] + bridge[10]*AT[ 4] + bridge[11]*AT[ 5];
                dataDst[(rowSeg1+1)*cols + (j+1)] = bridge[ 6]*AT[ 6] + bridge[ 7]*AT[ 7] + bridge[ 8]*AT[ 8] + bridge[ 9]*AT[ 9] + bridge[10]*AT[10] + bridge[11]*AT[11];
                dataDst[(rowSeg1+1)*cols + (j+2)] = bridge[ 6]*AT[12] + bridge[ 7]*AT[13] + bridge[ 8]*AT[14] + bridge[ 9]*AT[15] + bridge[10]*AT[16] + bridge[11]*AT[17];
                dataDst[(rowSeg1+1)*cols + (j+3)] = bridge[ 6]*AT[18] + bridge[ 7]*AT[19] + bridge[ 8]*AT[20] + bridge[ 9]*AT[21] + bridge[10]*AT[22] + bridge[11]*AT[23];
                dataDst[(rowSeg1+2)*cols + (j+0)] = bridge[12]*AT[ 0] + bridge[13]*AT[ 1] + bridge[14]*AT[ 2] + bridge[15]*AT[ 3] + bridge[16]*AT[ 4] + bridge[17]*AT[ 5];
                dataDst[(rowSeg1+2)*cols + (j+1)] = bridge[12]*AT[ 6] + bridge[13]*AT[ 7] + bridge[14]*AT[ 8] + bridge[15]*AT[ 9] + bridge[16]*AT[10] + bridge[17]*AT[11];
                dataDst[(rowSeg1+2)*cols + (j+2)] = bridge[12]*AT[12] + bridge[13]*AT[13] + bridge[14]*AT[14] + bridge[15]*AT[15] + bridge[16]*AT[16] + bridge[17]*AT[17];
                dataDst[(rowSeg1+2)*cols + (j+3)] = bridge[12]*AT[18] + bridge[13]*AT[19] + bridge[14]*AT[20] + bridge[15]*AT[21] + bridge[16]*AT[22] + bridge[17]*AT[23];
                tileCount++; 
            }
        }

        // Process row&col tail
        if((rowSeg2 != 0) && (colSeg2 != 0)){
            GET_OUTPUT_TILE(tmp, dataSrc, tileCount, OSTRIDE4X3);
            // First inverse transfrom for output data by AT
            TRANS_AT_FST(AT, tmp, bridge);
            // Second inverse transfrom for output data by AT
            TRANS_AT_SED(bridge, AT, middle);

            ACSAGetFinalOutput(dataDst, middle, 4, 4, rowSeg1, colSeg1, cols, 0, ZERO_LENGTH(rowSeg2), 0, ZERO_LENGTH(colSeg2));
            tileCount++; 
        }
    }
}

/* Process the data tail. */
static void tailPreProcess(ACSATensor4d *tensorOut, ACSATailMessage &tailMess, int &ntiles)
{
    int out_h = tensorOut->h_;
    int out_w = tensorOut->w_;
    int tail_h = out_h % 4;
    int tail_w = out_w % 4;

    tailMess.tail_h_ = tail_h;
    tailMess.tail_w_ = tail_w;

    if(tail_h == 0)
        ntiles = out_h/4;
    else
        ntiles = out_h/4 + 1;

    if(tail_w == 0)
        ntiles *= out_w/4;
    else
        ntiles *= out_w/4 +1;
}

/* API for winograd F(6,3). */
    template<typename Dtype>
ACSAStatus ACSAWinoConvolution_4x3(const Dtype *in, const Dtype *filter, Dtype *out,
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
    const int bb4x3 = winoMess->batch_block_;
    const int mg4x3 = winoMess->merge_;
    const int outHeight = tensorOut->h_; 
    const int outWidth = tensorOut->w_; 
    //const int ntiles = ((outHeight)*0.25)*((outWidth)*0.25);

    int ntiles;
    ACSATailMessage tailMess;

    tailPreProcess(tensorOut, tailMess, ntiles);

    const Dtype *b_in;
    Dtype *b_out;

    int b_bts = 64;
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
            inByTransform_nopad(b_in, wino_in, b_bts, C, H, W, &tailMess, ntiles, mg4x3);
        else if(H*W > 1225)
            inByTransform_padBigScale(b_in, wino_in, b_bts, C, H, W, pad_h, pad_w, &tailMess, ntiles, mg4x3);
        else{
            Dtype *in_pad = (Dtype *)mkl_malloc(num_threads*(H+2*pad_h)*(W+2*pad_w)*sizeof(Dtype), 64);
            memset(in_pad, 0, num_threads*(H+2*pad_h)*(W+2*pad_w)*sizeof(Dtype));
            inByTransform_padSmallScale(b_in, in_pad, wino_in, N, C, H, W, pad_h, pad_w, &tailMess, ntiles, mg4x3);
            mkl_free(in_pad);
        }
        matrix_compute(wino_in, mg4x3*ntiles, C, wino_filter, C, K, wino_out, b_bts/mg4x3);
        outByTransform(b_out, wino_out, b_bts, K, outHeight, outWidth, &tailMess, ntiles, mg4x3);
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
template ACSAStatus ACSAWinoConvolution_4x3<float>(const float *, const float *, float *,
        ACSATensor4d*, ACSATensor4d*, ACSATensor4d*,
        ACSAConvMessage*, ACSAWinoMessage*);

template void inByTransform_nopad<double>(const double *, double *,
        const int, const int, const int, const int,
        ACSATailMessage *, const int, const int);
template void inByTransform_padBigScale<double>(const double *, double *,
        const int, const int, const int, const int,
        const int, const int,
        ACSATailMessage *, const int, const int);
template void inByTransform_padSmallScale<double>(const double *, double *, double *,
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
template ACSAStatus ACSAWinoConvolution_4x3<double>(const double *, const double *, double *,
        ACSATensor4d*, ACSATensor4d*, ACSATensor4d*,
        ACSAConvMessage*, ACSAWinoMessage*);
