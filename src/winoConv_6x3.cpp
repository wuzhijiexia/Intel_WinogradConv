/* F(6x6,3x3) implementations for winograd. */

#include <immintrin.h>
#include "dnn.hpp"

const long ISTRIDE6X3 = ISTRIDE/64*16;
const long FSTRIDE6X3 = FSTRIDE;
const long OSTRIDE6X3 = OSTRIDE/64*16;

static const float AT[48] = {
    1, 1,  1,  1,   1,      1,      1,  0,
    0, 1, -1,  2,  -2,  1.0/2, -1.0/2,  0,
    0, 1,  1,  4,   4,  1.0/4,  1.0/4,  0,
    0, 1, -1,  8,  -8,  1.0/8, -1.0/8,  0,
    0, 1,  1, 16,  16, 1.0/16,  1.0/16, 0,
    0, 1, -1, 32, -32, 1.0/32, -1.0/32, 1
};

static const float G[24] = {
    1,         0,      0,
    -2.0/9,   -2.0/9, -2.0/9,
    -2.0/9,    2.0/9, -2.0/9,
    1.0/90,   1.0/45, 2.0/45,
    1.0/90,  -1.0/45, 2.0/45,
    32.0/45,  16.0/45, 8.0/45,
    32.0/45, -16.0/45, 8.0/45,
    0,        0,      1
};

static const float BT[64] = {
    1,      0, -21.0/4,       0,  21.0/4,       0, -1, 0,
    0,      1,       1, -17.0/4, -17.0/4,       1,  1, 0,
    0,     -1,       1,  17.0/4, -17.0/4,      -1,  1, 0,
    0,  1.0/2,   1.0/4,  -5.0/2,  -5.0/4,       2,  1, 0,
    0, -1.0/2,   1.0/4,   5.0/2,  -5.0/4,      -2,  1, 0,
    0,      2,       4,  -5.0/2,      -5,   1.0/2,  1, 0,
    0,     -2,       4,   5.0/2,      -5,  -1.0/2,  1, 0,
    0,     -1,       0,  21.0/4,       0, -21.0/4,  0, 1
};

/* Compute transformed data for input by BT.
 * */
    template<typename Dtype>
static inline void transformByBT(Dtype *dsrc, Dtype *ddst, int tileCount)
{
    ddst[tileCount + 0*ISTRIDE6X3] = (dsrc[0 ] - dsrc[6 ] - dsrc[48] + dsrc[54]) + (-dsrc[2 ] + dsrc[4 ] - dsrc[16] + dsrc[22] + dsrc[32] - dsrc[38] + dsrc[50] - dsrc[52])*5.25 + (dsrc[18] - dsrc[20] - dsrc[34] + dsrc[36])*27.5625;
    ddst[tileCount + 1*ISTRIDE6X3] = (dsrc[1 ] + dsrc[2 ] + dsrc[5 ] + dsrc[6 ] - dsrc[49] - dsrc[50] - dsrc[53] - dsrc[54]) + (-dsrc[3 ] - dsrc[4 ] + dsrc[51] + dsrc[52])*4.25 + (-dsrc[17] - dsrc[18] - dsrc[21] - dsrc[22] + dsrc[33] + dsrc[34] + dsrc[37] + dsrc[38])*5.25 + (dsrc[19] + dsrc[20] - dsrc[35] - dsrc[36])*22.3125;
    ddst[tileCount + 2*ISTRIDE6X3] = (-dsrc[1 ] + dsrc[2 ] - dsrc[5 ] + dsrc[6 ] + dsrc[49] - dsrc[50] + dsrc[53] - dsrc[54]) + (dsrc[3 ] - dsrc[4 ] - dsrc[51] + dsrc[52])*4.25 + (dsrc[17] - dsrc[18] + dsrc[21] - dsrc[22] - dsrc[33] + dsrc[34] - dsrc[37] + dsrc[38])*5.25 + (-dsrc[19] + dsrc[20] + dsrc[35] - dsrc[36])*22.3125;
    ddst[tileCount + 3*ISTRIDE6X3] = (dsrc[2 ] - dsrc[50])*0.25 + (dsrc[1 ] - dsrc[49])*0.5 + (dsrc[6 ] - dsrc[54]) + (-dsrc[4 ] + dsrc[52])*1.25 + (-dsrc[18] + dsrc[34])*1.3125 + (dsrc[5 ] - dsrc[53])*2 + (-dsrc[3 ] + dsrc[51])*2.5 + (-dsrc[17] + dsrc[33])*2.625 + (-dsrc[22] + dsrc[38])*5.25 + (dsrc[20] - dsrc[36])*6.5625 + (-dsrc[21] + dsrc[37])*10.5 + (dsrc[19] - dsrc[35])*13.125;
    ddst[tileCount + 4*ISTRIDE6X3] = (dsrc[2 ] - dsrc[50])*0.25 + (-dsrc[1 ] + dsrc[49])*0.5 + (dsrc[6 ] - dsrc[54]) + (-dsrc[4 ] + dsrc[52])*1.25 + (-dsrc[18] + dsrc[34])*1.3125 + (-dsrc[5 ] + dsrc[53])*2 + (dsrc[3 ] - dsrc[51])*2.5 + (dsrc[17] - dsrc[33])*2.625 + (-dsrc[22] + dsrc[38])*5.25 + (dsrc[20] - dsrc[36])*6.5625 + (dsrc[21] - dsrc[37])*10.5 + (-dsrc[19] + dsrc[35])*13.125;
    ddst[tileCount + 5*ISTRIDE6X3] = (dsrc[5 ] - dsrc[53])*0.5 + (dsrc[6 ] - dsrc[54]) + (dsrc[1 ] - dsrc[49])*2 + (-dsrc[3 ] + dsrc[51])*2.5 + (-dsrc[21] + dsrc[37])*2.625 + (dsrc[2 ] - dsrc[50])*4 + (-dsrc[4 ] + dsrc[52])*5 + (-dsrc[22] + dsrc[38])*5.25 + (-dsrc[17] + dsrc[33])*10.5 + (dsrc[19] - dsrc[35])*13.125 + (-dsrc[18] + dsrc[34])*21 + (dsrc[20] - dsrc[36])*26.25;
    ddst[tileCount + 6*ISTRIDE6X3] = (-dsrc[5 ] + dsrc[53])*0.5 + (dsrc[6 ] - dsrc[54]) + (-dsrc[1 ] + dsrc[49])*2 + (dsrc[3 ] - dsrc[51])*2.5 + (dsrc[21] - dsrc[37])*2.625 + (dsrc[2 ] - dsrc[50])*4 + (-dsrc[4 ] + dsrc[52])*5 + (-dsrc[22] + dsrc[38])*5.25 + (dsrc[17] - dsrc[33])*10.5 + (-dsrc[19] + dsrc[35])*13.125 + (-dsrc[18] + dsrc[34])*21 + (dsrc[20] - dsrc[36])*26.25;
    ddst[tileCount + 7*ISTRIDE6X3] = (-dsrc[1 ] + dsrc[7 ] + dsrc[49] - dsrc[55]) + (dsrc[3 ] - dsrc[5 ] + dsrc[17] - dsrc[23] - dsrc[33] + dsrc[39] - dsrc[51] + dsrc[53])*5.25 + (-dsrc[19] + dsrc[21] + dsrc[35] - dsrc[37])*27.5625;
    ddst[tileCount + 8*ISTRIDE6X3] = (dsrc[8 ] - dsrc[14] + dsrc[16] - dsrc[22] + dsrc[40] - dsrc[46] + dsrc[48] - dsrc[54]) + (-dsrc[24] + dsrc[30] - dsrc[32] + dsrc[38])*4.25 + (-dsrc[10] + dsrc[12] - dsrc[18] + dsrc[20] - dsrc[42] + dsrc[44] - dsrc[50] + dsrc[52])*5.25 + (dsrc[26] - dsrc[28] + dsrc[34] - dsrc[36])*22.3125;
    ddst[tileCount + 9*ISTRIDE6X3] = (dsrc[9 ] + dsrc[10] + dsrc[13] + dsrc[14] + dsrc[17] + dsrc[18] + dsrc[21] + dsrc[22] + dsrc[41] + dsrc[42] + dsrc[45] + dsrc[46] + dsrc[49] + dsrc[50] + dsrc[53] + dsrc[54]) + (-dsrc[11] - dsrc[12] - dsrc[19] - dsrc[20] - dsrc[25] - dsrc[26] - dsrc[29] - dsrc[30] - dsrc[33] - dsrc[34] - dsrc[37] - dsrc[38] - dsrc[43] - dsrc[44] - dsrc[51] - dsrc[52])*4.25 + (dsrc[27] + dsrc[28] + dsrc[35] + dsrc[36])*18.0625;
    ddst[tileCount +10*ISTRIDE6X3] = (-dsrc[9 ] + dsrc[10] - dsrc[13] + dsrc[14] - dsrc[17] + dsrc[18] - dsrc[21] + dsrc[22] - dsrc[41] + dsrc[42] - dsrc[45] + dsrc[46] - dsrc[49] + dsrc[50] - dsrc[53] + dsrc[54]) + (dsrc[11] - dsrc[12] + dsrc[19] - dsrc[20] + dsrc[25] - dsrc[26] + dsrc[29] - dsrc[30] + dsrc[33] - dsrc[34] + dsrc[37] - dsrc[38] + dsrc[43] - dsrc[44] + dsrc[51] - dsrc[52])*4.25 + (-dsrc[27] + dsrc[28] - dsrc[35] + dsrc[36])*18.0625;
    ddst[tileCount +11*ISTRIDE6X3] = (dsrc[10] + dsrc[18] + dsrc[42] + dsrc[50])*0.25 + (dsrc[9 ] + dsrc[17] + dsrc[41] + dsrc[49])*0.5 + (dsrc[14] + dsrc[22] + dsrc[46] + dsrc[54]) + (-dsrc[26] - dsrc[34])*1.0625 + (-dsrc[12] - dsrc[20] - dsrc[44] - dsrc[52])*1.25 + (dsrc[13] + dsrc[21] + dsrc[45] + dsrc[53])*2 + (-dsrc[25] - dsrc[33])*2.125 + (-dsrc[11] - dsrc[19] - dsrc[43] - dsrc[51])*2.5 + (-dsrc[30] - dsrc[38])*4.25 + (dsrc[28] + dsrc[36])*5.3125 + (-dsrc[29] - dsrc[37])*8.5 + (dsrc[27] + dsrc[35])*10.625;
    ddst[tileCount +12*ISTRIDE6X3] = (dsrc[10] + dsrc[18] + dsrc[42] + dsrc[50])*0.25 + (-dsrc[9 ] - dsrc[17] - dsrc[41] - dsrc[49])*0.5 + (dsrc[14] + dsrc[22] + dsrc[46] + dsrc[54]) + (-dsrc[26] - dsrc[34])*1.0625 + (-dsrc[12] - dsrc[20] - dsrc[44] - dsrc[52])*1.25 + (-dsrc[13] - dsrc[21] - dsrc[45] - dsrc[53])*2 + (dsrc[25] + dsrc[33])*2.125 + (dsrc[11] + dsrc[19] + dsrc[43] + dsrc[51])*2.5 + (-dsrc[30] - dsrc[38])*4.25 + (dsrc[28] + dsrc[36])*5.3125 + (dsrc[29] + dsrc[37])*8.5 + (-dsrc[27] - dsrc[35])*10.625;
    ddst[tileCount +13*ISTRIDE6X3] = (dsrc[13] + dsrc[21] + dsrc[45] + dsrc[53])*0.5 + (dsrc[14] + dsrc[22] + dsrc[46] + dsrc[54]) + (dsrc[9 ] + dsrc[17] + dsrc[41] + dsrc[49])*2 + (-dsrc[29] - dsrc[37])*2.125 + (-dsrc[11] - dsrc[19] - dsrc[43] - dsrc[51])*2.5 + (dsrc[10] + dsrc[18] + dsrc[42] + dsrc[50])*4 + (-dsrc[30] - dsrc[38])*4.25 + (-dsrc[12] - dsrc[20] - dsrc[44] - dsrc[52])*5 + (-dsrc[25] - dsrc[33])*8.5 + (dsrc[27] + dsrc[35])*10.625 + (-dsrc[26] - dsrc[34])*17 + (dsrc[28] + dsrc[36])*21.25;
    ddst[tileCount +14*ISTRIDE6X3] = (-dsrc[13] - dsrc[21] - dsrc[45] - dsrc[53])*0.5 + (dsrc[14] + dsrc[22] + dsrc[46] + dsrc[54]) + (-dsrc[9 ] - dsrc[17] - dsrc[41] - dsrc[49])*2 + (dsrc[29] + dsrc[37])*2.125 + (dsrc[11] + dsrc[19] + dsrc[43] + dsrc[51])*2.5 + (dsrc[10] + dsrc[18] + dsrc[42] + dsrc[50])*4 + (-dsrc[30] - dsrc[38])*4.25 + (-dsrc[12] - dsrc[20] - dsrc[44] - dsrc[52])*5 + (dsrc[25] + dsrc[33])*8.5 + (-dsrc[27] - dsrc[35])*10.625 + (-dsrc[26] - dsrc[34])*17 + (dsrc[28] + dsrc[36])*21.25;
    ddst[tileCount +15*ISTRIDE6X3] = (-dsrc[9 ] + dsrc[15] - dsrc[17] + dsrc[23] - dsrc[41] + dsrc[47] - dsrc[49] + dsrc[55]) + (dsrc[25] - dsrc[31] + dsrc[33] - dsrc[39])*4.25 + (dsrc[11] - dsrc[13] + dsrc[19] - dsrc[21] + dsrc[43] - dsrc[45] + dsrc[51] - dsrc[53])*5.25 + (-dsrc[27] + dsrc[29] - dsrc[35] + dsrc[37])*22.3125;
    ddst[tileCount +16*ISTRIDE6X3] = (-dsrc[8 ] + dsrc[14] + dsrc[16] - dsrc[22] - dsrc[40] + dsrc[46] + dsrc[48] - dsrc[54]) + (dsrc[24] - dsrc[30] - dsrc[32] + dsrc[38])*4.25 + (dsrc[10] - dsrc[12] - dsrc[18] + dsrc[20] + dsrc[42] - dsrc[44] - dsrc[50] + dsrc[52])*5.25 + (-dsrc[26] + dsrc[28] + dsrc[34] - dsrc[36])*22.3125;
    ddst[tileCount +17*ISTRIDE6X3] = (-dsrc[9 ] - dsrc[10] - dsrc[13] - dsrc[14] + dsrc[17] + dsrc[18] + dsrc[21] + dsrc[22] - dsrc[41] - dsrc[42] - dsrc[45] - dsrc[46] + dsrc[49] + dsrc[50] + dsrc[53] + dsrc[54]) + (dsrc[11] + dsrc[12] - dsrc[19] - dsrc[20] + dsrc[25] + dsrc[26] + dsrc[29] + dsrc[30] - dsrc[33] - dsrc[34] - dsrc[37] - dsrc[38] + dsrc[43] + dsrc[44] - dsrc[51] - dsrc[52])*4.25 + (-dsrc[27] - dsrc[28] + dsrc[35] + dsrc[36])*18.0625;
    ddst[tileCount +18*ISTRIDE6X3] = (dsrc[9 ] - dsrc[10] + dsrc[13] - dsrc[14] - dsrc[17] + dsrc[18] - dsrc[21] + dsrc[22] + dsrc[41] - dsrc[42] + dsrc[45] - dsrc[46] - dsrc[49] + dsrc[50] - dsrc[53] + dsrc[54]) + (-dsrc[11] + dsrc[12] + dsrc[19] - dsrc[20] - dsrc[25] + dsrc[26] - dsrc[29] + dsrc[30] + dsrc[33] - dsrc[34] + dsrc[37] - dsrc[38] - dsrc[43] + dsrc[44] + dsrc[51] - dsrc[52])*4.25 + (dsrc[27] - dsrc[28] - dsrc[35] + dsrc[36])*18.0625;
    ddst[tileCount +19*ISTRIDE6X3] = (-dsrc[10] + dsrc[18] - dsrc[42] + dsrc[50])*0.25 + (-dsrc[9 ] + dsrc[17] - dsrc[41] + dsrc[49])*0.5 + (-dsrc[14] + dsrc[22] - dsrc[46] + dsrc[54]) + (dsrc[26] - dsrc[34])*1.0625 + (dsrc[12] - dsrc[20] + dsrc[44] - dsrc[52])*1.25 + (-dsrc[13] + dsrc[21] - dsrc[45] + dsrc[53])*2 + (dsrc[25] - dsrc[33])*2.125 + (dsrc[11] - dsrc[19] + dsrc[43] - dsrc[51])*2.5 + (dsrc[30] - dsrc[38])*4.25 + (-dsrc[28] + dsrc[36])*5.3125 + (dsrc[29] - dsrc[37])*8.5 + (-dsrc[27] + dsrc[35])*10.625;
    ddst[tileCount +20*ISTRIDE6X3] = (-dsrc[10] + dsrc[18] - dsrc[42] + dsrc[50])*0.25 + (dsrc[9 ] - dsrc[17] + dsrc[41] - dsrc[49])*0.5 + (-dsrc[14] + dsrc[22] - dsrc[46] + dsrc[54]) + (dsrc[26] - dsrc[34])*1.0625 + (dsrc[12] - dsrc[20] + dsrc[44] - dsrc[52])*1.25 + (dsrc[13] - dsrc[21] + dsrc[45] - dsrc[53])*2 + (-dsrc[25] + dsrc[33])*2.125 + (-dsrc[11] + dsrc[19] - dsrc[43] + dsrc[51])*2.5 + (dsrc[30] - dsrc[38])*4.25 + (-dsrc[28] + dsrc[36])*5.3125 + (-dsrc[29] + dsrc[37])*8.5 + (dsrc[27] - dsrc[35])*10.625;
    ddst[tileCount +21*ISTRIDE6X3] = (-dsrc[13] + dsrc[21] - dsrc[45] + dsrc[53])*0.5 + (-dsrc[14] + dsrc[22] - dsrc[46] + dsrc[54]) + (-dsrc[9 ] + dsrc[17] - dsrc[41] + dsrc[49])*2 + (dsrc[29] - dsrc[37])*2.125 + (dsrc[11] - dsrc[19] + dsrc[43] - dsrc[51])*2.5 + (-dsrc[10] + dsrc[18] - dsrc[42] + dsrc[50])*4 + (dsrc[30] - dsrc[38])*4.25 + (dsrc[12] - dsrc[20] + dsrc[44] - dsrc[52])*5 + (dsrc[25] - dsrc[33])*8.5 + (-dsrc[27] + dsrc[35])*10.625 + (dsrc[26] - dsrc[34])*17 + (-dsrc[28] + dsrc[36])*21.25;
    ddst[tileCount +22*ISTRIDE6X3] = (dsrc[13] - dsrc[21] + dsrc[45] - dsrc[53])*0.5 + (-dsrc[14] + dsrc[22] - dsrc[46] + dsrc[54]) + (dsrc[9 ] - dsrc[17] + dsrc[41] - dsrc[49])*2 + (-dsrc[29] + dsrc[37])*2.125 + (-dsrc[11] + dsrc[19] - dsrc[43] + dsrc[51])*2.5 + (-dsrc[10] + dsrc[18] - dsrc[42] + dsrc[50])*4 + (dsrc[30] - dsrc[38])*4.25 + (dsrc[12] - dsrc[20] + dsrc[44] - dsrc[52])*5 + (-dsrc[25] + dsrc[33])*8.5 + (dsrc[27] - dsrc[35])*10.625 + (dsrc[26] - dsrc[34])*17 + (-dsrc[28] + dsrc[36])*21.25;
    ddst[tileCount +23*ISTRIDE6X3] = (dsrc[9 ] - dsrc[15] - dsrc[17] + dsrc[23] + dsrc[41] - dsrc[47] - dsrc[49] + dsrc[55]) + (-dsrc[25] + dsrc[31] + dsrc[33] - dsrc[39])*4.25 + (-dsrc[11] + dsrc[13] + dsrc[19] - dsrc[21] - dsrc[43] + dsrc[45] + dsrc[51] - dsrc[53])*5.25 + (dsrc[27] - dsrc[29] - dsrc[35] + dsrc[37])*22.3125;
    ddst[tileCount +24*ISTRIDE6X3] = (dsrc[16] - dsrc[22])*0.25 + (dsrc[8 ] - dsrc[14])*0.5 + (dsrc[48] - dsrc[54]) + (-dsrc[32] + dsrc[38])*1.25 + (-dsrc[18] + dsrc[20])*1.3125 + (dsrc[40] - dsrc[46])*2 + (-dsrc[24] + dsrc[30])*2.5 + (-dsrc[10] + dsrc[12])*2.625 + (-dsrc[50] + dsrc[52])*5.25 + (dsrc[34] - dsrc[36])*6.5625 + (-dsrc[42] + dsrc[44])*10.5 + (dsrc[26] - dsrc[28])*13.125;
    ddst[tileCount +25*ISTRIDE6X3] = (dsrc[17] + dsrc[18] + dsrc[21] + dsrc[22])*0.25 + (dsrc[9 ] + dsrc[10] + dsrc[13] + dsrc[14])*0.5 + (dsrc[49] + dsrc[50] + dsrc[53] + dsrc[54]) + (-dsrc[19] - dsrc[20])*1.0625 + (-dsrc[33] - dsrc[34] - dsrc[37] - dsrc[38])*1.25 + (dsrc[41] + dsrc[42] + dsrc[45] + dsrc[46])*2 + (-dsrc[11] - dsrc[12])*2.125 + (-dsrc[25] - dsrc[26] - dsrc[29] - dsrc[30])*2.5 + (-dsrc[51] - dsrc[52])*4.25 + (dsrc[35] + dsrc[36])*5.3125 + (-dsrc[43] - dsrc[44])*8.5 + (dsrc[27] + dsrc[28])*10.625;
    ddst[tileCount +26*ISTRIDE6X3] = (-dsrc[17] + dsrc[18] - dsrc[21] + dsrc[22])*0.25 + (-dsrc[9 ] + dsrc[10] - dsrc[13] + dsrc[14])*0.5 + (-dsrc[49] + dsrc[50] - dsrc[53] + dsrc[54]) + (dsrc[19] - dsrc[20])*1.0625 + (dsrc[33] - dsrc[34] + dsrc[37] - dsrc[38])*1.25 + (-dsrc[41] + dsrc[42] - dsrc[45] + dsrc[46])*2 + (dsrc[11] - dsrc[12])*2.125 + (dsrc[25] - dsrc[26] + dsrc[29] - dsrc[30])*2.5 + (dsrc[51] - dsrc[52])*4.25 + (-dsrc[35] + dsrc[36])*5.3125 + (dsrc[43] - dsrc[44])*8.5 + (-dsrc[27] + dsrc[28])*10.625;
    ddst[tileCount +27*ISTRIDE6X3] = (dsrc[18])*0.0625 + (dsrc[10] + dsrc[17])*0.125 + (dsrc[9 ] + dsrc[22] + dsrc[50])*0.25 + (-dsrc[20] - dsrc[34])*0.3125 + (dsrc[14] + dsrc[21] + dsrc[42] + dsrc[49])*0.5 + (-dsrc[12] - dsrc[19] - dsrc[26] - dsrc[33])*0.625 + (dsrc[13] + dsrc[41] + dsrc[54]) + (-dsrc[11] - dsrc[25] - dsrc[38] - dsrc[52])*1.25 + (dsrc[36])*1.5625 + (dsrc[46] + dsrc[53])*2 + (-dsrc[30] - dsrc[37] - dsrc[44] - dsrc[51])*2.5 + (dsrc[28] + dsrc[35])*3.125 + (dsrc[45])*4 + (-dsrc[29] - dsrc[43])*5 + (dsrc[27])*6.25;
    ddst[tileCount +28*ISTRIDE6X3] = (dsrc[18])*0.0625 + (dsrc[10] - dsrc[17])*0.125 + (-dsrc[9 ] + dsrc[22] + dsrc[50])*0.25 + (-dsrc[20] - dsrc[34])*0.3125 + (dsrc[14] - dsrc[21] + dsrc[42] - dsrc[49])*0.5 + (-dsrc[12] + dsrc[19] - dsrc[26] + dsrc[33])*0.625 + (-dsrc[13] - dsrc[41] + dsrc[54]) + (dsrc[11] + dsrc[25] - dsrc[38] - dsrc[52])*1.25 + (dsrc[36])*1.5625 + (dsrc[46] - dsrc[53])*2 + (-dsrc[30] + dsrc[37] - dsrc[44] + dsrc[51])*2.5 + (dsrc[28] - dsrc[35])*3.125 + (-dsrc[45])*4 + (dsrc[29] + dsrc[43])*5 + (-dsrc[27])*6.25;
    ddst[tileCount +29*ISTRIDE6X3] = (dsrc[21])*0.125 + (dsrc[13] + dsrc[22])*0.25 + (dsrc[14] + dsrc[17] + dsrc[53])*0.5 + (-dsrc[19] - dsrc[37])*0.625 + (dsrc[9 ] + dsrc[18] + dsrc[45] + dsrc[54]) + (-dsrc[11] - dsrc[20] - dsrc[29] - dsrc[38])*1.25 + (dsrc[10] + dsrc[46] + dsrc[49])*2 + (-dsrc[12] - dsrc[30] - dsrc[33] - dsrc[51])*2.5 + (dsrc[35])*3.125 + (dsrc[41] + dsrc[50])*4 + (-dsrc[25] - dsrc[34] - dsrc[43] - dsrc[52])*5 + (dsrc[27] + dsrc[36])*6.25 + (dsrc[42])*8 + (-dsrc[26] - dsrc[44])*10 + (dsrc[28])*12.5;
    ddst[tileCount +30*ISTRIDE6X3] = (-dsrc[21])*0.125 + (-dsrc[13] + dsrc[22])*0.25 + (dsrc[14] - dsrc[17] - dsrc[53])*0.5 + (dsrc[19] + dsrc[37])*0.625 + (-dsrc[9 ] + dsrc[18] - dsrc[45] + dsrc[54]) + (dsrc[11] - dsrc[20] + dsrc[29] - dsrc[38])*1.25 + (dsrc[10] + dsrc[46] - dsrc[49])*2 + (-dsrc[12] - dsrc[30] + dsrc[33] + dsrc[51])*2.5 + (-dsrc[35])*3.125 + (-dsrc[41] + dsrc[50])*4 + (dsrc[25] - dsrc[34] + dsrc[43] - dsrc[52])*5 + (-dsrc[27] + dsrc[36])*6.25 + (dsrc[42])*8 + (-dsrc[26] - dsrc[44])*10 + (dsrc[28])*12.5;
    ddst[tileCount +31*ISTRIDE6X3] = (-dsrc[17] + dsrc[23])*0.25 + (-dsrc[9 ] + dsrc[15])*0.5 + (-dsrc[49] + dsrc[55]) + (dsrc[33] - dsrc[39])*1.25 + (dsrc[19] - dsrc[21])*1.3125 + (-dsrc[41] + dsrc[47])*2 + (dsrc[25] - dsrc[31])*2.5 + (dsrc[11] - dsrc[13])*2.625 + (dsrc[51] - dsrc[53])*5.25 + (-dsrc[35] + dsrc[37])*6.5625 + (dsrc[43] - dsrc[45])*10.5 + (-dsrc[27] + dsrc[29])*13.125;
    ddst[tileCount +32*ISTRIDE6X3] = (dsrc[16] - dsrc[22])*0.25 + (-dsrc[8 ] + dsrc[14])*0.5 + (dsrc[48] - dsrc[54]) + (-dsrc[32] + dsrc[38])*1.25 + (-dsrc[18] + dsrc[20])*1.3125 + (-dsrc[40] + dsrc[46])*2 + (dsrc[24] - dsrc[30])*2.5 + (dsrc[10] - dsrc[12])*2.625 + (-dsrc[50] + dsrc[52])*5.25 + (dsrc[34] - dsrc[36])*6.5625 + (dsrc[42] - dsrc[44])*10.5 + (-dsrc[26] + dsrc[28])*13.125;
    ddst[tileCount +33*ISTRIDE6X3] = (dsrc[17] + dsrc[18] + dsrc[21] + dsrc[22])*0.25 + (-dsrc[9 ] - dsrc[10] - dsrc[13] - dsrc[14])*0.5 + (dsrc[49] + dsrc[50] + dsrc[53] + dsrc[54]) + (-dsrc[19] - dsrc[20])*1.0625 + (-dsrc[33] - dsrc[34] - dsrc[37] - dsrc[38])*1.25 + (-dsrc[41] - dsrc[42] - dsrc[45] - dsrc[46])*2 + (dsrc[11] + dsrc[12])*2.125 + (dsrc[25] + dsrc[26] + dsrc[29] + dsrc[30])*2.5 + (-dsrc[51] - dsrc[52])*4.25 + (dsrc[35] + dsrc[36])*5.3125 + (dsrc[43] + dsrc[44])*8.5 + (-dsrc[27] - dsrc[28])*10.625;
    ddst[tileCount +34*ISTRIDE6X3] = (-dsrc[17] + dsrc[18] - dsrc[21] + dsrc[22])*0.25 + (dsrc[9 ] - dsrc[10] + dsrc[13] - dsrc[14])*0.5 + (-dsrc[49] + dsrc[50] - dsrc[53] + dsrc[54]) + (dsrc[19] - dsrc[20])*1.0625 + (dsrc[33] - dsrc[34] + dsrc[37] - dsrc[38])*1.25 + (dsrc[41] - dsrc[42] + dsrc[45] - dsrc[46])*2 + (-dsrc[11] + dsrc[12])*2.125 + (-dsrc[25] + dsrc[26] - dsrc[29] + dsrc[30])*2.5 + (dsrc[51] - dsrc[52])*4.25 + (-dsrc[35] + dsrc[36])*5.3125 + (-dsrc[43] + dsrc[44])*8.5 + (dsrc[27] - dsrc[28])*10.625;
    ddst[tileCount +35*ISTRIDE6X3] = (dsrc[18])*0.0625 + (-dsrc[10] + dsrc[17])*0.125 + (-dsrc[9 ] + dsrc[22] + dsrc[50])*0.25 + (-dsrc[20] - dsrc[34])*0.3125 + (-dsrc[14] + dsrc[21] - dsrc[42] + dsrc[49])*0.5 + (dsrc[12] - dsrc[19] + dsrc[26] - dsrc[33])*0.625 + (-dsrc[13] - dsrc[41] + dsrc[54]) + (dsrc[11] + dsrc[25] - dsrc[38] - dsrc[52])*1.25 + (dsrc[36])*1.5625 + (-dsrc[46] + dsrc[53])*2 + (dsrc[30] - dsrc[37] + dsrc[44] - dsrc[51])*2.5 + (-dsrc[28] + dsrc[35])*3.125 + (-dsrc[45])*4 + (dsrc[29] + dsrc[43])*5 + (-dsrc[27])*6.25;
    ddst[tileCount +36*ISTRIDE6X3] = (dsrc[18])*0.0625 + (-dsrc[10] - dsrc[17])*0.125 + (dsrc[9 ] + dsrc[22] + dsrc[50])*0.25 + (-dsrc[20] - dsrc[34])*0.3125 + (-dsrc[14] - dsrc[21] - dsrc[42] - dsrc[49])*0.5 + (dsrc[12] + dsrc[19] + dsrc[26] + dsrc[33])*0.625 + (dsrc[13] + dsrc[41] + dsrc[54]) + (-dsrc[11] - dsrc[25] - dsrc[38] - dsrc[52])*1.25 + (dsrc[36])*1.5625 + (-dsrc[46] - dsrc[53])*2 + (dsrc[30] + dsrc[37] + dsrc[44] + dsrc[51])*2.5 + (-dsrc[28] - dsrc[35])*3.125 + (dsrc[45])*4 + (-dsrc[29] - dsrc[43])*5 + (dsrc[27])*6.25;
    ddst[tileCount +37*ISTRIDE6X3] = (dsrc[21])*0.125 + (-dsrc[13] + dsrc[22])*0.25 + (-dsrc[14] + dsrc[17] + dsrc[53])*0.5 + (-dsrc[19] - dsrc[37])*0.625 + (-dsrc[9 ] + dsrc[18] - dsrc[45] + dsrc[54]) + (dsrc[11] - dsrc[20] + dsrc[29] - dsrc[38])*1.25 + (-dsrc[10] - dsrc[46] + dsrc[49])*2 + (dsrc[12] + dsrc[30] - dsrc[33] - dsrc[51])*2.5 + (dsrc[35])*3.125 + (-dsrc[41] + dsrc[50])*4 + (dsrc[25] - dsrc[34] + dsrc[43] - dsrc[52])*5 + (-dsrc[27] + dsrc[36])*6.25 + (-dsrc[42])*8 + (dsrc[26] + dsrc[44])*10 + (-dsrc[28])*12.5;
    ddst[tileCount +38*ISTRIDE6X3] = (-dsrc[21])*0.125 + (dsrc[13] + dsrc[22])*0.25 + (-dsrc[14] - dsrc[17] - dsrc[53])*0.5 + (dsrc[19] + dsrc[37])*0.625 + (dsrc[9 ] + dsrc[18] + dsrc[45] + dsrc[54]) + (-dsrc[11] - dsrc[20] - dsrc[29] - dsrc[38])*1.25 + (-dsrc[10] - dsrc[46] - dsrc[49])*2 + (dsrc[12] + dsrc[30] + dsrc[33] + dsrc[51])*2.5 + (-dsrc[35])*3.125 + (dsrc[41] + dsrc[50])*4 + (-dsrc[25] - dsrc[34] - dsrc[43] - dsrc[52])*5 + (dsrc[27] + dsrc[36])*6.25 + (-dsrc[42])*8 + (dsrc[26] + dsrc[44])*10 + (-dsrc[28])*12.5;
    ddst[tileCount +39*ISTRIDE6X3] = (-dsrc[17] + dsrc[23])*0.25 + (dsrc[9 ] - dsrc[15])*0.5 + (-dsrc[49] + dsrc[55]) + (dsrc[33] - dsrc[39])*1.25 + (dsrc[19] - dsrc[21])*1.3125 + (dsrc[41] - dsrc[47])*2 + (-dsrc[25] + dsrc[31])*2.5 + (-dsrc[11] + dsrc[13])*2.625 + (dsrc[51] - dsrc[53])*5.25 + (-dsrc[35] + dsrc[37])*6.5625 + (-dsrc[43] + dsrc[45])*10.5 + (dsrc[27] - dsrc[29])*13.125;
    ddst[tileCount +40*ISTRIDE6X3] = (dsrc[40] - dsrc[46])*0.5 + (dsrc[48] - dsrc[54]) + (dsrc[8 ] - dsrc[14])*2 + (-dsrc[24] + dsrc[30])*2.5 + (-dsrc[42] + dsrc[44])*2.625 + (dsrc[16] - dsrc[22])*4 + (-dsrc[32] + dsrc[38])*5 + (-dsrc[50] + dsrc[52])*5.25 + (-dsrc[10] + dsrc[12])*10.5 + (dsrc[26] - dsrc[28])*13.125 + (-dsrc[18] + dsrc[20])*21 + (dsrc[34] - dsrc[36])*26.25;
    ddst[tileCount +41*ISTRIDE6X3] = (dsrc[41] + dsrc[42] + dsrc[45] + dsrc[46])*0.5 + (dsrc[49] + dsrc[50] + dsrc[53] + dsrc[54]) + (dsrc[9 ] + dsrc[10] + dsrc[13] + dsrc[14])*2 + (-dsrc[43] - dsrc[44])*2.125 + (-dsrc[25] - dsrc[26] - dsrc[29] - dsrc[30])*2.5 + (dsrc[17] + dsrc[18] + dsrc[21] + dsrc[22])*4 + (-dsrc[51] - dsrc[52])*4.25 + (-dsrc[33] - dsrc[34] - dsrc[37] - dsrc[38])*5 + (-dsrc[11] - dsrc[12])*8.5 + (dsrc[27] + dsrc[28])*10.625 + (-dsrc[19] - dsrc[20])*17 + (dsrc[35] + dsrc[36])*21.25;
    ddst[tileCount +42*ISTRIDE6X3] = (-dsrc[41] + dsrc[42] - dsrc[45] + dsrc[46])*0.5 + (-dsrc[49] + dsrc[50] - dsrc[53] + dsrc[54]) + (-dsrc[9 ] + dsrc[10] - dsrc[13] + dsrc[14])*2 + (dsrc[43] - dsrc[44])*2.125 + (dsrc[25] - dsrc[26] + dsrc[29] - dsrc[30])*2.5 + (-dsrc[17] + dsrc[18] - dsrc[21] + dsrc[22])*4 + (dsrc[51] - dsrc[52])*4.25 + (dsrc[33] - dsrc[34] + dsrc[37] - dsrc[38])*5 + (dsrc[11] - dsrc[12])*8.5 + (-dsrc[27] + dsrc[28])*10.625 + (dsrc[19] - dsrc[20])*17 + (-dsrc[35] + dsrc[36])*21.25;
    ddst[tileCount +43*ISTRIDE6X3] = (dsrc[42])*0.125 + (dsrc[41] + dsrc[50])*0.25 + (dsrc[10] + dsrc[46] + dsrc[49])*0.5 + (-dsrc[26] - dsrc[44])*0.625 + (dsrc[9 ] + dsrc[18] + dsrc[45] + dsrc[54]) + (-dsrc[25] - dsrc[34] - dsrc[43] - dsrc[52])*1.25 + (dsrc[14] + dsrc[17] + dsrc[53])*2 + (-dsrc[12] - dsrc[30] - dsrc[33] - dsrc[51])*2.5 + (dsrc[28])*3.125 + (dsrc[13] + dsrc[22])*4 + (-dsrc[11] - dsrc[20] - dsrc[29] - dsrc[38])*5 + (dsrc[27] + dsrc[36])*6.25 + (dsrc[21])*8 + (-dsrc[19] - dsrc[37])*10 + (dsrc[35])*12.5;
    ddst[tileCount +44*ISTRIDE6X3] = (dsrc[42])*0.125 + (-dsrc[41] + dsrc[50])*0.25 + (dsrc[10] + dsrc[46] - dsrc[49])*0.5 + (-dsrc[26] - dsrc[44])*0.625 + (-dsrc[9 ] + dsrc[18] - dsrc[45] + dsrc[54]) + (dsrc[25] - dsrc[34] + dsrc[43] - dsrc[52])*1.25 + (dsrc[14] - dsrc[17] - dsrc[53])*2 + (-dsrc[12] - dsrc[30] + dsrc[33] + dsrc[51])*2.5 + (dsrc[28])*3.125 + (-dsrc[13] + dsrc[22])*4 + (dsrc[11] - dsrc[20] + dsrc[29] - dsrc[38])*5 + (-dsrc[27] + dsrc[36])*6.25 + (-dsrc[21])*8 + (dsrc[19] + dsrc[37])*10 + (-dsrc[35])*12.5;
    ddst[tileCount +45*ISTRIDE6X3] = (dsrc[45])*0.25 + (dsrc[46] + dsrc[53])*0.5 + (dsrc[13] + dsrc[41] + dsrc[54]) + (-dsrc[29] - dsrc[43])*1.25 + (dsrc[14] + dsrc[21] + dsrc[42] + dsrc[49])*2 + (-dsrc[30] - dsrc[37] - dsrc[44] - dsrc[51])*2.5 + (dsrc[9 ] + dsrc[22] + dsrc[50])*4 + (-dsrc[11] - dsrc[25] - dsrc[38] - dsrc[52])*5 + (dsrc[27])*6.25 + (dsrc[10] + dsrc[17])*8 + (-dsrc[12] - dsrc[19] - dsrc[26] - dsrc[33])*10 + (dsrc[28] + dsrc[35])*12.5 + (dsrc[18])*16 + (-dsrc[20] - dsrc[34])*20 + (dsrc[36])*25;
    ddst[tileCount +46*ISTRIDE6X3] = (-dsrc[45])*0.25 + (dsrc[46] - dsrc[53])*0.5 + (-dsrc[13] - dsrc[41] + dsrc[54]) + (dsrc[29] + dsrc[43])*1.25 + (dsrc[14] - dsrc[21] + dsrc[42] - dsrc[49])*2 + (-dsrc[30] + dsrc[37] - dsrc[44] + dsrc[51])*2.5 + (-dsrc[9 ] + dsrc[22] + dsrc[50])*4 + (dsrc[11] + dsrc[25] - dsrc[38] - dsrc[52])*5 + (-dsrc[27])*6.25 + (dsrc[10] - dsrc[17])*8 + (-dsrc[12] + dsrc[19] - dsrc[26] + dsrc[33])*10 + (dsrc[28] - dsrc[35])*12.5 + (dsrc[18])*16 + (-dsrc[20] - dsrc[34])*20 + (dsrc[36])*25;
    ddst[tileCount +47*ISTRIDE6X3] = (-dsrc[41] + dsrc[47])*0.5 + (-dsrc[49] + dsrc[55]) + (-dsrc[9 ] + dsrc[15])*2 + (dsrc[25] - dsrc[31])*2.5 + (dsrc[43] - dsrc[45])*2.625 + (-dsrc[17] + dsrc[23])*4 + (dsrc[33] - dsrc[39])*5 + (dsrc[51] - dsrc[53])*5.25 + (dsrc[11] - dsrc[13])*10.5 + (-dsrc[27] + dsrc[29])*13.125 + (dsrc[19] - dsrc[21])*21 + (-dsrc[35] + dsrc[37])*26.25;
    ddst[tileCount +48*ISTRIDE6X3] = (-dsrc[40] + dsrc[46])*0.5 + (dsrc[48] - dsrc[54]) + (-dsrc[8 ] + dsrc[14])*2 + (dsrc[24] - dsrc[30])*2.5 + (dsrc[42] - dsrc[44])*2.625 + (dsrc[16] - dsrc[22])*4 + (-dsrc[32] + dsrc[38])*5 + (-dsrc[50] + dsrc[52])*5.25 + (dsrc[10] - dsrc[12])*10.5 + (-dsrc[26] + dsrc[28])*13.125 + (-dsrc[18] + dsrc[20])*21 + (dsrc[34] - dsrc[36])*26.25;
    ddst[tileCount +49*ISTRIDE6X3] = (-dsrc[41] - dsrc[42] - dsrc[45] - dsrc[46])*0.5 + (dsrc[49] + dsrc[50] + dsrc[53] + dsrc[54]) + (-dsrc[9 ] - dsrc[10] - dsrc[13] - dsrc[14])*2 + (dsrc[43] + dsrc[44])*2.125 + (dsrc[25] + dsrc[26] + dsrc[29] + dsrc[30])*2.5 + (dsrc[17] + dsrc[18] + dsrc[21] + dsrc[22])*4 + (-dsrc[51] - dsrc[52])*4.25 + (-dsrc[33] - dsrc[34] - dsrc[37] - dsrc[38])*5 + (dsrc[11] + dsrc[12])*8.5 + (-dsrc[27] - dsrc[28])*10.625 + (-dsrc[19] - dsrc[20])*17 + (dsrc[35] + dsrc[36])*21.25;
    ddst[tileCount +50*ISTRIDE6X3] = (dsrc[41] - dsrc[42] + dsrc[45] - dsrc[46])*0.5 + (-dsrc[49] + dsrc[50] - dsrc[53] + dsrc[54]) + (dsrc[9 ] - dsrc[10] + dsrc[13] - dsrc[14])*2 + (-dsrc[43] + dsrc[44])*2.125 + (-dsrc[25] + dsrc[26] - dsrc[29] + dsrc[30])*2.5 + (-dsrc[17] + dsrc[18] - dsrc[21] + dsrc[22])*4 + (dsrc[51] - dsrc[52])*4.25 + (dsrc[33] - dsrc[34] + dsrc[37] - dsrc[38])*5 + (-dsrc[11] + dsrc[12])*8.5 + (dsrc[27] - dsrc[28])*10.625 + (dsrc[19] - dsrc[20])*17 + (-dsrc[35] + dsrc[36])*21.25;
    ddst[tileCount +51*ISTRIDE6X3] = (-dsrc[42])*0.125 + (-dsrc[41] + dsrc[50])*0.25 + (-dsrc[10] - dsrc[46] + dsrc[49])*0.5 + (dsrc[26] + dsrc[44])*0.625 + (-dsrc[9 ] + dsrc[18] - dsrc[45] + dsrc[54]) + (dsrc[25] - dsrc[34] + dsrc[43] - dsrc[52])*1.25 + (-dsrc[14] + dsrc[17] + dsrc[53])*2 + (dsrc[12] + dsrc[30] - dsrc[33] - dsrc[51])*2.5 + (-dsrc[28])*3.125 + (-dsrc[13] + dsrc[22])*4 + (dsrc[11] - dsrc[20] + dsrc[29] - dsrc[38])*5 + (-dsrc[27] + dsrc[36])*6.25 + (dsrc[21])*8 + (-dsrc[19] - dsrc[37])*10 + (dsrc[35])*12.5;
    ddst[tileCount +52*ISTRIDE6X3] = (-dsrc[42])*0.125 + (dsrc[41] + dsrc[50])*0.25 + (-dsrc[10] - dsrc[46] - dsrc[49])*0.5 + (dsrc[26] + dsrc[44])*0.625 + (dsrc[9 ] + dsrc[18] + dsrc[45] + dsrc[54]) + (-dsrc[25] - dsrc[34] - dsrc[43] - dsrc[52])*1.25 + (-dsrc[14] - dsrc[17] - dsrc[53])*2 + (dsrc[12] + dsrc[30] + dsrc[33] + dsrc[51])*2.5 + (-dsrc[28])*3.125 + (dsrc[13] + dsrc[22])*4 + (-dsrc[11] - dsrc[20] - dsrc[29] - dsrc[38])*5 + (dsrc[27] + dsrc[36])*6.25 + (-dsrc[21])*8 + (dsrc[19] + dsrc[37])*10 + (-dsrc[35])*12.5;
    ddst[tileCount +53*ISTRIDE6X3] = (-dsrc[45])*0.25 + (-dsrc[46] + dsrc[53])*0.5 + (-dsrc[13] - dsrc[41] + dsrc[54]) + (dsrc[29] + dsrc[43])*1.25 + (-dsrc[14] + dsrc[21] - dsrc[42] + dsrc[49])*2 + (dsrc[30] - dsrc[37] + dsrc[44] - dsrc[51])*2.5 + (-dsrc[9 ] + dsrc[22] + dsrc[50])*4 + (dsrc[11] + dsrc[25] - dsrc[38] - dsrc[52])*5 + (-dsrc[27])*6.25 + (-dsrc[10] + dsrc[17])*8 + (dsrc[12] - dsrc[19] + dsrc[26] - dsrc[33])*10 + (-dsrc[28] + dsrc[35])*12.5 + (dsrc[18])*16 + (-dsrc[20] - dsrc[34])*20 + (dsrc[36])*25;
    ddst[tileCount +54*ISTRIDE6X3] = (dsrc[45])*0.25 + (-dsrc[46] - dsrc[53])*0.5 + (dsrc[13] + dsrc[41] + dsrc[54]) + (-dsrc[29] - dsrc[43])*1.25 + (-dsrc[14] - dsrc[21] - dsrc[42] - dsrc[49])*2 + (dsrc[30] + dsrc[37] + dsrc[44] + dsrc[51])*2.5 + (dsrc[9 ] + dsrc[22] + dsrc[50])*4 + (-dsrc[11] - dsrc[25] - dsrc[38] - dsrc[52])*5 + (dsrc[27])*6.25 + (-dsrc[10] - dsrc[17])*8 + (dsrc[12] + dsrc[19] + dsrc[26] + dsrc[33])*10 + (-dsrc[28] - dsrc[35])*12.5 + (dsrc[18])*16 + (-dsrc[20] - dsrc[34])*20 + (dsrc[36])*25;
    ddst[tileCount +55*ISTRIDE6X3] = (dsrc[41] - dsrc[47])*0.5 + (-dsrc[49] + dsrc[55]) + (dsrc[9 ] - dsrc[15])*2 + (-dsrc[25] + dsrc[31])*2.5 + (-dsrc[43] + dsrc[45])*2.625 + (-dsrc[17] + dsrc[23])*4 + (dsrc[33] - dsrc[39])*5 + (dsrc[51] - dsrc[53])*5.25 + (-dsrc[11] + dsrc[13])*10.5 + (dsrc[27] - dsrc[29])*13.125 + (dsrc[19] - dsrc[21])*21 + (-dsrc[35] + dsrc[37])*26.25;
    ddst[tileCount +56*ISTRIDE6X3] = (-dsrc[8 ] + dsrc[14] + dsrc[56] - dsrc[62]) + (dsrc[10] - dsrc[12] + dsrc[24] - dsrc[30] - dsrc[40] + dsrc[46] - dsrc[58] + dsrc[60])*5.25 + (-dsrc[26] + dsrc[28] + dsrc[42] - dsrc[44])*27.5625;
    ddst[tileCount +57*ISTRIDE6X3] = (-dsrc[9 ] - dsrc[10] - dsrc[13] - dsrc[14] + dsrc[57] + dsrc[58] + dsrc[61] + dsrc[62]) + (dsrc[11] + dsrc[12] - dsrc[59] - dsrc[60])*4.25 + (dsrc[25] + dsrc[26] + dsrc[29] + dsrc[30] - dsrc[41] - dsrc[42] - dsrc[45] - dsrc[46])*5.25 + (-dsrc[27] - dsrc[28] + dsrc[43] + dsrc[44])*22.3125;
    ddst[tileCount +58*ISTRIDE6X3] = (dsrc[9 ] - dsrc[10] + dsrc[13] - dsrc[14] - dsrc[57] + dsrc[58] - dsrc[61] + dsrc[62]) + (-dsrc[11] + dsrc[12] + dsrc[59] - dsrc[60])*4.25 + (-dsrc[25] + dsrc[26] - dsrc[29] + dsrc[30] + dsrc[41] - dsrc[42] + dsrc[45] - dsrc[46])*5.25 + (dsrc[27] - dsrc[28] - dsrc[43] + dsrc[44])*22.3125;
    ddst[tileCount +59*ISTRIDE6X3] = (-dsrc[10] + dsrc[58])*0.25 + (-dsrc[9 ] + dsrc[57])*0.5 + (-dsrc[14] + dsrc[62]) + (dsrc[12] - dsrc[60])*1.25 + (dsrc[26] - dsrc[42])*1.3125 + (-dsrc[13] + dsrc[61])*2 + (dsrc[11] - dsrc[59])*2.5 + (dsrc[25] - dsrc[41])*2.625 + (dsrc[30] - dsrc[46])*5.25 + (-dsrc[28] + dsrc[44])*6.5625 + (dsrc[29] - dsrc[45])*10.5 + (-dsrc[27] + dsrc[43])*13.125;
    ddst[tileCount +60*ISTRIDE6X3] = (-dsrc[10] + dsrc[58])*0.25 + (dsrc[9 ] - dsrc[57])*0.5 + (-dsrc[14] + dsrc[62]) + (dsrc[12] - dsrc[60])*1.25 + (dsrc[26] - dsrc[42])*1.3125 + (dsrc[13] - dsrc[61])*2 + (-dsrc[11] + dsrc[59])*2.5 + (-dsrc[25] + dsrc[41])*2.625 + (dsrc[30] - dsrc[46])*5.25 + (-dsrc[28] + dsrc[44])*6.5625 + (-dsrc[29] + dsrc[45])*10.5 + (dsrc[27] - dsrc[43])*13.125;
    ddst[tileCount +61*ISTRIDE6X3] = (-dsrc[13] + dsrc[61])*0.5 + (-dsrc[14] + dsrc[62]) + (-dsrc[9 ] + dsrc[57])*2 + (dsrc[11] - dsrc[59])*2.5 + (dsrc[29] - dsrc[45])*2.625 + (-dsrc[10] + dsrc[58])*4 + (dsrc[12] - dsrc[60])*5 + (dsrc[30] - dsrc[46])*5.25 + (dsrc[25] - dsrc[41])*10.5 + (-dsrc[27] + dsrc[43])*13.125 + (dsrc[26] - dsrc[42])*21 + (-dsrc[28] + dsrc[44])*26.25;
    ddst[tileCount +62*ISTRIDE6X3] = (dsrc[13] - dsrc[61])*0.5 + (-dsrc[14] + dsrc[62]) + (dsrc[9 ] - dsrc[57])*2 + (-dsrc[11] + dsrc[59])*2.5 + (-dsrc[29] + dsrc[45])*2.625 + (-dsrc[10] + dsrc[58])*4 + (dsrc[12] - dsrc[60])*5 + (dsrc[30] - dsrc[46])*5.25 + (-dsrc[25] + dsrc[41])*10.5 + (dsrc[27] - dsrc[43])*13.125 + (dsrc[26] - dsrc[42])*21 + (-dsrc[28] + dsrc[44])*26.25;
    ddst[tileCount +63*ISTRIDE6X3] = (dsrc[9 ] - dsrc[15] - dsrc[57] + dsrc[63]) + (-dsrc[11] + dsrc[13] - dsrc[25] + dsrc[31] + dsrc[41] - dsrc[47] + dsrc[59] - dsrc[61])*5.25 + (dsrc[27] - dsrc[29] - dsrc[43] + dsrc[45])*27.5625;
}

    template <typename Dtype>
inline void transformByBT_first(Dtype *dsrc, Dtype *ddst)
{
    ddst[ 0] = BT[ 0]*dsrc[ 0] + BT[ 1]*dsrc[ 8] + BT[ 2]*dsrc[16] + BT[ 3]*dsrc[24] + BT[ 4]*dsrc[32] + BT[ 5]*dsrc[40] + BT[ 6]*dsrc[48] + BT[ 7]*dsrc[56];
    ddst[ 1] = BT[ 0]*dsrc[ 1] + BT[ 1]*dsrc[ 9] + BT[ 2]*dsrc[17] + BT[ 3]*dsrc[25] + BT[ 4]*dsrc[33] + BT[ 5]*dsrc[41] + BT[ 6]*dsrc[49] + BT[ 7]*dsrc[57];
    ddst[ 2] = BT[ 0]*dsrc[ 2] + BT[ 1]*dsrc[10] + BT[ 2]*dsrc[18] + BT[ 3]*dsrc[26] + BT[ 4]*dsrc[34] + BT[ 5]*dsrc[42] + BT[ 6]*dsrc[50] + BT[ 7]*dsrc[58];
    ddst[ 3] = BT[ 0]*dsrc[ 3] + BT[ 1]*dsrc[11] + BT[ 2]*dsrc[19] + BT[ 3]*dsrc[27] + BT[ 4]*dsrc[35] + BT[ 5]*dsrc[43] + BT[ 6]*dsrc[51] + BT[ 7]*dsrc[59];
    ddst[ 4] = BT[ 0]*dsrc[ 4] + BT[ 1]*dsrc[12] + BT[ 2]*dsrc[20] + BT[ 3]*dsrc[28] + BT[ 4]*dsrc[36] + BT[ 5]*dsrc[44] + BT[ 6]*dsrc[52] + BT[ 7]*dsrc[60];
    ddst[ 5] = BT[ 0]*dsrc[ 5] + BT[ 1]*dsrc[13] + BT[ 2]*dsrc[21] + BT[ 3]*dsrc[29] + BT[ 4]*dsrc[37] + BT[ 5]*dsrc[45] + BT[ 6]*dsrc[53] + BT[ 7]*dsrc[61];
    ddst[ 6] = BT[ 0]*dsrc[ 6] + BT[ 1]*dsrc[14] + BT[ 2]*dsrc[22] + BT[ 3]*dsrc[30] + BT[ 4]*dsrc[38] + BT[ 5]*dsrc[46] + BT[ 6]*dsrc[54] + BT[ 7]*dsrc[62];
    ddst[ 7] = BT[ 0]*dsrc[ 7] + BT[ 1]*dsrc[15] + BT[ 2]*dsrc[23] + BT[ 3]*dsrc[31] + BT[ 4]*dsrc[39] + BT[ 5]*dsrc[47] + BT[ 6]*dsrc[55] + BT[ 7]*dsrc[63];
    ddst[ 8] = BT[ 8]*dsrc[ 0] + BT[ 9]*dsrc[ 8] + BT[10]*dsrc[16] + BT[11]*dsrc[24] + BT[12]*dsrc[32] + BT[13]*dsrc[40] + BT[14]*dsrc[48] + BT[15]*dsrc[56];
    ddst[ 9] = BT[ 8]*dsrc[ 1] + BT[ 9]*dsrc[ 9] + BT[10]*dsrc[17] + BT[11]*dsrc[25] + BT[12]*dsrc[33] + BT[13]*dsrc[41] + BT[14]*dsrc[49] + BT[15]*dsrc[57];
    ddst[10] = BT[ 8]*dsrc[ 2] + BT[ 9]*dsrc[10] + BT[10]*dsrc[18] + BT[11]*dsrc[26] + BT[12]*dsrc[34] + BT[13]*dsrc[42] + BT[14]*dsrc[50] + BT[15]*dsrc[58];
    ddst[11] = BT[ 8]*dsrc[ 3] + BT[ 9]*dsrc[11] + BT[10]*dsrc[19] + BT[11]*dsrc[27] + BT[12]*dsrc[35] + BT[13]*dsrc[43] + BT[14]*dsrc[51] + BT[15]*dsrc[59];
    ddst[12] = BT[ 8]*dsrc[ 4] + BT[ 9]*dsrc[12] + BT[10]*dsrc[20] + BT[11]*dsrc[28] + BT[12]*dsrc[36] + BT[13]*dsrc[44] + BT[14]*dsrc[52] + BT[15]*dsrc[60];
    ddst[13] = BT[ 8]*dsrc[ 5] + BT[ 9]*dsrc[13] + BT[10]*dsrc[21] + BT[11]*dsrc[29] + BT[12]*dsrc[37] + BT[13]*dsrc[45] + BT[14]*dsrc[53] + BT[15]*dsrc[61];
    ddst[14] = BT[ 8]*dsrc[ 6] + BT[ 9]*dsrc[14] + BT[10]*dsrc[22] + BT[11]*dsrc[30] + BT[12]*dsrc[38] + BT[13]*dsrc[46] + BT[14]*dsrc[54] + BT[15]*dsrc[62];
    ddst[15] = BT[ 8]*dsrc[ 7] + BT[ 9]*dsrc[15] + BT[10]*dsrc[23] + BT[11]*dsrc[31] + BT[12]*dsrc[39] + BT[13]*dsrc[47] + BT[14]*dsrc[55] + BT[15]*dsrc[63];
    ddst[16] = BT[16]*dsrc[ 0] + BT[17]*dsrc[ 8] + BT[18]*dsrc[16] + BT[19]*dsrc[24] + BT[20]*dsrc[32] + BT[21]*dsrc[40] + BT[22]*dsrc[48] + BT[23]*dsrc[56];
    ddst[17] = BT[16]*dsrc[ 1] + BT[17]*dsrc[ 9] + BT[18]*dsrc[17] + BT[19]*dsrc[25] + BT[20]*dsrc[33] + BT[21]*dsrc[41] + BT[22]*dsrc[49] + BT[23]*dsrc[57];
    ddst[18] = BT[16]*dsrc[ 2] + BT[17]*dsrc[10] + BT[18]*dsrc[18] + BT[19]*dsrc[26] + BT[20]*dsrc[34] + BT[21]*dsrc[42] + BT[22]*dsrc[50] + BT[23]*dsrc[58];
    ddst[19] = BT[16]*dsrc[ 3] + BT[17]*dsrc[11] + BT[18]*dsrc[19] + BT[19]*dsrc[27] + BT[20]*dsrc[35] + BT[21]*dsrc[43] + BT[22]*dsrc[51] + BT[23]*dsrc[59];
    ddst[20] = BT[16]*dsrc[ 4] + BT[17]*dsrc[12] + BT[18]*dsrc[20] + BT[19]*dsrc[28] + BT[20]*dsrc[36] + BT[21]*dsrc[44] + BT[22]*dsrc[52] + BT[23]*dsrc[60];
    ddst[21] = BT[16]*dsrc[ 5] + BT[17]*dsrc[13] + BT[18]*dsrc[21] + BT[19]*dsrc[29] + BT[20]*dsrc[37] + BT[21]*dsrc[45] + BT[22]*dsrc[53] + BT[23]*dsrc[61];
    ddst[22] = BT[16]*dsrc[ 6] + BT[17]*dsrc[14] + BT[18]*dsrc[22] + BT[19]*dsrc[30] + BT[20]*dsrc[38] + BT[21]*dsrc[46] + BT[22]*dsrc[54] + BT[23]*dsrc[62];
    ddst[23] = BT[16]*dsrc[ 7] + BT[17]*dsrc[15] + BT[18]*dsrc[23] + BT[19]*dsrc[31] + BT[20]*dsrc[39] + BT[21]*dsrc[47] + BT[22]*dsrc[55] + BT[23]*dsrc[63];
    ddst[24] = BT[24]*dsrc[ 0] + BT[25]*dsrc[ 8] + BT[26]*dsrc[16] + BT[27]*dsrc[24] + BT[28]*dsrc[32] + BT[29]*dsrc[40] + BT[30]*dsrc[48] + BT[31]*dsrc[56];
    ddst[25] = BT[24]*dsrc[ 1] + BT[25]*dsrc[ 9] + BT[26]*dsrc[17] + BT[27]*dsrc[25] + BT[28]*dsrc[33] + BT[29]*dsrc[41] + BT[30]*dsrc[49] + BT[31]*dsrc[57];
    ddst[26] = BT[24]*dsrc[ 2] + BT[25]*dsrc[10] + BT[26]*dsrc[18] + BT[27]*dsrc[26] + BT[28]*dsrc[34] + BT[29]*dsrc[42] + BT[30]*dsrc[50] + BT[31]*dsrc[58];
    ddst[27] = BT[24]*dsrc[ 3] + BT[25]*dsrc[11] + BT[26]*dsrc[19] + BT[27]*dsrc[27] + BT[28]*dsrc[35] + BT[29]*dsrc[43] + BT[30]*dsrc[51] + BT[31]*dsrc[59];
    ddst[28] = BT[24]*dsrc[ 4] + BT[25]*dsrc[12] + BT[26]*dsrc[20] + BT[27]*dsrc[28] + BT[28]*dsrc[36] + BT[29]*dsrc[44] + BT[30]*dsrc[52] + BT[31]*dsrc[60];
    ddst[29] = BT[24]*dsrc[ 5] + BT[25]*dsrc[13] + BT[26]*dsrc[21] + BT[27]*dsrc[29] + BT[28]*dsrc[37] + BT[29]*dsrc[45] + BT[30]*dsrc[53] + BT[31]*dsrc[61];
    ddst[30] = BT[24]*dsrc[ 6] + BT[25]*dsrc[14] + BT[26]*dsrc[22] + BT[27]*dsrc[30] + BT[28]*dsrc[38] + BT[29]*dsrc[46] + BT[30]*dsrc[54] + BT[31]*dsrc[62];
    ddst[31] = BT[24]*dsrc[ 7] + BT[25]*dsrc[15] + BT[26]*dsrc[23] + BT[27]*dsrc[31] + BT[28]*dsrc[39] + BT[29]*dsrc[47] + BT[30]*dsrc[55] + BT[31]*dsrc[63];
    ddst[32] = BT[32]*dsrc[ 0] + BT[33]*dsrc[ 8] + BT[34]*dsrc[16] + BT[35]*dsrc[24] + BT[36]*dsrc[32] + BT[37]*dsrc[40] + BT[38]*dsrc[48] + BT[39]*dsrc[56];
    ddst[33] = BT[32]*dsrc[ 1] + BT[33]*dsrc[ 9] + BT[34]*dsrc[17] + BT[35]*dsrc[25] + BT[36]*dsrc[33] + BT[37]*dsrc[41] + BT[38]*dsrc[49] + BT[39]*dsrc[57];
    ddst[34] = BT[32]*dsrc[ 2] + BT[33]*dsrc[10] + BT[34]*dsrc[18] + BT[35]*dsrc[26] + BT[36]*dsrc[34] + BT[37]*dsrc[42] + BT[38]*dsrc[50] + BT[39]*dsrc[58];
    ddst[35] = BT[32]*dsrc[ 3] + BT[33]*dsrc[11] + BT[34]*dsrc[19] + BT[35]*dsrc[27] + BT[36]*dsrc[35] + BT[37]*dsrc[43] + BT[38]*dsrc[51] + BT[39]*dsrc[59];
    ddst[36] = BT[32]*dsrc[ 4] + BT[33]*dsrc[12] + BT[34]*dsrc[20] + BT[35]*dsrc[28] + BT[36]*dsrc[36] + BT[37]*dsrc[44] + BT[38]*dsrc[52] + BT[39]*dsrc[60];
    ddst[37] = BT[32]*dsrc[ 5] + BT[33]*dsrc[13] + BT[34]*dsrc[21] + BT[35]*dsrc[29] + BT[36]*dsrc[37] + BT[37]*dsrc[45] + BT[38]*dsrc[53] + BT[39]*dsrc[61];
    ddst[38] = BT[32]*dsrc[ 6] + BT[33]*dsrc[14] + BT[34]*dsrc[22] + BT[35]*dsrc[30] + BT[36]*dsrc[38] + BT[37]*dsrc[46] + BT[38]*dsrc[54] + BT[39]*dsrc[62];
    ddst[39] = BT[32]*dsrc[ 7] + BT[33]*dsrc[15] + BT[34]*dsrc[23] + BT[35]*dsrc[31] + BT[36]*dsrc[39] + BT[37]*dsrc[47] + BT[38]*dsrc[55] + BT[39]*dsrc[63];
    ddst[40] = BT[40]*dsrc[ 0] + BT[41]*dsrc[ 8] + BT[42]*dsrc[16] + BT[43]*dsrc[24] + BT[44]*dsrc[32] + BT[45]*dsrc[40] + BT[46]*dsrc[48] + BT[47]*dsrc[56];
    ddst[41] = BT[40]*dsrc[ 1] + BT[41]*dsrc[ 9] + BT[42]*dsrc[17] + BT[43]*dsrc[25] + BT[44]*dsrc[33] + BT[45]*dsrc[41] + BT[46]*dsrc[49] + BT[47]*dsrc[57];
    ddst[42] = BT[40]*dsrc[ 2] + BT[41]*dsrc[10] + BT[42]*dsrc[18] + BT[43]*dsrc[26] + BT[44]*dsrc[34] + BT[45]*dsrc[42] + BT[46]*dsrc[50] + BT[47]*dsrc[58];
    ddst[43] = BT[40]*dsrc[ 3] + BT[41]*dsrc[11] + BT[42]*dsrc[19] + BT[43]*dsrc[27] + BT[44]*dsrc[35] + BT[45]*dsrc[43] + BT[46]*dsrc[51] + BT[47]*dsrc[59];
    ddst[44] = BT[40]*dsrc[ 4] + BT[41]*dsrc[12] + BT[42]*dsrc[20] + BT[43]*dsrc[28] + BT[44]*dsrc[36] + BT[45]*dsrc[44] + BT[46]*dsrc[52] + BT[47]*dsrc[60];
    ddst[45] = BT[40]*dsrc[ 5] + BT[41]*dsrc[13] + BT[42]*dsrc[21] + BT[43]*dsrc[29] + BT[44]*dsrc[37] + BT[45]*dsrc[45] + BT[46]*dsrc[53] + BT[47]*dsrc[61];
    ddst[46] = BT[40]*dsrc[ 6] + BT[41]*dsrc[14] + BT[42]*dsrc[22] + BT[43]*dsrc[30] + BT[44]*dsrc[38] + BT[45]*dsrc[46] + BT[46]*dsrc[54] + BT[47]*dsrc[62];
    ddst[47] = BT[40]*dsrc[ 7] + BT[41]*dsrc[15] + BT[42]*dsrc[23] + BT[43]*dsrc[31] + BT[44]*dsrc[39] + BT[45]*dsrc[47] + BT[46]*dsrc[55] + BT[47]*dsrc[63];
    ddst[48] = BT[48]*dsrc[ 0] + BT[49]*dsrc[ 8] + BT[50]*dsrc[16] + BT[51]*dsrc[24] + BT[52]*dsrc[32] + BT[53]*dsrc[40] + BT[54]*dsrc[48] + BT[55]*dsrc[56];
    ddst[49] = BT[48]*dsrc[ 1] + BT[49]*dsrc[ 9] + BT[50]*dsrc[17] + BT[51]*dsrc[25] + BT[52]*dsrc[33] + BT[53]*dsrc[41] + BT[54]*dsrc[49] + BT[55]*dsrc[57];
    ddst[50] = BT[48]*dsrc[ 2] + BT[49]*dsrc[10] + BT[50]*dsrc[18] + BT[51]*dsrc[26] + BT[52]*dsrc[34] + BT[53]*dsrc[42] + BT[54]*dsrc[50] + BT[55]*dsrc[58];
    ddst[51] = BT[48]*dsrc[ 3] + BT[49]*dsrc[11] + BT[50]*dsrc[19] + BT[51]*dsrc[27] + BT[52]*dsrc[35] + BT[53]*dsrc[43] + BT[54]*dsrc[51] + BT[55]*dsrc[59];
    ddst[52] = BT[48]*dsrc[ 4] + BT[49]*dsrc[12] + BT[50]*dsrc[20] + BT[51]*dsrc[28] + BT[52]*dsrc[36] + BT[53]*dsrc[44] + BT[54]*dsrc[52] + BT[55]*dsrc[60];
    ddst[53] = BT[48]*dsrc[ 5] + BT[49]*dsrc[13] + BT[50]*dsrc[21] + BT[51]*dsrc[29] + BT[52]*dsrc[37] + BT[53]*dsrc[45] + BT[54]*dsrc[53] + BT[55]*dsrc[61];
    ddst[54] = BT[48]*dsrc[ 6] + BT[49]*dsrc[14] + BT[50]*dsrc[22] + BT[51]*dsrc[30] + BT[52]*dsrc[38] + BT[53]*dsrc[46] + BT[54]*dsrc[54] + BT[55]*dsrc[62];
    ddst[55] = BT[48]*dsrc[ 7] + BT[49]*dsrc[15] + BT[50]*dsrc[23] + BT[51]*dsrc[31] + BT[52]*dsrc[39] + BT[53]*dsrc[47] + BT[54]*dsrc[55] + BT[55]*dsrc[63];
    ddst[56] = BT[56]*dsrc[ 0] + BT[57]*dsrc[ 8] + BT[58]*dsrc[16] + BT[59]*dsrc[24] + BT[60]*dsrc[32] + BT[61]*dsrc[40] + BT[62]*dsrc[48] + BT[63]*dsrc[56];
    ddst[57] = BT[56]*dsrc[ 1] + BT[57]*dsrc[ 9] + BT[58]*dsrc[17] + BT[59]*dsrc[25] + BT[60]*dsrc[33] + BT[61]*dsrc[41] + BT[62]*dsrc[49] + BT[63]*dsrc[57];
    ddst[58] = BT[56]*dsrc[ 2] + BT[57]*dsrc[10] + BT[58]*dsrc[18] + BT[59]*dsrc[26] + BT[60]*dsrc[34] + BT[61]*dsrc[42] + BT[62]*dsrc[50] + BT[63]*dsrc[58];
    ddst[59] = BT[56]*dsrc[ 3] + BT[57]*dsrc[11] + BT[58]*dsrc[19] + BT[59]*dsrc[27] + BT[60]*dsrc[35] + BT[61]*dsrc[43] + BT[62]*dsrc[51] + BT[63]*dsrc[59];
    ddst[60] = BT[56]*dsrc[ 4] + BT[57]*dsrc[12] + BT[58]*dsrc[20] + BT[59]*dsrc[28] + BT[60]*dsrc[36] + BT[61]*dsrc[44] + BT[62]*dsrc[52] + BT[63]*dsrc[60];
    ddst[61] = BT[56]*dsrc[ 5] + BT[57]*dsrc[13] + BT[58]*dsrc[21] + BT[59]*dsrc[29] + BT[60]*dsrc[37] + BT[61]*dsrc[45] + BT[62]*dsrc[53] + BT[63]*dsrc[61];
    ddst[62] = BT[56]*dsrc[ 6] + BT[57]*dsrc[14] + BT[58]*dsrc[22] + BT[59]*dsrc[30] + BT[60]*dsrc[38] + BT[61]*dsrc[46] + BT[62]*dsrc[54] + BT[63]*dsrc[62];
    ddst[63] = BT[56]*dsrc[ 7] + BT[57]*dsrc[15] + BT[58]*dsrc[23] + BT[59]*dsrc[31] + BT[60]*dsrc[39] + BT[61]*dsrc[47] + BT[62]*dsrc[55] + BT[63]*dsrc[63];
}

    template <typename Dtype>
inline void transformByBT_second(Dtype *dsrc, Dtype *ddst, int tileCount)
{
    ddst[tileCount +  0*ISTRIDE6X3] = dsrc[ 0]*BT[ 0] + dsrc[ 1]*BT[ 1] + dsrc[ 2]*BT[ 2] + dsrc[ 3]*BT[ 3] + dsrc[ 4]*BT[ 4] + dsrc[ 5]*BT[ 5] + dsrc[ 6]*BT[ 6] + dsrc[ 7]*BT[ 7];
    ddst[tileCount +  1*ISTRIDE6X3] = dsrc[ 0]*BT[ 8] + dsrc[ 1]*BT[ 9] + dsrc[ 2]*BT[10] + dsrc[ 3]*BT[11] + dsrc[ 4]*BT[12] + dsrc[ 5]*BT[13] + dsrc[ 6]*BT[14] + dsrc[ 7]*BT[15];
    ddst[tileCount +  2*ISTRIDE6X3] = dsrc[ 0]*BT[16] + dsrc[ 1]*BT[17] + dsrc[ 2]*BT[18] + dsrc[ 3]*BT[19] + dsrc[ 4]*BT[20] + dsrc[ 5]*BT[21] + dsrc[ 6]*BT[22] + dsrc[ 7]*BT[23];
    ddst[tileCount +  3*ISTRIDE6X3] = dsrc[ 0]*BT[24] + dsrc[ 1]*BT[25] + dsrc[ 2]*BT[26] + dsrc[ 3]*BT[27] + dsrc[ 4]*BT[28] + dsrc[ 5]*BT[29] + dsrc[ 6]*BT[30] + dsrc[ 7]*BT[31];
    ddst[tileCount +  4*ISTRIDE6X3] = dsrc[ 0]*BT[32] + dsrc[ 1]*BT[33] + dsrc[ 2]*BT[34] + dsrc[ 3]*BT[35] + dsrc[ 4]*BT[36] + dsrc[ 5]*BT[37] + dsrc[ 6]*BT[38] + dsrc[ 7]*BT[39];
    ddst[tileCount +  5*ISTRIDE6X3] = dsrc[ 0]*BT[40] + dsrc[ 1]*BT[41] + dsrc[ 2]*BT[42] + dsrc[ 3]*BT[43] + dsrc[ 4]*BT[44] + dsrc[ 5]*BT[45] + dsrc[ 6]*BT[46] + dsrc[ 7]*BT[47];
    ddst[tileCount +  6*ISTRIDE6X3] = dsrc[ 0]*BT[48] + dsrc[ 1]*BT[49] + dsrc[ 2]*BT[50] + dsrc[ 3]*BT[51] + dsrc[ 4]*BT[52] + dsrc[ 5]*BT[53] + dsrc[ 6]*BT[54] + dsrc[ 7]*BT[55];
    ddst[tileCount +  7*ISTRIDE6X3] = dsrc[ 0]*BT[56] + dsrc[ 1]*BT[57] + dsrc[ 2]*BT[58] + dsrc[ 3]*BT[59] + dsrc[ 4]*BT[60] + dsrc[ 5]*BT[61] + dsrc[ 6]*BT[62] + dsrc[ 7]*BT[63];
    ddst[tileCount +  8*ISTRIDE6X3] = dsrc[ 8]*BT[ 0] + dsrc[ 9]*BT[ 1] + dsrc[10]*BT[ 2] + dsrc[11]*BT[ 3] + dsrc[12]*BT[ 4] + dsrc[13]*BT[ 5] + dsrc[14]*BT[ 6] + dsrc[15]*BT[ 7];
    ddst[tileCount +  9*ISTRIDE6X3] = dsrc[ 8]*BT[ 8] + dsrc[ 9]*BT[ 9] + dsrc[10]*BT[10] + dsrc[11]*BT[11] + dsrc[12]*BT[12] + dsrc[13]*BT[13] + dsrc[14]*BT[14] + dsrc[15]*BT[15];
    ddst[tileCount + 10*ISTRIDE6X3] = dsrc[ 8]*BT[16] + dsrc[ 9]*BT[17] + dsrc[10]*BT[18] + dsrc[11]*BT[19] + dsrc[12]*BT[20] + dsrc[13]*BT[21] + dsrc[14]*BT[22] + dsrc[15]*BT[23];
    ddst[tileCount + 11*ISTRIDE6X3] = dsrc[ 8]*BT[24] + dsrc[ 9]*BT[25] + dsrc[10]*BT[26] + dsrc[11]*BT[27] + dsrc[12]*BT[28] + dsrc[13]*BT[29] + dsrc[14]*BT[30] + dsrc[15]*BT[31];
    ddst[tileCount + 12*ISTRIDE6X3] = dsrc[ 8]*BT[32] + dsrc[ 9]*BT[33] + dsrc[10]*BT[34] + dsrc[11]*BT[35] + dsrc[12]*BT[36] + dsrc[13]*BT[37] + dsrc[14]*BT[38] + dsrc[15]*BT[39];
    ddst[tileCount + 13*ISTRIDE6X3] = dsrc[ 8]*BT[40] + dsrc[ 9]*BT[41] + dsrc[10]*BT[42] + dsrc[11]*BT[43] + dsrc[12]*BT[44] + dsrc[13]*BT[45] + dsrc[14]*BT[46] + dsrc[15]*BT[47];
    ddst[tileCount + 14*ISTRIDE6X3] = dsrc[ 8]*BT[48] + dsrc[ 9]*BT[49] + dsrc[10]*BT[50] + dsrc[11]*BT[51] + dsrc[12]*BT[52] + dsrc[13]*BT[53] + dsrc[14]*BT[54] + dsrc[15]*BT[55];
    ddst[tileCount + 15*ISTRIDE6X3] = dsrc[ 8]*BT[56] + dsrc[ 9]*BT[57] + dsrc[10]*BT[58] + dsrc[11]*BT[59] + dsrc[12]*BT[60] + dsrc[13]*BT[61] + dsrc[14]*BT[62] + dsrc[15]*BT[63];
    ddst[tileCount + 16*ISTRIDE6X3] = dsrc[16]*BT[ 0] + dsrc[17]*BT[ 1] + dsrc[18]*BT[ 2] + dsrc[19]*BT[ 3] + dsrc[20]*BT[ 4] + dsrc[21]*BT[ 5] + dsrc[22]*BT[ 6] + dsrc[23]*BT[ 7];
    ddst[tileCount + 17*ISTRIDE6X3] = dsrc[16]*BT[ 8] + dsrc[17]*BT[ 9] + dsrc[18]*BT[10] + dsrc[19]*BT[11] + dsrc[20]*BT[12] + dsrc[21]*BT[13] + dsrc[22]*BT[14] + dsrc[23]*BT[15];
    ddst[tileCount + 18*ISTRIDE6X3] = dsrc[16]*BT[16] + dsrc[17]*BT[17] + dsrc[18]*BT[18] + dsrc[19]*BT[19] + dsrc[20]*BT[20] + dsrc[21]*BT[21] + dsrc[22]*BT[22] + dsrc[23]*BT[23];
    ddst[tileCount + 19*ISTRIDE6X3] = dsrc[16]*BT[24] + dsrc[17]*BT[25] + dsrc[18]*BT[26] + dsrc[19]*BT[27] + dsrc[20]*BT[28] + dsrc[21]*BT[29] + dsrc[22]*BT[30] + dsrc[23]*BT[31];
    ddst[tileCount + 20*ISTRIDE6X3] = dsrc[16]*BT[32] + dsrc[17]*BT[33] + dsrc[18]*BT[34] + dsrc[19]*BT[35] + dsrc[20]*BT[36] + dsrc[21]*BT[37] + dsrc[22]*BT[38] + dsrc[23]*BT[39];
    ddst[tileCount + 21*ISTRIDE6X3] = dsrc[16]*BT[40] + dsrc[17]*BT[41] + dsrc[18]*BT[42] + dsrc[19]*BT[43] + dsrc[20]*BT[44] + dsrc[21]*BT[45] + dsrc[22]*BT[46] + dsrc[23]*BT[47];
    ddst[tileCount + 22*ISTRIDE6X3] = dsrc[16]*BT[48] + dsrc[17]*BT[49] + dsrc[18]*BT[50] + dsrc[19]*BT[51] + dsrc[20]*BT[52] + dsrc[21]*BT[53] + dsrc[22]*BT[54] + dsrc[23]*BT[55];
    ddst[tileCount + 23*ISTRIDE6X3] = dsrc[16]*BT[56] + dsrc[17]*BT[57] + dsrc[18]*BT[58] + dsrc[19]*BT[59] + dsrc[20]*BT[60] + dsrc[21]*BT[61] + dsrc[22]*BT[62] + dsrc[23]*BT[63];
    ddst[tileCount + 24*ISTRIDE6X3] = dsrc[24]*BT[ 0] + dsrc[25]*BT[ 1] + dsrc[26]*BT[ 2] + dsrc[27]*BT[ 3] + dsrc[28]*BT[ 4] + dsrc[29]*BT[ 5] + dsrc[30]*BT[ 6] + dsrc[31]*BT[ 7];
    ddst[tileCount + 25*ISTRIDE6X3] = dsrc[24]*BT[ 8] + dsrc[25]*BT[ 9] + dsrc[26]*BT[10] + dsrc[27]*BT[11] + dsrc[28]*BT[12] + dsrc[29]*BT[13] + dsrc[30]*BT[14] + dsrc[31]*BT[15];
    ddst[tileCount + 26*ISTRIDE6X3] = dsrc[24]*BT[16] + dsrc[25]*BT[17] + dsrc[26]*BT[18] + dsrc[27]*BT[19] + dsrc[28]*BT[20] + dsrc[29]*BT[21] + dsrc[30]*BT[22] + dsrc[31]*BT[23];
    ddst[tileCount + 27*ISTRIDE6X3] = dsrc[24]*BT[24] + dsrc[25]*BT[25] + dsrc[26]*BT[26] + dsrc[27]*BT[27] + dsrc[28]*BT[28] + dsrc[29]*BT[29] + dsrc[30]*BT[30] + dsrc[31]*BT[31];
    ddst[tileCount + 28*ISTRIDE6X3] = dsrc[24]*BT[32] + dsrc[25]*BT[33] + dsrc[26]*BT[34] + dsrc[27]*BT[35] + dsrc[28]*BT[36] + dsrc[29]*BT[37] + dsrc[30]*BT[38] + dsrc[31]*BT[39];
    ddst[tileCount + 29*ISTRIDE6X3] = dsrc[24]*BT[40] + dsrc[25]*BT[41] + dsrc[26]*BT[42] + dsrc[27]*BT[43] + dsrc[28]*BT[44] + dsrc[29]*BT[45] + dsrc[30]*BT[46] + dsrc[31]*BT[47];
    ddst[tileCount + 30*ISTRIDE6X3] = dsrc[24]*BT[48] + dsrc[25]*BT[49] + dsrc[26]*BT[50] + dsrc[27]*BT[51] + dsrc[28]*BT[52] + dsrc[29]*BT[53] + dsrc[30]*BT[54] + dsrc[31]*BT[55];
    ddst[tileCount + 31*ISTRIDE6X3] = dsrc[24]*BT[56] + dsrc[25]*BT[57] + dsrc[26]*BT[58] + dsrc[27]*BT[59] + dsrc[28]*BT[60] + dsrc[29]*BT[61] + dsrc[30]*BT[62] + dsrc[31]*BT[63];
    ddst[tileCount + 32*ISTRIDE6X3] = dsrc[32]*BT[ 0] + dsrc[33]*BT[ 1] + dsrc[34]*BT[ 2] + dsrc[35]*BT[ 3] + dsrc[36]*BT[ 4] + dsrc[37]*BT[ 5] + dsrc[38]*BT[ 6] + dsrc[39]*BT[ 7];
    ddst[tileCount + 33*ISTRIDE6X3] = dsrc[32]*BT[ 8] + dsrc[33]*BT[ 9] + dsrc[34]*BT[10] + dsrc[35]*BT[11] + dsrc[36]*BT[12] + dsrc[37]*BT[13] + dsrc[38]*BT[14] + dsrc[39]*BT[15];
    ddst[tileCount + 34*ISTRIDE6X3] = dsrc[32]*BT[16] + dsrc[33]*BT[17] + dsrc[34]*BT[18] + dsrc[35]*BT[19] + dsrc[36]*BT[20] + dsrc[37]*BT[21] + dsrc[38]*BT[22] + dsrc[39]*BT[23];
    ddst[tileCount + 35*ISTRIDE6X3] = dsrc[32]*BT[24] + dsrc[33]*BT[25] + dsrc[34]*BT[26] + dsrc[35]*BT[27] + dsrc[36]*BT[28] + dsrc[37]*BT[29] + dsrc[38]*BT[30] + dsrc[39]*BT[31];
    ddst[tileCount + 36*ISTRIDE6X3] = dsrc[32]*BT[32] + dsrc[33]*BT[33] + dsrc[34]*BT[34] + dsrc[35]*BT[35] + dsrc[36]*BT[36] + dsrc[37]*BT[37] + dsrc[38]*BT[38] + dsrc[39]*BT[39];
    ddst[tileCount + 37*ISTRIDE6X3] = dsrc[32]*BT[40] + dsrc[33]*BT[41] + dsrc[34]*BT[42] + dsrc[35]*BT[43] + dsrc[36]*BT[44] + dsrc[37]*BT[45] + dsrc[38]*BT[46] + dsrc[39]*BT[47];
    ddst[tileCount + 38*ISTRIDE6X3] = dsrc[32]*BT[48] + dsrc[33]*BT[49] + dsrc[34]*BT[50] + dsrc[35]*BT[51] + dsrc[36]*BT[52] + dsrc[37]*BT[53] + dsrc[38]*BT[54] + dsrc[39]*BT[55];
    ddst[tileCount + 39*ISTRIDE6X3] = dsrc[32]*BT[56] + dsrc[33]*BT[57] + dsrc[34]*BT[58] + dsrc[35]*BT[59] + dsrc[36]*BT[60] + dsrc[37]*BT[61] + dsrc[38]*BT[62] + dsrc[39]*BT[63];
    ddst[tileCount + 40*ISTRIDE6X3] = dsrc[40]*BT[ 0] + dsrc[41]*BT[ 1] + dsrc[42]*BT[ 2] + dsrc[43]*BT[ 3] + dsrc[44]*BT[ 4] + dsrc[45]*BT[ 5] + dsrc[46]*BT[ 6] + dsrc[47]*BT[ 7];
    ddst[tileCount + 41*ISTRIDE6X3] = dsrc[40]*BT[ 8] + dsrc[41]*BT[ 9] + dsrc[42]*BT[10] + dsrc[43]*BT[11] + dsrc[44]*BT[12] + dsrc[45]*BT[13] + dsrc[46]*BT[14] + dsrc[47]*BT[15];
    ddst[tileCount + 42*ISTRIDE6X3] = dsrc[40]*BT[16] + dsrc[41]*BT[17] + dsrc[42]*BT[18] + dsrc[43]*BT[19] + dsrc[44]*BT[20] + dsrc[45]*BT[21] + dsrc[46]*BT[22] + dsrc[47]*BT[23];
    ddst[tileCount + 43*ISTRIDE6X3] = dsrc[40]*BT[24] + dsrc[41]*BT[25] + dsrc[42]*BT[26] + dsrc[43]*BT[27] + dsrc[44]*BT[28] + dsrc[45]*BT[29] + dsrc[46]*BT[30] + dsrc[47]*BT[31];
    ddst[tileCount + 44*ISTRIDE6X3] = dsrc[40]*BT[32] + dsrc[41]*BT[33] + dsrc[42]*BT[34] + dsrc[43]*BT[35] + dsrc[44]*BT[36] + dsrc[45]*BT[37] + dsrc[46]*BT[38] + dsrc[47]*BT[39];
    ddst[tileCount + 45*ISTRIDE6X3] = dsrc[40]*BT[40] + dsrc[41]*BT[41] + dsrc[42]*BT[42] + dsrc[43]*BT[43] + dsrc[44]*BT[44] + dsrc[45]*BT[45] + dsrc[46]*BT[46] + dsrc[47]*BT[47];
    ddst[tileCount + 46*ISTRIDE6X3] = dsrc[40]*BT[48] + dsrc[41]*BT[49] + dsrc[42]*BT[50] + dsrc[43]*BT[51] + dsrc[44]*BT[52] + dsrc[45]*BT[53] + dsrc[46]*BT[54] + dsrc[47]*BT[55];
    ddst[tileCount + 47*ISTRIDE6X3] = dsrc[40]*BT[56] + dsrc[41]*BT[57] + dsrc[42]*BT[58] + dsrc[43]*BT[59] + dsrc[44]*BT[60] + dsrc[45]*BT[61] + dsrc[46]*BT[62] + dsrc[47]*BT[63];
    ddst[tileCount + 48*ISTRIDE6X3] = dsrc[48]*BT[ 0] + dsrc[49]*BT[ 1] + dsrc[50]*BT[ 2] + dsrc[51]*BT[ 3] + dsrc[52]*BT[ 4] + dsrc[53]*BT[ 5] + dsrc[54]*BT[ 6] + dsrc[55]*BT[ 7];
    ddst[tileCount + 49*ISTRIDE6X3] = dsrc[48]*BT[ 8] + dsrc[49]*BT[ 9] + dsrc[50]*BT[10] + dsrc[51]*BT[11] + dsrc[52]*BT[12] + dsrc[53]*BT[13] + dsrc[54]*BT[14] + dsrc[55]*BT[15];
    ddst[tileCount + 50*ISTRIDE6X3] = dsrc[48]*BT[16] + dsrc[49]*BT[17] + dsrc[50]*BT[18] + dsrc[51]*BT[19] + dsrc[52]*BT[20] + dsrc[53]*BT[21] + dsrc[54]*BT[22] + dsrc[55]*BT[23];
    ddst[tileCount + 51*ISTRIDE6X3] = dsrc[48]*BT[24] + dsrc[49]*BT[25] + dsrc[50]*BT[26] + dsrc[51]*BT[27] + dsrc[52]*BT[28] + dsrc[53]*BT[29] + dsrc[54]*BT[30] + dsrc[55]*BT[31];
    ddst[tileCount + 52*ISTRIDE6X3] = dsrc[48]*BT[32] + dsrc[49]*BT[33] + dsrc[50]*BT[34] + dsrc[51]*BT[35] + dsrc[52]*BT[36] + dsrc[53]*BT[37] + dsrc[54]*BT[38] + dsrc[55]*BT[39];
    ddst[tileCount + 53*ISTRIDE6X3] = dsrc[48]*BT[40] + dsrc[49]*BT[41] + dsrc[50]*BT[42] + dsrc[51]*BT[43] + dsrc[52]*BT[44] + dsrc[53]*BT[45] + dsrc[54]*BT[46] + dsrc[55]*BT[47];
    ddst[tileCount + 54*ISTRIDE6X3] = dsrc[48]*BT[48] + dsrc[49]*BT[49] + dsrc[50]*BT[50] + dsrc[51]*BT[51] + dsrc[52]*BT[52] + dsrc[53]*BT[53] + dsrc[54]*BT[54] + dsrc[55]*BT[55];
    ddst[tileCount + 55*ISTRIDE6X3] = dsrc[48]*BT[56] + dsrc[49]*BT[57] + dsrc[50]*BT[58] + dsrc[51]*BT[59] + dsrc[52]*BT[60] + dsrc[53]*BT[61] + dsrc[54]*BT[62] + dsrc[55]*BT[63];
    ddst[tileCount + 56*ISTRIDE6X3] = dsrc[56]*BT[ 0] + dsrc[57]*BT[ 1] + dsrc[58]*BT[ 2] + dsrc[59]*BT[ 3] + dsrc[60]*BT[ 4] + dsrc[61]*BT[ 5] + dsrc[62]*BT[ 6] + dsrc[63]*BT[ 7];
    ddst[tileCount + 57*ISTRIDE6X3] = dsrc[56]*BT[ 8] + dsrc[57]*BT[ 9] + dsrc[58]*BT[10] + dsrc[59]*BT[11] + dsrc[60]*BT[12] + dsrc[61]*BT[13] + dsrc[62]*BT[14] + dsrc[63]*BT[15];
    ddst[tileCount + 58*ISTRIDE6X3] = dsrc[56]*BT[16] + dsrc[57]*BT[17] + dsrc[58]*BT[18] + dsrc[59]*BT[19] + dsrc[60]*BT[20] + dsrc[61]*BT[21] + dsrc[62]*BT[22] + dsrc[63]*BT[23];
    ddst[tileCount + 59*ISTRIDE6X3] = dsrc[56]*BT[24] + dsrc[57]*BT[25] + dsrc[58]*BT[26] + dsrc[59]*BT[27] + dsrc[60]*BT[28] + dsrc[61]*BT[29] + dsrc[62]*BT[30] + dsrc[63]*BT[31];
    ddst[tileCount + 60*ISTRIDE6X3] = dsrc[56]*BT[32] + dsrc[57]*BT[33] + dsrc[58]*BT[34] + dsrc[59]*BT[35] + dsrc[60]*BT[36] + dsrc[61]*BT[37] + dsrc[62]*BT[38] + dsrc[63]*BT[39];
    ddst[tileCount + 61*ISTRIDE6X3] = dsrc[56]*BT[40] + dsrc[57]*BT[41] + dsrc[58]*BT[42] + dsrc[59]*BT[43] + dsrc[60]*BT[44] + dsrc[61]*BT[45] + dsrc[62]*BT[46] + dsrc[63]*BT[47];
    ddst[tileCount + 62*ISTRIDE6X3] = dsrc[56]*BT[48] + dsrc[57]*BT[49] + dsrc[58]*BT[50] + dsrc[59]*BT[51] + dsrc[60]*BT[52] + dsrc[61]*BT[53] + dsrc[62]*BT[54] + dsrc[63]*BT[55];
    ddst[tileCount + 63*ISTRIDE6X3] = dsrc[56]*BT[56] + dsrc[57]*BT[57] + dsrc[58]*BT[58] + dsrc[59]*BT[59] + dsrc[60]*BT[60] + dsrc[61]*BT[61] + dsrc[62]*BT[62] + dsrc[63]*BT[63];
}

/*Compute transformed data for output by AT.
 * */
    template <typename Dtype>
inline void transformByAT_first(Dtype *dsrc, Dtype *ddst)
{
    ddst[ 0] = AT[ 0]*dsrc[ 0] + AT[ 1]*dsrc[ 8] + AT[ 2]*dsrc[16] + AT[ 3]*dsrc[24] + AT[ 4]*dsrc[32] + AT[ 5]*dsrc[40] + AT[ 6]*dsrc[48] + AT[ 7]*dsrc[56];
    ddst[ 1] = AT[ 0]*dsrc[ 1] + AT[ 1]*dsrc[ 9] + AT[ 2]*dsrc[17] + AT[ 3]*dsrc[25] + AT[ 4]*dsrc[33] + AT[ 5]*dsrc[41] + AT[ 6]*dsrc[49] + AT[ 7]*dsrc[57];
    ddst[ 2] = AT[ 0]*dsrc[ 2] + AT[ 1]*dsrc[10] + AT[ 2]*dsrc[18] + AT[ 3]*dsrc[26] + AT[ 4]*dsrc[34] + AT[ 5]*dsrc[42] + AT[ 6]*dsrc[50] + AT[ 7]*dsrc[58];
    ddst[ 3] = AT[ 0]*dsrc[ 3] + AT[ 1]*dsrc[11] + AT[ 2]*dsrc[19] + AT[ 3]*dsrc[27] + AT[ 4]*dsrc[35] + AT[ 5]*dsrc[43] + AT[ 6]*dsrc[51] + AT[ 7]*dsrc[59];
    ddst[ 4] = AT[ 0]*dsrc[ 4] + AT[ 1]*dsrc[12] + AT[ 2]*dsrc[20] + AT[ 3]*dsrc[28] + AT[ 4]*dsrc[36] + AT[ 5]*dsrc[44] + AT[ 6]*dsrc[52] + AT[ 7]*dsrc[60];
    ddst[ 5] = AT[ 0]*dsrc[ 5] + AT[ 1]*dsrc[13] + AT[ 2]*dsrc[21] + AT[ 3]*dsrc[29] + AT[ 4]*dsrc[37] + AT[ 5]*dsrc[45] + AT[ 6]*dsrc[53] + AT[ 7]*dsrc[61];
    ddst[ 6] = AT[ 0]*dsrc[ 6] + AT[ 1]*dsrc[14] + AT[ 2]*dsrc[22] + AT[ 3]*dsrc[30] + AT[ 4]*dsrc[38] + AT[ 5]*dsrc[46] + AT[ 6]*dsrc[54] + AT[ 7]*dsrc[62];
    ddst[ 7] = AT[ 0]*dsrc[ 7] + AT[ 1]*dsrc[15] + AT[ 2]*dsrc[23] + AT[ 3]*dsrc[31] + AT[ 4]*dsrc[39] + AT[ 5]*dsrc[47] + AT[ 6]*dsrc[55] + AT[ 7]*dsrc[63];
    ddst[ 8] = AT[ 8]*dsrc[ 0] + AT[ 9]*dsrc[ 8] + AT[10]*dsrc[16] + AT[11]*dsrc[24] + AT[12]*dsrc[32] + AT[13]*dsrc[40] + AT[14]*dsrc[48] + AT[15]*dsrc[56];
    ddst[ 9] = AT[ 8]*dsrc[ 1] + AT[ 9]*dsrc[ 9] + AT[10]*dsrc[17] + AT[11]*dsrc[25] + AT[12]*dsrc[33] + AT[13]*dsrc[41] + AT[14]*dsrc[49] + AT[15]*dsrc[57];
    ddst[10] = AT[ 8]*dsrc[ 2] + AT[ 9]*dsrc[10] + AT[10]*dsrc[18] + AT[11]*dsrc[26] + AT[12]*dsrc[34] + AT[13]*dsrc[42] + AT[14]*dsrc[50] + AT[15]*dsrc[58];
    ddst[11] = AT[ 8]*dsrc[ 3] + AT[ 9]*dsrc[11] + AT[10]*dsrc[19] + AT[11]*dsrc[27] + AT[12]*dsrc[35] + AT[13]*dsrc[43] + AT[14]*dsrc[51] + AT[15]*dsrc[59];
    ddst[12] = AT[ 8]*dsrc[ 4] + AT[ 9]*dsrc[12] + AT[10]*dsrc[20] + AT[11]*dsrc[28] + AT[12]*dsrc[36] + AT[13]*dsrc[44] + AT[14]*dsrc[52] + AT[15]*dsrc[60];
    ddst[13] = AT[ 8]*dsrc[ 5] + AT[ 9]*dsrc[13] + AT[10]*dsrc[21] + AT[11]*dsrc[29] + AT[12]*dsrc[37] + AT[13]*dsrc[45] + AT[14]*dsrc[53] + AT[15]*dsrc[61];
    ddst[14] = AT[ 8]*dsrc[ 6] + AT[ 9]*dsrc[14] + AT[10]*dsrc[22] + AT[11]*dsrc[30] + AT[12]*dsrc[38] + AT[13]*dsrc[46] + AT[14]*dsrc[54] + AT[15]*dsrc[62];
    ddst[15] = AT[ 8]*dsrc[ 7] + AT[ 9]*dsrc[15] + AT[10]*dsrc[23] + AT[11]*dsrc[31] + AT[12]*dsrc[39] + AT[13]*dsrc[47] + AT[14]*dsrc[55] + AT[15]*dsrc[63];
    ddst[16] = AT[16]*dsrc[ 0] + AT[17]*dsrc[ 8] + AT[18]*dsrc[16] + AT[19]*dsrc[24] + AT[20]*dsrc[32] + AT[21]*dsrc[40] + AT[22]*dsrc[48] + AT[23]*dsrc[56];
    ddst[17] = AT[16]*dsrc[ 1] + AT[17]*dsrc[ 9] + AT[18]*dsrc[17] + AT[19]*dsrc[25] + AT[20]*dsrc[33] + AT[21]*dsrc[41] + AT[22]*dsrc[49] + AT[23]*dsrc[57];
    ddst[18] = AT[16]*dsrc[ 2] + AT[17]*dsrc[10] + AT[18]*dsrc[18] + AT[19]*dsrc[26] + AT[20]*dsrc[34] + AT[21]*dsrc[42] + AT[22]*dsrc[50] + AT[23]*dsrc[58];
    ddst[19] = AT[16]*dsrc[ 3] + AT[17]*dsrc[11] + AT[18]*dsrc[19] + AT[19]*dsrc[27] + AT[20]*dsrc[35] + AT[21]*dsrc[43] + AT[22]*dsrc[51] + AT[23]*dsrc[59];
    ddst[20] = AT[16]*dsrc[ 4] + AT[17]*dsrc[12] + AT[18]*dsrc[20] + AT[19]*dsrc[28] + AT[20]*dsrc[36] + AT[21]*dsrc[44] + AT[22]*dsrc[52] + AT[23]*dsrc[60];
    ddst[21] = AT[16]*dsrc[ 5] + AT[17]*dsrc[13] + AT[18]*dsrc[21] + AT[19]*dsrc[29] + AT[20]*dsrc[37] + AT[21]*dsrc[45] + AT[22]*dsrc[53] + AT[23]*dsrc[61];
    ddst[22] = AT[16]*dsrc[ 6] + AT[17]*dsrc[14] + AT[18]*dsrc[22] + AT[19]*dsrc[30] + AT[20]*dsrc[38] + AT[21]*dsrc[46] + AT[22]*dsrc[54] + AT[23]*dsrc[62];
    ddst[23] = AT[16]*dsrc[ 7] + AT[17]*dsrc[15] + AT[18]*dsrc[23] + AT[19]*dsrc[31] + AT[20]*dsrc[39] + AT[21]*dsrc[47] + AT[22]*dsrc[55] + AT[23]*dsrc[63];
    ddst[24] = AT[24]*dsrc[ 0] + AT[25]*dsrc[ 8] + AT[26]*dsrc[16] + AT[27]*dsrc[24] + AT[28]*dsrc[32] + AT[29]*dsrc[40] + AT[30]*dsrc[48] + AT[31]*dsrc[56];
    ddst[25] = AT[24]*dsrc[ 1] + AT[25]*dsrc[ 9] + AT[26]*dsrc[17] + AT[27]*dsrc[25] + AT[28]*dsrc[33] + AT[29]*dsrc[41] + AT[30]*dsrc[49] + AT[31]*dsrc[57];
    ddst[26] = AT[24]*dsrc[ 2] + AT[25]*dsrc[10] + AT[26]*dsrc[18] + AT[27]*dsrc[26] + AT[28]*dsrc[34] + AT[29]*dsrc[42] + AT[30]*dsrc[50] + AT[31]*dsrc[58];
    ddst[27] = AT[24]*dsrc[ 3] + AT[25]*dsrc[11] + AT[26]*dsrc[19] + AT[27]*dsrc[27] + AT[28]*dsrc[35] + AT[29]*dsrc[43] + AT[30]*dsrc[51] + AT[31]*dsrc[59];
    ddst[28] = AT[24]*dsrc[ 4] + AT[25]*dsrc[12] + AT[26]*dsrc[20] + AT[27]*dsrc[28] + AT[28]*dsrc[36] + AT[29]*dsrc[44] + AT[30]*dsrc[52] + AT[31]*dsrc[60];
    ddst[29] = AT[24]*dsrc[ 5] + AT[25]*dsrc[13] + AT[26]*dsrc[21] + AT[27]*dsrc[29] + AT[28]*dsrc[37] + AT[29]*dsrc[45] + AT[30]*dsrc[53] + AT[31]*dsrc[61];
    ddst[30] = AT[24]*dsrc[ 6] + AT[25]*dsrc[14] + AT[26]*dsrc[22] + AT[27]*dsrc[30] + AT[28]*dsrc[38] + AT[29]*dsrc[46] + AT[30]*dsrc[54] + AT[31]*dsrc[62];
    ddst[31] = AT[24]*dsrc[ 7] + AT[25]*dsrc[15] + AT[26]*dsrc[23] + AT[27]*dsrc[31] + AT[28]*dsrc[39] + AT[29]*dsrc[47] + AT[30]*dsrc[55] + AT[31]*dsrc[63];
    ddst[32] = AT[32]*dsrc[ 0] + AT[33]*dsrc[ 8] + AT[34]*dsrc[16] + AT[35]*dsrc[24] + AT[36]*dsrc[32] + AT[37]*dsrc[40] + AT[38]*dsrc[48] + AT[39]*dsrc[56];
    ddst[33] = AT[32]*dsrc[ 1] + AT[33]*dsrc[ 9] + AT[34]*dsrc[17] + AT[35]*dsrc[25] + AT[36]*dsrc[33] + AT[37]*dsrc[41] + AT[38]*dsrc[49] + AT[39]*dsrc[57];
    ddst[34] = AT[32]*dsrc[ 2] + AT[33]*dsrc[10] + AT[34]*dsrc[18] + AT[35]*dsrc[26] + AT[36]*dsrc[34] + AT[37]*dsrc[42] + AT[38]*dsrc[50] + AT[39]*dsrc[58];
    ddst[35] = AT[32]*dsrc[ 3] + AT[33]*dsrc[11] + AT[34]*dsrc[19] + AT[35]*dsrc[27] + AT[36]*dsrc[35] + AT[37]*dsrc[43] + AT[38]*dsrc[51] + AT[39]*dsrc[59];
    ddst[36] = AT[32]*dsrc[ 4] + AT[33]*dsrc[12] + AT[34]*dsrc[20] + AT[35]*dsrc[28] + AT[36]*dsrc[36] + AT[37]*dsrc[44] + AT[38]*dsrc[52] + AT[39]*dsrc[60];
    ddst[37] = AT[32]*dsrc[ 5] + AT[33]*dsrc[13] + AT[34]*dsrc[21] + AT[35]*dsrc[29] + AT[36]*dsrc[37] + AT[37]*dsrc[45] + AT[38]*dsrc[53] + AT[39]*dsrc[61];
    ddst[38] = AT[32]*dsrc[ 6] + AT[33]*dsrc[14] + AT[34]*dsrc[22] + AT[35]*dsrc[30] + AT[36]*dsrc[38] + AT[37]*dsrc[46] + AT[38]*dsrc[54] + AT[39]*dsrc[62];
    ddst[39] = AT[32]*dsrc[ 7] + AT[33]*dsrc[15] + AT[34]*dsrc[23] + AT[35]*dsrc[31] + AT[36]*dsrc[39] + AT[37]*dsrc[47] + AT[38]*dsrc[55] + AT[39]*dsrc[63];
    ddst[40] = AT[40]*dsrc[ 0] + AT[41]*dsrc[ 8] + AT[42]*dsrc[16] + AT[43]*dsrc[24] + AT[44]*dsrc[32] + AT[45]*dsrc[40] + AT[46]*dsrc[48] + AT[47]*dsrc[56];
    ddst[41] = AT[40]*dsrc[ 1] + AT[41]*dsrc[ 9] + AT[42]*dsrc[17] + AT[43]*dsrc[25] + AT[44]*dsrc[33] + AT[45]*dsrc[41] + AT[46]*dsrc[49] + AT[47]*dsrc[57];
    ddst[42] = AT[40]*dsrc[ 2] + AT[41]*dsrc[10] + AT[42]*dsrc[18] + AT[43]*dsrc[26] + AT[44]*dsrc[34] + AT[45]*dsrc[42] + AT[46]*dsrc[50] + AT[47]*dsrc[58];
    ddst[43] = AT[40]*dsrc[ 3] + AT[41]*dsrc[11] + AT[42]*dsrc[19] + AT[43]*dsrc[27] + AT[44]*dsrc[35] + AT[45]*dsrc[43] + AT[46]*dsrc[51] + AT[47]*dsrc[59];
    ddst[44] = AT[40]*dsrc[ 4] + AT[41]*dsrc[12] + AT[42]*dsrc[20] + AT[43]*dsrc[28] + AT[44]*dsrc[36] + AT[45]*dsrc[44] + AT[46]*dsrc[52] + AT[47]*dsrc[60];
    ddst[45] = AT[40]*dsrc[ 5] + AT[41]*dsrc[13] + AT[42]*dsrc[21] + AT[43]*dsrc[29] + AT[44]*dsrc[37] + AT[45]*dsrc[45] + AT[46]*dsrc[53] + AT[47]*dsrc[61];
    ddst[46] = AT[40]*dsrc[ 6] + AT[41]*dsrc[14] + AT[42]*dsrc[22] + AT[43]*dsrc[30] + AT[44]*dsrc[38] + AT[45]*dsrc[46] + AT[46]*dsrc[54] + AT[47]*dsrc[62];
    ddst[47] = AT[40]*dsrc[ 7] + AT[41]*dsrc[15] + AT[42]*dsrc[23] + AT[43]*dsrc[31] + AT[44]*dsrc[39] + AT[45]*dsrc[47] + AT[46]*dsrc[55] + AT[47]*dsrc[63];
}

    template <typename Dtype>
inline void transformByAT_second(Dtype *dsrc, Dtype *ddst, int rowIdx, int colIdx, int colNum)
{
    ddst[(rowIdx+0)*colNum + (colIdx+0)] = dsrc[ 0]*AT[ 0] + dsrc[ 1]*AT[ 1] + dsrc[ 2]*AT[ 2] + dsrc[ 3]*AT[ 3] + dsrc[ 4]*AT[ 4] + dsrc[ 5]*AT[ 5] + dsrc[ 6]*AT[ 6] + dsrc[ 7]*AT[ 7];
    ddst[(rowIdx+0)*colNum + (colIdx+1)] = dsrc[ 0]*AT[ 8] + dsrc[ 1]*AT[ 9] + dsrc[ 2]*AT[10] + dsrc[ 3]*AT[11] + dsrc[ 4]*AT[12] + dsrc[ 5]*AT[13] + dsrc[ 6]*AT[14] + dsrc[ 7]*AT[15];
    ddst[(rowIdx+0)*colNum + (colIdx+2)] = dsrc[ 0]*AT[16] + dsrc[ 1]*AT[17] + dsrc[ 2]*AT[18] + dsrc[ 3]*AT[19] + dsrc[ 4]*AT[20] + dsrc[ 5]*AT[21] + dsrc[ 6]*AT[22] + dsrc[ 7]*AT[23];
    ddst[(rowIdx+0)*colNum + (colIdx+3)] = dsrc[ 0]*AT[24] + dsrc[ 1]*AT[25] + dsrc[ 2]*AT[26] + dsrc[ 3]*AT[27] + dsrc[ 4]*AT[28] + dsrc[ 5]*AT[29] + dsrc[ 6]*AT[30] + dsrc[ 7]*AT[31];
    ddst[(rowIdx+0)*colNum + (colIdx+4)] = dsrc[ 0]*AT[32] + dsrc[ 1]*AT[33] + dsrc[ 2]*AT[34] + dsrc[ 3]*AT[35] + dsrc[ 4]*AT[36] + dsrc[ 5]*AT[37] + dsrc[ 6]*AT[38] + dsrc[ 7]*AT[39];
    ddst[(rowIdx+0)*colNum + (colIdx+5)] = dsrc[ 0]*AT[40] + dsrc[ 1]*AT[41] + dsrc[ 2]*AT[42] + dsrc[ 3]*AT[43] + dsrc[ 4]*AT[44] + dsrc[ 5]*AT[45] + dsrc[ 6]*AT[46] + dsrc[ 7]*AT[47];
    ddst[(rowIdx+1)*colNum + (colIdx+0)] = dsrc[ 8]*AT[ 0] + dsrc[ 9]*AT[ 1] + dsrc[10]*AT[ 2] + dsrc[11]*AT[ 3] + dsrc[12]*AT[ 4] + dsrc[13]*AT[ 5] + dsrc[14]*AT[ 6] + dsrc[15]*AT[ 7];
    ddst[(rowIdx+1)*colNum + (colIdx+1)] = dsrc[ 8]*AT[ 8] + dsrc[ 9]*AT[ 9] + dsrc[10]*AT[10] + dsrc[11]*AT[11] + dsrc[12]*AT[12] + dsrc[13]*AT[13] + dsrc[14]*AT[14] + dsrc[15]*AT[15];
    ddst[(rowIdx+1)*colNum + (colIdx+2)] = dsrc[ 8]*AT[16] + dsrc[ 9]*AT[17] + dsrc[10]*AT[18] + dsrc[11]*AT[19] + dsrc[12]*AT[20] + dsrc[13]*AT[21] + dsrc[14]*AT[22] + dsrc[15]*AT[23];
    ddst[(rowIdx+1)*colNum + (colIdx+3)] = dsrc[ 8]*AT[24] + dsrc[ 9]*AT[25] + dsrc[10]*AT[26] + dsrc[11]*AT[27] + dsrc[12]*AT[28] + dsrc[13]*AT[29] + dsrc[14]*AT[30] + dsrc[15]*AT[31];
    ddst[(rowIdx+1)*colNum + (colIdx+4)] = dsrc[ 8]*AT[32] + dsrc[ 9]*AT[33] + dsrc[10]*AT[34] + dsrc[11]*AT[35] + dsrc[12]*AT[36] + dsrc[13]*AT[37] + dsrc[14]*AT[38] + dsrc[15]*AT[39];
    ddst[(rowIdx+1)*colNum + (colIdx+5)] = dsrc[ 8]*AT[40] + dsrc[ 9]*AT[41] + dsrc[10]*AT[42] + dsrc[11]*AT[43] + dsrc[12]*AT[44] + dsrc[13]*AT[45] + dsrc[14]*AT[46] + dsrc[15]*AT[47];
    ddst[(rowIdx+2)*colNum + (colIdx+0)] = dsrc[16]*AT[ 0] + dsrc[17]*AT[ 1] + dsrc[18]*AT[ 2] + dsrc[19]*AT[ 3] + dsrc[20]*AT[ 4] + dsrc[21]*AT[ 5] + dsrc[22]*AT[ 6] + dsrc[23]*AT[ 7];
    ddst[(rowIdx+2)*colNum + (colIdx+1)] = dsrc[16]*AT[ 8] + dsrc[17]*AT[ 9] + dsrc[18]*AT[10] + dsrc[19]*AT[11] + dsrc[20]*AT[12] + dsrc[21]*AT[13] + dsrc[22]*AT[14] + dsrc[23]*AT[15];
    ddst[(rowIdx+2)*colNum + (colIdx+2)] = dsrc[16]*AT[16] + dsrc[17]*AT[17] + dsrc[18]*AT[18] + dsrc[19]*AT[19] + dsrc[20]*AT[20] + dsrc[21]*AT[21] + dsrc[22]*AT[22] + dsrc[23]*AT[23];
    ddst[(rowIdx+2)*colNum + (colIdx+3)] = dsrc[16]*AT[24] + dsrc[17]*AT[25] + dsrc[18]*AT[26] + dsrc[19]*AT[27] + dsrc[20]*AT[28] + dsrc[21]*AT[29] + dsrc[22]*AT[30] + dsrc[23]*AT[31];
    ddst[(rowIdx+2)*colNum + (colIdx+4)] = dsrc[16]*AT[32] + dsrc[17]*AT[33] + dsrc[18]*AT[34] + dsrc[19]*AT[35] + dsrc[20]*AT[36] + dsrc[21]*AT[37] + dsrc[22]*AT[38] + dsrc[23]*AT[39];
    ddst[(rowIdx+2)*colNum + (colIdx+5)] = dsrc[16]*AT[40] + dsrc[17]*AT[41] + dsrc[18]*AT[42] + dsrc[19]*AT[43] + dsrc[20]*AT[44] + dsrc[21]*AT[45] + dsrc[22]*AT[46] + dsrc[23]*AT[47];
    ddst[(rowIdx+3)*colNum + (colIdx+0)] = dsrc[24]*AT[ 0] + dsrc[25]*AT[ 1] + dsrc[26]*AT[ 2] + dsrc[27]*AT[ 3] + dsrc[28]*AT[ 4] + dsrc[29]*AT[ 5] + dsrc[30]*AT[ 6] + dsrc[31]*AT[ 7];
    ddst[(rowIdx+3)*colNum + (colIdx+1)] = dsrc[24]*AT[ 8] + dsrc[25]*AT[ 9] + dsrc[26]*AT[10] + dsrc[27]*AT[11] + dsrc[28]*AT[12] + dsrc[29]*AT[13] + dsrc[30]*AT[14] + dsrc[31]*AT[15];
    ddst[(rowIdx+3)*colNum + (colIdx+2)] = dsrc[24]*AT[16] + dsrc[25]*AT[17] + dsrc[26]*AT[18] + dsrc[27]*AT[19] + dsrc[28]*AT[20] + dsrc[29]*AT[21] + dsrc[30]*AT[22] + dsrc[31]*AT[23];
    ddst[(rowIdx+3)*colNum + (colIdx+3)] = dsrc[24]*AT[24] + dsrc[25]*AT[25] + dsrc[26]*AT[26] + dsrc[27]*AT[27] + dsrc[28]*AT[28] + dsrc[29]*AT[29] + dsrc[30]*AT[30] + dsrc[31]*AT[31];
    ddst[(rowIdx+3)*colNum + (colIdx+4)] = dsrc[24]*AT[32] + dsrc[25]*AT[33] + dsrc[26]*AT[34] + dsrc[27]*AT[35] + dsrc[28]*AT[36] + dsrc[29]*AT[37] + dsrc[30]*AT[38] + dsrc[31]*AT[39];
    ddst[(rowIdx+3)*colNum + (colIdx+5)] = dsrc[24]*AT[40] + dsrc[25]*AT[41] + dsrc[26]*AT[42] + dsrc[27]*AT[43] + dsrc[28]*AT[44] + dsrc[29]*AT[45] + dsrc[30]*AT[46] + dsrc[31]*AT[47];
    ddst[(rowIdx+4)*colNum + (colIdx+0)] = dsrc[32]*AT[ 0] + dsrc[33]*AT[ 1] + dsrc[34]*AT[ 2] + dsrc[35]*AT[ 3] + dsrc[36]*AT[ 4] + dsrc[37]*AT[ 5] + dsrc[38]*AT[ 6] + dsrc[39]*AT[ 7];
    ddst[(rowIdx+4)*colNum + (colIdx+1)] = dsrc[32]*AT[ 8] + dsrc[33]*AT[ 9] + dsrc[34]*AT[10] + dsrc[35]*AT[11] + dsrc[36]*AT[12] + dsrc[37]*AT[13] + dsrc[38]*AT[14] + dsrc[39]*AT[15];
    ddst[(rowIdx+4)*colNum + (colIdx+2)] = dsrc[32]*AT[16] + dsrc[33]*AT[17] + dsrc[34]*AT[18] + dsrc[35]*AT[19] + dsrc[36]*AT[20] + dsrc[37]*AT[21] + dsrc[38]*AT[22] + dsrc[39]*AT[23];
    ddst[(rowIdx+4)*colNum + (colIdx+3)] = dsrc[32]*AT[24] + dsrc[33]*AT[25] + dsrc[34]*AT[26] + dsrc[35]*AT[27] + dsrc[36]*AT[28] + dsrc[37]*AT[29] + dsrc[38]*AT[30] + dsrc[39]*AT[31];
    ddst[(rowIdx+4)*colNum + (colIdx+4)] = dsrc[32]*AT[32] + dsrc[33]*AT[33] + dsrc[34]*AT[34] + dsrc[35]*AT[35] + dsrc[36]*AT[36] + dsrc[37]*AT[37] + dsrc[38]*AT[38] + dsrc[39]*AT[39];
    ddst[(rowIdx+4)*colNum + (colIdx+5)] = dsrc[32]*AT[40] + dsrc[33]*AT[41] + dsrc[34]*AT[42] + dsrc[35]*AT[43] + dsrc[36]*AT[44] + dsrc[37]*AT[45] + dsrc[38]*AT[46] + dsrc[39]*AT[47];
    ddst[(rowIdx+5)*colNum + (colIdx+0)] = dsrc[40]*AT[ 0] + dsrc[41]*AT[ 1] + dsrc[42]*AT[ 2] + dsrc[43]*AT[ 3] + dsrc[44]*AT[ 4] + dsrc[45]*AT[ 5] + dsrc[46]*AT[ 6] + dsrc[47]*AT[ 7];
    ddst[(rowIdx+5)*colNum + (colIdx+1)] = dsrc[40]*AT[ 8] + dsrc[41]*AT[ 9] + dsrc[42]*AT[10] + dsrc[43]*AT[11] + dsrc[44]*AT[12] + dsrc[45]*AT[13] + dsrc[46]*AT[14] + dsrc[47]*AT[15];
    ddst[(rowIdx+5)*colNum + (colIdx+2)] = dsrc[40]*AT[16] + dsrc[41]*AT[17] + dsrc[42]*AT[18] + dsrc[43]*AT[19] + dsrc[44]*AT[20] + dsrc[45]*AT[21] + dsrc[46]*AT[22] + dsrc[47]*AT[23];
    ddst[(rowIdx+5)*colNum + (colIdx+3)] = dsrc[40]*AT[24] + dsrc[41]*AT[25] + dsrc[42]*AT[26] + dsrc[43]*AT[27] + dsrc[44]*AT[28] + dsrc[45]*AT[29] + dsrc[46]*AT[30] + dsrc[47]*AT[31];
    ddst[(rowIdx+5)*colNum + (colIdx+4)] = dsrc[40]*AT[32] + dsrc[41]*AT[33] + dsrc[42]*AT[34] + dsrc[43]*AT[35] + dsrc[44]*AT[36] + dsrc[45]*AT[37] + dsrc[46]*AT[38] + dsrc[47]*AT[39];
    ddst[(rowIdx+5)*colNum + (colIdx+5)] = dsrc[40]*AT[40] + dsrc[41]*AT[41] + dsrc[42]*AT[42] + dsrc[43]*AT[43] + dsrc[44]*AT[44] + dsrc[45]*AT[45] + dsrc[46]*AT[46] + dsrc[47]*AT[47];
}

/* Don't use pad: 
 * compute the bridge data for in, and transform to form matrix A.
 * */
    template<typename Dtype>
static void inByTransform_nopad(const Dtype *in, Dtype *dataDst,
        const int N, const int C, const int rows, const int cols,
        const int ntiles, const int mg6x3)
{   
    int d1, d2;
    int sizeI = rows*cols;

#pragma omp parallel for private(d1) 
    for(d1 = 0; d1 < N*C; d1++){
        int i, j, k; 
        Dtype tmp[64] __attribute__((aligned(64)));
        Dtype bridge[64] __attribute__((aligned(64)));

        const int t1 = d1/(C*mg6x3);
        const int t2 = (d1%(C*mg6x3))/mg6x3;
        const int t3 = d1%mg6x3;

        // merge value influence the sequence of in data.
        const Dtype *data = in + (t1*mg6x3*C + t3*C + t2)*sizeI;
        int tileCount = d1*ntiles;

        for(i = 0; i < (rows-2); i += 6){
            //#pragma simd
            for(j = 0; j < (cols-2); j += 6){
                // Get the input data
                tmp[ 0] = data[(i+0)*cols + (j+0)]; 
                tmp[ 1] = data[(i+0)*cols + (j+1)]; 
                tmp[ 2] = data[(i+0)*cols + (j+2)]; 
                tmp[ 3] = data[(i+0)*cols + (j+3)]; 
                tmp[ 4] = data[(i+0)*cols + (j+4)]; 
                tmp[ 5] = data[(i+0)*cols + (j+5)]; 
                tmp[ 6] = data[(i+0)*cols + (j+6)]; 
                tmp[ 7] = data[(i+0)*cols + (j+7)]; 

                tmp[ 8] = data[(i+1)*cols + (j+0)]; 
                tmp[ 9] = data[(i+1)*cols + (j+1)]; 
                tmp[10] = data[(i+1)*cols + (j+2)]; 
                tmp[11] = data[(i+1)*cols + (j+3)]; 
                tmp[12] = data[(i+1)*cols + (j+4)]; 
                tmp[13] = data[(i+1)*cols + (j+5)]; 
                tmp[14] = data[(i+1)*cols + (j+6)]; 
                tmp[15] = data[(i+1)*cols + (j+7)]; 

                tmp[16] = data[(i+2)*cols + (j+0)]; 
                tmp[17] = data[(i+2)*cols + (j+1)]; 
                tmp[18] = data[(i+2)*cols + (j+2)]; 
                tmp[19] = data[(i+2)*cols + (j+3)]; 
                tmp[20] = data[(i+2)*cols + (j+4)]; 
                tmp[21] = data[(i+2)*cols + (j+5)]; 
                tmp[22] = data[(i+2)*cols + (j+6)]; 
                tmp[23] = data[(i+2)*cols + (j+7)]; 

                tmp[24] = data[(i+3)*cols + (j+0)]; 
                tmp[25] = data[(i+3)*cols + (j+1)]; 
                tmp[26] = data[(i+3)*cols + (j+2)]; 
                tmp[27] = data[(i+3)*cols + (j+3)]; 
                tmp[28] = data[(i+3)*cols + (j+4)]; 
                tmp[29] = data[(i+3)*cols + (j+5)]; 
                tmp[30] = data[(i+3)*cols + (j+6)]; 
                tmp[31] = data[(i+3)*cols + (j+7)]; 

                tmp[32] = data[(i+4)*cols + (j+0)]; 
                tmp[33] = data[(i+4)*cols + (j+1)]; 
                tmp[34] = data[(i+4)*cols + (j+2)]; 
                tmp[35] = data[(i+4)*cols + (j+3)]; 
                tmp[36] = data[(i+4)*cols + (j+4)]; 
                tmp[37] = data[(i+4)*cols + (j+5)]; 
                tmp[38] = data[(i+4)*cols + (j+6)]; 
                tmp[39] = data[(i+4)*cols + (j+7)]; 

                tmp[40] = data[(i+5)*cols + (j+0)]; 
                tmp[41] = data[(i+5)*cols + (j+1)]; 
                tmp[42] = data[(i+5)*cols + (j+2)]; 
                tmp[43] = data[(i+5)*cols + (j+3)]; 
                tmp[44] = data[(i+5)*cols + (j+4)]; 
                tmp[45] = data[(i+5)*cols + (j+5)]; 
                tmp[46] = data[(i+5)*cols + (j+6)]; 
                tmp[47] = data[(i+5)*cols + (j+7)]; 

                tmp[48] = data[(i+6)*cols + (j+0)]; 
                tmp[49] = data[(i+6)*cols + (j+1)]; 
                tmp[50] = data[(i+6)*cols + (j+2)]; 
                tmp[51] = data[(i+6)*cols + (j+3)]; 
                tmp[52] = data[(i+6)*cols + (j+4)]; 
                tmp[53] = data[(i+6)*cols + (j+5)]; 
                tmp[54] = data[(i+6)*cols + (j+6)]; 
                tmp[55] = data[(i+6)*cols + (j+7)]; 

                tmp[56] = data[(i+7)*cols + (j+0)]; 
                tmp[57] = data[(i+7)*cols + (j+1)]; 
                tmp[58] = data[(i+7)*cols + (j+2)]; 
                tmp[59] = data[(i+7)*cols + (j+3)]; 
                tmp[60] = data[(i+7)*cols + (j+4)]; 
                tmp[61] = data[(i+7)*cols + (j+5)]; 
                tmp[62] = data[(i+7)*cols + (j+6)]; 
                tmp[63] = data[(i+7)*cols + (j+7)]; 

                // The tranformation manually simplified
#if 0
                transformByBT(tmp, dataDst, tileCount);
#else
                transformByBT_first(tmp, bridge);
                transformByBT_second(bridge, dataDst, tileCount);
#endif
                tileCount++; 
            }
        }
    }
}

/* Compute the bridge data for filter, and transform to form matrix B. */
    template<typename Dtype>
static void filterByTransform(const Dtype *filter, Dtype *dataDst,
        const int C, const int K)
{
    int d1, d2, d3; 
    const Dtype *F;

#pragma omp parallel for collapse(2) private(d1, d2, d3, F)
    //#pragma prefetch out:1:1
    //#pragma simd
    for(d1 = 0; d1 < K; d1++){
        for(d2 = 0; d2 < C; d2++){
            Dtype ddt[64] __attribute__((aligned(64))); 
            F = filter+d2*3*3 + d1*3*3*C; 

            ddt[ 0] = F[0 ];
            ddt[ 1] = (-F[0 ] - F[1 ] - F[2 ])*2/9;
            ddt[ 2] = (-F[0 ] + F[1 ] - F[2 ])*2/9;
            ddt[ 3] = (F[0 ])/90 + (F[1 ])/45 + (F[2 ])*2/45;
            ddt[ 4] = (F[0 ])/90 + (-F[1 ])/45 + (F[2 ])*2/45;
            ddt[ 5] = (F[2 ])*8/45 + (F[1 ])*16/45 + (F[0 ])*32/45;
            ddt[ 6] = (F[2 ])*8/45 + (-F[1 ])*16/45 + (F[0 ])*32/45;
            ddt[ 7] = F[2 ];
            ddt[ 8] = (-F[0 ] - F[3 ] - F[6 ])*2/9;
            ddt[ 9] = (F[0 ] + F[1 ] + F[2 ] + F[3 ] + F[4 ] + F[5 ] + F[6 ] + F[7 ] + F[8 ])*4/81;
            ddt[10] = (F[0 ] - F[1 ] + F[2 ] + F[3 ] - F[4 ] + F[5 ] + F[6 ] - F[7 ] + F[8 ])*4/81;
            ddt[11] = (-F[0 ] - F[3 ] - F[6 ])/405 + (-F[1 ] - F[4 ] - F[7 ])*2/405 + (-F[2 ] - F[5 ] - F[8 ])*4/405;
            ddt[12] = (-F[0 ] - F[3 ] - F[6 ])/405 + (F[1 ] + F[4 ] + F[7 ])*2/405 + (-F[2 ] - F[5 ] - F[8 ])*4/405;
            ddt[13] = (-F[2 ] - F[5 ] - F[8 ])*16/405 + (-F[1 ] - F[4 ] - F[7 ])*32/405 + (-F[0 ] - F[3 ] - F[6 ])*64/405;
            ddt[14] = (-F[2 ] - F[5 ] - F[8 ])*16/405 + (F[1 ] + F[4 ] + F[7 ])*32/405 + (-F[0 ] - F[3 ] - F[6 ])*64/405;
            ddt[15] = (-F[2 ] - F[5 ] - F[8 ])*2/9;
            ddt[16] = (-F[0 ] + F[3 ] - F[6 ])*2/9;
            ddt[17] = (F[0 ] + F[1 ] + F[2 ] - F[3 ] - F[4 ] - F[5 ] + F[6 ] + F[7 ] + F[8 ])*4/81;
            ddt[18] = (F[0 ] - F[1 ] + F[2 ] - F[3 ] + F[4 ] - F[5 ] + F[6 ] - F[7 ] + F[8 ])*4/81;
            ddt[19] = (-F[0 ] + F[3 ] - F[6 ])/405 + (-F[1 ] + F[4 ] - F[7 ])*2/405 + (-F[2 ] + F[5 ] - F[8 ])*4/405;
            ddt[20] = (-F[0 ] + F[3 ] - F[6 ])/405 + (F[1 ] - F[4 ] + F[7 ])*2/405 + (-F[2 ] + F[5 ] - F[8 ])*4/405;
            ddt[21] = (-F[2 ] + F[5 ] - F[8 ])*16/405 + (-F[1 ] + F[4 ] - F[7 ])*32/405 + (-F[0 ] + F[3 ] - F[6 ])*64/405;
            ddt[22] = (-F[2 ] + F[5 ] - F[8 ])*16/405 + (F[1 ] - F[4 ] + F[7 ])*32/405 + (-F[0 ] + F[3 ] - F[6 ])*64/405;
            ddt[23] = (-F[2 ] + F[5 ] - F[8 ])*2/9;
            ddt[24] = (F[0 ])/90 + (F[3 ])/45 + (F[6 ])*2/45;
            ddt[25] = (-F[0 ] - F[1 ] - F[2 ])/405 + (-F[3 ] - F[4 ] - F[5 ])*2/405 + (-F[6 ] - F[7 ] - F[8 ])*4/405;
            ddt[26] = (-F[0 ] + F[1 ] - F[2 ])/405 + (-F[3 ] + F[4 ] - F[5 ])*2/405 + (-F[6 ] + F[7 ] - F[8 ])*4/405;
            ddt[27] = (F[0 ])/8100 + (F[1 ] + F[3 ])/4050 + (F[2 ] + F[4 ] + F[6 ])/2025 + (F[5 ] + F[7 ])*2/2025 + (F[8 ])*4/2025;
            ddt[28] = (F[0 ])/8100 + (-F[1 ] + F[3 ])/4050 + (F[2 ] - F[4 ] + F[6 ])/2025 + (F[5 ] - F[7 ])*2/2025 + (F[8 ])*4/2025;
            ddt[29] = (F[2 ])*4/2025 + (F[1 ] + F[5 ])*8/2025 + (F[0 ] + F[4 ] + F[8 ])*16/2025 + (F[3 ] + F[7 ])*32/2025 + (F[6 ])*64/2025;
            ddt[30] = (F[2 ])*4/2025 + (-F[1 ] + F[5 ])*8/2025 + (F[0 ] - F[4 ] + F[8 ])*16/2025 + (F[3 ] - F[7 ])*32/2025 + (F[6 ])*64/2025;
            ddt[31] = (F[2 ])/90 + (F[5 ])/45 + (F[8 ])*2/45;
            ddt[32] = (F[0 ])/90 + (-F[3 ])/45 + (F[6 ])*2/45;
            ddt[33] = (-F[0 ] - F[1 ] - F[2 ])/405 + (F[3 ] + F[4 ] + F[5 ])*2/405 + (-F[6 ] - F[7 ] - F[8 ])*4/405;
            ddt[34] = (-F[0 ] + F[1 ] - F[2 ])/405 + (F[3 ] - F[4 ] + F[5 ])*2/405 + (-F[6 ] + F[7 ] - F[8 ])*4/405;
            ddt[35] = (F[0 ])/8100 + (F[1 ] - F[3 ])/4050 + (F[2 ] - F[4 ] + F[6 ])/2025 + (-F[5 ] + F[7 ])*2/2025 + (F[8 ])*4/2025;
            ddt[36] = (F[0 ])/8100 + (-F[1 ] - F[3 ])/4050 + (F[2 ] + F[4 ] + F[6 ])/2025 + (-F[5 ] - F[7 ])*2/2025 + (F[8 ])*4/2025;
            ddt[37] = (F[2 ])*4/2025 + (F[1 ] - F[5 ])*8/2025 + (F[0 ] - F[4 ] + F[8 ])*16/2025 + (-F[3 ] + F[7 ])*32/2025 + (F[6 ])*64/2025;
            ddt[38] = (F[2 ])*4/2025 + (-F[1 ] - F[5 ])*8/2025 + (F[0 ] + F[4 ] + F[8 ])*16/2025 + (-F[3 ] - F[7 ])*32/2025 + (F[6 ])*64/2025;
            ddt[39] = (F[2 ])/90 + (-F[5 ])/45 + (F[8 ])*2/45;
            ddt[40] = (F[6 ])*8/45 + (F[3 ])*16/45 + (F[0 ])*32/45;
            ddt[41] = (-F[6 ] - F[7 ] - F[8 ])*16/405 + (-F[3 ] - F[4 ] - F[5 ])*32/405 + (-F[0 ] - F[1 ] - F[2 ])*64/405;
            ddt[42] = (-F[6 ] + F[7 ] - F[8 ])*16/405 + (-F[3 ] + F[4 ] - F[5 ])*32/405 + (-F[0 ] + F[1 ] - F[2 ])*64/405;
            ddt[43] = (F[6 ])*4/2025 + (F[3 ] + F[7 ])*8/2025 + (F[0 ] + F[4 ] + F[8 ])*16/2025 + (F[1 ] + F[5 ])*32/2025 + (F[2 ])*64/2025;
            ddt[44] = (F[6 ])*4/2025 + (F[3 ] - F[7 ])*8/2025 + (F[0 ] - F[4 ] + F[8 ])*16/2025 + (-F[1 ] + F[5 ])*32/2025 + (F[2 ])*64/2025;
            ddt[45] = (F[8 ])*64/2025 + (F[5 ] + F[7 ])*128/2025 + (F[2 ] + F[4 ] + F[6 ])*256/2025 + (F[1 ] + F[3 ])*512/2025 + (F[0 ])*1024/2025;
            ddt[46] = (F[8 ])*64/2025 + (F[5 ] - F[7 ])*128/2025 + (F[2 ] - F[4 ] + F[6 ])*256/2025 + (-F[1 ] + F[3 ])*512/2025 + (F[0 ])*1024/2025;
            ddt[47] = (F[8 ])*8/45 + (F[5 ])*16/45 + (F[2 ])*32/45;
            ddt[48] = (F[6 ])*8/45 + (-F[3 ])*16/45 + (F[0 ])*32/45;
            ddt[49] = (-F[6 ] - F[7 ] - F[8 ])*16/405 + (F[3 ] + F[4 ] + F[5 ])*32/405 + (-F[0 ] - F[1 ] - F[2 ])*64/405;
            ddt[50] = (-F[6 ] + F[7 ] - F[8 ])*16/405 + (F[3 ] - F[4 ] + F[5 ])*32/405 + (-F[0 ] + F[1 ] - F[2 ])*64/405;
            ddt[51] = (F[6 ])*4/2025 + (-F[3 ] + F[7 ])*8/2025 + (F[0 ] - F[4 ] + F[8 ])*16/2025 + (F[1 ] - F[5 ])*32/2025 + (F[2 ])*64/2025;
            ddt[52] = (F[6 ])*4/2025 + (-F[3 ] - F[7 ])*8/2025 + (F[0 ] + F[4 ] + F[8 ])*16/2025 + (-F[1 ] - F[5 ])*32/2025 + (F[2 ])*64/2025;
            ddt[53] = (F[8 ])*64/2025 + (-F[5 ] + F[7 ])*128/2025 + (F[2 ] - F[4 ] + F[6 ])*256/2025 + (F[1 ] - F[3 ])*512/2025 + (F[0 ])*1024/2025;
            ddt[54] = (F[8 ])*64/2025 + (-F[5 ] - F[7 ])*128/2025 + (F[2 ] + F[4 ] + F[6 ])*256/2025 + (-F[1 ] - F[3 ])*512/2025 + (F[0 ])*1024/2025;
            ddt[55] = (F[8 ])*8/45 + (-F[5 ])*16/45 + (F[2 ])*32/45;
            ddt[56] = (F[6 ])*1;
            ddt[57] = (-F[6 ] - F[7 ] - F[8 ])*2/9;
            ddt[58] = (-F[6 ] + F[7 ] - F[8 ])*2/9;
            ddt[59] = (F[6 ])/90 + (F[7 ])/45 + (F[8 ])*2/45;
            ddt[60] = (F[6 ])/90 + (-F[7 ])/45 + (F[8 ])*2/45;
            ddt[61] = (F[8 ])*8/45 + (F[7 ])*16/45 + (F[6 ])*32/45;
            ddt[62] = (F[8 ])*8/45 + (-F[7 ])*16/45 + (F[6 ])*32/45;
            ddt[63] = F[8 ];

            // scatter
#pragma unroll(64)
            for(d3 = 0; d3 < 64; d3++){
                dataDst[d3*FSTRIDE6X3+d1*C+d2] = ddt[d3]; 
            }
        }
    }
}

/* Kernel compute for bridge data by sgemm.
 * Number of sgemm calls is 64*BATCH. 
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
    for(d1 = 0; d1 < 64; d1++){
        for(d2 = 0; d2 < batch; d2++){
            const Dtype* pin = in+d1*ISTRIDE6X3+d2*irows*icols; 
            const Dtype* pft = filter+d1*FSTRIDE6X3; 
            Dtype* pot = out+d1*OSTRIDE6X3+d2*irows*fcols; 
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
        const int ntiles, const int mg6x3)
{
    int d1; 
    int sizeO = rows * cols;

#pragma omp parallel for private(d1)
    for(d1 = 0; d1 < N*K; d1++){
        int i, j;    
        Dtype ddt[64] __attribute__((aligned(64)));
        Dtype bridge[48] __attribute__((aligned(64)));

        const int t1 = d1/(K*mg6x3);
        const int t2 = (d1%(K*mg6x3))/mg6x3;
        const int t3 = d1%mg6x3;

        Dtype *data = out + (t1*mg6x3*K + t3*K + t2)*sizeO;
        int tileCount = d1*ntiles; 

        for(i = 0; i < rows; i += 6){
            //#pragma simd
            for(j = 0; j < cols; j += 6){
                ddt[ 0] = dataSrc[tileCount+  0*OSTRIDE6X3]; 
                ddt[ 1] = dataSrc[tileCount+  1*OSTRIDE6X3]; 
                ddt[ 2] = dataSrc[tileCount+  2*OSTRIDE6X3]; 
                ddt[ 3] = dataSrc[tileCount+  3*OSTRIDE6X3]; 
                ddt[ 4] = dataSrc[tileCount+  4*OSTRIDE6X3]; 
                ddt[ 5] = dataSrc[tileCount+  5*OSTRIDE6X3]; 
                ddt[ 6] = dataSrc[tileCount+  6*OSTRIDE6X3]; 
                ddt[ 7] = dataSrc[tileCount+  7*OSTRIDE6X3]; 

                ddt[ 8] = dataSrc[tileCount+  8*OSTRIDE6X3]; 
                ddt[ 9] = dataSrc[tileCount+  9*OSTRIDE6X3]; 
                ddt[10] = dataSrc[tileCount+ 10*OSTRIDE6X3]; 
                ddt[11] = dataSrc[tileCount+ 11*OSTRIDE6X3]; 
                ddt[12] = dataSrc[tileCount+ 12*OSTRIDE6X3]; 
                ddt[13] = dataSrc[tileCount+ 13*OSTRIDE6X3]; 
                ddt[14] = dataSrc[tileCount+ 14*OSTRIDE6X3]; 
                ddt[15] = dataSrc[tileCount+ 15*OSTRIDE6X3]; 

                ddt[16] = dataSrc[tileCount+ 16*OSTRIDE6X3]; 
                ddt[17] = dataSrc[tileCount+ 17*OSTRIDE6X3]; 
                ddt[18] = dataSrc[tileCount+ 18*OSTRIDE6X3]; 
                ddt[19] = dataSrc[tileCount+ 19*OSTRIDE6X3]; 
                ddt[20] = dataSrc[tileCount+ 20*OSTRIDE6X3]; 
                ddt[21] = dataSrc[tileCount+ 21*OSTRIDE6X3]; 
                ddt[22] = dataSrc[tileCount+ 22*OSTRIDE6X3]; 
                ddt[23] = dataSrc[tileCount+ 23*OSTRIDE6X3]; 

                ddt[24] = dataSrc[tileCount+ 24*OSTRIDE6X3]; 
                ddt[25] = dataSrc[tileCount+ 25*OSTRIDE6X3]; 
                ddt[26] = dataSrc[tileCount+ 26*OSTRIDE6X3]; 
                ddt[27] = dataSrc[tileCount+ 27*OSTRIDE6X3]; 
                ddt[28] = dataSrc[tileCount+ 28*OSTRIDE6X3]; 
                ddt[29] = dataSrc[tileCount+ 29*OSTRIDE6X3]; 
                ddt[30] = dataSrc[tileCount+ 30*OSTRIDE6X3]; 
                ddt[31] = dataSrc[tileCount+ 31*OSTRIDE6X3]; 

                ddt[32] = dataSrc[tileCount+ 32*OSTRIDE6X3]; 
                ddt[33] = dataSrc[tileCount+ 33*OSTRIDE6X3]; 
                ddt[34] = dataSrc[tileCount+ 34*OSTRIDE6X3]; 
                ddt[35] = dataSrc[tileCount+ 35*OSTRIDE6X3]; 
                ddt[36] = dataSrc[tileCount+ 36*OSTRIDE6X3]; 
                ddt[37] = dataSrc[tileCount+ 37*OSTRIDE6X3]; 
                ddt[38] = dataSrc[tileCount+ 38*OSTRIDE6X3]; 
                ddt[39] = dataSrc[tileCount+ 39*OSTRIDE6X3]; 

                ddt[40] = dataSrc[tileCount+ 40*OSTRIDE6X3]; 
                ddt[41] = dataSrc[tileCount+ 41*OSTRIDE6X3]; 
                ddt[42] = dataSrc[tileCount+ 42*OSTRIDE6X3]; 
                ddt[43] = dataSrc[tileCount+ 43*OSTRIDE6X3]; 
                ddt[44] = dataSrc[tileCount+ 44*OSTRIDE6X3]; 
                ddt[45] = dataSrc[tileCount+ 45*OSTRIDE6X3]; 
                ddt[46] = dataSrc[tileCount+ 46*OSTRIDE6X3]; 
                ddt[47] = dataSrc[tileCount+ 47*OSTRIDE6X3]; 

                ddt[48] = dataSrc[tileCount+ 48*OSTRIDE6X3]; 
                ddt[49] = dataSrc[tileCount+ 49*OSTRIDE6X3]; 
                ddt[50] = dataSrc[tileCount+ 50*OSTRIDE6X3]; 
                ddt[51] = dataSrc[tileCount+ 51*OSTRIDE6X3]; 
                ddt[52] = dataSrc[tileCount+ 52*OSTRIDE6X3]; 
                ddt[53] = dataSrc[tileCount+ 53*OSTRIDE6X3]; 
                ddt[54] = dataSrc[tileCount+ 54*OSTRIDE6X3]; 
                ddt[55] = dataSrc[tileCount+ 55*OSTRIDE6X3]; 

                ddt[56] = dataSrc[tileCount+ 56*OSTRIDE6X3]; 
                ddt[57] = dataSrc[tileCount+ 57*OSTRIDE6X3]; 
                ddt[58] = dataSrc[tileCount+ 58*OSTRIDE6X3]; 
                ddt[59] = dataSrc[tileCount+ 59*OSTRIDE6X3]; 
                ddt[60] = dataSrc[tileCount+ 60*OSTRIDE6X3]; 
                ddt[61] = dataSrc[tileCount+ 61*OSTRIDE6X3]; 
                ddt[62] = dataSrc[tileCount+ 62*OSTRIDE6X3]; 
                ddt[63] = dataSrc[tileCount+ 63*OSTRIDE6X3]; 

#if 0
                data[(i+0)*cols + (j+0)] = (ddt[0 ] + ddt[1 ] + ddt[2 ] + ddt[3 ] + ddt[4 ] + ddt[5 ] + ddt[6 ] + ddt[8 ] + ddt[9 ] + ddt[10] + ddt[11] + ddt[12] + ddt[13] + ddt[14] + ddt[16] + ddt[17] + ddt[18] + ddt[19] + ddt[20] + ddt[21] + ddt[22] + ddt[24] + ddt[25] + ddt[26] + ddt[27] + ddt[28] + ddt[29] + ddt[30] + ddt[32] + ddt[33] + ddt[34] + ddt[35] + ddt[36] + ddt[37] + ddt[38] + ddt[40] + ddt[41] + ddt[42] + ddt[43] + ddt[44] + ddt[45] + ddt[46] + ddt[48] + ddt[49] + ddt[50] + ddt[51] + ddt[52] + ddt[53] + ddt[54])*1;
                data[(i+0)*cols + (j+1)] = (ddt[5 ] - ddt[6 ] + ddt[13] - ddt[14] + ddt[21] - ddt[22] + ddt[29] - ddt[30] + ddt[37] - ddt[38] + ddt[45] - ddt[46] + ddt[53] - ddt[54])*0.5 + (ddt[1 ] - ddt[2 ] + ddt[9 ] - ddt[10] + ddt[17] - ddt[18] + ddt[25] - ddt[26] + ddt[33] - ddt[34] + ddt[41] - ddt[42] + ddt[49] - ddt[50]) + (ddt[3 ] - ddt[4 ] + ddt[11] - ddt[12] + ddt[19] - ddt[20] + ddt[27] - ddt[28] + ddt[35] - ddt[36] + ddt[43] - ddt[44] + ddt[51] - ddt[52])*2;
                data[(i+0)*cols + (j+2)] = (ddt[5 ] + ddt[6 ] + ddt[13] + ddt[14] + ddt[21] + ddt[22] + ddt[29] + ddt[30] + ddt[37] + ddt[38] + ddt[45] + ddt[46] + ddt[53] + ddt[54])*0.25 + (ddt[1 ] + ddt[2 ] + ddt[9 ] + ddt[10] + ddt[17] + ddt[18] + ddt[25] + ddt[26] + ddt[33] + ddt[34] + ddt[41] + ddt[42] + ddt[49] + ddt[50]) + (ddt[3 ] + ddt[4 ] + ddt[11] + ddt[12] + ddt[19] + ddt[20] + ddt[27] + ddt[28] + ddt[35] + ddt[36] + ddt[43] + ddt[44] + ddt[51] + ddt[52])*4;
                data[(i+0)*cols + (j+3)] = (ddt[5 ] - ddt[6 ] + ddt[13] - ddt[14] + ddt[21] - ddt[22] + ddt[29] - ddt[30] + ddt[37] - ddt[38] + ddt[45] - ddt[46] + ddt[53] - ddt[54])*0.125 + (ddt[1 ] - ddt[2 ] + ddt[9 ] - ddt[10] + ddt[17] - ddt[18] + ddt[25] - ddt[26] + ddt[33] - ddt[34] + ddt[41] - ddt[42] + ddt[49] - ddt[50]) + (ddt[3 ] - ddt[4 ] + ddt[11] - ddt[12] + ddt[19] - ddt[20] + ddt[27] - ddt[28] + ddt[35] - ddt[36] + ddt[43] - ddt[44] + ddt[51] - ddt[52])*8;
                data[(i+0)*cols + (j+4)] = (ddt[5 ] + ddt[6 ] + ddt[13] + ddt[14] + ddt[21] + ddt[22] + ddt[29] + ddt[30] + ddt[37] + ddt[38] + ddt[45] + ddt[46] + ddt[53] + ddt[54])*0.0625 + (ddt[1 ] + ddt[2 ] + ddt[9 ] + ddt[10] + ddt[17] + ddt[18] + ddt[25] + ddt[26] + ddt[33] + ddt[34] + ddt[41] + ddt[42] + ddt[49] + ddt[50]) + (ddt[3 ] + ddt[4 ] + ddt[11] + ddt[12] + ddt[19] + ddt[20] + ddt[27] + ddt[28] + ddt[35] + ddt[36] + ddt[43] + ddt[44] + ddt[51] + ddt[52])*16;
                data[(i+0)*cols + (j+5)] = (ddt[5 ] - ddt[6 ] + ddt[13] - ddt[14] + ddt[21] - ddt[22] + ddt[29] - ddt[30] + ddt[37] - ddt[38] + ddt[45] - ddt[46] + ddt[53] - ddt[54])*0.03125 + (ddt[1 ] - ddt[2 ] + ddt[7 ] + ddt[9 ] - ddt[10] + ddt[15] + ddt[17] - ddt[18] + ddt[23] + ddt[25] - ddt[26] + ddt[31] + ddt[33] - ddt[34] + ddt[39] + ddt[41] - ddt[42] + ddt[47] + ddt[49] - ddt[50] + ddt[55]) + (ddt[3 ] - ddt[4 ] + ddt[11] - ddt[12] + ddt[19] - ddt[20] + ddt[27] - ddt[28] + ddt[35] - ddt[36] + ddt[43] - ddt[44] + ddt[51] - ddt[52])*32;

                data[(i+1)*cols + (j+0)] = (ddt[40] + ddt[41] + ddt[42] + ddt[43] + ddt[44] + ddt[45] + ddt[46] - ddt[48] - ddt[49] - ddt[50] - ddt[51] - ddt[52] - ddt[53] - ddt[54])*0.5 + (ddt[8 ] + ddt[9 ] + ddt[10] + ddt[11] + ddt[12] + ddt[13] + ddt[14] - ddt[16] - ddt[17] - ddt[18] - ddt[19] - ddt[20] - ddt[21] - ddt[22]) + (ddt[24] + ddt[25] + ddt[26] + ddt[27] + ddt[28] + ddt[29] + ddt[30] - ddt[32] - ddt[33] - ddt[34] - ddt[35] - ddt[36] - ddt[37] - ddt[38])*2;
                data[(i+1)*cols + (j+1)] = (ddt[45] - ddt[46] - ddt[53] + ddt[54])*0.25 + (ddt[13] - ddt[14] - ddt[21] + ddt[22] + ddt[41] - ddt[42] - ddt[49] + ddt[50])*0.5 + (ddt[9 ] - ddt[10] - ddt[17] + ddt[18] + ddt[29] - ddt[30] - ddt[37] + ddt[38] + ddt[43] - ddt[44] - ddt[51] + ddt[52]) + (ddt[11] - ddt[12] - ddt[19] + ddt[20] + ddt[25] - ddt[26] - ddt[33] + ddt[34])*2 + (ddt[27] - ddt[28] - ddt[35] + ddt[36])*4;
                data[(i+1)*cols + (j+2)] = (ddt[45] + ddt[46] - ddt[53] - ddt[54])*0.125 + (ddt[13] + ddt[14] - ddt[21] - ddt[22])*0.25 + (ddt[29] + ddt[30] - ddt[37] - ddt[38] + ddt[41] + ddt[42] - ddt[49] - ddt[50])*0.5 + (ddt[9 ] + ddt[10] - ddt[17] - ddt[18]) + (ddt[25] + ddt[26] - ddt[33] - ddt[34] + ddt[43] + ddt[44] - ddt[51] - ddt[52])*2 + (ddt[11] + ddt[12] - ddt[19] - ddt[20])*4 + (ddt[27] + ddt[28] - ddt[35] - ddt[36])*8;
                data[(i+1)*cols + (j+3)] = (ddt[45] - ddt[46] - ddt[53] + ddt[54])*0.0625 + (ddt[13] - ddt[14] - ddt[21] + ddt[22])*0.125 + (ddt[29] - ddt[30] - ddt[37] + ddt[38])*0.25 + (ddt[41] - ddt[42] - ddt[49] + ddt[50])*0.5 + (ddt[9 ] - ddt[10] - ddt[17] + ddt[18]) + (ddt[25] - ddt[26] - ddt[33] + ddt[34])*2 + (ddt[43] - ddt[44] - ddt[51] + ddt[52])*4 + (ddt[11] - ddt[12] - ddt[19] + ddt[20])*8 + (ddt[27] - ddt[28] - ddt[35] + ddt[36])*16;
                data[(i+1)*cols + (j+4)] = (ddt[45] + ddt[46] - ddt[53] - ddt[54])*0.03125 + (ddt[13] + ddt[14] - ddt[21] - ddt[22])*0.0625 + (ddt[29] + ddt[30] - ddt[37] - ddt[38])*0.125 + (ddt[41] + ddt[42] - ddt[49] - ddt[50])*0.5 + (ddt[9 ] + ddt[10] - ddt[17] - ddt[18]) + (ddt[25] + ddt[26] - ddt[33] - ddt[34])*2 + (ddt[43] + ddt[44] - ddt[51] - ddt[52])*8 + (ddt[11] + ddt[12] - ddt[19] - ddt[20])*16 + (ddt[27] + ddt[28] - ddt[35] - ddt[36])*32;
                data[(i+1)*cols + (j+5)] = (ddt[45] - ddt[46] - ddt[53] + ddt[54])*0.015625 + (ddt[13] - ddt[14] - ddt[21] + ddt[22])*0.03125 + (ddt[29] - ddt[30] - ddt[37] + ddt[38])*0.0625 + (ddt[41] - ddt[42] + ddt[47] - ddt[49] + ddt[50] - ddt[55])*0.5 + (ddt[9 ] - ddt[10] + ddt[15] - ddt[17] + ddt[18] - ddt[23]) + (ddt[25] - ddt[26] + ddt[31] - ddt[33] + ddt[34] - ddt[39])*2 + (ddt[43] - ddt[44] - ddt[51] + ddt[52])*16 + (ddt[11] - ddt[12] - ddt[19] + ddt[20])*32 + (ddt[27] - ddt[28] - ddt[35] + ddt[36])*64;

                data[(i+2)*cols + (j+0)] = (ddt[40] + ddt[41] + ddt[42] + ddt[43] + ddt[44] + ddt[45] + ddt[46] + ddt[48] + ddt[49] + ddt[50] + ddt[51] + ddt[52] + ddt[53] + ddt[54])*0.25 + (ddt[8 ] + ddt[9 ] + ddt[10] + ddt[11] + ddt[12] + ddt[13] + ddt[14] + ddt[16] + ddt[17] + ddt[18] + ddt[19] + ddt[20] + ddt[21] + ddt[22]) + (ddt[24] + ddt[25] + ddt[26] + ddt[27] + ddt[28] + ddt[29] + ddt[30] + ddt[32] + ddt[33] + ddt[34] + ddt[35] + ddt[36] + ddt[37] + ddt[38])*4;
                data[(i+2)*cols + (j+1)] = (ddt[45] - ddt[46] + ddt[53] - ddt[54])*0.125 + (ddt[41] - ddt[42] + ddt[49] - ddt[50])*0.25 + (ddt[13] - ddt[14] + ddt[21] - ddt[22] + ddt[43] - ddt[44] + ddt[51] - ddt[52])*0.5 + (ddt[9 ] - ddt[10] + ddt[17] - ddt[18]) + (ddt[11] - ddt[12] + ddt[19] - ddt[20] + ddt[29] - ddt[30] + ddt[37] - ddt[38])*2 + (ddt[25] - ddt[26] + ddt[33] - ddt[34])*4 + (ddt[27] - ddt[28] + ddt[35] - ddt[36])*8;
                data[(i+2)*cols + (j+2)] = (ddt[45] + ddt[46] + ddt[53] + ddt[54])*0.0625 + (ddt[13] + ddt[14] + ddt[21] + ddt[22] + ddt[41] + ddt[42] + ddt[49] + ddt[50])*0.25 + (ddt[9 ] + ddt[10] + ddt[17] + ddt[18] + ddt[29] + ddt[30] + ddt[37] + ddt[38] + ddt[43] + ddt[44] + ddt[51] + ddt[52]) + (ddt[11] + ddt[12] + ddt[19] + ddt[20] + ddt[25] + ddt[26] + ddt[33] + ddt[34])*4 + (ddt[27] + ddt[28] + ddt[35] + ddt[36])*16;
                data[(i+2)*cols + (j+3)] = (ddt[45] - ddt[46] + ddt[53] - ddt[54])*0.03125 + (ddt[13] - ddt[14] + ddt[21] - ddt[22])*0.125 + (ddt[41] - ddt[42] + ddt[49] - ddt[50])*0.25 + (ddt[29] - ddt[30] + ddt[37] - ddt[38])*0.5 + (ddt[9 ] - ddt[10] + ddt[17] - ddt[18]) + (ddt[43] - ddt[44] + ddt[51] - ddt[52])*2 + (ddt[25] - ddt[26] + ddt[33] - ddt[34])*4 + (ddt[11] - ddt[12] + ddt[19] - ddt[20])*8 + (ddt[27] - ddt[28] + ddt[35] - ddt[36])*32;
                data[(i+2)*cols + (j+4)] = (ddt[45] + ddt[46] + ddt[53] + ddt[54])*0.015625 + (ddt[13] + ddt[14] + ddt[21] + ddt[22])*0.0625 + (ddt[29] + ddt[30] + ddt[37] + ddt[38] + ddt[41] + ddt[42] + ddt[49] + ddt[50])*0.25 + (ddt[9 ] + ddt[10] + ddt[17] + ddt[18]) + (ddt[25] + ddt[26] + ddt[33] + ddt[34] + ddt[43] + ddt[44] + ddt[51] + ddt[52])*4 + (ddt[11] + ddt[12] + ddt[19] + ddt[20])*16 + (ddt[27] + ddt[28] + ddt[35] + ddt[36])*64;
                data[(i+2)*cols + (j+5)] = (ddt[45] - ddt[46] + ddt[53] - ddt[54])*0.0078125 + (ddt[13] - ddt[14] + ddt[21] - ddt[22])*0.03125 + (ddt[29] - ddt[30] + ddt[37] - ddt[38])*0.125 + (ddt[41] - ddt[42] + ddt[47] + ddt[49] - ddt[50] + ddt[55])*0.25 + (ddt[9 ] - ddt[10] + ddt[15] + ddt[17] - ddt[18] + ddt[23]) + (ddt[25] - ddt[26] + ddt[31] + ddt[33] - ddt[34] + ddt[39])*4 + (ddt[43] - ddt[44] + ddt[51] - ddt[52])*8 + (ddt[11] - ddt[12] + ddt[19] - ddt[20])*32 + (ddt[27] - ddt[28] + ddt[35] - ddt[36])*128;

                data[(i+3)*cols + (j+0)] = (ddt[40] + ddt[41] + ddt[42] + ddt[43] + ddt[44] + ddt[45] + ddt[46] - ddt[48] - ddt[49] - ddt[50] - ddt[51] - ddt[52] - ddt[53] - ddt[54])*0.125 + (ddt[8 ] + ddt[9 ] + ddt[10] + ddt[11] + ddt[12] + ddt[13] + ddt[14] - ddt[16] - ddt[17] - ddt[18] - ddt[19] - ddt[20] - ddt[21] - ddt[22]) + (ddt[24] + ddt[25] + ddt[26] + ddt[27] + ddt[28] + ddt[29] + ddt[30] - ddt[32] - ddt[33] - ddt[34] - ddt[35] - ddt[36] - ddt[37] - ddt[38])*8;
                data[(i+3)*cols + (j+1)] = (ddt[45] - ddt[46] - ddt[53] + ddt[54])*0.0625 + (ddt[41] - ddt[42] - ddt[49] + ddt[50])*0.125 + (ddt[43] - ddt[44] - ddt[51] + ddt[52])*0.25 + (ddt[13] - ddt[14] - ddt[21] + ddt[22])*0.5 + (ddt[9 ] - ddt[10] - ddt[17] + ddt[18]) + (ddt[11] - ddt[12] - ddt[19] + ddt[20])*2 + (ddt[29] - ddt[30] - ddt[37] + ddt[38])*4 + (ddt[25] - ddt[26] - ddt[33] + ddt[34])*8 + (ddt[27] - ddt[28] - ddt[35] + ddt[36])*16;
                data[(i+3)*cols + (j+2)] = (ddt[45] + ddt[46] - ddt[53] - ddt[54])*0.03125 + (ddt[41] + ddt[42] - ddt[49] - ddt[50])*0.125 + (ddt[13] + ddt[14] - ddt[21] - ddt[22])*0.25 + (ddt[43] + ddt[44] - ddt[51] - ddt[52])*0.5 + (ddt[9 ] + ddt[10] - ddt[17] - ddt[18]) + (ddt[29] + ddt[30] - ddt[37] - ddt[38])*2 + (ddt[11] + ddt[12] - ddt[19] - ddt[20])*4 + (ddt[25] + ddt[26] - ddt[33] - ddt[34])*8 + (ddt[27] + ddt[28] - ddt[35] - ddt[36])*32;
                data[(i+3)*cols + (j+3)] = (ddt[45] - ddt[46] - ddt[53] + ddt[54])*0.015625 + (ddt[13] - ddt[14] - ddt[21] + ddt[22] + ddt[41] - ddt[42] - ddt[49] + ddt[50])*0.125 + (ddt[9 ] - ddt[10] - ddt[17] + ddt[18] + ddt[29] - ddt[30] - ddt[37] + ddt[38] + ddt[43] - ddt[44] - ddt[51] + ddt[52]) + (ddt[11] - ddt[12] - ddt[19] + ddt[20] + ddt[25] - ddt[26] - ddt[33] + ddt[34])*8 + (ddt[27] - ddt[28] - ddt[35] + ddt[36])*64;
                data[(i+3)*cols + (j+4)] = (ddt[45] + ddt[46] - ddt[53] - ddt[54])*0.0078125 + (ddt[13] + ddt[14] - ddt[21] - ddt[22])*0.0625 + (ddt[41] + ddt[42] - ddt[49] - ddt[50])*0.125 + (ddt[29] + ddt[30] - ddt[37] - ddt[38])*0.5 + (ddt[9 ] + ddt[10] - ddt[17] - ddt[18]) + (ddt[43] + ddt[44] - ddt[51] - ddt[52])*2 + (ddt[25] + ddt[26] - ddt[33] - ddt[34])*8 + (ddt[11] + ddt[12] - ddt[19] - ddt[20])*16 + (ddt[27] + ddt[28] - ddt[35] - ddt[36])*128;
                data[(i+3)*cols + (j+5)] = (ddt[45] - ddt[46] - ddt[53] + ddt[54])*0.00390625 + (ddt[13] - ddt[14] - ddt[21] + ddt[22])*0.03125 + (ddt[41] - ddt[42] + ddt[47] - ddt[49] + ddt[50] - ddt[55])*0.125 + (ddt[29] - ddt[30] - ddt[37] + ddt[38])*0.25 + (ddt[9 ] - ddt[10] + ddt[15] - ddt[17] + ddt[18] - ddt[23]) + (ddt[43] - ddt[44] - ddt[51] + ddt[52])*4 + (ddt[25] - ddt[26] + ddt[31] - ddt[33] + ddt[34] - ddt[39])*8 + (ddt[11] - ddt[12] - ddt[19] + ddt[20])*32 + (ddt[27] - ddt[28] - ddt[35] + ddt[36])*256;

                data[(i+4)*cols + (j+0)] = (ddt[40] + ddt[41] + ddt[42] + ddt[43] + ddt[44] + ddt[45] + ddt[46] + ddt[48] + ddt[49] + ddt[50] + ddt[51] + ddt[52] + ddt[53] + ddt[54])*0.0625 + (ddt[8 ] + ddt[9 ] + ddt[10] + ddt[11] + ddt[12] + ddt[13] + ddt[14] + ddt[16] + ddt[17] + ddt[18] + ddt[19] + ddt[20] + ddt[21] + ddt[22]) + (ddt[24] + ddt[25] + ddt[26] + ddt[27] + ddt[28] + ddt[29] + ddt[30] + ddt[32] + ddt[33] + ddt[34] + ddt[35] + ddt[36] + ddt[37] + ddt[38])*16;
                data[(i+4)*cols + (j+1)] = (ddt[45] - ddt[46] + ddt[53] - ddt[54])*0.03125 + (ddt[41] - ddt[42] + ddt[49] - ddt[50])*0.0625 + (ddt[43] - ddt[44] + ddt[51] - ddt[52])*0.125 + (ddt[13] - ddt[14] + ddt[21] - ddt[22])*0.5 + (ddt[9 ] - ddt[10] + ddt[17] - ddt[18]) + (ddt[11] - ddt[12] + ddt[19] - ddt[20])*2 + (ddt[29] - ddt[30] + ddt[37] - ddt[38])*8 + (ddt[25] - ddt[26] + ddt[33] - ddt[34])*16 + (ddt[27] - ddt[28] + ddt[35] - ddt[36])*32;
                data[(i+4)*cols + (j+2)] = (ddt[45] + ddt[46] + ddt[53] + ddt[54])*0.015625 + (ddt[41] + ddt[42] + ddt[49] + ddt[50])*0.0625 + (ddt[13] + ddt[14] + ddt[21] + ddt[22] + ddt[43] + ddt[44] + ddt[51] + ddt[52])*0.25 + (ddt[9 ] + ddt[10] + ddt[17] + ddt[18]) + (ddt[11] + ddt[12] + ddt[19] + ddt[20] + ddt[29] + ddt[30] + ddt[37] + ddt[38])*4 + (ddt[25] + ddt[26] + ddt[33] + ddt[34])*16 + (ddt[27] + ddt[28] + ddt[35] + ddt[36])*64;
                data[(i+4)*cols + (j+3)] = (ddt[45] - ddt[46] + ddt[53] - ddt[54])*0.0078125 + (ddt[41] - ddt[42] + ddt[49] - ddt[50])*0.0625 + (ddt[13] - ddt[14] + ddt[21] - ddt[22])*0.125 + (ddt[43] - ddt[44] + ddt[51] - ddt[52])*0.5 + (ddt[9 ] - ddt[10] + ddt[17] - ddt[18]) + (ddt[29] - ddt[30] + ddt[37] - ddt[38])*2 + (ddt[11] - ddt[12] + ddt[19] - ddt[20])*8 + (ddt[25] - ddt[26] + ddt[33] - ddt[34])*16 + (ddt[27] - ddt[28] + ddt[35] - ddt[36])*128;
                data[(i+4)*cols + (j+4)] = (ddt[45] + ddt[46] + ddt[53] + ddt[54])*0.00390625 + (ddt[13] + ddt[14] + ddt[21] + ddt[22] + ddt[41] + ddt[42] + ddt[49] + ddt[50])*0.0625 + (ddt[9 ] + ddt[10] + ddt[17] + ddt[18] + ddt[29] + ddt[30] + ddt[37] + ddt[38] + ddt[43] + ddt[44] + ddt[51] + ddt[52]) + (ddt[11] + ddt[12] + ddt[19] + ddt[20] + ddt[25] + ddt[26] + ddt[33] + ddt[34])*16 + (ddt[27] + ddt[28] + ddt[35] + ddt[36])*256;
                data[(i+4)*cols + (j+5)] = (ddt[45] - ddt[46] + ddt[53] - ddt[54])*0.00195312 + (ddt[13] - ddt[14] + ddt[21] - ddt[22])*0.03125 + (ddt[41] - ddt[42] + ddt[47] + ddt[49] - ddt[50] + ddt[55])*0.0625 + (ddt[29] - ddt[30] + ddt[37] - ddt[38])*0.5 + (ddt[9 ] - ddt[10] + ddt[15] + ddt[17] - ddt[18] + ddt[23]) + (ddt[43] - ddt[44] + ddt[51] - ddt[52])*2 + (ddt[25] - ddt[26] + ddt[31] + ddt[33] - ddt[34] + ddt[39])*16 + (ddt[11] - ddt[12] + ddt[19] - ddt[20])*32 + (ddt[27] - ddt[28] + ddt[35] - ddt[36])*512;

                data[(i+5)*cols + (j+0)] = (ddt[40] + ddt[41] + ddt[42] + ddt[43] + ddt[44] + ddt[45] + ddt[46] - ddt[48] - ddt[49] - ddt[50] - ddt[51] - ddt[52] - ddt[53] - ddt[54])*0.03125 + (ddt[8 ] + ddt[9 ] + ddt[10] + ddt[11] + ddt[12] + ddt[13] + ddt[14] - ddt[16] - ddt[17] - ddt[18] - ddt[19] - ddt[20] - ddt[21] - ddt[22] + ddt[56] + ddt[57] + ddt[58] + ddt[59] + ddt[60] + ddt[61] + ddt[62]) + (ddt[24] + ddt[25] + ddt[26] + ddt[27] + ddt[28] + ddt[29] + ddt[30] - ddt[32] - ddt[33] - ddt[34] - ddt[35] - ddt[36] - ddt[37] - ddt[38])*32;
                data[(i+5)*cols + (j+1)] = (ddt[45] - ddt[46] - ddt[53] + ddt[54])*0.015625 + (ddt[41] - ddt[42] - ddt[49] + ddt[50])*0.03125 + (ddt[43] - ddt[44] - ddt[51] + ddt[52])*0.0625 + (ddt[13] - ddt[14] - ddt[21] + ddt[22] + ddt[61] - ddt[62])*0.5 + (ddt[9 ] - ddt[10] - ddt[17] + ddt[18] + ddt[57] - ddt[58]) + (ddt[11] - ddt[12] - ddt[19] + ddt[20] + ddt[59] - ddt[60])*2 + (ddt[29] - ddt[30] - ddt[37] + ddt[38])*16 + (ddt[25] - ddt[26] - ddt[33] + ddt[34])*32 + (ddt[27] - ddt[28] - ddt[35] + ddt[36])*64;
                data[(i+5)*cols + (j+2)] = (ddt[45] + ddt[46] - ddt[53] - ddt[54])*0.0078125 + (ddt[41] + ddt[42] - ddt[49] - ddt[50])*0.03125 + (ddt[43] + ddt[44] - ddt[51] - ddt[52])*0.125 + (ddt[13] + ddt[14] - ddt[21] - ddt[22] + ddt[61] + ddt[62])*0.25 + (ddt[9 ] + ddt[10] - ddt[17] - ddt[18] + ddt[57] + ddt[58]) + (ddt[11] + ddt[12] - ddt[19] - ddt[20] + ddt[59] + ddt[60])*4 + (ddt[29] + ddt[30] - ddt[37] - ddt[38])*8 + (ddt[25] + ddt[26] - ddt[33] - ddt[34])*32 + (ddt[27] + ddt[28] - ddt[35] - ddt[36])*128;
                data[(i+5)*cols + (j+3)] = (ddt[45] - ddt[46] - ddt[53] + ddt[54])*0.00390625 + (ddt[41] - ddt[42] - ddt[49] + ddt[50])*0.03125 + (ddt[13] - ddt[14] - ddt[21] + ddt[22] + ddt[61] - ddt[62])*0.125 + (ddt[43] - ddt[44] - ddt[51] + ddt[52])*0.25 + (ddt[9 ] - ddt[10] - ddt[17] + ddt[18] + ddt[57] - ddt[58]) + (ddt[29] - ddt[30] - ddt[37] + ddt[38])*4 + (ddt[11] - ddt[12] - ddt[19] + ddt[20] + ddt[59] - ddt[60])*8 + (ddt[25] - ddt[26] - ddt[33] + ddt[34])*32 + (ddt[27] - ddt[28] - ddt[35] + ddt[36])*256;
                data[(i+5)*cols + (j+4)] = (ddt[45] + ddt[46] - ddt[53] - ddt[54])*0.00195312 + (ddt[41] + ddt[42] - ddt[49] - ddt[50])*0.03125 + (ddt[13] + ddt[14] - ddt[21] - ddt[22] + ddt[61] + ddt[62])*0.0625 + (ddt[43] + ddt[44] - ddt[51] - ddt[52])*0.5 + (ddt[9 ] + ddt[10] - ddt[17] - ddt[18] + ddt[57] + ddt[58]) + (ddt[29] + ddt[30] - ddt[37] - ddt[38])*2 + (ddt[11] + ddt[12] - ddt[19] - ddt[20] + ddt[59] + ddt[60])*16 + (ddt[25] + ddt[26] - ddt[33] - ddt[34])*32 + (ddt[27] + ddt[28] - ddt[35] - ddt[36])*512;
                data[(i+5)*cols + (j+5)] = (ddt[45] - ddt[46] - ddt[53] + ddt[54])*0.000976562 + (ddt[13] - ddt[14] - ddt[21] + ddt[22] + ddt[41] - ddt[42] + ddt[47] - ddt[49] + ddt[50] - ddt[55] + ddt[61] - ddt[62])*0.03125 + (ddt[9 ] - ddt[10] + ddt[15] - ddt[17] + ddt[18] - ddt[23] + ddt[29] - ddt[30] - ddt[37] + ddt[38] + ddt[43] - ddt[44] - ddt[51] + ddt[52] + ddt[57] - ddt[58] + ddt[63]) + (ddt[11] - ddt[12] - ddt[19] + ddt[20] + ddt[25] - ddt[26] + ddt[31] - ddt[33] + ddt[34] - ddt[39] + ddt[59] - ddt[60])*32 + (ddt[27] - ddt[28] - ddt[35] + ddt[36])*1024;
#else
                transformByAT_first(ddt, bridge);
                transformByAT_second(bridge, data, i, j, cols);
#endif
                tileCount++; 
            }
        }
    }
}

/* API for winograd F(6,3). */
    template<typename Dtype>
ACSAStatus ACSAWinoConvolution_6x3(const Dtype *in, const Dtype *filter, Dtype *out,
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
    const int bb6x3 = winoMess->batch_block_;
    const int mg6x3 = winoMess->merge_;
    const int outHeight = tensorOut->h_; 
    const int outWidth = tensorOut->w_; 
    const int ntiles = (outHeight/6)*(outWidth/6);

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
    ACSA_CHECK(((outHeight%6 == 0) && (outWidth%6 == 0)));

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
            inByTransform_nopad(b_in, wino_in, b_bts, C, H, W, ntiles, mg6x3);
#if 0
        else if(H*W > 1225)
            //else if(H*W > 1)
            inByTransform_padBigScale(b_in, wino_in, b_bts, C, H, W, pad_h, pad_w, ntiles, mg6x3);
        else{
            Dtype *in_pad = (Dtype *)mkl_malloc(num_threads*(H+2*pad_h)*(W+2*pad_w)*sizeof(Dtype), 64);
            memset(in_pad, 0, num_threads*(H+2*pad_h)*(W+2*pad_w)*sizeof(Dtype));
            inByTransform_padSmallScale(b_in, in_pad, wino_in, N, C, H, W, pad_h, pad_w, ntiles, mg6x3);
            mkl_free(in_pad);
        }
#endif
        matrix_compute(wino_in, mg6x3*ntiles, C, wino_filter, C, K, wino_out, b_bts/mg6x3);
        outByTransform(b_out, wino_out, b_bts, K, outHeight, outWidth, ntiles, mg6x3);
    }

    return ACSASUCCESS;
}

/* Instantiate Template */
template inline void transformByBT<float>(float *, float *, int);
template inline void transformByBT_first(float *, float *);
template inline void transformByBT_second(float *, float *, int);
template inline void transformByAT_first(float *, float *);
template inline void transformByAT_second(float *, float *, int, int, int);
#if 0
template void inByTransform_pad<float>(const float *, float *,
        const int, const int, const int, const int,
        const int, const int,
        const int, const int);
template void inByTransform_padBigScale<float>(const float *, float *,
        const int, const int, const int, const int,
        const int, const int,
        const int, const int);
template void inByTransform_padSmallScale<float>(const float *, float *, float *,
        const int, const int, const int, const int,
        const int, const int,
        const int, const int);
#endif
template void inByTransform_nopad<float>(const float *, float *,
        const int, const int, const int, const int,
        const int, const int);
template void filterByTransform<float>(const float *, float *,
        const int, const int);
template void matrix_compute<float>(const float *, const int, const int,
        const float *, const int, const int,
        float *, const int);
template void outByTransform<float>(float *, const float *,
        const int, const int, const int, const int,
        const int, const int);
template ACSAStatus ACSAWinoConvolution_6x3<float>(const float *, const float *, float *,
        ACSATensor4d*, ACSATensor4d*, ACSATensor4d*,
        ACSAConvMessage*, ACSAWinoMessage*);

template inline void transformByBT<double>(double *, double *, int);
template inline void transformByBT_first(double *, double *);
template inline void transformByBT_second(double *, double *, int);
template inline void transformByAT_first(double *, double *);
template inline void transformByAT_second(double *, double *, int, int, int);
template void inByTransform_nopad<double>(const double *, double *,
        const int, const int, const int, const int,
        const int, const int);
#if 0
template void inByTransform_pad<double>(const double *, double *,
        const int, const int, const int, const int,
        const int, const int,
        const int, const int);
template void inByTransform_padBigScale<double>(const double *, double *,
        const int, const int, const int, const int,
        const int, const int,
        const int, const int);
template void inByTransform_padSmallScale<double>(const double *, double *, double *,
        const int, const int, const int, const int,
        const int, const int,
        const int, const int);
#endif
template void filterByTransform<double>(const double *, double *,
        const int, const int);
template void matrix_compute<double>(const double *, const int, const int,
        const double *, const int, const int,
        double *, const int);
template void outByTransform<double>(double *, const double *,
        const int, const int, const int, const int,
        const int, const int);
template ACSAStatus ACSAWinoConvolution_6x3<double>(const double *, const double *, double *,
        ACSATensor4d*, ACSATensor4d*, ACSATensor4d*,
        ACSAConvMessage*, ACSAWinoMessage*);
