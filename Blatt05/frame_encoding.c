#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ppp_image.h"

#include "frame_encoding.h"

#include <xmmintrin.h>

/*
 * An image is dividing it into 8x8 macro blocks.
 * Each macro block is encoded individually (i.e.,
 * independently of the other macro blocks in the frame).
 *
 * The encoding of a macro block happens in several steps:
 *  1) The unsigned gray values in the range 0..255 are
 *     transformed to signed values in the range -128..127,
 *     i.e., 128 is subtracted from each unsigned gray value
 *     (see uint8_image_to_blocks()).
 *
 *  2) The 8x8 values of the block are transformed 
 *     by discrete cosine transform (DCT), see function
 *     qdct_block() below.
 *     This yields a new 8x8 matrix of float values.
 *
 *  3) Each value in the result of the DCT is divided by a certain
 *     factor (found in matrix 'quantization_factors') and
 *     rounded to the nearest integer. In addition, the
 *     ordering of the values is changed (according to 'permut')
 *     to achieve better compression.
 *     This yields a new 8x8 matrix of signed integer values.
 *
 *  4) The 64 values in the 8x8 matrix after DCT are
 *     compressed (entroy encoded) using the compression
 *     algorithm described below and implemented in
 *     compress_block() and uncompress_block().
 *
 * Function encode_frame() below implements the necessary
 * steps to encode an image.
 */


/*
 * Divide a by b. The remainder is always positive.
 */
static inline div_t quotRem(int a, unsigned b) {
    div_t d = div(a, b);
    if (d.rem < 0) {
        d.rem  += b;
        d.quot--;
    }
    return d;
}

/*
 * Compute (block,offset)-pair from (y,x) coordinate
 * for an image with 'columns' pixels per row.
 * 'y' and 'x' can be negative.
 */
void row_col_to_block(int y, int x, int columns, int *block, int *offset) {
    div_t xx = quotRem(x, 8);
    div_t yy = quotRem(y, 8);
    *offset = 8*yy.rem + xx.rem;
    *block  = yy.quot * (columns/8) + xx.quot;
}

/*
 * 'block' and 'offset' are assumed to be non-negative.
 */
void block_to_row_col(int block, int offset, int columns, int *y, int *x) {
    int blocks_per_row = columns/8;
    *y = 8*(block/blocks_per_row) + offset/8;
    *x = 8*(block%blocks_per_row) + offset%8;
}

/*
 * Reorder the pixels in 'image' from row-wise ordering to
 * macro block order, i.e., the 64 pixels of a macro block (processed
 * row-wise) are stored adjacent in 'blocks'. The blocks are output
 * in row major order (row-wise).
 */
void uint8_image_to_blocks(const uint8_t *image, int rows, int columns, int16_t *blocks) {
    int idx = 0;
    for (int yy=0; yy<rows; yy+=8) {
        for (int xx=0; xx<columns; xx+=8) {
            for (int y=0; y<8; y++) {
                for (int x=0; x<8; x++)
                    blocks[idx++] = (int16_t)image[(yy+y)*columns + xx+x] - 128;
            }
        }
    }
}

/*
 * Matrices with coefficients for computing (I)DCTs.
 * Initialized in init_qdct().
 */
static float dct_coeffs[8][8];
static float dct_coeffs_tr[8][8];

/*
 * Permutation of 64 values in a macro block.
 * Used in qdct_block() and iqdct_block().
 */
static const int permut[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

/*
 * Quantization factors for the results of the DCT
 * of a macro block. Used in qdct_block() and iqdct_block().
 */
static const int quantization_factors[64] = {
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
};

/*
 * Initialize the matrices
 *   dct_coeffs
 *   dct_coeffs_tr
 * for the (I)DCT implementation below.
 *
 * This function must be called before using
 * qdct_block() or iqdct_block().
 */
void init_qdct(void) {
    const float inv_sqrt_2 = 0.70710677f; /* 1/sqrtf(2) */
    const float pi_16 = M_PI / 16.0f;

    for (int k=0; k<8; k++) {
        const float f =  k==0 ? 0.5f*inv_sqrt_2 : 0.5f;
        for (int n=0; n<8; n++) {
            float coeff = f * cosf(((2*n+1)*k)*pi_16);
            dct_coeffs[k][n] = coeff;
            dct_coeffs_tr[n][k] = coeff;
        }
    }
}

/*
 * Given two 8x8 matrices A and B (stored in row-major),
 * compute C := A*B*A^tr
 * (where A^tr denotes the transposition of A).
 */
static void mmm(float A[8][8], const float *B, float *C) {
    float BAtr_tr[8][8];

    /* Compute B*A^tr and store it result (in transposed
     * form) in BAtr_tr. The result is stored in transposed
     * form becase the second step below can access BAtr_tr
     * row-wise, then.
     */
    for (int r=0; r<8; r++) {       /* row of B */
        for (int c=0; c<8; c++) {   /* column of A^tr */
            float acc = 0.0f;
            for (int i=0; i<8; i++)
                acc += B[8*r+i] * A[c][i];
            BAtr_tr[c][r] = acc;
        }
    }

    /* Compute A*(B*A_tr). Since (B*A_tr) is stored
     * in BAtr_tr in transposed form, BAtr_tr is accessed
     * row-wise. */
    for (int r=0; r<8; r++) {      /* row of A */
        for (int c=0; c<8; c++) {  /* column of B*A^tr (= row of BAtr_tr) */
            float acc = 0.0f;
            for (int i=0; i<8; i++)
                acc += A[r][i] * BAtr_tr[c][i];
            C[8*r+c] = acc;
        }
    }
}

/*
 * This version of 'mmm' also computes
 *    C := A*B*A^tr
 * but the loops on 'r' and 'c' are exchanged to expose
 * SIMD parallelism (the SIMD parallelism is exploited
 * using SSE intrinsics in the version 'mmm_sse' below).
 * To retain a row-wise memory access pattern, we use
 * a little trick (remeber that (A*B)^tr = B^tr * A^tr holds):
 * C=A*B*A^tr implies C^tr = A*B^tr*A^tr, i.e., we can access
 * B row-wise (because we actually use B^tr in the computation)
 * and store C row-wise (as we actually compute C^tr).
 *
 * In both loop nests, the loops on 'c' and 'r' are independent,
 * i.e., their iterations could be executed in parallel.
 *
 * Note that instead of A (as with 'mmm'), this version takes
 * A^tr as first parameter.
 */
static void mmm_tr(float A_tr[8][8], const float *B, float *C) {
    float ABtr_tr[8][8];

    /* Compute A*B_tr into ABtr_tr */
    for (int c=0; c<8; c++) {      /* column of B_tr */
        for (int r=0; r<8; r++) {  /* row of A */
            float acc = 0.0f;
            for (int i=0; i<8; i++)
                acc += A_tr[i][r] * B[8*c+i];
            ABtr_tr[c][r] = acc;
        }
    }

    /* Compute (A*B_tr)*A_tr */
    for (int c=0; c<8; c++) {      /* column of A_tr */
        for (int r=0; r<8; r++) {  /* row of A*B_tr */
            float acc = 0.0f;
            for (int i=0; i<8; i++)
                acc += ABtr_tr[i][r] * A_tr[i][c];
            C[8*c+r] = acc;
        }
    }
}

/*
 * SSE version of 'mmm_tr'.
 * An SSE processor register can store 4 float values.
 * The type __m128 denotes a packed 4-float value (which can be
 * stored in an SSE register). We transform the 8 iterations of
 * the loop on 'r' (in 'mmm_tr') into 2 __m128 values
 * (corresponding to the 8 values the 8 iterations compute).
 *
 * The following SSE intrinsics directly map to 1
 * (or 2 in case of _mm_set_ps1) SSE instructions:
 *
 *   __m128 _mm_set_ps1(float v)
 *              set all 4 floats in the result to v
 *
 *   __m128 _mm_loadu_ps(float *p)
 *              load the 4 floats p[0],...,p[3] and return then as
 *              a __m128 value
 *
 *   void _mm_storeu_ps(float *p, __m128 x)
 *              store the 4 floats in 'x' to p[0],...,p[3]
 *
 *   __m128 _mm_add_ps(__m128 a, __m128 b)
 *   __m128 _mm_mul_ps(__m128 a, __m128 b)
 *              add/multiply the values of a and b element-wise
 *
 */
static void mmm_sse(float A_tr[8][8], const float *B, float *C) {
    __m128 ABtr_tr[8][2];

    /* Compute A*B_tr into ABtr_tr */
    for (int c=0; c<8; c++) {      /* column of B_tr */
        __m128 acc1, acc2;
        acc1 = acc2 = _mm_set_ps1(0.0f);

        for (int i=0; i<8; i++) {
            __m128 A_tr1 = _mm_loadu_ps(&A_tr[i][0]);
            __m128 A_tr2 = _mm_loadu_ps(&A_tr[i][4]);
            __m128 B_val = _mm_set_ps1(B[8*c+i]);
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(A_tr1, B_val));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(A_tr2, B_val));
        }
        ABtr_tr[c][0] = acc1;
        ABtr_tr[c][1] = acc2;
    }

    /* Compute (A*B_tr)*A_tr and store the result (which is C_tr)
     * in transposed form into C
     */
    for (int c=0; c<8; c++) {      /* column of A_tr */
        __m128 acc1, acc2;
        acc1 = acc2 = _mm_set_ps1(0.0f);
        for (int i=0; i<8; i++) {
            __m128 Atr_val = _mm_set_ps1(A_tr[i][c]);
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(ABtr_tr[i][0], Atr_val));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(ABtr_tr[i][1], Atr_val));
        }
        _mm_storeu_ps(&C[8*c+0], acc1);
        _mm_storeu_ps(&C[8*c+4], acc2);
    }
}

/*
 * Compute an 8x8 DCT of the 64 values stored in 'f',
 * divide by the quantization factors and permute the values
 * yielding the 64 result values in 'ff'.
 *
 * The DCT is computed by the matrix product
 *    qdct_coeffs * f * qdct_coeffs_tr
 * (using one of the 'mmm*' functions).
 */
void qdct_block(const int16_t *f, int16_t *ff) {
    float B[64], C[64];

    /* Convert the values to float. */
    for (int i=0; i<64; i++)
        B[i] = f[i];

    /*
     * Compute the DCT.
     *
     * Altervative implemenations:
     *   mmm    (dct_coeffs,    B, C);
     *   mmm_tr (dct_coeffs_tr, B, C);
     */
    mmm_sse(dct_coeffs_tr, B, C);

    /* Divide DCT results by quantization factors and permute
     * the values (and round to nearest integer).
     */
    for (int i=0; i<64; i++)
        ff[permut[i]] = lrint(C[i] / quantization_factors[i]);
}

/*
 * Inverse of qdct_block. Undo the permutation, 
 * multiply by the quantization factors and apply
 * an 8x8 IDCT to the 64 values given in 'ff' and
 * return the 64 result values in 'f'.
 *
 * The DCT is computed by the matrix product
 *    qdct_coeffs_tr * ff * qdct_coeffs
 * (using one of the 'mmm*' functions).
 * (Due to the special properties of (I)DCT,
 * the transpose of qdct_coeffs is also its own inverse.)
 */
void iqdct_block(const int16_t *ff, int16_t *f) {
    float B[64], C[64];

    /* Undo permutation and multiply by quantization factors. */
    for (int i=0; i<64; i++)
        B[i] = ff[permut[i]] * quantization_factors[i];

    /*
     * Apply the inverse DCT.
     *
     * Altervatives implmentations:
     *   mmm   (dct_coeffs_tr, B, C);
     *   mmm_tr(dct_coeffs,    B, C);
     */
    mmm_sse(dct_coeffs, B, C);

    /* Round results to nearest integer. */
    for (int i=0; i<64; i++)
        f[i] = lrint(C[i]);
}



/*
 * The number of bytes that are maximally required
 * to stored the compression result of 'n_values' values.
 */
int max_encoded_length(int n_values) {
    return (3*n_values+1)/2;
}

int uncompress_data(const uint8_t *codes, int max_input_length,
                    int n_values, int16_t *output) {
    const int max_nibbles = 2 * max_input_length;
    uint8_t nibble(int n) {
        uint8_t v;
       if (n >= max_nibbles)
            return 0;
        v = codes[n/2];
        return  (n & 1) ? (v&0xF) : (v>>4);
    }
    
    int i = 0, pos = 0;
    while (i < n_values) {
        uint8_t c = nibble(pos++);
        if (c == 0) {
            memset(output+i, 0, (n_values-i)*sizeof(*output));
            i = n_values;
        } else if (c < 8) {
            memset(output+i, 0, c*sizeof(*output));
            i += c;
        } else if (c == 0x8)
            output[i++] = 1;
        else if (c == 0x9)
            output[i++] = 2;
        else if (c == 0xA)
            output[i++] = -1;
        else if (c == 0xB)
            output[i++] = -2;
        else if (c == 0xC)
            output[i++] = nibble(pos++) + 3;
        else if (c == 0xD)
            output[i++] = - (int16_t)(nibble(pos++) + 3);
        else if (c == 0xE) {
            int16_t v;
            v = nibble(pos++) << 4;
            v |= nibble(pos++);
            output[i++] = v + 19;
        } else {
            int16_t v;
            v = nibble(pos++) << 4;
            v |= nibble(pos++);
            output[i++] = -(v + 19);
        }
    }
    return (pos+1)/2;
}

/*
 * Uncompress a block. The compressed codes are given in 'codes';
 * the decompressed 64 values are stored in 'output'.
 * Up to 96 bytes from 'codes' are conusmed to obtain the 64
 * values. Return the number of bytes consumed from 'codes'.
 */
int uncompress_block(const uint8_t *codes, int16_t *output) {
    return uncompress_data(codes, 96, 64, output);
}

/*
 * Call uncompress_block and iqdct_block on 'input'
 * returning the result in 'block'.
 */
int uncompress_iqdct_block(const uint8_t *input, int16_t *block) {
    int16_t dct[64];
    int len = uncompress_block(input, dct);
    iqdct_block(dct, block);
    return len;
}



/*
 ********************************
 * Compression (Entropy Coding) *
 ********************************
 */

/*
 * A sequence of of 64 values is stored as a sequence of
 * 4-, 8-, and 12-bit codes, i.e., each code consists of
 * 1, 2 or 3 nibbles (4-bit values). The first nibble
 * unambiguously indicates the type of the code.
 *
 * Let n1 be the next nibble in the encoded sequence and n2, n3 the
 * nibbles following n1 (if needed). The decoding yiels:
 *
 * Nibble n1
 *   0x0:          all remaining values in the sequence of 64 are 0.
 *   0x1 .. 0x7:   indicate n1 zeros  
 *   0x8:          next value us +1
 *   0x9:          next value is -1
 *   0xA:          next value is +2
 *   0xB:          next value is -2
 *   0xC:          next value is  (n2 + 3)
 *   0xD:          next value is -(n2 + 3)
 *   0xE:          next value is  (n2 + (n3<<4) + 19)
 *   0xF:          next value is -(n2 + (n3<<4) + 19)
 *
 * Example:
 * The encoded sequence 0xC 0xB 0x2 0x9 0xF 0x1 0x2 0x0
 * is decoded as:
 *   0xC 0xB:     (0xB + 3) = 14
 *   0x2:         two zeros, i.e., 0 0
 *   0x9:         -1
 *   0xF 0x1 0x2: -(0x1 + (0x2<<4) + 19) = -(0x21 + 19) = -52
 *   0x0:         all remaining values are 0
 * That is, the encoded sequence represents the following original
 * sequence (in decimal):
 *   14, 0, 0, -1, -52, 0,   ...,    0
 *                       \ 59 zeros /
 *
 * The nibbles in the compressed sequence are stored 2 nibbles per
 * byte (of course). The first nibble is stored in the higher bits
 * of the byte. In the above example, the sequence of bytes would be:
 *   0xCB 0x29 0xF1 0x20
 * When the number of nibbles is odd, a nibble 0x0 is added (which
 * is stored in the lower bits of the last byte).
 *
 * Since each of the 64 values in an input sequence is represented
 * by at most 3 nibbles in the output, the maximal length
 * of the compressed sequence is 3*64 nibbles = 96 bytes.
 */

/*
 * Return the value for the first nibble of the
 * encoding of 'val' (val /= 0).
 */
static uint8_t first_nibble(int16_t val) {
    if (val == 1)
        return 0x8;
    else if (val == 2)
        return 0x9;
    else if (val == -1)
        return 0xA;
    else if (val == -2)
        return 0xB;
    else if (val >= 19)
        return 0xE;
    else if (val <= -19)
        return 0xF;
    else if (val <= -3)
        return 0xD;
    else
        return 0xC;
}

/*
 * Compress the given input values (in 'input') storing
 * the compressed data in 'codes'. 'n_values' values are read from
 * 'input'. Returns the number of bytes used in 'codes' (i.e.,
 * the length of the compressed data in bytes).
 */
int compress_data(const int16_t *input, int n_values, uint8_t *codes) {
    uint8_t nibbles[3*n_values];

    /* Walk through the values. */
    int zeros = 0, pos = 0;
    for (int i=0; i<n_values; i++) {
        int16_t val = input[i];
        if (val == 0)
            zeros++;
        else {
            /* When we meet a non-zero value, we output an appropriate
             * number of codes for the zeros preceding the current
             * non-zero value.
             */
            int16_t absval = val < 0 ? -val : val;
            while (zeros > 0) {
                int z = zeros > 7 ? 7 : zeros;
                nibbles[pos++] = z;
                zeros -= z;
            }
            nibbles[pos++] = first_nibble(val);
            if (absval >= 19) {
                uint8_t code = absval - 19;
                nibbles[pos++] = code >> 4;
                nibbles[pos++] = code & 0xF;
            } else if (absval >= 3) {
                nibbles[pos++] = absval - 3;
            }
        }
    }

    /* When the sequence ends with zeros, terminate the
     * sequence with 0x0.
     */
    if (zeros > 0)
        nibbles[pos++] = 0;
    
    /* Add a nibble 0x0 if the number of nibbles in the code
     * is odd.
     */
    if ((pos & 1) != 0)
        nibbles[pos++] = 0;
    
    /*
     * Pack the nibbles into bytes. The first nibble of each
     * pair of nibbles goes into the upper half of the byte.
     */
    for (int i=0; i<pos/2; i++)
        codes[i] = (nibbles[2*i] << 4) | nibbles[2*i+1];

    /* Return the number of bytes used. */
    return pos/2;
}

/*
 * Compress a sequence of 64 uncompressed values in 'input'
 * to the compressed representation at 'codes'.
 * Return the number of bytes used in the compressed
 * result. At most 96 bytes are used in 'codes'.
 */
int compress_block(const int16_t *input, uint8_t *codes) {
    return compress_data(input, 64, codes);
}

/*
 * Call qdct_block and compress_block on 'block'
 * returning the result in 'output'.
 */
int qdct_compress_block(const int16_t *block, uint8_t *output) {
    int16_t dctres[64];
    qdct_block(block, dctres);
    return compress_block(dctres, output);
}



/*
 * Example function that encodes an image according
 * to the requested image format (given in 'format').
 * The pixels must be stored in macro block order, i.e.,
 * the first 64 values are the pixels of the first 8x8 macro block
 * and so on (the macro blocks are stored row-wise).
 * The result is stored in the given ppp_frame 'frame'.
 */
void encode_frame(const int16_t *blocks, int rows, int columns,
                  enum ppp_image_format format, ppp_frame *frame) {
    int16_t qdctres[64];
    const int pixels = rows * columns;
    const int n_blocks = pixels / 64;

    switch(format) {
    case PPP_IMGFMT_UNCOMPRESSED_BLOCKS:
        for (int i=0; i<pixels; i++)
            ((int8_t *)frame->data)[i] = blocks[i];
        frame->length = pixels;
        break;
    case PPP_IMGFMT_UNCOMPRESSED_DCT:
        for (int b=0; b<n_blocks; b++) {
            qdct_block(blocks+64*b, qdctres);
            for (int i=0; i<64; i++)
                ((int8_t *)frame->data)[64*b+i] = qdctres[i];
        }
        frame->length = pixels;
        break;
    case PPP_IMGFMT_COMPRESSED_DCT:
        frame->length = 0;
        for (int b=0; b<n_blocks; b++)
            frame->length += qdct_compress_block(blocks+64*b, frame->data+frame->length);
        break;
    }
}
