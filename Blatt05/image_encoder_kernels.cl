/* Hey, Emacs, this file contains -*- c -*- code. */

/*
 * Define int16_t and uint8_t (so we can use the same types as
 * in the host code).
 */
typedef short int16_t;
typedef uchar uint8_t;


/*
 * OpenCL extension: allow printf() in kernels
 * (AMD/ATI implementation only)
 */
// #pragma OPENCL EXTENSION cl_amd_printf: enable


/*
 * Coefficients of the matrix A^tr to be used in DCT computation
 */
constant float dct_coeffs_tr[64] = {
    0.35355338f,  0.49039263f ,  0.46193978f ,  0.4157348f   ,
                  0.35355338f ,  0.2777851f  ,  0.19134171f  , 9.754512e-2f,
    0.35355338f,  0.4157348f  ,  0.19134171f , -9.754516e-2f ,
                 -0.35355338f , -0.49039266f , -0.46193978f  , -0.277785f,
    0.35355338f,  0.2777851f  , -0.19134176f , -0.49039266f  ,
                 -0.35355332f ,  9.754521e-2f, 0.46193978f   , 0.41573468f,
    0.35355338f,  9.754512e-2f, -0.46193978f , -0.277785f    ,
                  0.3535535f  ,  0.41573468f , -0.19134195f  , -0.4903926f,
    0.35355338f, -9.754516e-2f, -0.46193978f ,  0.2777852f   ,
                  0.35355338f , -0.4157349f  , -0.19134149f  , 0.4903927f,
    0.35355338f, -0.27778518f , -0.19134156f ,  0.49039263f  ,
                 -0.35355362f , -9.754511e-2f,  0.46193966f  , -0.4157348f,
    0.35355338f, -0.41573483f ,  0.1913418f  ,  9.7545035e-2f,
                 -0.35355327f ,  0.49039266f , -0.46193987f  , 0.27778557f,
    0.35355338f, -0.49039266f ,  0.46193978f , -0.4157349f   ,
                  0.3535534f ,  -0.27778542f ,  0.19134195f  , -9.754577e-2f
};

/*
 * Permutation of 64 values in a macro block.
 * Used in qdct_block() and iqdct_block().
 */
constant int permut[64] = {
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
constant int quantization_factors[64] = {
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
};


kernel void encode_frame(global uint8_t *image,
                         uint rows, uint columns, uint format,
                         global uint8_t *frame) {
    /* Copy the input image to the output element-by-element
     * in a single thread.
     * (This is, of course, not a correct implementation as it
     * does not put the pixels in the correct location. The ordering
     * of the pixels must be changed to follow the macro block order.
     * In addition, processing should happen in parallel...)
     */
    if (get_global_id(0) == 0 && get_global_id(1) == 0) {
        for (uint i=0; i<rows*columns; i++)
            frame[i] = image[i] - 128;
    }
}
