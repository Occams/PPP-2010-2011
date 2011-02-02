/* Hey, Emacs, this file contains -*- c -*- code. */

/*
 * Define int16_t and uint8_t (so we can use the same types as
 * in the host code).
 */
typedef uint  uint32_t;
typedef short int16_t;
typedef uchar uint8_t;

/*
 * Use atom_or (instead of nibble[] array) when constructing
 * compressed nibbles. Only works on reger (the NVIDIA card
 * in ravel does not support atomic operations on local memory).
 * Using atom_or seems to slightly faster than using the
 * nibbles[] array.
 */
// #define COMPR_ATOM_OR

/*
 * OpenCL extension: allow printf() in kernels
 * (AMD/ATI implementation only)
 */
// #pragma OPENCL EXTENSION cl_amd_printf: enable

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#ifdef COMPR_ATOM_OR
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable
#endif



uint8_t first_nibble(int16_t val) {
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
 * Compress the given 64 input values (in 'input') storing the
 * compressed data in 'codes'. Returns the number of bytes used in
 * 'codes' (i.e., the length of the compressed data in bytes).
 * This is a naive implementation where all the threads do the
 * whole computation (i.e., it is like being single-threaded).
 */
int compress_block_simple(const local int16_t *input, local uint8_t *codes) {
    uint8_t nibbles[192];

    /* Walk through the values. */
    int zeros = 0, pos = 0;
    for (int i=0; i<64; i++) {
        int16_t val = input[i];
        if (val == 0)
            zeros++;
        else {
            int16_t absval = abs(val);
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
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Return the number of bytes used. */
    return pos/2;
}


/*
 * Compress 64 values from 'input' to 'output' returning
 * the number of bytes in 'totalBytes'.
 * Uses scans (parallel prefix sums) to parallelise the
 * computation. Return the compressed size (in bytes).
 * 'temp' is temporary storage for scans.
 */
int compress_block_red(local const int16_t input[64],
                       local uint8_t *codes,
                       local int temp[192]) {
    const size_t self = get_local_id(0) + get_local_id(1)*get_local_size(0);
    const size_t np   = get_local_size(0) * get_local_size(1);

    const int16_t val = input[self];

    /* buffers for the scans (parallel prefix sums) */
    local int *prevNonZero = temp, *prevNonZero2 = temp+64;

    /*
     * To determine which thread writes codes for zeros, we compute,
     * for each index i=0,...,63 where the last (rightmost) non-zero
     * value preceding i (including i if input[i]!=0) is. When no
     * non-zero value precedes input[i], the result shall be -1.
     * Formally:
     *   prevNonZero[i] := max({-1} u {j | input[j]!=0, j<=i})
     *
     * This enables us to compute the number of consecutive zeros
     * input[i] is part of (when input[i]==0) by  i-prevNonZeros[i].
     *
     * To compute prevNonZeros by a parallel scan (parallel prefix
     * sum), we need to define an associative operator. Conceptually,
     * we perform the scan on a list of pairs (length,offset) where
     * 'length' is the number of elements in a segment of 'input'
     * already processed and 'offset' gives the position (relative
     * offset from the start of the segment) of the last (rightmost)
     * non-zero value in the segment.
     *
     * Example: For input = { 1, 0, 0, 2, 3, 0, 0, 0 } we start the
     * reduction with
     *     { (1,0), (1,-1), (1,-1) (1,0), (1,0), (1,-1), (1,-1), (1,-1) }
     * Note that all the segments have length 1 and non-zero values
     * have offset 0 whereas zero values have offset -1 (because there
     * is no last non-zero in the segment).
     *
     * The reduction operator & we now use is:
     *
     *                            { (l1+l2, off1)     if off2 == -1  (*1*)
     *  (l1,off1) & (l2,off2)  =  {
     *                            { (l1+l2, l1+off2)  otherwise      (*2*)
     *
     * That is, when two (neighbouring) segments are combined into
     * one segment, the lengths are added (of course). The offset of
     * the last non-zero value is given by 'off1' if there is no
     * non-zero value in the right interval (i.e., off2==-1) or
     * by l1+off2 if the last non-zero value is in the right interval.
     *
     * This operator is associative (as can be easily verified) and
     * yields the offset of the last non-zero value. By using it as
     * the operator of a (parallel) scan, we get the offset of the
     * last non-zero value for every element in the input.
     *
     * The following implementation stores the offset in prevNonZero.
     * The lengths can be computed from the step size of the parallel
     * scan as the segments start with length 1 and the lengths double
     * in each step. Only the segments at the beginning cannot grow
     * bigger (i.e., the segment size is actually min(other+1,s)).
     */
    prevNonZero[self] = val == 0 ? -1 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s=1; s<64; s*=2) {
        /* 'self' refers to the right interval,
         * 'other' to the left interval */
        int other = self-s;
        int myPrevNZ = prevNonZero[self];
        if (self >= s) {
            if (myPrevNZ == -1)
                myPrevNZ = prevNonZero[other];   // (*1*) above
            else
                myPrevNZ += min(other+1,s);      // (*2*) above
        }
        prevNonZero2[self] = myPrevNZ;
        barrier(CLK_LOCAL_MEM_FENCE);

        local int *tmp = prevNonZero;
        prevNonZero = prevNonZero2;
        prevNonZero2 = tmp;
    }

    /*
     * prevNonZero[i] now contains the index of the last non-zero
     * before inputs[i] (including inputs[i] if inputs[i]!=0) or
     * -1 if there is no preceding zero.
     * In other words, if prevNonZero[i]==-1, then inputs[0],...,inputs[i]
     * are all 0. If prevNonZero[i]>=0 then inputs[prevNonZero[i]]!=0 and
     * inputs[prevNonZero[i]+1],...,inputs[i-1] are 0.
     */

    int myPNZ = prevNonZero[self];
    int pnz63 = prevNonZero[63];
    /* barrier(0) would suffice but does not work on NVIDIA */
    barrier(CLK_LOCAL_MEM_FENCE); /* we overwrite the scan buffers below */

    /* Number of zeros between the last non-zero value and the
     * current value (including the current value if it is 0).
     */
    int zeros = self - myPNZ;

    int len = 0;  /* Number of nibbles this thread writes */

    /* Nibble codes to output */
    uint32_t c0=0, c1=0, c2=0;

    local int *offsets  = prevNonZero;
    local int *offsets2 = prevNonZero2;
    if (val == 0) {
        /* If the value is 0 and we are the last thread, we write the
         * terminating 0.
         * Otherwise, we output a code for zero if we are not part of
         * the trailing zeros (because thread 63 writes '0' in this case),
         * and we are the 7th zero after the last code (zero or non-zero)
         * to be written or we are the last zero in the current sequence
         * of zeros (i.e., input[self+1]!=0).
         */
        if (self == 63) { 
            c0 = 0;
            len = 1;
        } else if (myPNZ < pnz63 && (zeros%7 == 0 || input[self+1] != 0)) {
            c0 = zeros%7 == 0 ? 7 : zeros%7;
            len = 1;
        }
    } else {
        int16_t absval = abs(val);
        c0 = first_nibble(val);
        if (absval >= 19) {
            uint8_t code = absval - 19;
            c1 = code >> 4;
            c2 = code & 0xF;
            len = 3;
        } else if (absval >= 3) {
            c1 = absval-3;
            len = 2;
        } else
            len = 1;
    }
    offsets[self] = len;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Sum up all 'len' values using parallel prefix sum.
     * Then, offsets[self] holds the number of nibbles written by
     * up to and including 'self'.
     */
    for (int s=1; s<64; s*=2) {
        int other = self-s;
        if (self >= s)
            offsets2[self] = offsets[other] + offsets[self];
        else
            offsets2[self] = offsets[self];
        barrier(CLK_LOCAL_MEM_FENCE);
        local int *tmp = offsets; offsets = offsets2; offsets2 = tmp;
    }
    int pos = offsets[self] - len;  /* Our position for writing our nibbles */
    int totalBytes = (offsets[63]+1)/2;
    barrier(CLK_LOCAL_MEM_FENCE);


#ifdef COMPR_ATOM_OR
    /* Clear output space */
    for (int i=self; i<totalBytes; i+=np)
        codes[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Construct one or two uint32_t values which can be
     * logically or'ed (at the appropriate, aligned offset)
     * to construct the output codes.
     * This code relies on the device being little-endian.
     */

    int off   = pos/8;
    int shift = (pos%8)*4;
    uint32_t code0, code1=0;
    if ((shift%8) == 4) {
        code0 = (c0 | (c1<<12) | (c2<<8)) << (shift-4);
        if (shift == 28)
            code1 = (c1<<4) | c2;
    } else {
        code0 = ((c0<<4) | c1 | (c2<<12)) << shift;
        if (shift == 24)
            code1 = c2<<4;
    }

    if (len > 0) {
        local uint32_t *codes32 = (local uint32_t *)codes;
        atom_or(&codes32[off], code0);
        if (code1 != 0)
            atom_or(&codes32[off+1], code1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    /* Put the nibbles into an array and then copy them together
     * to the final output.
     */
    local uint *nibbles = (local uint *)offsets;
    if (len >= 1)
        nibbles[pos] = c0;
    if (len >= 2)
        nibbles[pos+1] = c1;
    if (len >= 3)
        nibbles[pos+2] = c2;
    if (self == 63 && pos%2 == 0)
        nibbles[pos+1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    /*
     * Pack the nibbles into bytes. The first nibble of each
     * pair of nibbles goes into the upper half of the byte.
     */
    for (int i=self; i<totalBytes; i+=np)
        codes[i] = (nibbles[2*i]<<4) | nibbles[2*i+1];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    /* Return the number of bytes used. */
    return totalBytes;
}

/*
 * Coefficients of the matrix A to be used in DCT computation
 */
__attribute__((aligned(16)))
constant float dct_coeffs[64] = {
    0.35355338f ,  0.35355338f ,  0.35355338f ,  0.35355338f  ,
                   0.35355338f ,  0.35355338f ,  0.35355338f  ,  0.35355338f, 
    0.49039263f ,  0.4157348f  ,  0.2777851f  ,  9.754512e-2f ,
                  -9.754516e-2f, -0.27778518f , -0.41573483f  , -0.49039266f,
    0.46193978f ,  0.19134171f , -0.19134176f , -0.46193978f  ,
                  -0.46193978f , -0.19134156f ,  0.1913418f   ,  0.46193978f,
    0.4157348f  , -9.754516e-2f, -0.49039266f , -0.277785f    ,  
                   0.2777852f  ,  0.49039263f ,  9.7545035e-2f, -0.4157349f,   
    0.35355338f , -0.35355338f , -0.35355332f ,  0.3535535f   , 
                   0.35355338f , -0.35355362f , -0.35355327f  ,  0.3535534f,
    0.2777851f  , -0.49039266f ,  9.754521e-2f,  0.41573468f  ,
                  -0.4157349f  , -9.754511e-2f,  0.49039266f  , -0.27778542f, 
    0.19134171f , -0.46193978f ,  0.46193978f , -0.19134195f  ,
                  -0.19134149f ,  0.46193966f , -0.46193987f  ,  0.19134195f,
    9.754512e-2f, -0.277785f   ,  0.41573468f , -0.4903926f   ,
                   0.4903927f  , -0.4157348f  ,  0.27778557f  , -9.754577e-2f
};

/*
 * Coefficients of the matrix A^tr to be used in DCT computation
 */
__attribute__((aligned(16)))
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


/*
 * Compute C := A*B*A^tr.
 * Requires 64 threads.
 */
void mmm64(constant float A[64], local const float B[64],
           local float C[64]) {
    const uint r = get_local_id(0);
    const uint c = get_local_id(1);

    /* We store the intermediat matrix (for (B*A^tr)^tr)
     * in the same location as the result; this requires
     * an additional barrier, but does not need any
     * additional storage in the local memory.
     */
    local float *BAtr_tr = C;

    /* Compute B*A^tr */
    float acc = 0.0f;
    for (int i=0; i<8; i++)
        acc += B[8*r+i] * A[8*c+i];
    BAtr_tr[8*c+r] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute A * (B*A^tr) */
    acc = 0.0f;
    for (int i=0; i<8; i++)
        acc += A[8*r+i] * BAtr_tr[8*c+i];

    /* This barrier could be remove if BAtr_tr and C were
     * separate spaces
     */
    barrier(CLK_LOCAL_MEM_FENCE);

    C[8*r+c] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
 * Compute C := A*B*A^tr.
 * Requires 64 threads. Uses dot() to compute dot products.
 */
void mmm64dot(constant float A[64], local const float B[64],
              local float C[64]) {
    const uint r = get_local_id(0);
    const uint c = get_local_id(1);

    /* We store the intermediat matrix (for (B*A^tr)^tr)
     * in the same location as the result; this requires
     * an additional barrier, but does not need any
     * additional storage in the local memory.
     */
    local float *BAtr_tr = C;
    local float4 *BAtr_tr4 = (local float4 *)BAtr_tr;
    constant float4 *A4 = (constant float4 *)A;
    local float4 *B4 = (local float4 *)B;

    /* Compute B*A^tr */
    BAtr_tr[8*c+r] = dot(B4[2*r], A4[2*c]) + dot(B4[2*r+1], A4[2*c+1]);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute A * (B*A^tr) */
    float result = dot(A4[2*r], BAtr_tr4[2*c]) + dot(A4[2*r+1], BAtr_tr4[2*c+1]);

    /* This barrier could be remove if BAtr_tr and C were
     * separate spaces
     */
    barrier(CLK_LOCAL_MEM_FENCE);

    C[8*r+c] = result;
    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
 * Compute DCT of 'B' (with quantization and permutation)
 * and store result in 'ff'.
 */
void qdct_block(local float B[64], local int16_t ff[64],
                local float4 temp[16]) {
    const uint r = get_local_id(1);
    const uint c = get_local_id(0);
    const uint self = 8*r+c;

    constant float4 *A4 = (constant float4 *)dct_coeffs;
    local    float4 *B4 = (local float4 *)B;
    local    float4 *BAtr_tr4 = temp;
    local    float  *BAtr_tr  = (local float *)BAtr_tr4;

    /* Compute B*A^tr */
    BAtr_tr[8*c+r] = dot(B4[2*r], A4[2*c]) + dot(B4[2*r+1], A4[2*c+1]);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute A * (B*A^tr) */
    float Cval = dot(A4[2*r], BAtr_tr4[2*c]) + dot(A4[2*r+1], BAtr_tr4[2*c+1]);

    ff[permut[self]] = round(Cval / quantization_factors[self]);
    barrier(CLK_LOCAL_MEM_FENCE);
}


/*
 * Each block corresponds to a work group with 64 threads.
 * Each work group consists of 8x8 threads.
 * With the attribute 'reqd_work_size' we make sure the kernel
 * can only be called with a work group size of 8x8 (or 8x8x1).
 */
__attribute__((reqd_work_group_size(8, 8, 1)))
kernel void encode_frame(global uint8_t *image, int rows, int columns,
                         int format, global uint *size,
                         global uint *offsets_and_sizes,
                         global uint8_t *frame) {
    const size_t self = get_local_id(0) + get_local_id(1)*get_local_size(0);
    const size_t np   = get_local_size(0) * get_local_size(1);
    const uint blockX = get_group_id(0);
    const uint blockY = get_group_id(1);
    const uint block_nr = blockY*(columns/8) + blockX;
    const uint n_blocks = (columns/8)*(rows/8);

    local uint8_t result[96];
    size_t len;

    const bool compr  =  format == 2;  // Is compression (-c) requested?

    /* Initialize 'size'.
     * When not compressing, we know the size in advance.
     * When compression is on, each work group increments 'size' 
     * before storing its data, so we store all the data contiguously,
     * but not necessarily in the right order (the CPU reorders the data
     * later).
     */
    if (self == 0 && blockX == 0 && blockY == 0)
        *size = compr ? 0 : 64*n_blocks;
    barrier(CLK_GLOBAL_MEM_FENCE);

    global uint8_t *current = image + 8*blockY*columns + 8*blockX;

    local int16_t ff[64];

    /* Temporary space for compression (comprTemp) is not
     * used at the same time as 'temp' and B4 (which are for qdct_block())
     * but 'temp' and 'B4' are both used in qdct_block(), so we
     * can use a union to store 'comprTemp' in the same
     * memory area as 'temp' and 'B4' to save some space in local memory.
     */
    local union {
        struct {
            float4 temp[16];
            float4 B4[16];
        };
        int comprTemp[192];
    } t;
    local float *B = (local float *)t.B4;

    switch(format) {
    case 0: {
        /* Reorder block directly to the output location */
        result[self] = (int)current[(self/8)*columns + (self%8)] - 128;
        len = 64;
        break;
    }
    case 1: {
        /* Reorder block to scratchpad */
        B[self] = (int)current[(self/8)*columns + (self%8)] - 128;
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Compute DCT and store the result of the DCT to the output location */
        qdct_block(B, ff, t.temp);
        result[self] = (uint8_t)ff[self];
        len = 64;
        break;
    }
    case 2:
        /* Reorder block to scratchpad */
        B[self] = (int)current[(self/8)*columns + (self%8)] - 128;
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Compute DCT and call compression */
        qdct_block(B, ff, t.temp);
        // *size = compress_block_simple(qdct_res, result);
        len = compress_block_red(ff, result, t.comprTemp);
        break;
    default:
        len = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* When compression is not enabled, our data has to be put
     * at 64*block_nr. With compression, we store the output data
     * contiguously, but we cannot guarantee the order (the CPU
     * will reorder the data later).
     */
    size_t off;
    if (compr) {
        /* Atomically increment 'size' by 'len'. The return value
         * (l_off) is the old value of 'size', i.e., the offset where
         * to put our data.
         */
        local size_t l_off;
        if (self == 0)
            l_off = atom_add(size, len);
        barrier(CLK_LOCAL_MEM_FENCE);
        off = l_off;
    } else
        off = 64*block_nr;
    
    /* We could use a loop to copy our data from local memory
     * to global memory. We use the asynchronous copy function
     * to do so. We have to wait for the completion of the
     * copy using wait_group_events() below.
     */
    event_t ev = async_work_group_copy(&frame[off], result, len, 0);
    if (self == 0) {
        /* Write out the offset and the length of our data so
         * the CPU knows to find the data for our block.
         */
        offsets_and_sizes[2*block_nr]   = off;
        offsets_and_sizes[2*block_nr+1] = len;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    wait_group_events(1, &ev);
}

