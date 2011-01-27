/* Hey, Emacs, this file contains -*- c -*- code. */

#pragma OPENCL EXTENSION cl_amd_printf: enable

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable


typedef uint  uint32_t;
typedef short int16_t;
typedef uchar uint8_t;
typedef char int8_t;

typedef struct {
    int8_t motionY;
    int8_t motionX;
} ppp_motion;

typedef struct {
    int y;
    int x;
} pt;

constant ppp_motion PPP_MOTION_INTRA = { .motionY=-128, .motionX=-128 };

enum motion_search {
    MS_NONE=0, MS_00=1, MS_DIAMOND=2
};


/*
 * np:    number of threads per work group
 * self:  number of own thread
 * selfT: thread ID in first two coordinates only
 *        (i.e., 0 <= selfT <= 63)
 * myX, myY: x- and y-coordinate corresponding to selfT
 */
#define np    (get_local_size(0)*get_local_size(1)*get_local_size(2))
#define self  (get_local_id(0)+get_local_size(0)*(get_local_id(1)+get_local_size(1)*get_local_id(2)))
#define selfT (get_local_id(0)+get_local_size(0)*get_local_id(1))
#define myX   (get_local_id(0))
#define myY   (get_local_id(1))

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
 * Compress 64 values from 'input' to 'output' returning
 * the number of bytes in 'totalBytes'.
 * Uses scans (parallel prefix sums) to parallelise the
 * computation. Return the compressed size (in bytes).
 * 'temp' is temporary storage for scans.
 */
int compress_block_red(local const int16_t input[64],
                       local uint8_t *codes,
                       local int temp[192]) {
    const int16_t val = input[selfT];

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
    prevNonZero[selfT] = val == 0 ? -1 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s=1; s<64; s*=2) {
        /* 'selfT' refers to the right interval,
         * 'other' to the left interval */
        int other = selfT-s;
        int myPrevNZ = prevNonZero[selfT];
        if (selfT >= s) {
            if (myPrevNZ == -1)
                myPrevNZ = prevNonZero[other];   // (*1*) above
            else
                myPrevNZ += min(other+1,s);      // (*2*) above
        }
        prevNonZero2[selfT] = myPrevNZ;
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

    int myPNZ = prevNonZero[selfT];
    int pnz63 = prevNonZero[63];
    barrier(0); /* we overwrite the scan buffers below */

    /* Number of zeros between the last non-zero value and the
     * current value (including the current value if it is 0).
     */
    int zeros = selfT - myPNZ;

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
         * of zeros (i.e., input[selfT+1]!=0).
         */
        if (selfT == 63) { 
            c0 = 0;
            len = 1;
        } else if (myPNZ < pnz63 && (zeros%7 == 0 || input[selfT+1] != 0)) {
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
    offsets[selfT] = len;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Sum up all 'len' values using parallel prefix sum.
     * Then, offsets[selfT] holds the number of nibbles written by
     * up to and including 'selfT'.
     */
    for (int s=1; s<64; s*=2) {
        int other = selfT-s;
        if (selfT >= s)
            offsets2[selfT] = offsets[other] + offsets[selfT];
        else
            offsets2[selfT] = offsets[selfT];
        barrier(CLK_LOCAL_MEM_FENCE);
        local int *tmp = offsets; offsets = offsets2; offsets2 = tmp;
    }
    int pos = offsets[selfT] - len;  /* Our position for writing our nibbles */
    int totalBytes = (offsets[63]+1)/2;
    barrier(CLK_LOCAL_MEM_FENCE);

    //   if (get_group_id(0) == 0 && get_group_id(1) == 0)
//        printf("%d: len=%d, offsets=%d\n", selfT, len, offsets[selfT]);

#ifdef COMPR_ATOM_OR
    /* Clear output space */
    for (int i=selfT; i<totalBytes; i+=np)
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
    if (selfT == 63 && pos%2 == 0)
        nibbles[pos+1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    /*
     * Pack the nibbles into bytes. The first nibble of each
     * pair of nibbles goes into the upper half of the byte.
     */
    for (int i=selfT; i<totalBytes; i+=np)
        codes[i] = (nibbles[2*i]<<4) | nibbles[2*i+1];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    /* Return the number of bytes used. */
    return totalBytes;
}


inline int8_t clamp_pixel_value(int16_t val) {
    return val < -128 ? -128 : val > 127 ? 127 : val;
// clamp() does not seem to be defined for int in OpenCL 1.0
//    return clamp(val, -128, 127);
}


/*
 * Offsets to use in diamond search.
 */
constant pt diamond_offsets[8] = { { -1, -1 }, { -2,  0 }, { -1,  1 },
                                   {  0, -2 },             {  0,  2 },
                                   {  1, -1 }, {  2,  0 }, {  1,  1 } };

/*
 * Offsets in the last step of diamond search (when distance
 * has reached 1).
 */
constant pt vh_offsets[4] = { { -1,  0 }, {  0, -1 },
                              {  0,  1 }, {  1,  0 } };


void qdct_block(local const int16_t f[64], local int16_t ff[64],
                local float4 temp[32]) {
    const uint r = get_local_id(1);
    const uint c = get_local_id(0);

    constant float4 *A4 = (constant float4 *)dct_coeffs;
    local    float4 *B4 = temp;
    local    float  *B  = (local float *)B4;
    local    float4 *BAtr_tr4 = temp + 16;
    local    float  *BAtr_tr  = (local float *)BAtr_tr4;

    B[selfT] = f[selfT];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute B*A^tr */
    BAtr_tr[8*c+r] = dot(B4[2*r], A4[2*c]) + dot(B4[2*r+1], A4[2*c+1]);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute A * (B*A^tr) */
    float Cval = dot(A4[2*r], BAtr_tr4[2*c]) + dot(A4[2*r+1], BAtr_tr4[2*c+1]);

    ff[permut[selfT]] = round(Cval / quantization_factors[selfT]);
    barrier(CLK_LOCAL_MEM_FENCE);
}


void iqdct_block(local const int16_t ff[64], local int16_t f[64],
                 local float4 temp[32]) {
    const uint r = get_local_id(1);
    const uint c = get_local_id(0);

    constant float4 *Atr4   = (constant float4 *)dct_coeffs_tr;
    local    float4 *B4     = temp;
    local    float  *B      = (local float *)B4;
    local    float4 *BA_tr4 = temp + 16;
    local    float  *BA_tr  = (local float *)BA_tr4;

    B[selfT] = ff[permut[selfT]] * quantization_factors[selfT];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute B*A */
    BA_tr[8*c+r] = dot(B4[2*r], Atr4[2*c]) + dot(B4[2*r+1], Atr4[2*c+1]);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute A^tr * (B*A) */
    float Cval = dot(Atr4[2*r], BA_tr4[2*c]) + dot(Atr4[2*r+1], BA_tr4[2*c+1]);

    f[selfT] = round(Cval);
    barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void encode_video_frame(global uint8_t *image, global int8_t *old_image,
                               int motion_search,
                               int rows, int columns, int format,
                               global uint *size,
                               global uint *offsets_and_sizes,
                               global uint8_t *frame,
                               global ppp_motion *motions) {
    const size_t blockX = get_group_id(0), blockY = get_group_id(1);
    const size_t block_nr = get_group_id(0)+get_num_groups(0)*get_group_id(1);

    motions[block_nr] = PPP_MOTION_INTRA;
    barrier(CLK_GLOBAL_MEM_FENCE);

    local int16_t block[64];
    local int16_t intra_qdct[64];
    local uint8_t intra_compr[96];

    int yy, xx;
    int len;

    yy = blockY * 8;
    xx = blockX * 8;

    /*
     * Put macro block number 'b' into 'block' and
     * compute its DCT and compressed representation.
     * The length of the compressed block is stored in 'intra_len'.
     */
    if (self < 64)
        block[self] = (int)image[(yy+myY)*columns+xx+myX] - 128;
    barrier(CLK_LOCAL_MEM_FENCE);

    local float4 temp[32];    
    local int comprTemp[192];

    qdct_block(block, intra_qdct, temp);
    len = compress_block_red(intra_qdct, intra_compr, comprTemp);

    /* Determine location where to put our data in output
     * (CPU will reorder blocks)
     */
    local size_t l_off;
    if (self == 0)
        l_off = atom_add(size, len);
    barrier(CLK_LOCAL_MEM_FENCE);
    size_t off = l_off;

    /* Write data to output */
    event_t ev;
    global int8_t *current = (global int8_t *)&image    [yy*columns + xx];
    ev = async_work_group_copy(&frame[off], intra_compr, len, 0);
        
    /* Decode block and replace pixels in current frame */
    iqdct_block(intra_qdct, block, temp);
    int16_t newpixel = block[selfT];
    current[myY*columns + myX] = clamp_pixel_value(newpixel);

    /* Write offset and length so CPU knows where to find
     * data for this block.
     */
    if (self == 0) {
        offsets_and_sizes[2*block_nr]   = off;
        offsets_and_sizes[2*block_nr+1] = len;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    wait_group_events(1, &ev);        
}
