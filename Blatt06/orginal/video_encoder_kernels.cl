/* Hey, Emacs, this file contains -*- c -*- code. */

//#pragma OPENCL EXTENSION cl_amd_printf: enable

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
                local float4 temp[32], bool cond) {
    const uint r = get_local_id(1);
    const uint c = get_local_id(0);

    constant float4 *A4 = (constant float4 *)dct_coeffs;
    local    float4 *B4 = temp;
    local    float  *B  = (local float *)B4;
    local    float4 *BAtr_tr4 = temp + 16;
    local    float  *BAtr_tr  = (local float *)BAtr_tr4;
	
	if (cond)
		B[selfT] = f[selfT];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute B*A^tr */
	if (cond)
		BAtr_tr[8*c+r] = dot(B4[2*r], A4[2*c]) + dot(B4[2*r+1], A4[2*c+1]);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute A * (B*A^tr) */
	if (cond) {
    float Cval = dot(A4[2*r], BAtr_tr4[2*c]) + dot(A4[2*r+1], BAtr_tr4[2*c+1]);
    ff[permut[selfT]] = round(Cval / quantization_factors[selfT]);
	}
    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
 * Compress the given input values (in 'input') storing
 * the compressed data in 'codes'. 'n_values' values are read from
 * 'input'. Returns the number of bytes used in 'codes' (i.e.,
 * the length of the compressed data in bytes).
 */
int compress_data(const __local int16_t *input, local uint8_t *codes) {
    const int n_values = 64;
    uint8_t nibbles[3*64];

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


void iqdct_block(local const int16_t ff[64], local int16_t f[64],
                 local float4 temp[32], bool cond) {
    const uint r = get_local_id(1);
    const uint c = get_local_id(0);

    constant float4 *Atr4   = (constant float4 *)dct_coeffs_tr;
    local    float4 *B4     = temp;
    local    float  *B      = (local float *)B4;
    local    float4 *BA_tr4 = temp + 16;
    local    float  *BA_tr  = (local float *)BA_tr4;
	
	if (cond)
		B[selfT] = ff[permut[selfT]] * quantization_factors[selfT];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute B*A */
	if (cond)
		BA_tr[8*c+r] = dot(B4[2*r], Atr4[2*c]) + dot(B4[2*r+1], Atr4[2*c+1]);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Compute A^tr * (B*A) */
	if (cond){
		float Cval = dot(Atr4[2*r], BA_tr4[2*c]) + dot(Atr4[2*r+1], BA_tr4[2*c+1]);
		f[selfT] = round(Cval);
	}
	
    barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void encode_video_frame(global uint8_t *image, global int8_t *old_image,
                               int motion_search,
                               int rows, int columns, int format,
                               global uint8_t *frame,
                               global ppp_motion *motions) {
    const size_t blockX = get_group_id(0), blockY = get_group_id(1);
    const size_t block_nr = get_group_id(0)+get_num_groups(0)*get_group_id(1);
    local int16_t block[128];
    local int16_t intra_qdct[128],reconstruct[64];
	local float4 temp[64];  
    int yy = blockY * 8, xx = blockX * 8, pixel = (yy+myY)*columns+xx+myX;

	motions[block_nr] = PPP_MOTION_INTRA;

    if (self < 64) {
        block[selfT] = (int)image[pixel] - 128;
		block[selfT + 64] = block[selfT];
	}
	
	
    barrier(CLK_LOCAL_MEM_FENCE);  
	
	/* Parallel DCT */
	int offset[4] = {0,1,0,1};
	int offset32 = offset[get_local_id(2)]*32, offset64 = offset[get_local_id(2)]*64;
    qdct_block(block + offset64, intra_qdct + offset64, temp + offset32, get_local_id(2) == 0 || get_local_id(2) == 1);
	
	
	/* Do both compressions in parallel */
	local uint8_t codes_intra[97];
	local uint8_t codes_p[97];
	
    if(self == 0)
		codes_intra[96] = compress_data(intra_qdct, codes_intra);
	if(self == 65)
		codes_p[96] = compress_data(intra_qdct + 64, codes_p);
	
	barrier(CLK_LOCAL_MEM_FENCE);  
	
	/* Store compression result in frame array (coalesced global memory access)*/
	if(self < 97) {	  
		if (codes_intra[96] > codes_p[96]) {
				frame[block_nr*97 + self] = codes_intra[self];
		} else {
				frame[block_nr*97 + self] = codes_p[self];
		}
	}

	/* Reconstruction */
	iqdct_block(intra_qdct, reconstruct, temp, get_local_id(2) == 0 || get_local_id(2) == 1);
	
	if (self < 64)
		image[pixel] = (uint8_t) reconstruct[selfT];
	
}
