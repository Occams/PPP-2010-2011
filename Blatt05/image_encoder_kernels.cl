/* Hey, Emacs, this file contains -*- c -*- code. */

/*
 * Define int16_t and uint8_t (so we can use the same types as
 * in the host code).
 */
typedef short int16_t;
typedef uchar uint8_t;

enum ppp_image_format {
    PPP_IMGFMT_UNCOMPRESSED_BLOCKS = 0,
    PPP_IMGFMT_UNCOMPRESSED_DCT    = 1,
    PPP_IMGFMT_COMPRESSED_DCT      = 2
};

/*
 * OpenCL extension: allow printf() in kernels
 * (AMD/ATI implementation only)
 */
//#pragma OPENCL EXTENSION cl_amd_printf: enable

/*
 * Coefficients of the matrix A^tr to be used in DCT computation
 */
constant float dct_coeffs_tr[64] __attribute__ ((aligned(sizeof(float)))) = {
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

constant float8 dct_coeffs[8] __attribute__ ((aligned(sizeof(float)))) = {
	(float8) (0.35355338f,0.35355338f,0.35355338f,0.35355338f,0.35355338f,0.35355338f,0.35355338f,0.35355338f),
	(float8) (0.49039263f,0.4157348f,0.2777851f,9.754512e-2f,-9.754516e-2f,-0.27778518f,-0.41573483f,-0.49039266f),
	(float8) (0.46193978f,0.19134171f,-0.19134176f,-0.46193978f,-0.46193978f,-0.19134156f,0.1913418f,0.46193978f),
	(float8) (0.4157348f,-9.754516e-2f,-0.49039266f,-0.277785f,0.2777852f,0.49039263f,9.7545035e-2f,-0.4157349f),
	(float8) (0.35355338f,-0.35355338f,-0.35355332f,0.3535535f,0.35355338f,-0.35355362f,-0.35355327f,0.3535534f),
	(float8) (0.2777851f,-0.49039266f,9.754521e-2f,0.41573468f,-0.4157349f,-9.754511e-2f,0.49039266f,-0.27778542f),
	(float8) (0.19134171f,-0.46193978f,0.46193978f,-0.19134195f,-0.19134149f,0.46193966f,-0.46193987f,0.19134195f),
	(float8) (9.754512e-2f,-0.277785f,0.41573468f,-0.4903926f,0.4903927f,-0.4157348f,0.27778557f,-9.754577e-2f)
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
 * Computes M = A*M*A_tr
 */
void mm_tr(constant float A_tr[64], float *M) {
    float AM[64];

    /* Compute AM = A*M */
    for (int y=0; y<8; y++) {
        for (int x=0; x<8; x++) {
            float acc = 0.0f;
            for (int i=0; i<8; i++)
                acc += A_tr[i*8+y] * M[8*i+x]; /* A_tr must be addressed so that we get A */
            AM[y*8+x] = acc;
        }
    }

    /* Compute M = (AM)*A_tr */
    for (int y=0; y<8; y++) {
        for (int x=0; x<8; x++) {
            float acc = 0.0f;
            for (int i=0; i<8; i++)
                acc += AM[y*8+i] * A_tr[8*i+x]; /* A_tr must be addressed so that we get A */
            M[y*8+x] = acc;
        }
    }
}

/*
 * Return the value for the first nibble of the
 * encoding of 'val' (val /= 0).
 */
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
 * Compress the given input values (in 'input') storing
 * the compressed data in 'codes'. 'n_values' values are read from
 * 'input'. Returns the number of bytes used in 'codes' (i.e.,
 * the length of the compressed data in bytes).
 */
int compress_data(const __local int16_t *input, global uint8_t *codes) {
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

/* Union type to dynamically address single elements of the float8 vector type. */
union v_mask{ float8 v; float f[8]; };

kernel void encode_image(global uint8_t *image,
                         uint rows, uint columns, enum ppp_image_format format,
                         global uint8_t *frame) {
	local int16_t tmp[64];
	local union v_mask a[8] __attribute__ ((aligned(sizeof(float))));
	local union v_mask b[8] __attribute__ ((aligned(sizeof(float))));
	int b_col = get_global_id(0) / 8;
	int b_row = get_global_id(1) / 8;
	int b_num = b_col + (get_global_size(0)/8)*b_row;
	int b_col_offset = get_local_id(0);
	int b_row_offset = get_local_id(1);
	int local_idx = b_row_offset*8 + b_col_offset;
	int idx = b_row * 8 * columns + b_col * 64 + local_idx;
	
	/* 
	*	FIRST STEP: Macroblock-major order of bytes.
	*	The respective macroblock is stored in transposed form to
	*	access it row-wise later on.
	*/
	a[b_col_offset].f[b_row_offset] = image[get_global_id(1) * columns +  get_global_id(0)] - 128;
	
	if(format == PPP_IMGFMT_UNCOMPRESSED_BLOCKS) {
    	frame[idx] = a[b_col_offset].f[b_row_offset];
    	
    	/* Kernel is done in this case! */
    	return;
    }
	
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* SECOND STEP: DCT transformation. */
     
	/* B = DCT*A */
	b[b_row_offset].f[b_col_offset] = dot(dct_coeffs[b_row_offset].lo,a[b_col_offset].v.lo)
		+ dot(dct_coeffs[b_row_offset].hi, a[b_col_offset].v.hi);
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	/* TMP = B*DCT^tr and quantization as well as permutation. */	
	tmp[permut[local_idx]] = (int16_t) rint((dot(b[b_row_offset].v.lo,dct_coeffs[b_col_offset].lo)
		+ dot(b[b_row_offset].v.hi, dct_coeffs[b_col_offset].hi)) / quantization_factors[local_idx]);
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
    if(format == PPP_IMGFMT_UNCOMPRESSED_DCT) {
        frame[idx] = tmp[local_idx];
		
        /* We're done in this case. */
        return;
    }
    
    /* THIRD STEP: Compression */
	
    if(b_col_offset == 0 && b_row_offset == 0) {
		frame[(b_num+1)*96 + b_num] = compress_data(tmp, &(frame[b_num*97]));
	}
}
