Abgabe PPP Blatt 5
--------------------------------------------------------------------------------
Huber Bastian       51432
Watzinger Daniel    51746


#######################
### image_encoder.c ###
#######################

#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "compression_stats.h"
#include "frame_encoding.h"
#include "ocl_init.h"
#include "ppp_image.h"
#include "ppp_pnm.h"


enum impl_type {
	IMPL_SEQ, IMPL_CPU, IMPL_GPU
};

double seconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}  

/*
* Encode a frame using the CPU implementation.
*/
static ppp_frame *encode(uint8_t *image, const ppp_image_info *info) {
	ppp_frame *frame;
	int16_t *blocks;
	const int rows = info->rows;
	const int columns = info->columns;
	const enum ppp_image_format format = info->format;

	/*
	* max_enc_bytes is the maximal number of bytes needed 
	* in any of the supported formats.
	*/
	const int max_enc_bytes = max_encoded_length(rows*columns);

	/*
	* Allocate space for the macro blocks.
	*/
	blocks = malloc(rows*columns * sizeof(int16_t));
	if (blocks == NULL)
	return NULL;

	/*
	* Allocate a frame for the result.
	*/
	frame = ppp_frame_alloc(max_enc_bytes);
	if (frame == NULL) {
		free(blocks);
		return NULL;
	}

	double t = seconds();

	/*
	* Convert the image to blocks (i.e., change the ordering of
	* the pixels from row-wise to macro block order).
	*/
	uint8_image_to_blocks(image, rows, columns, blocks);

	/*
	* Encode the blocks according to 'format'.
	*/
	encode_frame(blocks, rows, columns, format, frame);
	
	t = seconds() - t;
	printf("Duration: %.3f ms\n", t*1000);

	free(blocks);

	return frame;
}


/*
* Encode a frame using the OpenCL implementation.
*/
static ppp_frame *encode_opencl(uint8_t *image, const ppp_image_info *info,
		bool use_gpu) {

	ppp_frame *frame;
	const cl_uint rows = info->rows;
	const cl_uint columns = info->columns;
	const cl_uint format = info->format;

	/*
	* max_enc_bytes is the maximal number of bytes needed 
	* in any of the supported formats.
	*/
	const int max_enc_bytes = max_encoded_length(rows*columns);
	const int blocks = rows*columns/64;
	cl_context context;
	cl_device_id devid;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem imageGPU, frameGPU;
	cl_int res;


	/* Allocate space for the result. Reserve space for length values. */
	frame = ppp_frame_alloc(max_enc_bytes + blocks);
	if (frame == NULL)
		return NULL;

	context = create_cl_context(use_gpu, &devid);
	
	/* Create a command queue (which allows timing) */
	queue = clCreateCommandQueue(context, devid, CL_QUEUE_PROFILING_ENABLE,
		&res);
	if (queue == NULL)
		error_and_abort("Could not create command queue", res);
	
	/* Allocate the buffer memory object for the input image.
	* We request that the image (from the host) is copied to the device.
	*/
	imageGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	    sizeof(*image)*rows*columns, (void *)image, &res);
	if (res != CL_SUCCESS)
		error_and_abort("Could not allocate imageGPU", res);
	
	/* Allocate the buffer memory object for the result.
	* We need at most 'max_enc_bytes' to represent the result.
	*/
	frameGPU = clCreateBuffer(context, CL_MEM_READ_WRITE,
	    max_enc_bytes + blocks, NULL, &res);
	
	if (res != CL_SUCCESS)
		error_and_abort("Could not allocate frameGPU", res);

	/*
	* Load the OpenCL program from file "image_encoder_kernels.cl"
	* and compile it.
	*/
	program = build_program("image_encoder_kernels.cl", context, devid);
	
	/* Find kernel "blockwise_order" in the compiled program. */
	kernel = clCreateKernel(program, "encode_image", NULL);
	if (kernel == NULL) {
		fprintf(stderr, "Could not create kernel 'encode_frame'.\n");
		exit(1);
	}

	/* Set the arguments for the kernel invocation.
	* We pass the pointers to the input image, the number of rows
	* and columns, the format and the pointer to the result.
	*/
	clSetKernelArg(kernel, 0, sizeof(cl_mem),  &imageGPU);
	clSetKernelArg(kernel, 1, sizeof(cl_uint), &rows);
	clSetKernelArg(kernel, 2, sizeof(cl_uint), &columns);
	clSetKernelArg(kernel, 3, sizeof(cl_uint), &format);
	clSetKernelArg(kernel, 4, sizeof(cl_mem),  &frameGPU);
	// clSetKernelArg(kernel, 5, sizeof(int16_t) * 64,  NULL);
	// clSetKernelArg(kernel, 6, sizeof(float) * 64,  NULL);
	
	/* Set the work group size and global number of work items for the kernel. */
	size_t work_dims = 2;
	size_t global_work_size[] = {columns, rows};
	size_t local_work_size[] = {8,8};

	cl_event kernelEvent;
    res = clEnqueueNDRangeKernel(queue, kernel, work_dims, NULL,
                                 global_work_size, local_work_size,
                                 0, NULL, &kernelEvent);
                                 
	if (res != CL_SUCCESS)
		error_and_abort("Could not enqueue kernel invocation", res);
		

    size_t size;
    if(info->format != PPP_IMGFMT_COMPRESSED_DCT) {
        size = rows * columns;
        frame->length = size;
    } else {
        size = max_enc_bytes + blocks;
    }
    

	/* Copy the result from the device to the host. */
	res = clEnqueueReadBuffer(queue, frameGPU, CL_TRUE, 0,
		size, frame->data, 0, NULL, NULL);

	if (res != CL_SUCCESS)
		error_and_abort("Could not enqueue buffer read", res);
		
    /* Compress encoded values by using the encoding index */
    if(info->format == PPP_IMGFMT_COMPRESSED_DCT) {
	
        frame->length = 0;
        for(int i = 0; i < blocks; i++) {
			int lenght = (int) frame->data[(i+1)*96 + i];
			memcpy(&(frame->data[frame->length]), &(frame->data[i*97]), lenght);
			frame->length += lenght;
        }
    }
	
	res = clFinish(queue);
	if (res != CL_SUCCESS)
		error_and_abort("Error at clFinish", res);
	
	cl_ulong nanos = get_event_end_nanos(kernelEvent) -
       get_event_start_nanos(kernelEvent);
	printf("Duration: %.3f ms\n", nanos/1.0e6);

	clReleaseMemObject(imageGPU);
	clReleaseMemObject(frameGPU);
	clReleaseKernel(kernel);
	clReleaseContext(context);

	return frame;
}


static void usage(const char *progname) {
	fprintf(stderr, "USAGE: %s -i IN -o OUT [-d | -c] [-z IMPL]\n", progname);
	fprintf(stderr,
	"  -d: with DCT\n"
	"  -c: with DCT and compression\n"
	"  -z: select implementation:\n"
	"        seq: CPU sequential (default)\n"
	"        cpu: CPU via OpenCL\n"
	"        gpu: GPU via OpenCL\n"
	"  -s: show encoder statistics (only with -c)\n"
	"  -h: show this help\n"
	"\n"
	);
}

int main(int argc, char *argv[]) {
	enum pnm_kind kind;
	enum ppp_image_format format;
	int rows, columns, maxcolor;
	uint8_t *image;
	int option;

	char *infile, *outfile;
	bool dct, compress, show_stats;
	enum impl_type implementation;

	init_qdct();

	infile = NULL;
	outfile = NULL;
	dct = false;
	compress = false;
	show_stats = false;
	implementation = IMPL_SEQ;
	while ((option = getopt(argc,argv,"cdhi:o:sz:")) != -1) {
		switch(option) {
		case 'd': dct = true; break;
		case 'c': compress = true; break;
		case 'i': infile = strdup(optarg); break;
		case 'o': outfile = strdup(optarg); break;
		case 's': show_stats = true; break;
		case 'z':
			if (strcmp(optarg, "seq") == 0) {
				implementation = IMPL_SEQ;
				break;
			} else if (strcmp(optarg, "cpu") == 0) {
				implementation = IMPL_CPU;
				break;
			} else if (strcmp(optarg, "gpu") == 0) {
				implementation = IMPL_GPU;
				break;
			}
			/* fall through */
		case 'h':
		default:
			usage(argv[0]);
			return 1;
		}
	}
	
	if (infile == NULL || outfile == NULL || (dct && compress)) {
		usage(argv[0]);
		return 1;
	}
	
	if (dct)
	format = PPP_IMGFMT_UNCOMPRESSED_DCT;
	else if (compress)
	format = PPP_IMGFMT_COMPRESSED_DCT;
	else
	format = PPP_IMGFMT_UNCOMPRESSED_BLOCKS;
	
	image = ppp_pnm_read(infile, &kind, &rows, &columns, &maxcolor);
	
	if (image != NULL) {
		if (rows%8 != 0 || columns%8 != 0) {
			fprintf(stderr, "Error: number of rows and columns must be "
			"multiples of 8\n");
		} else if (kind == PNM_KIND_PGM) {
			ppp_image_info info;
			ppp_frame *frame;
			info.rows = rows;
			info.columns = columns;
			info.format = format;

			if (implementation == IMPL_SEQ)
			frame = encode(image, &info);
			else
			frame = encode_opencl(image, &info,
			implementation == IMPL_GPU);

			if (frame != NULL) {
				if (show_stats && format == PPP_IMGFMT_COMPRESSED_DCT) {
					encoder_stats_init();
					encoder_stats(frame->data, frame->length);
					encoder_stats_print(stderr);
				}

				if (ppp_image_write(outfile, &info, frame) != 0)
				fprintf(stderr, "could not write image\n");
				free(frame);
			} else
			fprintf(stderr, "error while encoding\n");
		} else
		fprintf(stderr, "not a PGM image\n");
		free(image);
	} else
	fprintf(stderr, "could not load image\n");
	
	return 0;
}



################################
### image_encoder_kernels.cl ###
################################

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
	
	if(format == PPP_IMGFMT_UNCOMPRESSED_BLOCKS) {
    	frame[idx] =  image[get_global_id(1) * columns +  get_global_id(0)] - 128;
    	
    	/* Kernel is done in this case! */
    	return;
    }
	
	a[b_col_offset].f[b_row_offset] = image[get_global_id(1) * columns +  get_global_id(0)] - 128;
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
