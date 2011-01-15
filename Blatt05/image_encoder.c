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

	cl_context context;
	cl_device_id devid;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem imageGPU, frameGPU, frameGPUComprIx;
	cl_int res;


	/* Allocate space for the result. */
	frame = ppp_frame_alloc(max_enc_bytes);
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
	    max_enc_bytes, NULL, &res);
	
	if (res != CL_SUCCESS)
		error_and_abort("Could not allocate frameGPU", res);
		
    if(info->format == PPP_IMGFMT_COMPRESSED_DCT) {
        frameGPUComprIx = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              (rows*columns)/64, NULL, &res);
                              
        if (res != CL_SUCCESS)
            error_and_abort("Could not allocate frameGPUComprIx", res);
    }

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
	if(format == PPP_IMGFMT_COMPRESSED_DCT) {
	    printf("Pass frame index\n");
    	clSetKernelArg(kernel, 5, sizeof(cl_mem),  &frameGPUComprIx);
    } else {
        clSetKernelArg(kernel, 5, sizeof(cl_mem),  &frameGPU);
    }
	clSetKernelArg(kernel, 6, sizeof(int16_t) * 64,  NULL);
	clSetKernelArg(kernel, 7, sizeof(float) * 64,  NULL);
	
	/* Set the work group size and global number of work items for blockwise_order kernel. */
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
        size = max_enc_bytes;
    }
    

	/* Copy the result from the device to the host. */
	res = clEnqueueReadBuffer(queue, frameGPU, CL_TRUE, 0,
		size, frame->data, 0, NULL, NULL);

	if (res != CL_SUCCESS)
		error_and_abort("Could not enqueue buffer read", res);
		
    /* Compress encoded values by using the encoding index */
    if(info->format == PPP_IMGFMT_COMPRESSED_DCT) {
        int encix_size = (rows*columns)/64;
        uint8_t encix[encix_size];
        
        /* Copy the encoding index from the device to the host. */
        res = clEnqueueReadBuffer(queue, frameGPUComprIx, CL_TRUE, 0,
                                  encix_size, encix, 0, NULL, NULL);
        if (res != CL_SUCCESS)
            error_and_abort("Could not enqueue buffer (frameGPUComprIx) read", res);
        
        frame->length = 0;
        for(int i = 0; i < encix_size; i++) {
            if(frame->length > 0) {
                memcpy(&(frame->data[frame->length]), &(frame->data[i*96]), encix[i]);
            }
            frame->length += encix[i];
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
    if(info->format == PPP_IMGFMT_COMPRESSED_DCT)
        clReleaseMemObject(frameGPUComprIx);
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

