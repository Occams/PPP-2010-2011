#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <CL/cl.h>

#include "ocl_init.h"
#include "ppp_pnm.h"
#include "ppp_video.h"

#include "frame_encoding.h"

#include "compression_stats.h"
#include "motion_stats.h"

/* Which implementation (CPU sequential, CPU OpenCL, GPU OpenCL) to use */
enum impl_type {
	IMPL_SEQ, IMPL_CPU, IMPL_GPU
};

/* How to estimate motion between frames */
enum prediction {
	MS_NONE=0,   /* no motion estimation (i.e., use only intra-coded frames) */
	MS_00=1,     /* try offset (0,0) in previous frame only */
	MS_DIAMOND=2 /* use diamond search to find best prediction */
};

/* Options controlling the encoding process */
typedef struct {
	int frames_to_skip;   /* Number of frames to skip at beginning of input */
	int max_frames;       /* Number of frames to encode */
	enum prediction motion_search;   /* Motion estimation method */
	bool show_stats;      /* Print statistics after encoding */
} options;

/* Representation for 2-dimensional coordinates
* (this halves the number of function arguments related to coordinates).
*/
typedef struct {
	int y, x;
} pt;


static double seconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

/* Update encoder and motion estimation statistics for current output frame */
static void do_stats(const ppp_frame *frame, const ppp_motion *motions,
int n_blocks) {
	encoder_stats(frame->data, frame->length);
	motion_stats(motions, n_blocks);
}

/* Restrict a 16-bit pixel value to the range -128..127 */
inline static int8_t clamp_pixel_value(int16_t val) {
	return val < -128 ? -128 : val > 127 ? 127 : val;
}

/*
* Compute the pixel-wise difference between a macro block
* given by 'block' and the 8x8 rectangle at 'd' in 'old'
* (note that old is stored row-wise, i.e., not in macro-block order).
* 'columns' is the number of columns of image 'old'.
* Return differences in 'deltas'.
*/
static void compute_deltas(const int16_t block[64], const int8_t *old,
int columns, pt p, int16_t deltas[64]) {
	for (int y=0; y<8; y++) {
		for (int x=0; x<8; x++) {
			int i = 8*y+x;
			deltas[i] = block[i] - old[(p.y+y)*columns + p.x+x];
		}
	}
}

/*
* Evaluate the pixel-wise difference between macro block 'block'
* and the 8x8 rectangle at 'd' in image 'old'
* (see also compute_deltas()).
* Lower return values indicate better prediction.
*/
static int evaluate_deltas(const int16_t block[64], const int8_t *old,
int columns, pt p) {
	int s = 0;

	for (int y=0; y<8; y++) {
		for (int x=0; x<8; x++) {
			s += abs(block[8*y+x] - old[(p.y+y)*columns + p.x+x]);
		}
	}

	return s;
}

/*
* Offsets to use in diamond search.
*/
static const pt diamond_offsets[8] = { { -1, -1 }, { -2,  0 }, { -1,  1 },
	{  0, -2 },             {  0,  2 },
	{  1, -1 }, {  2,  0 }, {  1,  1 } };

/*
* Offsets in the last step of diamond search (when distance
* has reached 1).
*/
static const pt vh_offsets[4] = { { -1,  0 }, {  0, -1 },
	{  0,  1 }, {  1,  0 } };

/*
* Evaluate the differences between 'block' and the
* 'n_offs' 8x8 regions in 'old' around 'center' with
* offsets found in 'offs'. Offsets are multiplied by 'dist'.
* 'center' is set to the point with the lowest value for
* evaluate_deltas(). 'min_err' contains the minimum of the
* evaluation function so far and is updated if a lower
* value is found.
* (If two or more offsets yield the same minimal evaluation,
* any of them can be chosen.)
*/
static void eval_offsets(const pt offs[], int n_offs,
const int16_t block[64], const int8_t *old,
int rows, int columns, int dist,
int *min_err, pt *center) {
	pt curr = *center;
	for (int o=0; o<n_offs; o++) {
		/* Point 'p' is the next point (top-left corner of an 8x8 region)
		* to evaluate as predictor.
		*/
		pt p;
		p.y = center->y + offs[o].y * dist;
		p.x = center->x + offs[o].x * dist;

		/* Skip the current offset if it points beyond the borders */
		if (p.y < 0 || p.x < 0 || p.y > rows-8 || p.x > columns-8)
		continue;

		/* Compute the "error" between the current and the
		* candidate region by calling the evaluation function.
		* Lower "error" values are better.
		*/
		int err = evaluate_deltas(block, old, columns, p);
		if (err < *min_err) {
			curr = p;
			*min_err = err;
		}
	}
	*center = curr;
}

/*
* Search for a good motion vector to predict the current macro block
* 'block' from the previous frame 'old' using diamond search.
* 'max_dist' gives the initial distance to use in diamond search.
* 'center' has to be initialized with the center for the initial
* search. The final location is also returned in 'center'.  The
* results of quantized DCT and compression are returned in
* 'deltas_qdct' and 'deltas_compr', respectively.  Returns the number
* of bytes used in deltas_compr.
*/
static int diamond_search(const int16_t block[64],
const int8_t *old,
int rows, int columns, int max_dist,
int16_t deltas_qdct[64], uint8_t deltas_compr[96],
pt *center) {
	int16_t deltas[64];
	int min_err;

	if (max_dist > 0) {
	
		/* Initialise the "error" to the value of the
		* evaluation function at the center.
		*/
		min_err = evaluate_deltas(block, old, columns, *center);
		
		/* Perform diamond search. eval_offsets() updates 'center'
		* and 'min_err'.
		*/
		for (int dist=max_dist; dist>=1; dist=dist/2) {
			eval_offsets(diamond_offsets, 8, block, old,
			rows, columns, dist, 
			&min_err, center);
		}
		
		/* Finally, evaluate the location horizontally and
		* vertically neighbouring the center.
		*/
		eval_offsets(vh_offsets, 4, block, old,
		rows, columns, 1, 
		&min_err, center);
	}

	/*
	* Compute deltas, their DCT and the compression result
	* for the just found candidate.
	*/
	compute_deltas(block, old, columns, *center, deltas);
	qdct_block(deltas, deltas_qdct);
	return compress_block(deltas_qdct, deltas_compr);
}

/*
* Encode the current frame given by 'image'. The pixels in 'image'
* are stored row-wise (_not_ in macroblock order) and are unsigned,
* i.e., the values are in the range 0..255.
* 'old_image' points to the reference frame (the previous frame);
* it is also stored row-wise (_not_ in macroblock order) but
* its pixel values are signed gray values in the range -128..127.
* 'frame' takes the output (the encoded blocks) and 'motions'
* takes, for each block, the the motion information (i.e., whether
* a block is intra-coded or predicted and when predicted, the value
* of the motion vector).
* 'motion_search' specifies which motion search to perform.
* The pixels in 'image' are updated to contain the reconstructed
* version of the current frame, i.e., 'image' will be what the
* decoder sees as the current frame. The returned pixels in 'image'
* are _signed_ (range -128..127), so 'image' can directly be used
* as 'old_image' in the next call to encode_video_frame().
*/
static void encode_video_frame(uint8_t *image, const int8_t *old_image,
const ppp_image_info *info,
ppp_frame *frame, ppp_motion *motions,
enum prediction motion_search) {
	const int columns  = info->columns;
	const int rows     = info->rows;
	const int n_blocks = rows*columns/64;
	uint8_t *output = frame->data;

	uint8_t intra_compr[96], delta_compr[96];

	/* Set motion information for all blocks to "intra-coded".
	* When we decide to use prediction below, we replace
	* motions[i] with the code for the corresponding motion.
	*/
	for (int i=0; i<n_blocks; i++)
	motions[i] = PPP_MOTION_INTRA;

	/* When asked for uncompressed blocks or DCT, we simply call
	* encode_frame() to do the work.
	*/
	if (info->format == PPP_IMGFMT_UNCOMPRESSED_BLOCKS ||
			info->format == PPP_IMGFMT_UNCOMPRESSED_DCT) {
		encode_frame(image, info->rows, info->columns, info->format, frame);
		return;
	}

	/* 'ptr' points to the location where to output the bytes
	* for the next macro block.
	*/
	uint8_t *ptr = output;
	for (int block_nr=0; block_nr<n_blocks; block_nr++) {
		int16_t block[64];
		int yy, xx, max_dist;
		int16_t intra_qdct[64], delta_qdct[64];
		int intra_len, delta_len;

		/* Get the coordinates of the top-left corner of the block in (xx,yy) */
		block_to_row_col(block_nr, 0, columns, &yy, &xx);

		/*
		* Put macro block number 'b' into 'block' and
		* compute its DCT and compressed representation.
		* The length of the compressed block is stored in 'intra_len'.
		*/
		uint8_image_get_block(image, columns, block_nr, block);
		qdct_block(block, intra_qdct);
		intra_len = compress_block(intra_qdct, intra_compr);

		/*
		* Stores the center for the diamond search and
		* (after search has finished) the absolute coordinates
		* of the (top-left corner of the) 8x8 region used to
		* predict the current block.
		*/
		pt center;

		/* When diamond search is enabled, we start with a
		* search distance of 32 because in the horizontal and vertical
		* directions the multiplier is 2, i.e., we look 64 pixels aside.
		* This means that the maximum distance at the end of the search
		* can be 64+32+...+1 = 127.
		*/
		switch (motion_search) {
		case MS_DIAMOND: max_dist = 32; break;
		default: max_dist = 0;
		}

		if (motion_search != MS_NONE) {
			/* Perform motion search. We start the search at the
			* (top-left corner of the) current block and
			* get the (absolute) location of the best predictor
			* in 'center', the DCT and compression results of
			* the differenecs in 'delta_qdct' and 'delta_compr'
			* and the length of the compressed deltas in 'delta_len'.
			*/
			center.y = yy;
			center.x = xx;
			delta_len = diamond_search(block, old_image,
			rows, columns, max_dist,
			delta_qdct, delta_compr,
			&center);
		} else
		delta_len = intra_len;

		/* Prefer intra coding when lengths are equal (or there
		* is no motion estimation). Copy the respective compressed
		* result to the output frame (at 'ptr'), decode the
		* block again and put the decoded result in 'image'
		* (at 'current', which points to the top-left corner of
		* the current block in image). This changes the
		* pixels in 'image' to signed values (-128..127) and,
		* therefore, we declare 'current' as a pointer to int8_t.
		*/
		int8_t *current = (int8_t *)&image[yy*columns + xx];
		if (intra_len <= delta_len) {
			/* Copy intra-coded block to the output */
			memcpy(ptr, intra_compr, intra_len);
			ptr += intra_len;

			/* Decode compressed intra-coded block and replace the pixels
			* in 'image' with the result of the decoding.
			*/
			iqdct_block(intra_qdct, block);
			for (int y=0; y<8; y++) {
				for (int x=0; x<8; x++) {
					int16_t newpixel = block[8*y+x];
					current[y*columns + x] = clamp_pixel_value(newpixel);
				}
			}
		} else {
			/* Copy the compressed predicted block to the output */
			memcpy(ptr, delta_compr, delta_len);
			ptr += delta_len;

			/* Decode compressed predicted block and replace the
			* pixels in 'image' with the result of the decoding plus
			* the value of the corresponding pixels in 'old_image'.
			*/
			iqdct_block(delta_qdct, block);
			for (int y=0; y<8; y++) {
				for (int x=0; x<8; x++) {
					int16_t newpixel = block[8*y+x];
					newpixel += old_image[(center.y+y)*columns + center.x+x];
					current[y*columns + x] = clamp_pixel_value(newpixel);
				}
			}

			/* Set motion information for current block (i.e., mark
			* the block as predicted instead of intra-coded.
			* The motion vector is given in relative coordinates
			* (as offset relative to the current block).
			*/
			motions[block_nr] = motion_code(center.y-yy, center.x-xx);
		}
	}

	/* Determine the total length of the encoded frame */
	frame->length = ptr - output;
}

/*
* Encode a video stream (read from 'video') according to 'opts'
* and store the result in file stream 'f'. 'info' holds the
* image properties, e.g., image format.
*/
static int encode_video(video *video, FILE *f, const ppp_image_info *info,
const options *opts) {
	const int rows    = info->rows;
	const int columns = info->columns;
	const int n_blocks = (rows/8)*(columns/8);

	/* Number of frames to skip in the input */
	const int frames_to_skip = opts->frames_to_skip;

	/* Number of frames to encode */
	const int max_frames     = opts->max_frames;

	ppp_frame *frame, *old_frame, *enc_frame;
	ppp_motion motions[n_blocks];

	const size_t max_enc_bytes = max_encoded_length(64)*n_blocks;

	encoder_stats_init();

	frame = video_alloc_frame(video);
	if (frame == NULL) {
		fprintf(stderr, "could not allocate frame buffer\n");
		return -1;
	}

	old_frame = video_alloc_frame(video);
	if (old_frame == NULL) {
		fprintf(stderr, "could not allocate frame buffer for old frame\n");
		free(frame);
		return -1;
	}

	enc_frame = ppp_frame_alloc(max_enc_bytes);
	if (enc_frame == NULL) {
		fprintf(stderr, "could not allocate buffer for encoded frame\n");
		free(old_frame);
		free(frame);
		return -1;
	}

	/* Sequentially walk through the frames of the input video. */
	int frame_nr = 0;
	double t = seconds();
	while (frame_nr < max_frames && video_get_next_frame(video, frame) == 0) {
		if (frame_nr >= frames_to_skip) {
			/* When we are at the beginning of a GOP, we always set the
			* motion seach method to MS_NONE to force an intra-coded
			* frame.
			*/
			enum prediction motion_search =
			(frame_nr-frames_to_skip)%64==0 ? MS_NONE : opts->motion_search;

			/* encode_video_frame() take the current frame (frame->data)
			* as unsigned gray values (in range 0..255) as input. It
			* expects the previous frame (old_frame->data) to be
			* in signed values (range -128..127) and updates
			* the current frame (frame->data) to contain signed values.
			* Therefore, we need to cast old_frame->data to a pointer to
			* int8_t.
			*/
			encode_video_frame(frame->data, (const int8_t *)old_frame->data,
			info, enc_frame, motions, motion_search);

			/* Swap 'frame' and 'old_frame' */
			ppp_frame *tmp = frame;
			frame = old_frame;
			old_frame = tmp;

			/* Write 'enc_frame' with motion vectors 'motions' to
			* output file (and do statistics if requested).
			*/
			ppp_video_frame_write(f, n_blocks, enc_frame, motions);
			if (opts->show_stats)
			do_stats(enc_frame, motions, n_blocks);
		}
		printf("frame %d\n", frame_nr);
		fflush(stdout);
		frame_nr++;
	}
	t = seconds() - t;
	printf("\nDuration: %.3f ms\n", t*1000);
	free(enc_frame);
	free(old_frame);
	free(frame);
	return 0;
}

/*
* Call OpenCL version of encode_video().
*/
static int encode_video_cl(video *video, FILE *f, const ppp_image_info *info,
const options *opts, bool use_gpu) {
	const int rows    = info->rows;
	const int columns = info->columns;
	const int n_blocks = (rows*columns)/64;

	/* Number of frames to skip in the input */
	const int frames_to_skip = opts->frames_to_skip;

	/* Number of frames to encode */
	const int max_frames = opts->max_frames;

	ppp_frame *frame, *enc_frame;
	ppp_motion motions[n_blocks];

	const cl_uint format = info->format;
	const bool compr     = format == PPP_IMGFMT_COMPRESSED_DCT;
	const size_t max_enc_bytes = max_encoded_length(64)*n_blocks;
	
	if (!compr) {
		fprintf(stderr, "Compression must be enabled for OpenCL implementation\n");
		return -1;
	}

	encoder_stats_init();

	frame = video_alloc_frame(video);
	if (frame == NULL) {
		fprintf(stderr, "could not allocate frame buffer\n");
		return -1;
	}

	enc_frame = ppp_frame_alloc(max_enc_bytes + n_blocks);
	if (enc_frame == NULL) {
		fprintf(stderr, "could not allocate buffer for encoded frame\n");
		free(frame);
		return -1;
	}
	
	cl_event startEvent, endEvent, imageWriteEvent;
	cl_event *event_wait_list = (cl_event *) malloc(sizeof(cl_event));
	
	if (event_wait_list == NULL) {
		fprintf(stderr, "could not allocate buffer for cl_waitlist\n");
		free(frame);
		free(enc_frame);
		return -1;
	}
	
	/* Fast forward to first frame if available */
	bool frames_available = true;
	int frame_nr;
	frames_available &= video_get_next_frame(video, frame) == 0;
	
	for (frame_nr = 0; frame_nr < frames_to_skip; frame_nr++)
		frames_available &= video_get_next_frame(video, frame) == 0;
		
	/* Return if there are no more frames available or the maximum number of frames to be encoded is already reached */
	if (!frames_available || frame_nr >= max_frames) {
		printf("No frames have been encoded.\n");
		free(frame);
		free(enc_frame);
		return 0;
	}

	cl_context context;
	cl_device_id devid;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernelEncode;
	cl_mem imageGPU, old_imageGPU, newImageGPU, frameGPU;
	cl_mem motionsGPU;
	cl_int res;

	/* Set the work group size and global number of work items.
	*/
	size_t work_dims = 3;
	size_t global_work_size[] = {columns, rows, 8};
	size_t local_work_size[] = {8, 8, 8};

	context = create_cl_context(use_gpu, &devid);
	
	/* Create a command queue (which allows timing) */
	queue = clCreateCommandQueue(context, devid, CL_QUEUE_PROFILING_ENABLE, &res);
	if (queue == NULL)
		error_and_abort("Could not create command queue", res);
	
	/* Allocate the buffer memory object for the input image.
	* We request that the image (from the host) is copied to the device.
	*/
	imageGPU = clCreateBuffer(context, CL_MEM_READ_WRITE,
	sizeof(uint8_t)*rows*columns, NULL, &res);
	if (res != CL_SUCCESS)
		error_and_abort("Could not allocate imageGPU", res);
		
	newImageGPU = clCreateBuffer(context, CL_MEM_READ_WRITE,
	sizeof(uint8_t)*rows*columns, NULL, &res);
	if (res != CL_SUCCESS)
		error_and_abort("Could not allocate newImageGPU", res);

	old_imageGPU = clCreateBuffer(context, CL_MEM_READ_WRITE,
	sizeof(int8_t)*rows*columns, NULL, &res);
	if (res != CL_SUCCESS)
		error_and_abort("Could not allocate old_imageGPU", res);

	/* Allocate the buffer memory object for the result.
	* We need at most 'max_enc_bytes' to represent the result.
	*/
	frameGPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
	max_enc_bytes + n_blocks, NULL, &res);
	if (res != CL_SUCCESS)
		error_and_abort("Could not allocate frameGPU", res);

	motionsGPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
	n_blocks*sizeof(ppp_motion), NULL, &res);
	if (res != CL_SUCCESS)
		error_and_abort("Could not allocate motionsGPU", res);

	/*
	* Load the OpenCL program from file "video_encoder_kernels.cl"
	* and compile it.
	*/
	program = build_program("video_encoder_kernels.cl", context, devid);
	
	/* Find kernel "encode_video_frame" in the compiled program. */
	kernelEncode = clCreateKernel(program, "encode_video_frame", &res);
	if (res != CL_SUCCESS)
		error_and_abort("Could not create kernel 'encode_video_frame'", res);

	/* Set the arguments for the kernel invocation.
	*/
	clSetKernelArg(kernelEncode, 3, sizeof(cl_int), &rows);
	clSetKernelArg(kernelEncode, 4, sizeof(cl_int), &columns);
	clSetKernelArg(kernelEncode, 5, sizeof(cl_int), &format);
	clSetKernelArg(kernelEncode, 6, sizeof(cl_mem), &frameGPU);
	clSetKernelArg(kernelEncode, 7, sizeof(cl_mem), &motionsGPU);

	clEnqueueMarker(queue, &startEvent);
	
	/* Load first frame onto the device using blocking writes to buffers. */
	res = clEnqueueWriteBuffer(queue, imageGPU, CL_TRUE, 0, frame->length, frame->data, 0, NULL, &imageWriteEvent);
	if (res != CL_SUCCESS)
		error_and_abort("Could not enqueue image buffer write", res);
		
	/* Init cl_wait_list */
	event_wait_list[0] = imageWriteEvent;

	/* Sequentially walk through the remaining frames of the input video. */
	while (frame_nr < max_frames && frames_available) {
		cl_int motion_search = (frame_nr-frames_to_skip)%64==0 ? MS_NONE : opts->motion_search;
		
		clSetKernelArg(kernelEncode, 0, sizeof(cl_mem), &imageGPU);
		clSetKernelArg(kernelEncode, 1, sizeof(cl_mem), &old_imageGPU);
		clSetKernelArg(kernelEncode, 2, sizeof(cl_int), &motion_search);

		/* Enqueue kernel, wait for buffer write events to finish */
		res = clEnqueueNDRangeKernel(queue, kernelEncode, work_dims, NULL, global_work_size, local_work_size, 1, event_wait_list, NULL);	
		if (res != CL_SUCCESS)
			error_and_abort("Could not enqueue kernel invocation", res);
		
		/* Non-blocking writes to buffer */
		frames_available &= video_get_next_frame(video, frame) == 0;
		
		if (frames_available) {
			res = clEnqueueWriteBuffer(queue, newImageGPU, CL_FALSE, 0, frame->length, frame->data, 0, NULL, &imageWriteEvent);
			if (res != CL_SUCCESS)
				error_and_abort("Could not enqueue image buffer write", res);
				
			/* Swap buffer pointers */
			cl_mem tmp = old_imageGPU;
			old_imageGPU = imageGPU;
			imageGPU = newImageGPU;
			newImageGPU = tmp;
		}
		
		/* Copy the result from the device to the host. */
		res = clEnqueueReadBuffer(queue, frameGPU, CL_TRUE, 0, max_enc_bytes + n_blocks, enc_frame->data, 0, NULL, NULL);

		if (res != CL_SUCCESS)
			error_and_abort("Could not enqueue buffer read", res);
		
		/* Compress encoded values by using the encoding index */
		enc_frame->length = 0;
		
        for(int i = 0; i < n_blocks; i++) {
			int length = (int) enc_frame->data[(i+1)*96 + i];
			memcpy(&(enc_frame->data[enc_frame->length]), &(enc_frame->data[i*97]), length);
			enc_frame->length += length;
        }
		
		printf("Size: %u\n", (unsigned int)enc_frame->length);

		res = clEnqueueReadBuffer(queue, motionsGPU, CL_TRUE, 0, n_blocks*sizeof(ppp_motion), motions, 0, NULL, NULL);
		if (res != CL_SUCCESS)
			error_and_abort("Could not enqueue buffer read", res);

		ppp_video_frame_write(f, n_blocks, enc_frame, motions);
		
		if (opts->show_stats)
			do_stats(enc_frame, motions, n_blocks);            
		
		printf("frame %d\n", frame_nr);
		fflush(stdout);
		frame_nr++;
	}

	clEnqueueMarker(queue, &endEvent);

	res = clFinish(queue);
	
	if (res != CL_SUCCESS)
		error_and_abort("Error at clFinish", res);

	cl_ulong nanos = get_event_start_nanos(endEvent) -
	get_event_end_nanos(startEvent);
	printf("\nDuration: %.3f ms\n\n", nanos/1.0e6);
	
	clReleaseMemObject(imageGPU);
	clReleaseMemObject(old_imageGPU);
	clReleaseMemObject(newImageGPU);
	clReleaseMemObject(frameGPU);
	clReleaseMemObject(motionsGPU);
	clReleaseKernel(kernelEncode);
	clReleaseContext(context);

	return 0;
}

static void usage(const char *progname) {
	fprintf(stderr, "USAGE: %s -i IN -o OUT [-d | -c] [-p METHOD] "
	"[-n FRAMES] [-b SKIP] [-h]\n", progname);
	fprintf(stderr,
	"  -d: with DCT\n"
	"  -c: with DCT and compression\n"
	"  -n: number of frames to encode\n"
	"  -b: number of frames to skip at beginning of video\n"
	"  -p: select motion estimation method (requires -c if not 'none')\n"
	"        none: no motion search (only intra-coded blocks)\n"
	"        zero: try prediction from same location only\n"
	"        full: use diamond search to find prediction location\n"
	"  -z: select implementation:\n"
	"        seq: CPU sequential (default)\n"
	"        cpu: CPU via OpenCL\n"
	"        gpu: GPU via OpenCL\n"
	"  -h: show this help\n"
	"\n"
	);
}

int main(int argc, char *argv[]) {
	enum ppp_image_format format;
	video *video;
	int option;

	char *infile, *outfile;
	bool dct, compress;
	options opts;
	enum impl_type implementation;

	init_qdct();
	encoder_stats_init();
	motion_stats_init();

	infile = NULL;
	outfile = NULL;
	dct = false;
	compress = false;
	opts.show_stats = false;
	opts.frames_to_skip = 0;
	opts.max_frames = INT_MAX;
	opts.motion_search = MS_NONE;
	implementation = IMPL_SEQ;
	while ((option = getopt(argc,argv,"b:cdhi:n:o:p:sz:")) != -1) {
		switch(option) {
		case 'd': dct = true; break;
		case 'c': compress = true; break;
		case 'b': opts.frames_to_skip = atoi(optarg); break;
		case 'n': opts.max_frames = atoi(optarg); break;
		case 's': opts.show_stats = true; break;
		case 'i': infile  = optarg; break;
		case 'o': outfile = optarg; break;
		case 'p': 
			if (strcmp(optarg, "none") == 0) {
				opts.motion_search = MS_NONE;
				break;
			} else if (strcmp(optarg, "zero") == 0) {
				opts.motion_search = MS_00;
				break;
			} else if (strcmp(optarg, "full") == 0) {
				opts.motion_search = MS_DIAMOND;
				break;
			}
			usage(argv[0]);
			return 1;
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
	
	if (infile == NULL || outfile == NULL || (dct && compress) ||
			(opts.motion_search != MS_NONE && !compress) ||
			opts.frames_to_skip < 0 || opts.max_frames < 0) {
		usage(argv[0]);
		return 1;
	}

	if (dct)
	format = PPP_IMGFMT_UNCOMPRESSED_DCT;
	else if (compress)
	format = PPP_IMGFMT_COMPRESSED_DCT;
	else
	format = PPP_IMGFMT_UNCOMPRESSED_BLOCKS;
	
	/* Open input video for reading (using FFmpeg) */
	video = video_open(infile);
	if (video != NULL) {
		ppp_image_info img_info;
		ppp_video_info vid_info;
		img_info.rows    = video_get_height(video);
		img_info.columns = video_get_width(video);
		img_info.format  = format;
		vid_info.fps     = lrintf(video_get_fps(video)*1000);
		if (img_info.rows%8 == 0 && img_info.columns%8 == 0) {
			FILE *f = ppp_video_write(outfile, &img_info, &vid_info);
			if (f != NULL) {
				if (implementation == IMPL_SEQ) 
				encode_video(video, f, &img_info, &opts);
				else
				encode_video_cl(video, f, &img_info, &opts,
				implementation == IMPL_GPU);
			} else
			fprintf(stderr, "could not open output file\n");
		} else
		fprintf(stderr, "width and height must be multiples of 8\n");
		video_close(video);
	} else
	fprintf(stderr, "could not open video\n");

	if (opts.show_stats) {
		encoder_stats_print(stderr);
		motion_stats_print(stderr);
	}

	return 0;
}

