#ifndef _PPP_IMAGE_H
#define _PPP_IMAGE_H

#include <stdint.h>

/*
 * Image format, i.e., how the image data for each
 * macro block is encoded. The image data is always
 * macro block by macro block and the blocks are
 * processed row-wise.
 *
 * UNCOMPRESSED_BLOCKS:
 *   The pixels of each macro block are stored
 *   row-wise. The value of each pixels is stored
 *   as-is as a signed 8-bit integer in the
 *   range -128..127.
 *
 * UNCOMPRESSED_DCT:
 *   Each macro block has been transformed with DCT,
 *   quantized and permuted. The resulting coefficients
 *   are stored row-wise as signed 8-bit integers.
 *
 * COMPRESSED_DCT:
 *   The coefficients after DCT and quantization
 *   are stored in the compressed format
 *   described in frame_encoding.c.
 *
 */
enum ppp_image_format {
    PPP_IMGFMT_UNCOMPRESSED_BLOCKS = 0,
    PPP_IMGFMT_UNCOMPRESSED_DCT    = 1,
    PPP_IMGFMT_COMPRESSED_DCT      = 2
};

/*
 * Image parameters.
 */
typedef struct _ppp_image_info {
    uint16_t rows;     /* number of rows in the image */
    uint16_t columns;  /* number of columns in the image */
    uint16_t format;   /* encoding format, see ppp_image_format */
} ppp_image_info;

/*
 * Header of the image file format.
 * The file starts with the string and is followed by
 * a ppp_image_info structure. The image data
 * (encoded as given by the 'format' field in 'image_info')
 * immediately follows the header.
 */
typedef struct _ppp_image_header {
    char ident[8];              /* "PPPIMAGE" */
    ppp_image_info image_info;
    uint8_t data[0];
} ppp_image_header;


/*
 * A frame, i.e., a (possibly encoded) image.
 * Field 'length' gives the number of bytes in the 'data'
 * area of the frame (i.e., the total number of bytes the encoded
 * macro blocks occupy). The field 'length' itself is not included
 * in the length.
 *
 * The 'data' field contains the data of the macro blocks
 * row-by-row, i.e., the macro block in row 0,
 * column 0 is stored first, then the macro block
 * in row 0, column 1 etc.
 */
typedef struct _ppp_frame {
    uint32_t length;       /* number of bytes in this frame */
    uint8_t data[0];       /* 'length' bytes of data */
} ppp_frame;


ppp_frame *ppp_frame_alloc(int databytes);

int ppp_image_write(const char *filename,
		    const ppp_image_info *info,
		    const ppp_frame *frame);
ppp_frame *ppp_image_read(const char *filename,
                          ppp_image_info *info);
ppp_frame *ppp_frame_read(FILE *f);


#endif
