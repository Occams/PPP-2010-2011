#ifndef _FRAME_ENCODING_H
#define _FRAME_ENCODING_H

#include <stdbool.h>
#include <stdint.h>
#include "ppp_image.h"

void row_col_to_block(int y, int x, int columns, int *block, int *offset);
void block_to_row_col(int block, int offset, int columns, int *y, int *x);

void uint8_image_to_blocks(const uint8_t *image, int rows, int columns, int16_t *blocks);

void init_qdct(void);
void qdct_block (const int16_t *block, int16_t *output);
void iqdct_block(const int16_t *input, int16_t *block);

int max_encoded_length(int n_values);
int compress_block  (const int16_t *block, uint8_t *output);
int uncompress_block(const uint8_t *input, int16_t *block, bool *dec_error);

int qdct_compress_block   (const int16_t *block, uint8_t *output);
int uncompress_iqdct_block(const uint8_t *input, int16_t *block, bool *dec_error);

void encode_frame(const int16_t *blocks, int rows, int columns,
                  enum ppp_image_format format, ppp_frame *frame);

#endif
