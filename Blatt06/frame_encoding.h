#ifndef _FRAME_ENCODING_H
#define _FRAME_ENCODING_H

#include <stdbool.h>
#include <stdint.h>
#include "ppp_image.h"

void row_col_to_block(int y, int x, int columns, int *block, int *offset);
void block_to_row_col(int block, int offset, int columns, int *y, int *x);

void uint8_image_get_block(const uint8_t *image, int columns,
                           int block_nr, int16_t block[64]);

void init_qdct(void);
void qdct_block (const int16_t block[64], int16_t *output);
void iqdct_block(const int16_t *input   , int16_t block[64]);

int max_encoded_length(int n_values);
int compress_block  (const int16_t block[64], uint8_t *output);
int uncompress_block(const uint8_t *input, int16_t block[64], bool *dec_error);
int uncompress_data (const uint8_t *input, int max_input_len, int n_values,
                     int16_t *block, bool *dec_error);

int qdct_compress_block   (const int16_t block[64], uint8_t *output);
int uncompress_iqdct_block(const uint8_t *input, int16_t block[64], bool *dec_error);

void encode_frame(const uint8_t *image, int rows, int columns,
                  enum ppp_image_format format, ppp_frame *frame);

#endif
