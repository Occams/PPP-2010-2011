#ifndef _COMPRESSION_STATS_H
#define _COMPRESSION_STATS_h

#include <stdint.h>
#include <stdio.h>

void encoder_stats_init(void);
void encoder_stats_print(FILE *f);
void encoder_stats(const uint8_t *encoded, int encoded_length);

#endif
