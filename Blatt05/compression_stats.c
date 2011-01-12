#include <string.h>

#include "compression_stats.h"
#include "frame_encoding.h"

/*
 * Functions to collect statistics for compressed blocks.
 */

static int stats_zeros, stats_one, stats_two;
static int stats_small, stats_big;
static int stats_zeros_lengths[65], stats_endzeros_lengths[65];
static int stats_block_lengths[97];

void encoder_stats_init(void) {
    memset(stats_zeros_lengths,    0, sizeof(stats_zeros_lengths));
    memset(stats_endzeros_lengths, 0, sizeof(stats_endzeros_lengths));
    memset(stats_block_lengths,    0, sizeof(stats_block_lengths));
    stats_zeros = stats_one = stats_two = 0;
    stats_small = stats_big = 0;
}

void encoder_stats_print(FILE *f) {
    fprintf(f, "Encoder statistics:\n"
            "  Frequency of codes:\n"
            "    zeros: %d, ones: %d, twos: %d\n"
            "    4-bit codes: %d, 8-bit codes: %d\n",
            stats_zeros, stats_one, stats_two,
            stats_small, stats_big);
    fprintf(f, "  Number of 0s per block (starts with 1):\n");
    for (int i=1; i<=64; i++) {
        fprintf(f, " %7d", stats_zeros_lengths[i]);
        if (i%8 == 0)
            fprintf(f, "\n");
    }
    fprintf(f, "  Number of tail 0s per block (starts with 1):\n");
    for (int i=1; i<=64; i++) {
        fprintf(f, " %7d", stats_endzeros_lengths[i]);
        if (i%8 == 0)
            fprintf(f, "\n");
    }

    int total_length = 0, total_blocks = 0;
    fprintf(f, "  Lengths of encoded blocks (starts with 1):\n");
    for (int i=1; i<=96; i++) {
        fprintf(f, " %7d", stats_block_lengths[i]);
        if (i%8 == 0)
            fprintf(f, "\n");
        total_length += i*stats_block_lengths[i];
        total_blocks +=   stats_block_lengths[i];
    }

    double bytes_per_block = (double)total_length / total_blocks;
    fprintf(f, " Average encoded block length: %g bytes\n"
               " Compression ratio:  1 : %g\n",
            bytes_per_block, 64.0/bytes_per_block);
}

/*
 * Add _encoded_ data 'encoded' with length 'enc_length' to statistics.
 */
void encoder_stats(const uint8_t *encoded, int enc_length) {
    int16_t input[64];
    while (enc_length > 0) {
        int len = uncompress_block(encoded, input);
        encoded    += len;
        enc_length -= len;

        int zl = 0;
        for (int i=0; i<64; i++) {
            int8_t val = input[i] < 0 ? -input[i] : input[i];
            if (val == 0) {
                stats_zeros++;
                zl++;
            } else if (val == 1)
                stats_one++;
            else if (val == 2)
                stats_two++;
            else if (val <= 18)
                stats_small++;
            else
                stats_big++;
            
            if (val != 0) {
                stats_zeros_lengths[zl]++;
                zl = 0;
            }
        }
        stats_zeros_lengths[zl]++;
        
        for (int i=63; i>=-1; i--) {
            if (i == -1 || input[i] != 0) {
                stats_endzeros_lengths[63-i]++;
                break;
            }
        }
        
        stats_block_lengths[len]++;
    }
}
