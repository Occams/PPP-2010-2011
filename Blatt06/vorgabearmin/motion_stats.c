#include <stdio.h>
#include <stdlib.h>

#include "frame_encoding.h"
#include "motion_stats.h"

static int stats_intra, stats_predicted, stats_motion00;
static int64_t stats_motion_sum_x, stats_motion_sum_y;

void motion_stats_init(void) {
    stats_intra = stats_predicted = stats_motion00 = 0;
    stats_motion_sum_x = stats_motion_sum_y = 0;
}

static void motion_stats_motion(int deltaY, int deltaX) {
    stats_predicted++;
    if (deltaY == 0 && deltaX == 0)
        stats_motion00++;
    else {
        stats_motion_sum_x += llabs(deltaX);
        stats_motion_sum_y += llabs(deltaY);
    }
}

void motion_stats(const ppp_motion *motions, int n_blocks) {
    for (int i=0; i<n_blocks; i++) {
        if (is_intra(motions[i]))
            stats_intra++;
        else {
            int mY, mX;
            decode_motion_code(motions[i], &mY, &mX);
            motion_stats_motion(mY, mX);
        }
    }
}

void motion_stats_print(FILE *f) {
    int motion_non0 = stats_predicted - stats_motion00;

    fprintf(f, "Macro block statistics:\n"
            "  intra coded blocks:     %d\n"
            "  predicted   blocks:     %d\n"
            "  non-0 predicted blocks: %d\n"
            "  Average absolute non-0 motion vector: (dx,dy)=(%g,%g)\n\n",
            stats_intra, stats_predicted, motion_non0,
            (double)stats_motion_sum_x / motion_non0,
            (double)stats_motion_sum_y / motion_non0
        );
}
