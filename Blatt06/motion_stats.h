#ifndef _MOTION_STATS_H
#define _MOTION_STATS_H

#include <stdio.h>

#include "ppp_image.h"
#include "ppp_video.h"

void motion_stats_init(void);
void motion_stats(const ppp_motion *motions, int n_blocks);
void motion_stats_print(FILE *f);

#endif
