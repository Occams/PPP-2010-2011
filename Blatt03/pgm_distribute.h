#ifndef PGM_DISTRIBUTE_H_INCLUDED
#define PGM_DISTRIBUTE_H_INCLUDED
#include "common.h"
#include "omp.h"

typedef struct {
	int rows; /* Real amount of rows of this part */
	int offset; /* Offset of rows until this part starts */
	int overlapping_top; /* The amount of rows this part overlaps with the one above */
	int overlapping_bot; /* The amount of rows this part overlaps with the one below */
} pgm_part;

void pgm_distribute_init(int processors);
void pgm_partinfo(int rows, int proc, pgm_part *info);
void renormalize(double *i, int length, int maxcolor);
void renormalize_parallel(double *i, int length, int maxcolor);

#endif
