#include <pgm_distribute.h>

static int pgm_procs;

void pgm_distribute_init(int processors) {
	pgm_procs = processors;
}


void pgm_partinfo(int rows, int proc, pgm_part *info) {
	/* Amount of rows */
	info->rows = (proc == 0 || proc == pgm_procs-1) ? rows/pgm_procs + 1 : rows/pgm_procs+2;
	if(proc == pgm_procs-1) info->rows += rows%pgm_procs;
	
	/* Offset */
	info->offset = (proc == 0) ? 0 : (rows/pgm_procs)*proc - 1;
	
	/* Overlapping */
	info->overlapping_top = (proc == 0) ? 0 : 1;
	info->overlapping_bot = (proc == pgm_procs - 1) ? 0 : 1;
}

void renormalize(double *i, int length, int maxcolor) {
	int x;
	double *img = i;
	
	for (x = 0; x < length; x++) {
		img[x] = MIN(img[x], maxcolor);
		img[x] = MAX(img[x], 0);
	}
}

void renormalize_parallel(double *i, int length, int maxcolor) {
	int x;
	double *img = i;
	
	#pragma omp parallel for
	for (x = 0; x < length; x++) {
		img[x] = MIN(img[x], maxcolor);
		img[x] = MAX(img[x], 0);
	}
}
