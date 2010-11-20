#ifndef SOBEL_H_INCLUDED
#define SOBEL_H_INCLUDED

#include <ppp_pnm.h>

void sobel_seq(int *image, int rows, int columns, int c, int maxcolor);

void sobel_mpi_init(int mpi_self, int mpi_processors);
int *sobel_mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length);
void sobel_parallel(int *image, int rows, int columns, int c, int maxcolor);

#endif
