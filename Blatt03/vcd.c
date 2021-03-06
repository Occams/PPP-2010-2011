#include <vcd.h>

static int vcd_mpi_self = 0, vcd_mpi_processors = 1;
int vcd_stop = false;

static double delta_edge(double *i, int x, int y, int rows, int cols);
static double delta(double *i, int x, int y, int rows, int cols);
static double s(double *img, int x, int y, int rows, int cols);
static void intToDoubleArray(int *src, double *dest, int length);
static void doubleToIntArray(double *src, int *dest, int length, int maxcolor);
static void intToDoubleArray_parallel(int *src, double *dest, int length);
static void doubleToIntArray_parallel(double *src, int *dest, int length, int maxcolor);

void vcd_parallel(int *image, int rows, int columns, int maxcolor) {
	MPI_Request send_top1, send_bottom1, recv_top1, recv_bottom1;
	MPI_Request send_top2, send_bottom2, recv_top2, recv_bottom2;
	MPI_Status st1, st2, st3, st4;
	int i,x,y, rows_l = vcd_mpi_self < vcd_mpi_processors-1 ? rows-1 : rows, idx, length = rows*columns;
	double img1[length], img2[length], *tmp, *img1_p = img1, *img2_p = img2, d;
	bool edge;
	
	/* Init MPI calls */
	if (vcd_mpi_self > 0) {
		MPI_Send_init(img1_p+columns, columns, MPI_DOUBLE, vcd_mpi_self - 1, 0, MPI_COMM_WORLD, &send_top1);
		MPI_Recv_init(img1_p, columns, MPI_DOUBLE, vcd_mpi_self - 1, 1, MPI_COMM_WORLD, &recv_bottom1);
		
		MPI_Send_init(img2_p+columns, columns, MPI_DOUBLE, vcd_mpi_self - 1, 0, MPI_COMM_WORLD, &send_top2);
		MPI_Recv_init(img2_p, columns, MPI_DOUBLE, vcd_mpi_self - 1, 1, MPI_COMM_WORLD, &recv_bottom2);
	}
	
	if (vcd_mpi_self < vcd_mpi_processors - 1) {
		MPI_Send_init(img1_p+(rows-2)*columns, columns, MPI_DOUBLE, vcd_mpi_self + 1, 1, MPI_COMM_WORLD, &send_bottom1);
		MPI_Recv_init(img1_p+(rows-1)*columns, columns, MPI_DOUBLE, vcd_mpi_self + 1, 0, MPI_COMM_WORLD, &recv_top1);
		
		MPI_Send_init(img2_p+(rows-2)*columns, columns, MPI_DOUBLE, vcd_mpi_self + 1, 1, MPI_COMM_WORLD, &send_bottom2);
		MPI_Recv_init(img2_p+(rows-1)*columns, columns, MPI_DOUBLE, vcd_mpi_self + 1, 0, MPI_COMM_WORLD, &recv_top2);
	}
	
	intToDoubleArray_parallel(image, img1_p, length);
	
	for (i = 0; i < N && !vcd_stop; i++) {
		vcd_stop = true;
		
		#pragma omp parallel for private (y,idx,edge,d) reduction (&& : vcd_stop)
		for (x = 0; x < rows_l; x++) {
			for (y = 0; y < columns; y++) {
				idx = x*columns+y;
				edge = x == 0 || y == 0 || x+1 == rows || y+1 == columns;
				d = edge ? delta_edge(img1_p, x, y, rows, columns) : delta(img1_p, x, y, rows, columns);
				vcd_stop = vcd_stop && (edge || ABS(d) <= EPSILON);
				img2_p[idx] = img1_p[idx] + KAPPA * DELTA_T * d;
			}
		}
		
		/* Switch array pointers */
		tmp = img1_p;
		img1_p = img2_p;
		img2_p = tmp;

		
		/* Share overlapping at the top */
		if(vcd_mpi_self > 0) {
			if(img1_p == img1) {
				MPI_Start(&send_top1);
			} else {
				MPI_Start(&send_top2);
			}
		}
		
		if (vcd_mpi_self < vcd_mpi_processors-1) {
			if(img1_p == img1) {
				MPI_Start(&recv_top1);
				MPI_Wait(&recv_top1, &st1);
			} else {
				MPI_Start(&recv_top2);
				MPI_Wait(&recv_top2, &st1);
			}
		}
		
		if(vcd_mpi_self > 0) {
			if(img1_p == img1) {
				MPI_Wait(&send_top1, &st2);
			} else {
				MPI_Wait(&send_top2, &st2);
			}
		}
		
		
		/* Share overlapping at the bottom */
		if(vcd_mpi_self < vcd_mpi_processors-1) {
			if(img1_p == img1) {
				MPI_Start(&send_bottom1);
			} else {
				MPI_Start(&send_bottom2);
			}
		}
		
		if(vcd_mpi_self > 0) {
			if(img1_p == img1) {
				MPI_Start(&recv_bottom1);
				MPI_Wait(&recv_bottom1, &st3);
			} else {
				MPI_Start(&recv_bottom2);
				MPI_Wait(&recv_bottom2, &st3);
			}
		}
		
		if(vcd_mpi_self < vcd_mpi_processors-1) {
			if(img1_p == img1) {
				MPI_Wait(&send_bottom1, &st4);
			} else {
				MPI_Wait(&send_bottom2, &st4);
			}
		}
		
		
		/* Update stop condition */
		MPI_Allreduce(&vcd_stop, &vcd_stop, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	}
	
	doubleToIntArray_parallel(img1_p, image, length, maxcolor);
	
	if (vcd_mpi_self > 0) {
		MPI_Request_free(&send_top1);
		MPI_Request_free(&recv_bottom1);
		MPI_Request_free(&send_top2);
		MPI_Request_free(&recv_bottom2);
	}
	
	if(vcd_mpi_self < vcd_mpi_processors-1) {
		MPI_Request_free(&send_bottom2);
		MPI_Request_free(&recv_top2);
		MPI_Request_free(&send_bottom1);
		MPI_Request_free(&recv_top1);
	}
}

void vcd_sequential(int *image, int rows, int columns, int maxcolor) {
	int i, x, y, idx, length = rows*columns;
	double img1[length], img2[length], *tmp, *img1_p = img1, *img2_p = img2, d;
	bool edge;
	
	intToDoubleArray(image, img1_p, length);
	
	for (i = 0; i < N && !vcd_stop; i++) {
		vcd_stop = true;
		
		for (x = 0; x < rows; x++) {
			for (y = 0; y < columns; y++) {
				idx = x*columns+y;
				edge = x == 0 || y == 0 || x+1 == rows || y+1 == columns;
				d = edge ? delta_edge(img1_p, x, y, rows, columns) : delta(img1_p, x, y, rows, columns);
				vcd_stop = vcd_stop && (edge || ABS(d) <= EPSILON);	
				img2_p[idx] = img1_p[idx] + KAPPA * DELTA_T * d;
			}
		}
		
		tmp = img1_p;
		img1_p = img2_p;
		img2_p = tmp;
	}
	
	doubleToIntArray(img1_p, image, length, maxcolor);
}

static void intToDoubleArray(int *src, double *dest, int length) {
	int x;
	
	for (x = 0; x < length; x++) {
		dest[x] = (int) src[x];
	}
}

static void intToDoubleArray_parallel(int *src, double *dest, int length) {
	int x;
	
	#pragma omp parallel for
	for (x = 0; x < length; x++) {
		dest[x] = src[x];
	}
}

static void doubleToIntArray(double *src, int *dest, int length, int maxcolor) {
	int x;
	
	for (x = 0; x < length; x++) {
		dest[x] = src[x] < 0 ? 0 : (int) MIN(src[x],maxcolor); 
	}
}

static void doubleToIntArray_parallel(double *src, int *dest, int length, int maxcolor) {
	int x;
	
	#pragma omp parallel for
	for (x = 0; x < length; x++) {
		dest[x] = src[x] < 0 ? 0 : (int) MIN(src[x], maxcolor); 
	}
}

static double delta(double *i, int x, int y, int rows, int cols) {
	double center = i[x*cols + y];
	
	return PHI(i[(x+1)*cols + y] - center) 
	- PHI(center - i[(x-1)*cols + y])
	+ PHI(i[x*cols + y + 1] - center)
	- PHI(center - i[x*cols + y - 1])
	+ XI(i[(x+1)*cols + y + 1] - center)
	- XI(center - i[(x-1)*cols + y - 1])
	+ XI(i[(x-1)*cols + y + 1] - center)
	- XI(center - i[(x+1)*cols + y - 1]);

}

static double delta_edge(double *i, int x, int y, int rows, int cols) {
	double center = i[x*cols + y];
	
	return PHI(s(i,x+1,y,rows,cols) - center) 
	- PHI(center - s(i,x-1,y,rows,cols))
	+ PHI(s(i,x,y+1,rows,cols) - center)
	- PHI(center - s(i,x,y-1,rows,cols))
	+ XI(s(i,x+1,y+1,rows,cols) - center)
	- XI(center - s(i,x-1,y-1,rows,cols))
	+ XI(s(i,x-1,y+1,rows,cols) - center)
	- XI(center - s(i,x+1,y-1,rows,cols));
}

static double s(double *img, int x, int y, int rows, int cols) {
	return x < 0 || x >= rows || y < 0 || y >= cols ? 0.0 : img[x*cols+y];
}

void vcd_mpi_init(int mpi_self, int mpi_processors) {
	vcd_mpi_self = mpi_self;
	vcd_mpi_processors = mpi_processors;
}


int *vcd_mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length) {
	pgm_part info;
	pgm_partinfo(rows, vcd_mpi_self, &info);
	
	*offset = info.offset*columns;
	*length = info.rows*columns;
	
	return (int*)malloc(info.rows*columns*sizeof(int));
}

