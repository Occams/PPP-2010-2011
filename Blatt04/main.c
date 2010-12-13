#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "omp.h"
#include <stdbool.h>
#include "ppp_pnm.h"
#include "common.h"
#include "nbody-readdat.h"
#include <math.h>
#include <stdarg.h>

#define MASTER 0

/* Globals */
int mpi_self = MASTER, mpi_processors;

typedef struct {
	long double x, y;
} vector;

typedef struct {
	long double max_x, max_y, offset;
	bool gen_img;
	int img_steps, width, heigth;
	char* img_prefix;
} imggen_info;

/* Prototypes */
int m_printf(char *format, ... );
void printhelp();
void printBodies(const body *bodies, int body_count);
double interactions(int body_count, int steps, double time);
void solve_sequential(body *bodies, int body_count, int steps, int delta, imggen_info info);
void solve_parallel(body *bodies, int body_count, int steps, int delta, imggen_info info);
bool examineBodies(const body *bodies, int body_count, long double *max_x, long double *max_y);

int main(int argc, char **argv) {
	int option, steps = 365, delta = 1, body_count;
	char *input = "init.dat", *output = "result.dat";
	bool parallel = false;
	double start;
	long double px, py;
	imggen_info img_info;
	body *bodies = NULL;
	
	/* PBM default values */
	img_info.gen_img = false;
	img_info.img_steps = 1000;
	img_info.img_prefix = "PBM";
	img_info.offset = 2.2;
	img_info.width = 400;
	img_info.heigth = 400;
	
	/* Read cmdline params */
	while ((option = getopt(argc,argv,"phs:d:f:o:i:x:g")) != -1) {
		
		switch(option) {
		case 'p': parallel = true; break;
		case 's': steps = atoi(optarg); break;
		case 'd': delta = atoi(optarg); break;
		case 'f': input = optarg; break;
		case 'o': output = optarg; break;
		case 'i': img_info.img_prefix = optarg; break;
		case 'x': img_info.img_steps = atoi(optarg); break;
		case 'g': img_info.gen_img = true; break;
		default:
			printhelp();
			return 1;
		}
	}
	
	/* Init MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_self);
	
	/* Validate params */
	if (steps < 0 && delta < 1) {
		m_printf("Wrong parameter value! Will exit!\n Steps: %i | Delta: %i | Input Filepath: %s", steps, delta, input);
	}
	
	
	bodies = readBodies(fopen(input,"r"), &body_count);
	
	if (bodies == NULL) {
		m_printf("Error while reading input file %s. Will exit now!\n", input);
		MPI_Finalize();
		return 1;
	}
	
	/* Assert that all masses > 0 and any two particles do not share the same position */
	if (!examineBodies(bodies, body_count, &img_info.max_x, &img_info.max_y)) {
		m_printf("Error while reading input file %s. Some value are not permitted!\n", input);
		printBodies(bodies, body_count);
		MPI_Finalize();
		return 1;
	}
	
	//m_printf("max_x = %Lf, max_y = %Lf\n",img_info.max_x, img_info.max_y);
	
	//printBodies(bodies, body_count);
	totalImpulse(bodies, body_count, &px, &py);
	m_printf("Initial total impulse: px = %Lf , py = %Lf\n", px, py);
	start = seconds();
	
	if (parallel) {
		m_printf("PARALLEL\n");
		solve_parallel(bodies, body_count, steps, delta, img_info);
	} else {
		m_printf("SEQUENTIAL\n");
		solve_sequential(bodies, body_count, steps, delta, img_info);
	}
	
	m_printf("Rate of interactions: %lf\n", interactions(body_count, steps, seconds() - start));
	totalImpulse(bodies, body_count, &px, &py);
	m_printf("Resulting total impulse: px = %Lf , py = %Lf\n", px, py);
	
	/* Write result to file */
	FILE *f = fopen(output,"w");
	
	if (f == NULL) {
		m_printf("Error while opening output file %s.\n", output);
		MPI_Finalize();
		return 1;
	}
	
	writeBodies(f, bodies, body_count);
	
	MPI_Finalize();
	return 0;
}

inline void solve_sequential(body *bodies, int body_count, int steps, int delta, imggen_info img_info) {
	int x, i, j;
	long double tmp2, tmp3, tmp4, constants[body_count][body_count], 
		delta_square = delta * delta * 0.5, meters = MAX(img_info.max_x, img_info.max_y);
	vector mutual_f[body_count][body_count], total_f[body_count];
	
	for (i = 0; i < body_count; i++)
		for (j = i + 1; j < body_count; j++)
			constants[i][j] =  G * bodies[j].mass * bodies[i].mass;
	
	
	for (x = 0; x < steps; x++) {
		
		for (i = 0; i < body_count; i++) {
			for(j = i + 1; j < body_count; j++) {
				tmp2 = bodies[j].x - bodies[i].x;
				tmp3 = bodies[j].y - bodies[i].y;
				tmp4 = sqrtl(tmp2*tmp2 + tmp3*tmp3);
				tmp4 = tmp4 * tmp4 * tmp4;
				mutual_f[i][j].x = constants[i][j] * (tmp2) / tmp4;
				mutual_f[i][j].y = constants[i][j] * (tmp3) / tmp4;
			}
		}
		
		for (i = 0; i < body_count; i++) {
			total_f[i].x = 0;
			total_f[i].y = 0;
			
			for(j = 0; j < body_count; j++) {
				
				if(j > i) {
					total_f[i].x += mutual_f[i][j].x;
					total_f[i].y += mutual_f[i][j].y;
				} else if( i != j ) {
					total_f[i].x -= mutual_f[j][i].x;
					total_f[i].y -= mutual_f[j][i].y;
				}
			}
			
			//printf("Total force: %i > (%Lf,%Lf)\n", i,total_f[i].x, total_f[i].y); 
			
			/* Acceleration */
			total_f[i].x = total_f[i].x / bodies[i].mass;
			total_f[i].y = total_f[i].y / bodies[i].mass;
			
			//printf("Acceleration: %i > (%Lf,%Lf)\n", i,total_f[i].x, total_f[i].y);
			
			/* Update position and velocity */
			bodies[i].x = bodies[i].x + bodies[i].vx  * delta + total_f[i].x * delta_square;
			bodies[i].y = bodies[i].y + bodies[i].vy  * delta + total_f[i].y * delta_square;
			bodies[i].vx = bodies[i].vx + total_f[i].x * delta;
			bodies[i].vy = bodies[i].vy + total_f[i].y * delta;	
		}
		
		/* Save an image of intermediate results. */
		if (img_info.gen_img && x % img_info.img_steps == 0) {
			saveImage(x, bodies, body_count, img_info.offset * meters,
			img_info.offset * meters, img_info.width, img_info.heigth, img_info.img_prefix);
		}
	}
}

inline void solve_parallel(body *bodies, int body_count, int steps, int delta, imggen_info img_info) {
int x, i, j;
	long double tmp2, tmp3, tmp4, constants[body_count][body_count], 
		delta_square = delta * delta * 0.5, meters = MAX(img_info.max_x, img_info.max_y);
	vector mutual_f[body_count][body_count], total_f[body_count];
	
	#pragma omp parallel for private (j)
	for (i = 0; i < body_count; i++)
		for (j = i + 1; j < body_count; j++)
			constants[i][j] =  G * bodies[j].mass * bodies[i].mass;
	
	
	for (x = 0; x < steps; x++) {
		
		#pragma omp parallel for private (j, tmp2, tmp3, tmp4)
		for (i = 0; i < body_count; i++) {
			for(j = i + 1; j < body_count; j++) {
				tmp2 = bodies[j].x - bodies[i].x;
				tmp3 = bodies[j].y - bodies[i].y;
				tmp4 = sqrtl(tmp2*tmp2 + tmp3*tmp3);
				tmp4 = tmp4 * tmp4 * tmp4;
				mutual_f[i][j].x = constants[i][j] * (tmp2) / tmp4;
				mutual_f[i][j].y = constants[i][j] * (tmp3) / tmp4;
			}
		}
		
		#pragma omp parallel for private (j)
		for (i = 0; i < body_count; i++) {
			total_f[i].x = 0;
			total_f[i].y = 0;
			
			for(j = 0; j < body_count; j++) {
				
				if(j > i) {
					total_f[i].x += mutual_f[i][j].x;
					total_f[i].y += mutual_f[i][j].y;
				} else if( i != j ) {
					total_f[i].x -= mutual_f[j][i].x;
					total_f[i].y -= mutual_f[j][i].y;
				}
			}
			
			//printf("Total force: %i > (%Lf,%Lf)\n", i,total_f[i].x, total_f[i].y); 
			
			/* Acceleration */
			total_f[i].x = total_f[i].x / bodies[i].mass;
			total_f[i].y = total_f[i].y / bodies[i].mass;
			
			//printf("Acceleration: %i > (%Lf,%Lf)\n", i,total_f[i].x, total_f[i].y);
			
			/* Update position and velocity */
			bodies[i].x = bodies[i].x + bodies[i].vx  * delta + total_f[i].x * delta_square;
			bodies[i].y = bodies[i].y + bodies[i].vy  * delta + total_f[i].y * delta_square;
			bodies[i].vx = bodies[i].vx + total_f[i].x * delta;
			bodies[i].vy = bodies[i].vy + total_f[i].y * delta;	
		}
		
		/* Save an image of intermediate results. */
		if (img_info.gen_img && x % img_info.img_steps == 0) {
			saveImage(x, bodies, body_count, img_info.offset * meters,
			img_info.offset * meters, img_info.width, img_info.heigth, img_info.img_prefix);
		}
	}
}

inline double interactions(int body_count, int steps, double time) {
	return (double) (body_count * (body_count-1) * steps) / time;
}

void printhelp() {
	m_printf("Usage:\n");
	m_printf("-f  Input path of initial setting (DEFAULT: init.dat)\n");
	m_printf("-i  Prefix for image creation (DEFAULT: PBM)\n");
	m_printf("-p  BOOL	Parallel execution (DEFAULT: false)\n");
	m_printf("-s  INT	Number of simulation steps (DEFAULT: 365)\n");
	m_printf("-d  INT	Step size of the simulation (DEFAULT: 1)\n");
	m_printf("-g  BOOL	Generate PBM images of intermediate results. (DEFAULT: false)\n");
	m_printf("-x  INT	Generate a PBM image of the current setting every x simulation steps (DEFAULT: 1000)\n");
	m_printf("-h  This message\n");
}

void printBodies(const body *bodies, int body_count) {
	int i;
	
	for (i = 0; i < body_count; i++)
	m_printf("%i> Mass: %Lf , Position: (%Lf,%Lf) , Velocity: (%Lf,%Lf)\n",i,
	bodies[i].mass, bodies[i].x, bodies[i].y, bodies[i].vx, bodies[i].vy);
}

bool examineBodies(const body *bodies, int body_count,long double *max_x, long double *max_y) {
	bool valid = true;
	int i,j;
	*max_x = 0;
	*max_y = 0;
	
	for (i = 0; i < body_count && valid; i++) {
		valid&= bodies[i].mass > 0;
		*max_x = MAX(*max_x,ABS(bodies[i].x));
		*max_y = MAX(*max_y,ABS(bodies[i].y));
		
		for (j = 0; j < body_count && valid; j++) {
			valid&= (bodies[i].x != bodies[j].x || bodies[i].y != bodies[j].y || i == j);
		}
	}
	
	return valid;
}

int m_printf(char *format, ...) {
	
	if (mpi_self == MASTER) {
		va_list args;
		va_start(args, format);
		int r = vprintf(format, args);
		va_end(args);
		return r;
	} else {
		return -1;
	}
}

