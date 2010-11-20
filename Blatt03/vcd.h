#ifndef VCD_H_INCLUDED
#define VCD_H_INCLUDED

#include <math.h>
#include <stdbool.h>
#include "common.h"
#include <stdio.h>
#include "pgm_distribute.h"

#define N 40
#define EPSILON 0.005
#define KAPPA 30.0
#define DELTA_T 0.1
#define PHI(v) ( CHI(v) * exp( -( CHI(v)* CHI(v) ) / 2.0 ) )
#define CHI(v) ( (v) / KAPPA ) 
#define XI(v) ( 1.0 / sqrt(2) * PSI(v) * exp( -( PSI(v) * PSI(v) ) / 2.0 ) )
#define PSI(v) ( (v) / (sqrt(2) * KAPPA))

int *vcd_sequential(int *image, int rows, int columns, int maxcolor);

#endif
