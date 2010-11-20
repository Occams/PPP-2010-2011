#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED

#include <stdlib.h>
#include <sys/time.h>

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) < (Y) ? (Y) : (X))
#endif

#ifndef ABS
#define ABS(x) ((x) < 0 ? (-x) : (x))
#endif

double seconds(void);
int *allocints(int size);

#endif
