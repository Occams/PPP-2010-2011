#include <common.h>


long double seconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long double)tv.tv_sec + ((long double)tv.tv_usec)/1000000.0;
}

int *allocints(int size) {
	int *p;
	p = (int *)malloc(size * sizeof(int));
	return p;
}
