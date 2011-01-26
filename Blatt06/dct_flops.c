#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include "frame_encoding.h"

uint64_t run_dct(int n) {
    int16_t input[64], output[64];

    for (int i=0; i<64; i++)
        input[i] = ((i*i+253)%256)-128;

    for (int i=0; i<n; i++)
        qdct_block(input, output);

    return 2*(2*8*8*8)*(uint64_t)n;
}

/* liefert die Sekunden seit dem 01.01.1970 */
double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

int main(void) {
    int n = 10000000;
    init_qdct();

    double t = seconds();
    uint64_t ops = run_dct(n);
    t = seconds() - t;

    printf("%f MFLOPS, %f DCTs per sec\n", ops/t/1.0e6, n/t);

    return 1;
}
