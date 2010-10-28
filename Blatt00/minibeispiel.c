#include <stdio.h>
#include <omp.h>

int main(void) {
    printf("Hier ist der Hauptthread.\n");

#pragma omp parallel
    {
	int np   = omp_get_num_threads();
	int self = omp_get_thread_num();
	printf("Hier ist Thread #%d von %d Threads insgesamt.\n", self, np);
    }

    printf("Hier ist wieder der Hauptthread.\n");

    return 0;
}
