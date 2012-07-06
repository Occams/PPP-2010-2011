#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>

int np;   /* Anzahl der MPI-Prozessore */
int self; /* Nummer des eigenen Prozesses (im Bereich 0,...,np-1) */

/* liefert die Sekunden seit dem 01.01.1970 */
double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec)/1000000.0;
}

/*
 * Speicherplatz fuer `size' viele int anfordern (d.h. ein
 * int-Array der Laenge `size').
 * Der Speicherplatz ist nach Benutzung mit  free  freizugeben:
 *    int *p = allocints(100);
 *    .... // p benutzen
 *    free(p);
 */
int *allocints(int size) {
    int *p;
    p = (int *)malloc(size * sizeof(int));
    return p;
}

int main(int argc, char *argv[])
{
    double start, end;
    int option;

    int option_a, option_b;
    int option_c, c_arg;

    /* MPI initialisieren und die Anzahl der Prozesse sowie
     * die eigene Prozessnummer bestimmen
     */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    /* Beispiel fuer Kommandozeilen-Optionen mit getopt.
     * Ein nachgestellter Doppelpunkt signalisiert eine
     * Option mit Argument (in diesem Beispiel bei "-c").
     */
    option_a = option_b = option_c = 0;
    while ((option = getopt(argc,argv,"abc:")) != -1) {
        switch(option) {
        case 'a': option_a = 1; break;
        case 'b': option_b = 1; break;
        case 'c': option_c = 1; c_arg = atoi(optarg); break;
        default:
            MPI_Finalize();
            return 1;
        }
    }

    if (option_a)
	printf("Option -a gesetzt\n");
    if (option_b)
	printf("Option -b gesetzt\n");
    if (option_c)
	printf("Option -c mit Argument %d gesetzt\n", c_arg);

    /* hier geht's los... */
    start = seconds();
    printf("rank = %d, size = %d\n", self, np);
    end = seconds();
    printf("Time: %f s\n", end-start);

    /* MPI beenden */
    MPI_Finalize();

    return 0;
}
