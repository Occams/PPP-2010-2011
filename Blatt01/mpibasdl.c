#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>

int np;   /* Anzahl der MPI-Prozessore */
int self; /* Nummer des eigenen Prozesses (im Bereich 0,...,np-1) */
double start, end;

void BCAST_MPI(int*, int, int);
void BCAST_SEND(int*, int, int);
void BCAST_TREE(int*, int, int);

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

    int option;

    int option_a, option_b, option_d;
    int option_c, c_arg;
    int option_s, s_arg;

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
    option_a = option_b = option_c = option_s = 0;
    while ((option = getopt(argc,argv,"abc:ds:")) != -1) {
        switch(option) {
        case 'a': option_a = 1; break;
        case 'b': option_b = 1; break;
        case 'c': option_c = 1; c_arg = atoi(optarg); break;
        case 'd': option_d = 1; break;
        case 's': option_s = 1; s_arg = atoi(optarg); break;
        default:
            MPI_Finalize();
            return 1;
        }
    }

	/*
	 * Array Size
	 */
    if (!option_c) {
		c_arg = 1;
	}
	
	/*
	 * Source
	 */
	if (!option_s) {
		s_arg = 0;
	}

	/*
	 * Init array
	 */
	int arr_count = 1<<c_arg; // Exponential size
    int arr[arr_count];

	if(option_a) {
		BCAST_MPI(arr, arr_count, s_arg);
	}
	
	if(option_b) {
		BCAST_SEND(arr, arr_count, s_arg);
	}
	
	if(option_d) {
		BCAST_TREE(arr, arr_count, s_arg);
	}


    /* MPI beenden */
    MPI_Finalize();

    return 0;
}

void BCAST_MPI(int* arr, int arr_count, int source) {
    /* Broadcast Method */
    if(self == source) {
    	MPI_Barrier(MPI_COMM_WORLD);
    	start = seconds();
    	MPI_Bcast(arr, arr_count, MPI_INT, source, MPI_COMM_WORLD);
    	MPI_Barrier(MPI_COMM_WORLD);
    	end = seconds();
    	printf("MPI_BCAST %i elements: %f, source %i\n", arr_count, end-start, source);
	} else {
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(arr, arr_count, MPI_INT, source, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void BCAST_SEND(int* arr, int arr_count, int source) {
	/* Single Send Method */
	if(self == source) {
		int i;
		start = seconds();
		for(i = 0; i < np; i++) {
			if(i != source) {
				MPI_Send(arr, arr_count, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		end = seconds();
    	printf("MPI_Send %i elements: %f, source %i\n", arr_count, end-start, source);
	} else {
		MPI_Status status;
		MPI_Recv(arr, arr_count, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void BCAST_TREE(int* arr, int arr_count, int source) {
	
}
