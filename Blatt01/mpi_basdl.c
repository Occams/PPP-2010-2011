#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>

#define true 1
#define false 0

int np;   /* Anzahl der MPI-Prozessore */
int self; /* Nummer des eigenen Prozesses (im Bereich 0,...,np-1) */
double start, end;

char *optarg = 0;

double BCAST_MPI(int*, int, int);
double BCAST_SEND(int*, int, int);
double BCAST_TREE(int*, int, int);
int BCAST_TREE_ConvAdr(int source, int adr, int fromRealToShifted);
void printhelp(void);

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

    int option_a, option_b, option_d, option_h;
    int option_c, c_arg;
    int option_s, s_arg;
    int option_f, f_arg;

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
    option_a = option_b = option_c = option_d = option_s = option_f = option_h = 0;
    while ((option = getopt(argc,argv,"abc:ds:f:h")) != -1) {
        switch(option) {
        case 'a': option_a = 1; break;
        case 'b': option_b = 1; break;
        case 'c': option_c = 1; c_arg = atoi(optarg); break;
        case 'd': option_d = 1; break;
        case 'f': option_f = 1; f_arg = atoi(optarg); break;
        case 'h': option_h = 1; break;
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
	 * Loop counter
	 */
	if (!option_f) {
		f_arg = 1;
	}

	/*
	 * Init array
	 */
	int arr_count = 1<<c_arg; // Exponential size
    int arr[arr_count];
    double average;
    int i;
    
    
	if(option_a) {
		average = 0;
		for(i = 0; i < f_arg; i++) {
			double d = BCAST_MPI(arr, arr_count, s_arg);
			if(self == s_arg) {
				average += d;
			}
		}
	
		if(self == s_arg) {
			average /= f_arg;
			printf("MPI_Bcast %f\n", average);
		}
	}
	
	if(option_b) {
		average = 0;
		for(i = 0; i < f_arg; i++) {
			double d = BCAST_SEND(arr, arr_count, s_arg);
			if(self == s_arg) {
				average += d;
			}
		}
		
		if(self == s_arg) {
			average /= f_arg;
			printf("SEND %f\n", average);
		}
	}
	
	if(option_d) {
		average = 0;
		for(i = 0; i < f_arg; i++) {
			double d = BCAST_TREE(arr, arr_count, s_arg);
			if(self == s_arg) {
				average += d;
			}
		}
		
		if(self == s_arg) {
			average /= f_arg;
			printf("TREE %f\n", average);
		}
	}
	
	if(option_h) {
		if(self == s_arg) printhelp();
	}

    /* MPI beenden */
    MPI_Finalize();

    return 0;
}

double BCAST_MPI(int* arr, int arr_count, int source) {
    /* Broadcast Method */
    if(self == source) {
    	MPI_Barrier(MPI_COMM_WORLD);
    	start = seconds();
    	MPI_Bcast(arr, arr_count, MPI_INT, source, MPI_COMM_WORLD);
    	MPI_Barrier(MPI_COMM_WORLD);
    	end = seconds();
    	return end-start;
	} else {
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(arr, arr_count, MPI_INT, source, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		return 0;
	}
}

double BCAST_SEND(int* arr, int arr_count, int source) {
	/* Single Send Method */
	if(self == source) {
		int i;
		MPI_Barrier(MPI_COMM_WORLD);
		start = seconds();
		for(i = 0; i < np; i++) {
			if(i != source) {
				MPI_Send(arr, arr_count, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		end = seconds();
		return end-start;
	} else {
		MPI_Status status;
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Recv(arr, arr_count, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
		MPI_Barrier(MPI_COMM_WORLD);
		return 0;
	}
}

double BCAST_TREE(int* arr, int arr_count, int source) {
	int pos = BCAST_TREE_ConvAdr(source, self, true);	
	int firstChild = 2*pos+1;
	
	if( self != source ) {
		MPI_Status status;
		MPI_Recv(arr, arr_count, MPI_INT, BCAST_TREE_ConvAdr(source, (pos - 1)/2, false), 0, MPI_COMM_WORLD, &status);
	} else {
		start = seconds();
	}
	
	if(firstChild < np) {
		MPI_Send(arr, arr_count, MPI_INT, BCAST_TREE_ConvAdr(source, firstChild, false), 0, MPI_COMM_WORLD);
	}
	
	firstChild++;
	
	if(firstChild < np) {
		MPI_Send(arr, arr_count, MPI_INT, BCAST_TREE_ConvAdr(source, firstChild, false), 0, MPI_COMM_WORLD);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(self == source) {
		end = seconds();
		return end-start;
	}
	return 0;
}

int BCAST_TREE_ConvAdr(int source, int adr, int fromRealToShifted) {
	if(fromRealToShifted) {
		return (adr + np - source) % np;
	} else {
		return (adr + source) % np;
	}
}

void printhelp() {
	printf("USAGE:\n");
	printf("-c INT	Amount of integers to be broadcasted\n");
	printf("-a	Broadcast with MPI_Bcast\n");
	printf("-b	Broadcast with MPI_Send with a star algorithm\n");
	printf("-d	Broadcast with MPI_Send with a tree algorithm\n");
	printf("-f INT	Amount of retries to measure average values\n");
	printf("-s INT	The source process\n");
	printf("-h	This help message\n");
}
