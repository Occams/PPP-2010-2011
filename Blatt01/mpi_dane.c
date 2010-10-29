#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include <stdbool.h>

int np, self; 

double seconds() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + ((double)time.tv_usec)/1000000.0;
}

int *allocints(int size) {
    int *p;
    p = (int *)malloc(size * sizeof(int));
    return p;
}

void mpi_broadcast(int *source,int *self,int *length,int *array,int *receive_b) {
	printf("grr");
	MPI_Barrier(MPI_COMM_WORLD);
	printf("grr2");
	if (*self == *source) {
		MPI_Bcast(array,*length,MPI_INT,*source,MPI_COMM_WORLD);
	} else {
		printf("Self: %i", *self);
		MPI_Status status;
		MPI_Recv(receive_b,*length,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		printf("Processor %i received array", *self);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
	enum b_type {MPI_BROADCAST, SIMULATED_BCAST, TREE_DISTRIBUTION} type = MPI_BROADCAST;
	bool simulated_blocking = false;
    int option = 0;
	int source = 0, length = 2,count = 1; 
	int *array, *receive_b;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
	
    while ((option = getopt(argc,argv,"s:l:bi:tn:")) != -1) {
	
        switch(option) {
		case 'b': type = MPI_BROADCAST; break;
		case 'i': type = SIMULATED_BCAST; simulated_blocking = atoi(optarg); break;
		case 't': type = TREE_DISTRIBUTION; break;
		case 'l': length = atoi(optarg); break;
		case 'n': count = atoi(optarg); break;
		case 's': source = atoi(optarg); break;
        default:
            MPI_Finalize();
            return 1;
        }
    }
	
	if (source < 0 || source > np || length < 1 || count < 1) {
		printf("Wrong paramter value!");
		return 1;
	}
	
	
	if (self == source) {
    if (type == MPI_BROADCAST)
	printf("MPI_BROADCAST\n");
    if (type == SIMULATED_BCAST)
	printf("SIMULATED_BCAST - %i\n",simulated_blocking);
    if (type == TREE_DISTRIBUTION)
	printf("TREE_DISTRIBUTION\n");
	printf("Source: %i - Length: %i - Count: %i\n",source,length,count);
	}
	
	array =  allocints(length);
	receive_b = allocints(length);
	
	switch(type) {
	case MPI_BROADCAST: mpi_broadcast(&source,&self,&length,array,receive_b); break;
	case SIMULATED_BCAST: break;
	case TREE_DISTRIBUTION: break;
	default:
		MPI_Finalize();
		return 1;
	}

	
	free(array);
	free(receive_b);
    /* MPI beenden */
    MPI_Finalize();

    return 0;
}
