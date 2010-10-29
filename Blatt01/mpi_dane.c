#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include <stdbool.h>

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

double mpi_broadcast(int *source,int *self,int *length,int *array,int *count) {
	double elapsed_time = 0;
	double start;
	int i = 0;
	
	for (i = 0; i < *count; i++) {
		
		if (*self == *source) {
			MPI_Barrier(MPI_COMM_WORLD);
			start = seconds();
			MPI_Bcast(array,*length,MPI_INT,*source,MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			elapsed_time += seconds() - start;
		} else {
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(array,*length,MPI_INT,*source,MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
	
	return elapsed_time / *count;
}

double simulated_bcast(int *source,int *self,int *length,int *array,int *count,bool *blocking, int *processors) {
	double elapsed_time = 0;
	double start;
	int i = 0,requests_index = 0;
	MPI_Status status;
	MPI_Request *requests = malloc((*processors-1)*sizeof(MPI_Request));
	MPI_Status *status_array = malloc((*processors-1)*sizeof(MPI_Status));;
	
	if (requests == NULL || status_array == NULL) {
		printf("Could not allocate memory. Will now exit.");
		return 1;
	}
	
	for (i = 0; i < *count; i++) {
		requests_index = 0;
		
		if (*self == *source) {
			MPI_Barrier(MPI_COMM_WORLD);
			start = seconds();
			
			int j = 0;
			for (j = 0; j < *processors; j++) {
				
				if (j != *self) {

					if (*blocking) {
						MPI_Isend(array,*length,MPI_INT,j,0, MPI_COMM_WORLD,&requests[requests_index]);
						requests_index++;
					} else {
						MPI_Send(array, *length, MPI_INT, j, 0 , MPI_COMM_WORLD);
					}
				}
			}
			
			/* Wait until all processes received their arrays */
			if (*blocking) {
				MPI_Waitall(*processors-1,requests,status_array);
			}
			
			MPI_Barrier(MPI_COMM_WORLD);
			elapsed_time += seconds() - start;
		} else {
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Recv(array,*length,MPI_INT,*source,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			//printf("Processor %i received array\n", *self);
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
	
	/* free memory */
	free(requests);
	free(status_array);
	
	return elapsed_time / *count;
}

int computeRank(int *source, int *self,int *processors, bool real) {
	if (!real) {
		return (*self + *source) % *processors;
	} else {
		return (*self + *processors - *source) % *processors;
	}
}

double tree_distribution(int *source,int *self,int *length,int *array,int *count, int *processors) {
	double elapsed_time = 0;
	double start;
	int i = 0;
	MPI_Request *requests = malloc(2 * sizeof(MPI_Request));
	MPI_Status *status_array = malloc(2 * sizeof(MPI_Status));
	MPI_Status status;
	
	if (requests == NULL || status_array == NULL) {
		printf("Could not allocate memory. Will now exit.");
		return 1;
	}
	
	for (i = 0; i < *count; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
		
		/* Wait for array if not source */
		if (*self != *source) {
			MPI_Recv(array,*length, MPI_INT, MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			//printf("Processor %i received array\n", *self);
		} else {
			start = seconds();
		}
	
			int firstRecipient = 2 * (computeRank(source,self,processors,true)) + 1;
			int sentMessages = 0;
			
			if (firstRecipient < *processors) {
			//printf("Processor %i sends array to processor %i\n", *self,computeRank(source,&firstRecipient,processors,false));
			MPI_Isend(array,*length,MPI_INT,computeRank(source,&firstRecipient,processors,false),0,MPI_COMM_WORLD,&requests[0]);
			sentMessages++;
			}
			
			firstRecipient++;
			
			if (firstRecipient < *processors) {
			//printf("Processor %i sends array to processor %i\n", *self,computeRank(source,&firstRecipient,processors,false));
			MPI_Isend(array,*length,MPI_INT,computeRank(source,&firstRecipient,processors,false),0,MPI_COMM_WORLD,&requests[1]);
			sentMessages++;
			}
			
			MPI_Waitall(sentMessages,requests,status_array);
			MPI_Barrier(MPI_COMM_WORLD);
			elapsed_time += seconds() - start;
	}
	
	/* free memory */
	free(requests);
	free(status_array);
	
	return elapsed_time / *count;
	
}

int main(int argc, char **argv) {
	enum b_type {MPI_BROADCAST, SIMULATED_BCAST, TREE_DISTRIBUTION} type = MPI_BROADCAST;
	bool simulated_blocking = false;
	char option;
	int processors,self,source = 0, length = 2,count = 1; 
	int *array, *receive_b;
	
	/* init MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &self);
	
	/* extract cmdline options */
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
	
	/* validate parameter values */
	if (source < 0 || source > processors || length < 1 || count < 1) {
		printf("Wrong parameter value!");
		return 1;
	}
	
	
	/* print status messages */
	if (self == source) {
		if (type == MPI_BROADCAST)
		printf("MPI_BROADCAST\n");
		if (type == SIMULATED_BCAST)
		printf("SIMULATED_BCAST - %i\n",simulated_blocking);
		if (type == TREE_DISTRIBUTION)
		printf("TREE_DISTRIBUTION\n");
		//printf("Source: %i - Length: %i - Count: %i\n",source,length,count);
	}
	
	/* allocate arrays */
	array =  allocints(length);
	receive_b = allocints(length);
	
	if (array == NULL || receive_b == NULL) {
		printf("Could not allocate memory. Will now exit.");
		return 1;
	}
	
	
	/* perform broadcast and measure required time */
	double time_elapsed = 0;
	
	switch(type) {
	case MPI_BROADCAST: time_elapsed = mpi_broadcast(&source,&self,&length,array,&count); break;
	case SIMULATED_BCAST: simulated_bcast(&source,&self,&length,array,&count,&simulated_blocking,&processors); break;
	case TREE_DISTRIBUTION: tree_distribution(&source,&self,&length,array,&count,&processors); break;
	default:
		MPI_Finalize();
		return 1;
	}
	
	if (self == source) {
		printf("Time Elapsed: %f\n", time_elapsed);
	}

	/* free memory */
	free(array);
	free(receive_b);
	
	/* MPI end */
	MPI_Finalize();

	return 0;
}
