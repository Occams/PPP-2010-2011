APP = mpi_basdl

.PHONY: all clean call

all: $(APP).c
	mpicc -Wall -o $(APP).bin $(APP).c

clean:
	rm -f $(APP).bin

call:
	mpiexec -n 2 $(APP)
