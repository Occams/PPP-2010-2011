APP = aufgabe1-geruest

.PHONY: all clean call

all: $(APP).c
	mpicc -Wall -o $(APP) $(APP).c

clean:
	rm -f $(APP)

call:
	mpiexec -n 2 $(APP)
