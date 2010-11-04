APP = basdl
LIBS = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -lppp_pnm
INCLUDES = -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-64


.PHONY: all clean dane

all: $(APP).c
	mpicc -Wall -o $(APP).bin $(APP).c $(INCLUDES) $(LIBS) -fopenmp
	
dane:
	mpicc -Wall -o dane.bin dane.c 

clean:
	rm -f $(APP).bin dane.bin
