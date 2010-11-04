APP = basdl
LIBS = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -lppp_pnm
INCLUDES = -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-64

.PHONY: all clean

all: $(APP).c
	mpicc -Wall -o $(APP).bin $(APP).c $(INCLUDES) $(LIBS) -fopenmp

clean:
	rm -f $(APP).bin
