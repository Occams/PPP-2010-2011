SOURCE = main
ADD_SOURCES = vcd.c sobel.c common.c

LIBS64 = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -lppp_pnm -Lm
INCLUDES64 = -L./ -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -I../../Common

LIBS32 = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-32 -lppp_pnm -Lm
INCLUDES32 = -L./ -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-32 -I../../Common

PARAMS = -Wall -std=gnu99 -O3 -fomit-frame-pointer -fopenmp

.PHONY: all clean hydra hermes
vpath % ./:../../Common

all: $(SOURCE).c
	mpicc $(PARAMS) -o $(SOURCE).bin $(SOURCE).c $(ADD_SOURCES) $(INCLUDES64) $(LIBS64)

hydra: 
	mpicc $(PARAMS) -o $(SOURCE)-hydra.bin $(SOURCE).c $(ADD_SOURCES) $(INCLUDES64) $(LIBS64)

hermes:
	mpigcc4.mpich2 $(PARAMS) -o $(SOURCE)-hermes.bin $(SOURCE).c $(ADD_SOURCES) $(INCLUDES32) $(LIBS32)

clean:
	rm -f $(SOURCE).bin $(SOURCE)-hydra.bin $(SOURCE)-hermes.bin output.pgm
