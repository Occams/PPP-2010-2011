SOURCE = main
COMMON = ../Common
ADD_SOURCES = nbody-readdat.c $(COMMON)/common.c

LIBS64 = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -lppp_pnm -lm
INCLUDES64 = -I./ -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -I$(COMMON)

LIBS32 = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-32 -lppp_pnm -lm
INCLUDES32 = -I./ -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-32 -I$(COMMON)

PARAMS = -Wall -std=gnu99 -O3 -fomit-frame-pointer -fopenmp


.PHONY: all clean hydra hermes

all: $(SOURCE).c
	rm -f $(SOURCE).bin $(SOURCE)-hydra.bin $(SOURCE)-hermes.bin PBM*;
	mpicc $(PARAMS) -o $(SOURCE).bin $(SOURCE).c $(ADD_SOURCES) $(INCLUDES64) $(LIBS64)

hydra: 
	mpicc $(PARAMS) -o $(SOURCE)-hydra.bin $(SOURCE).c $(ADD_SOURCES) $(INCLUDES64) $(LIBS64)

hermes:
	mpigcc4.mpich2 $(PARAMS) -o $(SOURCE)-hermes.bin $(SOURCE).c $(ADD_SOURCES) $(INCLUDES32) $(LIBS32)

clean:
	rm -f $(SOURCE).bin $(SOURCE)-hydra.bin $(SOURCE)-hermes.bin PBM*;
