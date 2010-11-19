SOURCE = app
APP = converter
COMMON = ../../Common/common.c
LIBS64 = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -lppp_pnm -Lm
INCLUDES64 = -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -I../../Common
LIBS32 = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-32 -lppp_pnm -Lm
INCLUDES32 = -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-32 -I../../Common


.PHONY: all clean hydra hermes

all: $(SOURCE).c
	mpicc -Wall -std=gnu99 -O3 -fomit-frame-pointer -fopenmp -o $(APP).bin $(SOURCE).c $(COMMON) $(INCLUDES64) $(LIBS64)

hydra: 
	mpicc -Wall -std=c99 -O3 -fomit-frame-pointer -fopenmp -o $(APP)-hydra.bin $(SOURCE).c $(COMMON) $(INCLUDES64) $(LIBS64)

hermes:
	mpigcc4.mpich2 -Wall -std=c99 -O3 -fomit-frame-pointer -fopenmp -o $(APP)-hermes.bin $(SOURCE).c $(COMMON) $(INCLUDES32) $(LIBS32)

clean:
	rm -f $(APP).bin $(APP)-hydra.bin $(APP)-hermes.bin output.pgm
