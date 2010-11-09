SOURCE = dane
APP = converter
LIBS64 = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -lppp_pnm
INCLUDES64 = -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-64 -I../../Common
LIBS32 = -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-32 -lppp_pnm
INCLUDES32 = -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-32 -I../../Common


.PHONY: all clean hydra hermes

all: $(SOURCE).c
	mpicc -Wall -O3 -fopenmp -o $(APP).bin $(SOURCE).c ../../Common/common.c $(INCLUDES64) $(LIBS64)

hydra: 
	mpicc -Wall -O3 -fopenmp -o $(APP)-hydra.bin $(SOURCE).c ../../Common/common.c $(INCLUDES64) $(LIBS64)

hermes:
	mpigcc4.mpich2 -Wall -O3 -fopenmp -o $(APP)-hermes.bin $(SOURCE).c ../../Common/common.c $(INCLUDES32) $(LIBS32)

clean:
	rm -f $(APP).bin $(APP)-hydra.bin $(APP)-hermes.bin
