#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <sobel.h>
#include <vcd.h>

/*
 * Declarations.
 */
void help(void);



int main(int argc, char *argv[]) {
	char *input, *output;
	
	
	int option;
    while ((option = getopt(argc,argv,"i:o:h")) != -1) {
        switch(option) {
        case 'i': input = optarg; break;
        case 'o': output = optarg; break;
        case 'h': help(); return 0; break;
        default:
            return 1;
        }
    }
}

void help() {
	printf("HELPTEXT\n");
}
