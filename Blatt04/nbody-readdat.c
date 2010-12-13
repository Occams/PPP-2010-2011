#include "nbody-readdat.h"

/*
* Kommentare (mit "# ...") in ".dat" Dateien ueberspringen.
*/
static void skipComments(FILE *f) {
	int n, error;
	do {
		n=0;
		if (fscanf(f, " #%n", &n) != -1);
		error = 0;
		if (n > 0) {
			if(fscanf(f, "%*[^\n]") != -1);
			error = 0;
			if(fscanf(f, "\n") != -1);
			error = 0;
		}
	} while (n>0 && !error);
	
	if (error)
	printf("I/O error while parsing file!");
}

/*
* Eine ".dat" Datei mit Beschreibungen der Koerper einlesen.
* (Format siehe Uebungsblatt).
*    f: Dateihandle, aus dem gelesen wird
*    n: Output-Parameter fuer die Anzahl der gelesenen Koerper
* Die Koerper werden in einem Array von body-Strukturen
* zurueckgeliefert. Im Fehlerfall wird NULL zurueckgegeben.
*/
body* readBodies(FILE *f, int *n) {
	int i, conv;
	body *bodies;
	
	if (f == NULL)
	return NULL;

	skipComments(f);
	if (fscanf(f, " %d", n) != 1)
	return NULL;
	bodies = (body *) malloc(sizeof(body) * *n);
	if (bodies == NULL)
	return NULL;

	for (i=0; i<*n; i++) {
		skipComments(f);
		conv = fscanf(f, " %Lf %Lf %Lf %Lf %Lf",
		&(bodies[i].mass),
		&(bodies[i].x), &(bodies[i].y),
		&(bodies[i].vx), &(bodies[i].vy));
		if (conv != 5) {
			free(bodies);
			return NULL;
		}
	}
	return bodies;
}

/*
* Schreibe 'n' Koerper aus dem Array 'bodies' in die
* durch das Dateihandle 'f' bezeichnete Datei im ".dat" Format.
*/
void writeBodies(FILE *f, const body *bodies, int n) {
	int i;
	fprintf(f, "%d\n", n);
	for (i=0; i<n; i++) {
		fprintf(f, "% 10.4Lg % 10.4Lg % 10.4Lg % 10.4Lg % 10.4Lg\n",
		bodies[i].mass, bodies[i].x, bodies[i].y,
		bodies[i].vx, bodies[i].vy);
	}
}

/*
* Berechne den Gesamtimpuls des Systems.
*   bodies:  Array der Koerper
*   nBodies: Anzahl der Koerper
*   (px,py): Output-Parameter fuer den Gesamtimpuls
*/
void totalImpulse(const body *bodies, int nBodies,
long double *px, long double *py)
{
	long double px_=0, py_=0;
	int i;

	for (i=0; i<nBodies; i++) {
		px_ += bodies[i].mass * bodies[i].vx;
		py_ += bodies[i].mass * bodies[i].vy;
	}
	
	*px = px_;
	*py = py_;
}

/*
* Einfache Routine zur Ausgabe der Koerper als Bild.
* Legt ein PBM (portable bitmap) Bild mit einem weissen
* Pixel fuer jeden Koerper an.
*   imgNum:  Nummer des Bildes (geht in den Dateinamen ein)
*   bodies:  Array der Koerper
*   nBodies: Anzahl der Koerper
*/
void saveImage(int imgNum, const body *bodies, int nBodies, long double width,
long double height, int imgWidth, int imgHeight, char *imgFilePrefix)
{
	int i, x, y;
	int *img = (int *) malloc(sizeof(int) * imgWidth * imgHeight);
	char name[strlen(imgFilePrefix)+15];

	if (img == NULL) {
		fprintf(stderr, "Oops: could not allocate memory for image\n");
		return;
	}

	sprintf(name, "%s%010d.pbm", imgFilePrefix, imgNum);
	for (i=0; i< imgWidth*imgHeight; i++)
	img[i] = 1;

	for (i=0; i<nBodies; i++) {
		x = imgWidth/2  + bodies[i].x*imgWidth/width;
		y = imgHeight/2 - bodies[i].y*imgHeight/height;

		if (x >= 0 && x < imgWidth && y >=0 && y < imgHeight)
		img[y*imgWidth + x] = 0;
	}

	ppp_pnm_write(name, PNM_KIND_PBM, imgHeight, imgWidth, 1, img);
	free(img);
}
