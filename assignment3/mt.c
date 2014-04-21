#include "simdocs.h"
#include <mpi.h>

FILE* file_open(char* filename) {
	FILE* fp = fopen(filename, "r");
	if(fp == NULL) {
		fprintf(stderr, "ERROR: while opening file %s, abort.\n", filename);
		exit(1);
	}
	return fp;
}

char* file_getline(char* buffer, FILE* fp) {
	buffer = fgets(buffer, 300, fp);
	return buffer;
}

int numberOfLines(char * filename)
{
	int nlines = 0;
	char c;
	FILE * file = file_open(filename);

	while ( (c = fgetc(file)) != EOF )
	{
        	if ( c == '\n' )
		{
            		nlines++;
		}
    	}
	fclose(file);
	return nlines;
}


char ** getDataA(char * filename, int n)
{
	char **a =  (char **) malloc(sizeof(char *) * n);
	char* line = malloc(300*sizeof(char));
	FILE * file = file_open(filename);
	int count = 0;

	while((line = file_getline(line,file)) != NULL)
	{
		char * strtemp = malloc(strlen(line));
		char * tok = strtok(line, "\n");
		strcpy(strtemp, tok);
		a[count] = strtemp;
		count++;
	}
	fclose(file);
	free(line);
	return a;

}

float * getDataB(char * filename, int n)
{
	float *a = (float *) malloc(sizeof(float) * n);
	char* line = malloc(300*sizeof(char));
	FILE * file = file_open(filename);
	int count = 0;

	while((line = file_getline(line,file)) != NULL)
	{
		a[count] = atof(line);
		count++;
	}
	fclose(file);
	free(line);
	return a;
}


int parseInput(char ** a, gk_csr_t *mat, int n, int nb)
{
	int i, j, k;
	int row;
	int col;
	int count = 0;
	float v;
	char * current;
	char * pch;
	int lastRow = 0;
	for (i = 0; i < n; i++)
	{
		current = a[i];
		pch = strtok(current, " ");
		if (pch == NULL)
		{
			return -1;
		}
		row = atoi(pch);
		pch = strtok(NULL, " ");
		if (pch == NULL)
		{
			return -1;
		}
		col = atoi(pch);
		pch = strtok(NULL, " ");
		if (pch == NULL)
		{
			return -1;
		}
		v = atof(pch);
		if (row != lastRow)				// computes the prefix sum as it goes.
		{
			for (j = lastRow; j < row; j++)		//in case the next row is more than one greater than the last.
			{
				mat->rowptr[j + 1] = count;
			}
		}
		count++;
		mat->rowind[i] = col;
		mat->rowval[i] = v;
		lastRow = row;
	}
	mat->rowptr[nb] = n;
	return 0;


}


int main(int argc, char** argv)
{
	int rank, p;
    	int n;
	int nb;
	int prow;
    	char ** a;
    	float *b;
	float * pb;
	gk_csr_t *mat;
	gk_csr_t *mmat;
	int ndata;
	int i, j;
	int * nEachRow;
	int * dspl;
	int *nbneeded;
	int *bneeded;
	int tneeded;
	int count;
	int * cptrCum;
	int * nbrec;
	int * nbrecCum;
	int * nbneededCum;
	int tsend;
	int *toSend;
	float *toRec;
	float *toSendBack;
	int rowOffset;
	int first;
	int temp;
	MPI_Init(&argc, &argv);
   	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	if (rank == 0)
	{
		if (argc != 3)
		{
			printf("error: invalid arguments\n");
			exit(-1);
		}
		mmat = (gk_csr_t *) malloc(sizeof(gk_csr_t));
		n = numberOfLines(argv[1]);
		nb = numberOfLines(argv[2]);
		mmat->nrows = nb;
		mmat->ncols = nb;
		a = getDataA(argv[1], n);
		b = getDataB(argv[2], nb);
		mmat->rowptr = (int *) calloc(nb + 1, sizeof(int));
		mmat->rowval = (float *) malloc(sizeof(float) * n);
		mmat->rowind = (int *) malloc(sizeof(int) * n);

		int start, end;
		prow = nb/p;
//		printf("%d\n",nb);
		if (prow * p != nb)
		{
			printf("ERROR: rows do not divide evenly into processors\n");
			exit(-1);
		}
		if (parseInput(a, mmat, n, nb) == -1)
		{
			printf("ERROR: Failed to parse data.\n");
			exit(-1);
		}
		nEachRow = (int *) malloc(sizeof(int) * p);
		dspl = (int *) malloc(sizeof(int) * p);
		for (i = 0; i < p; i++)
		{
			start = i * prow;
			end = i * prow + prow;
			dspl[i] = mmat->rowptr[start];
			nEachRow[i] = mmat->rowptr[end]-mmat->rowptr[start];
		}

	}
	MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(nEachRow, 1, MPI_INT, &ndata, 1, MPI_INT, 0, MPI_COMM_WORLD);
	prow = nb/p;
	mat = (gk_csr_t *) malloc(sizeof(gk_csr_t));
	mat->nrows = prow;
	mat->ncols = nb;
	mat->rowptr = (int *) malloc(sizeof(int) * prow + 1);
	pb = (float *) malloc(sizeof(float) * prow);
	mat->rowind = (int *) malloc(sizeof(int) * ndata + 1);
	mat->rowval = (float *) malloc(sizeof(float) * ndata);
	MPI_Scatter(mmat->rowptr, prow, MPI_INT, mat->rowptr, prow, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(b, prow, MPI_FLOAT, pb, prow, MPI_FLOAT, 0, MPI_COMM_WORLD);
	first = mat->rowptr[0];
	for (i = 0; i < prow; i++)
	{
		mat->rowptr[i] = mat->rowptr[i] -  first;
	}
	mat->rowptr[prow] = ndata;
	if (rank == 0)
	{
	//	printf("%d\n", rank);
	for (i = 0; i < p; i++)
	{
	//	printf("%d\n", dspl[i]);
	}

//		printf("%d\n",dspl[0]);
//		printf("%d\n",dspl[1]);
//		printf("%d\n",dspl[2]);
//		printf("%d\n",dspl[3]);
	}
	MPI_Scatterv(mmat->rowind, nEachRow, dspl, MPI_INT, mat->rowind, ndata, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(mmat->rowval, nEachRow, dspl, MPI_FLOAT, mat->rowval, ndata, MPI_FLOAT, 0, MPI_COMM_WORLD);


	gk_csr_CreateIndex(mat, GK_CSR_COL);

	tneeded = 0;
	nbneeded = (int *) calloc(p, sizeof(int));
	for (i = 0; i < nb; i++)
	{
		if (mat->colptr[i + 1] - mat->colptr[i] > 0)
		{
			int spot = (i*p)/nb;
			nbneeded[spot]++;
			tneeded++;
		}
	}
//printf("tneeded = %d, from %d\n",tneeded, rank);
	bneeded = (int *) malloc(sizeof(int) * tneeded);
	cptrCum = (int *) malloc(sizeof(int) * tneeded + 1);
	count = 0;
	for (i = 0; i < nb; i++)
	{
		if (mat->colptr[i + 1] - mat->colptr[i] > 0)
		{
			bneeded[count] = i;
			cptrCum[count] = mat->colptr[i];
			count++;
		}
	}
	cptrCum[tneeded] = mat->colptr[nb];
	nbrecCum = (int *) malloc(sizeof(int) * p + 1);
	nbrec = (int *) malloc(sizeof(int) * p);
	nbneededCum = (int *) malloc(sizeof(int) * p + 1);
	MPI_Alltoall(nbneeded, 1, MPI_INT, nbrec, 1, MPI_INT, MPI_COMM_WORLD);
	tsend = 0;
	nbrecCum[0] = 0;
	nbneededCum[0] = 0;
	for (i = 0; i < p; i++)
	{
		tsend += nbrec[i];
		nbrecCum[i+1] = nbrec[i];
		nbrecCum[i+1] += nbrecCum[i];
		nbneededCum[i+1] = nbneeded[i];
		nbneededCum[i+1] += nbneededCum[i];
	}

	toSend = (int *) malloc(sizeof(int) * tsend);
	toSendBack = (float *) malloc(sizeof(float) * tsend);
	toRec = (float *) malloc(sizeof(float) * tneeded);



	for (i = 0; i < p; i++)
	{
		MPI_Scatterv(bneeded, nbneeded, nbneededCum, MPI_INT, toSend + nbrecCum[i], nbrec[i], MPI_INT, i, MPI_COMM_WORLD);
	}

	rowOffset = rank*nb/p;

	for (i = 0; i < tsend; i++)
	{
		toSendBack[i] = pb[toSend[i] - rowOffset];
	}



	for (i = 0; i < p; i++)
	{
		MPI_Scatterv(toSendBack, nbrec, nbrecCum, MPI_FLOAT, toRec + nbneededCum[i], nbneeded[i], MPI_FLOAT, i, MPI_COMM_WORLD);
	}
	for (i = 0; i < tneeded + 1; i++)
	{
//		printf("%d, from %d\n", cptrCum[i], rank);
	}
//printf("nb = %d, p = %d, from %d\n",nb, p, rank);
MPI_Barrier(MPI_COMM_WORLD);
	count = 0;
	float * result = (float *) malloc(nb/p * sizeof(float));
	for (i = 0; i < nb/p; i++)
	{
//		result[i] = 0.0;
	}
	for (i = 0; i < tneeded; i++)
	{
//		printf("%d from %d\n", mat->colptr[i], rank);
		int total = cptrCum[i+1] - cptrCum[i];
		float btemp = toRec[i];
//		printf("%f from %d\n", btemp, rank);
		int offset = cptrCum[i];
		for (j = 0; j < total; j++)
		{
			int rowtemp = mat->colind[j + offset];
			float valtemp = mat->colval[j + offset];
			printf("");
			result[rowtemp] += btemp * valtemp;
//			printf("col %d, row %d, val %f rank %d\n", i,  temp, mat->colval[j + offset], rank);
		}
	}



//	float * result = (float *) malloc(sizeof(float) * nb);
if (rank == 0)
{
	for (i = 0; i < ndata; i++)
	{
//		printf("col %d, val %f rank %d\n", mat->colind[i] ,mat->colval[i], rank);
	}
	for (i = 0; i <= nb; i++)
	{
//		printf("%d rank %d\n", mat->colptr[i], rank);

	//	float output = 0.0;
	//	int total = mat->rowptr[i+1] - mat->rowptr[i];
	//	int * startind = mat->rowind + mat->rowptr[i];
	//	float * startval = mat->rowval + mat->rowptr[i];
	//	for (j = 0; j < total; j++)
	//	{
//			output += b[startind[j]] * startval[j];
//			printf("%f : val %f\n", b[startind[j]] ,startval[j]);
//			printf("row %d, col %d, val %f rank %d\n", i,  startind[j] ,startval[j], rank);
	//	}
//		result[i] = output;
	}
}




	for (i = 0; i < nb/p; i++)
	{
		printf("%f\n", result[i]);
	}


/*
	for (i = 0; i < nb; i++)
	{
		int total = mat->rowptr[i+1] - mat->rowptr[i];
		int * startind = mat->rowind + mat->rowptr[i];
		float * startval = mat->rowval + mat->rowptr[i];
		for (j = 0; j < total; j++)
		{
			printf("row %d col %d : val %f\n",i, startind[j] ,startval[j]);
		}
	}
	printf("cols\n\n");
	for (i = 0; i < nb; i++)
	{
		int total = mat->colptr[i+1] - mat->colptr[i];
		int * startind = mat->colind + mat->colptr[i];
		float * startval = mat->colval + mat->colptr[i];
		for (j = 0; j < total; j++)
		{
			printf("col %d row %d : val %f\n",i, startind[j] ,startval[j]);
		}
	}
*/
	MPI_Finalize();
return 1;

    }

