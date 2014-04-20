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


int parseInput(char ** a, gk_csr_t *mat, int n)
{
	int i, j, k;
	int row;
	int col;
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
				mat->rowptr[j + 1] = mat->rowptr[j];
			}
		}
		mat->rowptr[row]++;
		mat->rowind[i] = col;
		mat->rowval[i] = v;
		lastRow = row;
	}
	return 0;


}

int disData(gk_csr_t *mat, int nb, int p, int *ndata)
{
	int i, j;
	int start, end;
	int prow = nb/p;
	if (prow * p != nb)
	{
		printf("ERROR: rows do not evenly divide into processors\n");
		return -1;
	}
	int *nEachRow = (int *) malloc(sizeof(int) * p);
	for (i = 0; i < p, i++)
	{
		start = i * prow;
		end = i * prow + prow;
		int total = 0;
		for (j = start; j < end; j++)
		{
			int temp = mat->rowptr[j]-mat->rowptr[j - 1];
			total += temp;
		}
		nEachRow[i] = total;
	}
	MPI_Scatter(nEachRow, 1, MPI_INT, &ndata, 1, MPI_INT, 0);
}


int main(int argc, char** argv)
{
	int rank, p;
    	int n;
	int ndata;
	int nb;
    	char ** a;
    	float *b;
	gk_csr_t *mat;
	int i, j;
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
		mat = (gk_csr_t *) malloc(sizeof(gk_csr_t));
		n = numberOfLines(argv[1]);
		nb = numberOfLines(argv[2]);
		mat->nrows = nb;
		mat->ncols = nb;
		a = getDataA(argv[1], n);
		b = getDataB(argv[2], nb);
		mat->rowptr = (int *) malloc(sizeof(int) * nb);
		mat->rowval = (float *) malloc(sizeof(float) * n);
		mat->rowind = (int *) malloc(sizeof(int) * n);
		for (i = 0; i < nb; i++)
		{
			mat->rowptr[i] = 0;
		}

		if (parseInput(a, mat, n) == -1)
		{
			printf("ERROR: Failed to parse data.\n");
			exit(-1);
		}
		int start, end;
		int prow = nb/p;
		if (prow * p != nb)
		{
			printf("ERROR: rows do not evenly divide into processors\n");
			return -1;
		}
		int *nEachRow = (int *) malloc(sizeof(int) * p);
		for (i = 0; i < p, i++)
		{
			start = i * prow;
			end = i * prow + prow;
			int total = 0;
			for (j = start; j < end; j++)
			{
				int temp = mat->rowptr[j]-mat->rowptr[j - 1];
				total += temp;
			}
			nEachRow[i] = total;
		}
/*		if (disData(mat, nb, p, &ndata) == -1)
		{
			printf("ERROR: Failed to distribute data.\n");
			exit(-1);
		}
*/
	}
	MPI_Scatter(nEachRow, 1, MPI_INT, &ndata, 1, MPI_INT, 0);
	//distrubute data
	//find parts of b needed
	//get those
	//do this part differently
	float * result = (float *) malloc(sizeof(float) * nb);
	for (i = 0; i < mat->nrows; i++)
	{
		float output = 0.0;
		int total = mat->rowptr[i]-mat->rowptr[i - 1];
		int * startind = mat->rowind+mat->rowptr[i - 1];
		float * startval = mat->rowval+mat->rowptr[i - 1];
		for (j = 0; j < total; j++)
		{
			printf("i = %d, j = %d, val = %f, b = %f\n",i, j, startval[j], b[startind[j]]);
			output += b[startind[j]] * startval[j];
		}
		result[i] = output;
	}
	for (i = 0; i < nb; i++)
	{
		printf("%f\n", result[i]);
	}
	MPI_Finalize();
	return 1;

    }

