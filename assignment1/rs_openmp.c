#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>


FILE* file_open(char* filename) {
	FILE* fp = fopen(filename, "r");
	if(fp == NULL) {
		fprintf(stderr, "ERROR: while opening file %s, abort.\n", filename);
		exit(1);
	}
	return fp;
}

char* file_getline(char* buffer, FILE* fp) {
	buffer = fgets(buffer, 100, fp);
	return buffer;
}


int key(int number, int pass, int bits, int keep) {
    int temp = (number >> (pass * bits));
    temp = temp & (keep - 1);
  return temp;
}

void countingSort(int a[], int n, int bits, int pass, int p)
{
	int nbuckets = 1 << bits;
	int *c = (int *) malloc(sizeof(int)*nbuckets);
	int *c2 = (int *) malloc(sizeof(int)*nbuckets*p); 
	int i, j, temp, threadnum, start, end, startc, endc, offset;
	int *b = (int *) malloc(sizeof(int)*n);
	for (i = 0; i < nbuckets; i++)
	{
		c[i] = 0;
	}
	for (i = 0; i < nbuckets * p; i++)
	{
		c2[i] = 0;
	}
#pragma omp parallel num_threads(p) default(shared) private(i, temp, threadnum, start, end, startc, endc, offset)
{
	threadnum = omp_get_thread_num();
	start = (n/p)*threadnum;
	end = (n/p)*(threadnum + 1);
	startc = nbuckets*threadnum;
	endc = nbuckets*(threadnum + 1);
	offset = startc;
	for (i = start; i < end; i++)
	{
		b[i] = a[i];
		temp = key(a[i], pass, bits, nbuckets);
		c2[temp + offset] += 1;
	}

	for (i = startc + 1; i < endc; i++)
	{
		c2[i] += c2[i-1];
	}
	
}

	for (i = 0; i < nbuckets; i++)
	{
		for (j = 0; j < p; j++)
		{
			c[i] += c2[i + nbuckets * j];
		}
	}


	for (i = n-1; i >= 0; i--)
	{
		temp = key(b[i], pass, bits, nbuckets);
		a[c[temp]- 1] = b[i];
		c[temp]--;
	
		
	}
	free(b);
	free(c);

}

void radixSort(int a[], int n, int b, int p)
{

	int i;
	for (i = 0; i < 32/b; i++)
	{
	
		countingSort(a, n, b, i, p);

	}

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


int * getData(char * filename, int n)
{
	int *a = (int *) malloc(sizeof(int) * n);
	char* line = malloc(100*sizeof(char));
	FILE * file = file_open(filename);
	int count = 0;

	while((line = file_getline(line,file)) != NULL)
	{
		a[count] = atoi(line);
		count++;
	}
	fclose(file);
	free(line);
	return a;

}

int main(int argc, char* argv[])
{

	if (argc != 4)
	{
		printf("error: invalid arguments\n");
		exit(-1);
	}
	int n = numberOfLines(argv[1]);
	int *a = getData(argv[1], n) ;
	int b = atoi(argv[2]);
	int p = atoi(argv[3]);
	int  i;
	double time_start, time_end;
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv, &tz);
        time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

	radixSort(a,n,b,p);

	gettimeofday(&tv, &tz);
        time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
/*	for (i = 0; i < n; i++)
	{
		printf("%d\n", a[i]);
	}
*/
        printf("Total execution time: %lf\n", time_end - time_start);
	

	return 0;

}
