#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
 
const int N = 16; 
const int blocksize = 16; 


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

 
__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}
 
int main()
{
	int  i;
	double time_start, time_end;
        struct timeval tv;
        struct timezone tz;


	if (argc != 4)
	{
		printf("ERROR: Invalid arguments. Usage is ./cudaqsort fileToBeSorted elementsPerBlock threadsPerBlock\n");
		exit(-1);
	}
	int n = numberOfLines(argv[1]);
	int *a = getData(argv[1], n) ;
	int *aDevice; 
	const int asize = n*sizeof(int);

	int nPerBlock = atoi(argv[2]);
	int threads = atoi(argv[3]);

	cudaMalloc( (void**)&aDevice, asize ); 
	cudaMemcpy( aDevice, a, asize, cudaMemcpyHostToDevice ); 
	
        gettimeofday(&tv, &tz);

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
	cudaFree( ad );
	cudaFree( bd );
	
	printf("%s\n", a);
	return EXIT_SUCCESS;
}
