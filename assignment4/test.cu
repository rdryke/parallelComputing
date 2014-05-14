#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


struct linkedList
{
	int size;
	int done;
	int offset;
	struct linkedList * prev;
	struct linkedList * next;
};

typedef struct linkedList ll;

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
	char* line = (char *) malloc(100*sizeof(char));
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


__device__
int scan(int size, int *array)
{
	int i, j, k, step, nactive;
	int mytid = threadIdx.x
	int nthreads = blockDim.x
	int temp;
	__shared__ int sum;
	step = 1;
	for (nactive = (size>>3) ; nactive > 0; nactive>>=1)
	{
		__syncthreads();

		for (k = mytid; k<nactive; k+=nthreads)
		{
			i = step * (2 * k + 1) -1;
			j = i + step;

			array[j] += array[i];
		}
		step <<=1;
	}
	__syncthreads();
	if (mytid == 0)
	{
		sum = array[size - 1];
		array[size - 1] = 0;
	}
	__syncthreads();

	for (nactive = 1; nactive < size; nactive *= 2)
	{
		__syncthreads();
		step >>= 1;
		for (k = mytid; k < nactive; k += nthreads)
		{
			i = step * (2 * k + 1) - 1;
			j = i + step;
			temp = array[i];
			array[i] = array[j];
			array[j] += temp;
		}
	}
	__syncthreads();
	array[size] = sum;
	return sum;
}


__global__
void sort(int *a, int start, int size, int pivot)
{
//	a[threadIdx.x] += b[threadIdx.x];
	int blockStart = start + blockIdx.x * blockDim.x;
	int blockEnd = blockStart + size;
	int i, j;
	int index;
	int storeIndex = blockStart + threadIdx.x;
	int temp;
	int count = 0;
	int nthreads = blockDim.x
	for (i = blockStart; i < blockEnd; i += nthreads)
	{
		index = i + threadIdx.x;
		if (a[index] < pivot)
		{
			temp = a[index];
			a[index] = a[storeIndex];
			a[storeIndex] = temp;
			storeIndex += nthreads;
			count++;
		}
	}

}

__global__
void sortToCompletion(int *a, int start, int size, int nthreads)
{
//	a[threadIdx.x] += b[threadIdx.x];
}

int main(int argc, char ** argv)
{

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
	int  i;
	int nSubArrays = 1;
	int * sizeSubArray;
//	int offset;
	int currentSize;
//	int * currentSizeDevice;
//	int currentDone;
	int currentOffset;
//	int * currentOffsetDevice;
//	ll * doneArray;
//	int done;
	int pivot;
//	int *pivotDevice;
	int pivotSpot;

	cudaMalloc( (void**)&aDevice, asize );
	cudaMemcpy( aDevice, a, asize, cudaMemcpyHostToDevice );

        gettimeofday(&tv, &tz);
	time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

	ll * head;
	ll * current;
	ll * currentNext;
	ll * tail;
	head = (ll *) malloc(sizeof(ll));
	current = (ll *) malloc(sizeof(ll));
	currentNext = (ll *) malloc(sizeof(ll));
	tail = (ll *) malloc(sizeof(ll));
	head->size = n;
	head->done = 0;
	tail = head;
//	done = 1;


	while (current != NULL)
	{
		current = head;
		while (current != NULL)
		{
			currentNext = current->next;
			currentSize = current->size;
		//	currentDone = current->done;
			currentOffset = current->offset;
		//	done *= currentDone;

		//	if (currentDone == 1)
		//	{
		//		continue;
		//	}
		//	cudaMalloc( (void**)&currentSizeDevice, sizeof(int) );
		//	cudaMemcpy( currentSizeDevice, currentSize, sizeof(int), cudaMemcpyHostToDevice );
		//	cudaMalloc( (void**)&currentOffsetDevice, sizeof(int) );
		//	cudaMemcpy( currentOffsetDevice, currentOffset, sizeof(int), cudaMemcpyHostToDevice );

			if (currentSize < nPerBlock)
			{
				sortToCompletion<<<1, threads>>>(aDevice, currentOffset, currentSize, threads);
				//current->done == 1;
				if (current->prev != NULL && current->next != NULL)
				{
					current->prev->next = current->next;
					current->next->prev = current->prev;
				}
				else if (current->prev != NULL)
				{
					current->prev->next = NULL;
				}
				else if (current->next != NULL)
				{
					current->next->prev = NULL;
					head = head->next;
				}
				else
				{
					head = NULL;
				}
				current = current->next;
				nSubArrays--;
				continue;
			}

			pivotSpot = rand() % currentSize + currentOffset;
			pivot = aDevice[pivotSpot];
		//	cudaMalloc( (void**)&pivotDevice, sizeof(int) );
		//	cudaMemcpy( pivotDevice, *pivot, sizeof(int), cudaMemcpyHostToDevice );
			//figure out when (currentSize/nPerBlock) * nPerBlock != currentSize//
			//


			sort<<<currentSize/nPerBlock, threads>>>(aDevice, currentOffset, currentSize, pivot, threads);
			nSubArrays++;

			current = currentNext;
		}
	//	done = 1;
	}



/*
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
	cudaFree( ad );
	cudaFree( bd );

	printf("%s\n", a);

*/

	gettimeofday(&tv, &tz);
	time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	printf("Total execution time: %lf\n", time_end - time_start);
	return EXIT_SUCCESS;
}
