#include <stdio.h>
#include <stdlib.h>


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


void countingSort(int a[], int n, int bit)
{
	int c[2];
	int i;
	int *b = (int *) malloc(sizeof(int)*n);
	int d[n];
	for (i = 0; i < n; i++)
	{
		b[i] = a[i];
		d[i] = ((a[i] >> bit) & 1);
	}
	for (i = 0; i < 2; i++)
	{
		c[i] = 0;
	}
	for (i = 0; i < n; i++)
	{

		c[d[i]] += 1;
	}

	c[1] = c[1] + c[0];


	for (i = n-1; i >= 0; i--)
	{
		a[c[d[i]]- 1] = b[i];
		c[d[i]]--;
	
		
	}
	free(b);
}

void radixSort(int a[], int n, int b)
{

	int i;
	for (i = 0; i < b; i++)
	{
	
		countingSort(a, n, i);

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

	if (argc != 3)
	{
		printf("error: invalid arguments\n");
		exit(-1);
	}
	int n = numberOfLines(argv[1]);
	int *a = getData(argv[1], n) ;
	int b = atoi(argv[2]);
	int  i;
	radixSort(a,n,b);
	for (i = 0; i < n; i++)
	{
		printf("%d\n", a[i]);
	}

	return 0;

}
