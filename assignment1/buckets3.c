#include <stdio.h>
#include <stdlib.h>

int indexx(int a, int b, int n)
{
    return a*n+b;
}


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


void countingSort(int a[], int n, int bit, int q)
{
	int c[2];
	int i = 0;
        int j;
	int *b = (int *) malloc(sizeof(int)*n);
        int *d = (int *) malloc(sizeof(int)*n);
        
        c[0] = 0;
        c[1] = 0;
	for (i = 0; i < q; i++)
        {
		b[i] = a[i];
		d[i] = ((a[i] >> bit) & 1);
                c[d[i]]++;
	}
	

	c[1] = c[1] + c[0];


	for (j = q - 1; j >= 0; j--)
	{
		a[c[d[j]]- 1] = b[j];
		c[d[j]]--;
	
		
	}
	free(b);
        free(d);
}

void radixSort(int a[], int count, int n, int start, int end)
{

	int i;
        
	for (i = start; i < end; i++)
	{
	
		countingSort(a, n, i, count);

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

int key(int i, int shift, int keep, int p) {
    int temp = (i >> shift);
    temp = temp & ~(~0 << keep);
    int divideBy = 256/p;
    temp = temp/divideBy;
  return temp;
}


int * getBuckets(int a[], int **cnt, int pass, int n, int p)
{
        int *array1 = malloc(p * n * sizeof(int *));
        int i;
        int j;
        int * counter = (int *) malloc(sizeof(int)*p);
	for(i = 0; i < p; i++)
        {
            counter[i] = 0;
            
            for (j = 0; j < n; j++)
            {
                array1[indexx(i,j,n)] = -1;
            }
        }
        for (i = 0; i < n; i++)
        {
            int tempKey = key(a[i], pass * 8, 8, p);
            array1[indexx(tempKey, counter[tempKey], n)] = a[i];
            counter[tempKey]++;
        }
        
/*        for (i = 0; i < p; i++)
        {
                printf("bucket %d \n", i);
                printf("number of spots %d\n\n", counter[i]);
                for (j = 0; j < counter[i]; j++)
                {
                        printf("%d\n", array1[indexx(i,j, n)]);
                }
                printf("\n");
        }
*/
        *cnt = counter;

        return array1;
	
}




int main(int argc, char** argv) {

        if (argc != 3)
	{
		printf("error: invalid arguments\n");
		exit(-1);
	}
	int n = numberOfLines(argv[1]);
	int *a = getData(argv[1], n) ;
	int p = atoi(argv[2]);
	int  i;
        int j;
        int k;
        int * temp;
        int spot;
        int * cnt;
        for (i = 0; i < 32/p; i++)
        {
            spot = 0;
            temp = getBuckets(a, &cnt, i, n, p);
            for (j = 0; j < p; j++)
            {
                radixSort(&temp[j*n], cnt[j], n, i*8, i*8 + 8);
                for (k = 0; k < cnt[j]; k++)
                {
                    a[k+spot] = temp[indexx(j,k, n)];
                    
                }
                spot += cnt[j];
                   
                
            }
        }
        for (i = 0; i < n; i++)
        {
            printf("%d\n", a[i]);
        }
        
}


