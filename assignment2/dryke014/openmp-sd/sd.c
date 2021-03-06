#ifndef NTHREADS
#define NTHREADS 8 
#endif

/*!
\file  main.c
\brief This file is the entry point for paragon's various components
 
\date   Started 11/27/09
\author George
\version\verbatim $Id: omp_main.c 9585 2011-03-18 16:51:51Z karypis $ \endverbatim
*/


#include "simdocs.h"

//#include <omp.h>

/*************************************************************************/
/*! This is the entry point for finding simlar patents */
/**************************************************************************/
int main(int argc, char *argv[])
{
  params_t params;
  int rc = EXIT_SUCCESS;

  cmdline_parse(&params, argc, argv);

  printf("********************************************************************************\n");
  printf("sd (%d.%d.%d) Copyright 2011, GK.\n", VER_MAJOR, VER_MINOR, VER_SUBMINOR);
  printf("  nnbrs=%d, minsim=%.2f\n",
      params.nnbrs, params.minsim);

  gk_clearwctimer(params.timer_global);
  gk_clearwctimer(params.timer_1);
  gk_clearwctimer(params.timer_2);
  gk_clearwctimer(params.timer_3);
  gk_clearwctimer(params.timer_4);

  gk_startwctimer(params.timer_global);

  ComputeNeighbors(&params);

  gk_stopwctimer(params.timer_global);

  printf("    wclock: %.2lfs\n", gk_getwctimer(params.timer_global));
  printf("    timer1: %.2lfs\n", gk_getwctimer(params.timer_1));
  printf("    timer2: %.2lfs\n", gk_getwctimer(params.timer_2));
  printf("    timer3: %.2lfs\n", gk_getwctimer(params.timer_3));
  printf("    timer4: %.2lfs\n", gk_getwctimer(params.timer_4));
  printf("********************************************************************************\n");

  exit(rc);
}


/*************************************************************************/
/*! Reads and computes the neighbors of each document */
/**************************************************************************/
void ComputeNeighbors(params_t *params)
{
  int i, j, nhits;
  gk_csr_t *mat;
  int32_t *marker;
  gk_fkv_t *hits, *cand;
  FILE *fpout;
printf("threads = %d\n", NTHREADS);
  printf("Reading data for %s...\n", params->infstem);

  mat = gk_csr_Read(params->infstem, GK_CSR_FMT_CSR, 1, 0);

  printf("#docs: %d, #nnz: %d.\n", mat->nrows, mat->rowptr[mat->nrows]);

  /* compact the column-space of the matrices */
  gk_csr_CompactColumns(mat);

  /* perform auxiliary normalizations/pre-computations based on similarity */
  gk_csr_Normalize(mat, GK_CSR_ROW, 2);

  /* create the inverted index */
  gk_csr_CreateIndex(mat, GK_CSR_COL);

  /* create the output file */
  fpout = (params->outfile ? gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") : NULL);
  gk_startwctimer(params->timer_1);
#pragma omp parallel num_threads(NTHREADS) default(shared) private(i, j, hits, marker, cand, nhits)
{
  /* allocate memory for the necessary working arrays */
  hits   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: hits");
  marker = gk_i32smalloc(mat->nrows, -1, "ComputeNeighbors: marker");
  cand   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: cand");


  /* find the best neighbors for each query document */

  nhits = 0;
  i = 0;
  j = 0;
#pragma omp for schedule(dynamic) nowait
  for (i=0; i<mat->nrows; i+=4) {
//   if (params->verbosity > 0)
//      printf("Working on query %7d\n", i);

    /* find the neighbors of the ith document */ 
    nhits = gk_csr_GetSimilarRows(mat, 
                 mat->rowptr[i+1]-mat->rowptr[i], 
                 mat->rowind+mat->rowptr[i], 
                 mat->rowval+mat->rowptr[i], 
                 GK_CSR_COS, params->nnbrs, params->minsim, hits, 
                 marker, cand, 1);

    /* write the results in the file */

    if (fpout) {
#pragma omp critical
{
      for (j=0; j<nhits; j++) 
	{

        	fprintf(fpout, "%8d %8zd %.3f\n", i, hits[j].val, hits[j].key);
	}
}
nhits = gk_csr_GetSimilarRows(mat, 
                 mat->rowptr[i+2]-mat->rowptr[i+1], 
                 mat->rowind+mat->rowptr[i + 1], 
                 mat->rowval+mat->rowptr[i + 1], 
                 GK_CSR_COS, params->nnbrs, params->minsim, hits, 
                 marker, cand, 0);

    /* write the results in the file */

    if (fpout) 
    {
#pragma omp critical
{
      for (j=0; j<nhits; j++) 
	{

        	fprintf(fpout, "%8d %8zd %.3f\n", i + 1, hits[j].val, hits[j].key);
	}
}
    }
nhits = gk_csr_GetSimilarRows(mat, 
                 mat->rowptr[i+3]-mat->rowptr[i+2], 
                 mat->rowind+mat->rowptr[i+2], 
                 mat->rowval+mat->rowptr[i+2], 
                 GK_CSR_COS, params->nnbrs, params->minsim, hits, 
                 marker, cand, 1);

    /* write the results in the file */

    if (fpout) {
#pragma omp critical
{
      for (j=0; j<nhits; j++) 
	{

        	fprintf(fpout, "%8d %8zd %.3f\n", i + 2, hits[j].val, hits[j].key);
	}
}
nhits = gk_csr_GetSimilarRows(mat, 
                 mat->rowptr[i+4]-mat->rowptr[i+3], 
                 mat->rowind+mat->rowptr[i + 3], 
                 mat->rowval+mat->rowptr[i + 3], 
                 GK_CSR_COS, params->nnbrs, params->minsim, hits, 
                 marker, cand, 0);

    /* write the results in the file */

    if (fpout) 
    {
#pragma omp critical
{
      for (j=0; j<nhits; j++) 
	{

        	fprintf(fpout, "%8d %8zd %.3f\n", i + 3, hits[j].val, hits[j].key);
	}
}
    }
}

}

}

  gk_free((void **)&hits, &marker, &cand, LTERM);
}

  /* cleanup and exit */
//#pragma omp barrier
//#pragma omp master
//{
  gk_stopwctimer(params->timer_1);
  if (fpout) gk_fclose(fpout);



  gk_csr_Free(&mat);
//}

  return;
}
