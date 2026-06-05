// In-place cache-blocked LU decomposition of a square matrix using OpenMP
// Compile with: gcc -Wall -O2 -fopenmp 3_parallel_block_lu.c -o 3_parallel_block_lu -DN=1056


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>

#include <sys/time.h>


#ifndef TILESIZE
#define TILESIZE 32
#endif



void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}



void serial_lu(double *a) {

  // For each diagonal element   
  for (int i=0;i<N;i++) {
  
    // Compute L elements
    for (int j=0;j<i;j++) {
      for (int k=0;k<j;k++) {
        a[i*N+j] -= a[i*N+k] * a[k*N+j];
      }
      a[i*N+j] /= a[j*N+j];
    }
    
    // Compute U elements
    for (int j=i;j<N;j++) {
      for (int k=0;k<i;k++) {
        a[i*N+j] -= a[i*N+k]*a[k*N+j];
      }
    }
  }

}


void parallel_lu_cache_blocking_openmp_tasks(double *a) {

  int bs = TILESIZE;

    #pragma omp parallel
    #pragma omp single
    {
        for (int md = 0; md < N; md += bs) {

            // 1: main diagonal block
            #pragma omp task depend(inout: a[md*N+md])
            {
                for (int i = md; i < md+bs; i++){
                    for (int j = md; j < i; j++){
                        for (int k = md; k < j; k++){
                            a[i*N+j] -= a[i*N+k] * a[k*N+j];
                        }
                        a[i*N+j] /= a[j*N+j];
                    }
                    for (int j = i; j < md+bs; j++){
                        for (int k = md; k < i; k++){
                            a[i*N+j] -= a[i*N+k] * a[k*N+j];
                        }
                    }
                }
            }

            // 2L blocks - one task per block
            for (int ii = md+bs; ii < N; ii += bs) {
                #pragma omp task depend(in: a[md*N+md]) depend(out: a[ii*N+md])
                {
                    for(int i = ii; i < ii + bs; i++) {
                        for (int j = md; j < md+bs; j++){
                            for (int k = md; k < j; k++){
                                a[i*N+j] -= a[i*N+k] * a[k*N+j];
                            }
                            a[i*N+j] /= a[j*N+j];
                        }
                    }
                }
            }

            // 2U blocks - one task per block
            for (int jj = md+bs; jj < N; jj += bs) {
                #pragma omp task depend(in: a[md*N+md]) depend(out: a[md*N+jj])
                {
                    for (int i = md; i < md+bs; i++) {
                        for (int j = jj; j < jj + bs; j++) {
                            for (int k = md; k < i; k++){
                                a[i*N+j] -= a[i*N+k] * a[k*N+j];
                            }
                        }
                    }
                }
            }

            // 3 Rest blocks - one task per block
            for (int ii = md+bs; ii < N; ii += bs) {
                for (int jj = md+bs; jj < N; jj += bs) {
                    #pragma omp task depend(in: a[ii*N+md], a[md*N+jj]) depend(out: a[ii*N+jj])
                    {
                        for (int i = ii; i < ii + bs; i++) {
                            for (int j = jj; j < jj + bs; j++) {
                                for (int k = md; k < md+bs; k++){
                                    a[i*N+j] -= a[i*N+k] * a[k*N+j];
                                }
                            }
                        }
                    }
                }
            }

        } // end of for md

    } // end of single + parallel
}


int main() {
double ts,te;

double *a,*acheck;

  // allocate input/output and check arrays
  a = (double *)malloc(N*N*sizeof(double));
  if (a==NULL) {
    exit(1);
  }
  acheck = (double *)malloc(N*N*sizeof(double));
  if (acheck==NULL) {
    free(a);
    exit(1);
  }
  
  // init elements of arrays
  for (int row=0;row<N;row++) {
    double sum = 0.0;
    for (int col=0;col<N;col++) {
      if (row!=col) {
        sum += a[row*N+col] = acheck[row*N+col] = (double)rand()/RAND_MAX;
      }
    }
    a[row*N+row] = acheck[row*N+row] = sum + 0.1;
  }

  // get starting time (double, seconds) 
  get_walltime(&ts);

  // workload (LU decomposition)
  parallel_lu_cache_blocking_openmp_tasks(a);

  // get ending time
  get_walltime(&te);

  // print computation time
  printf("Computation time = %f sec\n",(te-ts));

  // check result
  serial_lu(acheck);
  int success = 1;
  for (int i=0;i<N*N;i++) {
    if (fabs(a[i]-acheck[i]) > 0.1) {
      printf("Error at index %d: %f != %f\n", i, a[i], acheck[i]);
      success = 0;
      break;
    }
  }
  if (success==1) printf("Success!\n");
  else printf("Error!\n");
  
  // free arrays
  free(acheck); 
  free(a);
  
  return te-ts;
}