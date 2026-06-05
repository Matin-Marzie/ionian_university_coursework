// In-place serial LU decomposition of a square matrix
// Compile with:  gcc -Wall -O2 1_lu_serial_cache_blocking.c -o 1_lu_serial_cache_blocking -DN=1056


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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


void serial_lu_cache_blocking(double *a) {

  int bs = TILESIZE;

    
    // For each main diagonal block md
    for (int md = 0; md < N; md += bs) {

        // 1: diagonal block
        for (int i = md; i < md+bs; i++){
            for (int j = md; j < i; j++){
                for (int k = md; k < j; k++){
                    a[i*N+j] -= a[i*N+k] * a[k*N+j];
                }
                a[i*N+j] /= a[j*N+j];
            }
        }
    
        // 2L blocks
        for (int ii = md+bs; ii<N; ii+=bs) {
            for(int i = ii; i < ii + bs; i++) {
                for (int j = md; j < md + bs; j++){
                    for (int k = md; k < j; k++){
                        a[i*N+j] -= a[i*N+k] * a[k*N+j];
                    }
                    a[i*N+j] /= a[j*N+j];
                }
          }
        }

        // 2U blocks
        for (int jj = md+bs; jj<N; jj+=bs) {
            for (int i = md; i < md + bs; i++) {
                for (int j = jj; j < jj + bs; j++) {
                    for (int k = md; k < i; k++){
                        a[i*N+j] -= a[i*N+k] * a[k*N+j];
                    }
                }
            }
        }

        // 3 Rest of the blocks
        for (int ii = md+bs; ii<N; ii+=bs) {
            for (int jj = md+bs; jj<N; jj+=bs){
                for (int i = ii; i < ii + bs; i++) {
                    for (int j = jj; j < jj + bs; j++) {
                        for (int k = md; k < md + bs; k++){
                            a[i*N+j] -= a[i*N+k] * a[k*N+j];
                        }
                    }
                }
            }
        }
    }

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
  serial_lu_cache_blocking(a);

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