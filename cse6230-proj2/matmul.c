#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>

#include <float.h>
#include <math.h>

#include "doyen.h"
#include "timing.h"

#if !defined(COMPILER)
#  define COMPILER "unknown"
#endif
#if !defined(FLAGS)
#  define FLAGS "unknown"
#endif

/*
  We try to run enough iterations to get reasonable timings.  The matrices
  are multiplied at least MIN_RUNS times.  If that doesn't take MIN_SECS
  seconds, then we double the number of iterations and try again.

  You may want to modify these to speed debugging...
*/
#define MIN_RUNS 8
#define MIN_SECS 1.0

/*
  Note the strange sizes...  You'll see some interesting effects
  around some of the powers-of-two.
*/
const int test_sizes[] = {
  1, 2, 8, 15, 16, 17, 18, 31, 32, 64, 69, 80, 96, 97, 112, 127,
  128, 129, 130, 144, 191, 192, 229,
#if defined(DEBUG_RUN)
# define MAX_SIZE 229u
#else
  255, 256, 257, 258, 260, 319, 320, 420, 479, 480, 511, 512, 528, 639, 640,
  767, 768, 777, 1024, 1911, 2000, 2048, 4000, 4096
# define MAX_SIZE 4096u
#endif
};

#define N_SIZES (sizeof (test_sizes) / sizeof (int))

static double A[MAX_SIZE * MAX_SIZE], B[MAX_SIZE * MAX_SIZE],
  C[MAX_SIZE * MAX_SIZE];

static void 
matrix_init(double *A){
	unsigned i;

	for(i = 0; i < MAX_SIZE*MAX_SIZE; ++i){
		A[i] = drand48 ();
	}
}

static void
matrix_clear(double *C){
	memset (C, 0, MAX_SIZE * MAX_SIZE * sizeof (double));
}

/*
  Dot products satisfy the following error bound:
  float(sum a_i * b_i) = sum a_i * b_i * (1 + delta_i)
  where delta_i <= n * epsilon.  In order to check your matrix
  multiply, we compute each element in term and make sure that
  your product is within three times the given error bound.
  We make it three times because there are three sources of
  error:

  - the roundoff error in your multiply
  - the roundoff error in our multiply
  - the roundoff error in computing the error bound

  That last source of error is not so significant, but that's a
  story for another day.
*/
static void
validate_dgemm (const int M,
                const double *A, const double *B, double *C){
  int i, j, k;

  matrix_clear (C);
  square_dgemm (M, A, B, C);

  for (i = 0; i < M; ++i) {
    for (j = 0; j < M; ++j) {

      double dotprod = 0;
      double errorbound = 0;
      double err;

      for (k = 0; k < M; ++k) {
	double prod = A[k*M + i] * B[j*M + k];
	dotprod += prod;
	errorbound += fabs(prod);
      }
      errorbound *= (M * FLT_EPSILON);

      err = fabs(C[j*M + i] - dotprod);
      if (err > 3*errorbound) {
	printf("Matrix multiply failed.\n");
	printf("C(%d,%d) should be %lg, was %lg\n", i, j,
	       dotprod, C[j*M + i]);
	printf("Error of %lg, acceptable limit %lg\n",
	       err, 3*errorbound);
	exit(EXIT_FAILURE);
      }
    }
  }
}

static double
time_dgemm (const int M, const double *A, const double *B, double *C){
  struct stopwatch_t* watch;
  double mflops, mflop_s;
  double secs = -1.0;

  int num_iterations = MIN_RUNS;
  int i;

  watch = stopwatch_create ();
  assert (watch);

  while (secs < MIN_SECS) {

    matrix_clear (C);
    stopwatch_start (watch);
    for (i = 0; i < num_iterations; ++i) {
      square_dgemm (M, A, B, C);
    }
    secs = stopwatch_stop (watch);

    mflops  = 2.0 * num_iterations * M * M * M / 1.0e6;
    mflop_s = mflops/secs;

    num_iterations *= 2;
  }

  stopwatch_destroy (watch);
  return mflop_s;
}

int main (void){
	double mflop_s;
	unsigned i;

	matrix_init (A);
	matrix_init (B);

	printf ("Compiler:\t%s %s\nOptions:\t%s\nDescription:\t%s\n\n",
		COMPILER, __VERSION__, FLAGS, dgemm_desc);
	fflush (stdout);

	stopwatch_init ();
	for (i = 0; i < N_SIZES; ++i) {
		const int M = test_sizes[i];

		#ifndef NO_VALIDATE
		validate_dgemm (M, A, B, C);
		#endif
		mflop_s = time_dgemm(M, A, B, C);    

		printf ("Size: %u \tmflop/s: %lg\n", M, mflop_s);
		fflush (stdout);
	}
	return 0;
}
