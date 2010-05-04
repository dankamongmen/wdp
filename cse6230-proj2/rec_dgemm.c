/*
  In case you're wondering, dgemm stands for Double-precision, GEneral
  Matrix-Matrix multiplication.
*/

const char* dgemm_desc = "Simple recursive dgemm.";

/* BLOCK_SIZE: You may want to change this value... */

/* First check is a preprocessor hack to test of "blank" BLOCK_SIZE definition */
#if defined(BLOCK_SIZE) && (!(BLOCK_SIZE+0))
#  undef BLOCK_SIZE
#endif

#if !defined(BLOCK_SIZE)
#  define BLOCK_SIZE ((int) 57)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

void
basic_dgemm (const int lda,
             const int M, const int N, const int K,
             const double *A, const double *B, double *C)
{
  int i, j, k;

  /*
    To optimize this, think about loop unrolling and software
    pipelining.  Hint:  For the majority of the matmuls, you
    know exactly how many iterations there are (the block size)...
  */

  for (i = 0; i < M; ++i) {
       const double *Ai_ = A + i;
       for (j = 0; j < N; ++j) {
            const double *B_j = B + j*lda;

            double cij = *(C + j*lda + i);

            for (k = 0; k < K; ++k) {
                 cij += *(Ai_ + k*lda) * *(B_j + k);
            }

            *(C + j*lda + i) = cij;
       }
  }
}

void
rec_dgemm (const int lda,
	   const int M, const int N, const int K,
	   const double* A, const double* B, double* C)
{
  if (M <= BLOCK_SIZE || N <= BLOCK_SIZE || K <= BLOCK_SIZE) {
    basic_dgemm (lda, M, N, K, A, B, C);
  } else { /* M, N, K > BLOCK_SIZE; split */
    const int M1 = M / 2;
    const int M2 = M - M1;
    const int N1 = N / 2;
    const int N2 = N - N1;
    const int K1 = K / 2;
    const int K2 = K - K1;

    const double* A11 = A;
    const double* A12 = A + K1*lda;
    const double* A21 = A + M1;
    const double* A22 = A + M1 + K1*lda;

    const double* B11 = B;
    const double* B12 = B + N1*lda;
    const double* B21 = B + K1;
    const double* B22 = B + K1 + N1*lda;

    double* C11 = C;
    double* C12 = C + N1*lda;
    double* C21 = C + M1;
    double* C22 = C + M1 + N1*lda;

    rec_dgemm (lda, M1, N1, K1, A11, B11, C11);
    rec_dgemm (lda, M1, N1, K2, A12, B21, C11);

    rec_dgemm (lda, M1, N2, K1, A11, B12, C12);
    rec_dgemm (lda, M1, N2, K2, A12, B22, C12);

    rec_dgemm (lda, M2, N1, K1, A21, B11, C21);
    rec_dgemm (lda, M2, N1, K2, A22, B21, C21);

    rec_dgemm (lda, M2, N2, K1, A21, B12, C22);
    rec_dgemm (lda, M2, N2, K2, A22, B22, C22);
  }
}

void
square_dgemm (const int M, 
              const double *A, const double *B, double *C)
{
  rec_dgemm (M, M, M, M, A, B, C);
}

/* eof */
