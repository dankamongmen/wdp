/*
  In case you're wondering, dgemm stands for Double-precision, GEneral
  Matrix-Matrix multiplication.
*/

const char* dgemm_desc = "Simple blocked dgemm.";

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
do_block (const int lda,
          const double *A, const double *B, double *C,
          const int i, const int j, const int k)
{
     /*
       Remember that you need to deal with the fringes in each
       dimension.

       If the matrix is 7x7 and the blocks are 3x3, you'll have 1x3,
       3x1, and 1x1 fringe blocks.

             xxxoooX
             xxxoooX
             xxxoooX
             oooxxxO
             oooxxxO
             oooxxxO
             XXXOOOX

       You won't get this to go fast until you figure out a `better'
       way to handle the fringe blocks.  The better way will be more
       machine-efficient, but very programmer-inefficient.
     */
     const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
     const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
     const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);

     basic_dgemm (lda, M, N, K,
                  A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void
square_dgemm (const int M, 
              const double *A, const double *B, double *C)
{
     const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
     int bi, bj, bk;

     for (bi = 0; bi < n_blocks; ++bi) {
          const int i = bi * BLOCK_SIZE;
          
          for (bj = 0; bj < n_blocks; ++bj) {
               const int j = bj * BLOCK_SIZE;

               for (bk = 0; bk < n_blocks; ++bk) {
                    const int k = bk * BLOCK_SIZE;
                    
                    do_block (M, A, B, C, i, j, k);
               }
          }
     }
}

/* eof */
