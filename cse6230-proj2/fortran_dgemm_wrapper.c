#include <stdlib.h>
#include "cfortran.h"

const char* dgemm_desc = "Fortran dgemm.";

PROTOCCALLSFSUB4(SDGEMM,sdgemm,INT,DOUBLEV,DOUBLEV,DOUBLEV)
#define FORTRAN_DGEMM(M, A, B, C) CCALLSFSUB4(SDGEMM,sdgemm,INT,DOUBLEV,DOUBLEV,DOUBLEV,M,A,B,C)

void
square_dgemm (int M, double* A, double* B, double* C)
{
  FORTRAN_DGEMM(M,A,B,C);
}
