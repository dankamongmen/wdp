#include <stdio.h>
#include "doyen.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include "asm/karma.h"

const char* dgemm_desc = "doyen dgemm.";

#define L0_BLOCK 16

// by assuming square matrices as input to doyen (not necessarily to
// basic_gemm(), however), aligning to even-numbered M, and using only even
// blocking parameters, we can skip the off-by-1-case on all axes.
static void
basic_dgemm (const int lda, const double * restrict A,
		const double * restrict B, double * restrict C) {
	int i;

	//assert(!(K % 2)); assert(!(J % 2));
	for(i = 0 ; i < L0_BLOCK ; ++i){ // Replace with Duff's Device?? FIXME
		const double * restrict Ai_ = A + i * lda;
		int j;

		for(j = 0 ; j < L0_BLOCK ; j += 4){
			transmatmul1x16x16x4(Ai_,B + j*lda,(C + j * lda + i),lda * sizeof(double));
		}
	}
}

// BLOCKING PARAMETERS MUST BE MULTIPLES of 16 -- SEE COMMENT IN BASIC_DGEMM()!
//
// We want a multiple of L0_BLOCK, but not to overflow 32k L1 with a block
// active from both A and B: 32 * 32 * 17 = 17k maxes these constraints.
#define L1_BLOCK 32
// We want a multiple of L1_BLOCK, but not to overflow 256k L2 with a block
// active from both A and B: 96 * 96 * 17 = 153k maxes these constraints. Since
// we're not actually blocking L1 yet, we'd go ahead and use 112 instead, but
// we can't fit three 112x112 blocks into the pages of our TLB1. Use of 96x96
// means we can pair copy-and-pack operations with L2 blocking iterations.
#define L2_BLOCK 96
// Figure we can take a little over 3/4 of pages references for our own data
// (why?). 96x96 (current L2_BLOCK) is precisely 18 pages. 2 input blocks of
// 96x96, together with one output block of 96x96, occupy 54 pages, or %84.375
// of the available TLB1 references (we're assuming 4KB pages). T0_BLOCK is
// thus equivalent to L2_BLOCK.
#define BLOCKSIN(bsize,insize) \
	((insize) / (bsize) + ((insize) % (bsize) ? 1 : 0))

static void
copy_and_pack(const double * restrict src,const unsigned M,const unsigned N,
		double * restrict dst,const unsigned slda,const unsigned dlda){
	unsigned i,j;

	printf("copy-and-packing [%d,%d]\n",M,N);
	for(i = 0 ; i < L2_BLOCK ; ++i){
		const unsigned dbase = i * dlda;
		const unsigned sbase = i * slda;

		for(j = 0 ; j < L2_BLOCK ; ++j){
			dst[dbase + j] = (i < M && j < N) ? src[sbase + j] : 0;
		}
	}
	printf("packed %d elements from %d\n",L2_BLOCK * L2_BLOCK,M*N);
}

static void
copy_and_transpose(const double * restrict src,const unsigned M,const unsigned N,
		double * restrict dst,const unsigned slda,const unsigned dlda){
	unsigned i,j;

	printf("copy-and-transposing [%d,%d]\n",M,N);
	for(j = 0 ; j < L2_BLOCK ; ++j){
		const unsigned sbase = j * slda;
		const unsigned dbase = j * dlda;

		for(i = 0 ; i < L2_BLOCK ; ++i){
			dst[dbase + i] = (j < M && i < N) ? src[sbase + i] : 0;
		}
	}
	printf("packed %d elements from %d\n",L2_BLOCK * L2_BLOCK,M*N);
}

static void
square_dgemm_pack(const int M, const double * restrict A,
		const double * restrict B, double * restrict C,
		double * restrict AP,double * restrict BP,
		double * restrict CP){
 const int blocks = BLOCKSIN(L2_BLOCK,M);
 // The number of C output blocks in each {row, col} of our (square) toplevel
 // target matrix. Also the number of rows of B we iterate over to generate C,
 // and also the number of columns of A with which we do the same.
 int ci, cj, ck, i, j, k;

 for(ci = 0 ; ci < blocks ; ++ci){ // row of output C block, input A row
  // number of target blocks to process along a subblock row
  const int cI = ((ci + 1) * L2_BLOCK > M ? M - (ci * L2_BLOCK) : L2_BLOCK);
          
  for(cj = 0 ; cj < blocks ; ++cj){ // col of output C block, input B col
   // number of target blocks to process along a subblock column
   const int cJ = ((cj + 1) * L2_BLOCK > M ? M - (cj * L2_BLOCK) : L2_BLOCK);

   for(ck = 0 ; ck < blocks ; ++ck){ // input A col, input B row
    const int cK = (((ck + 1) * L2_BLOCK > M) ? (M - (ck * L2_BLOCK)) : L2_BLOCK);
    const int sbi = BLOCKSIN(L0_BLOCK,cI);

    printf("M: %d c(i,j,k): %d,%d,%d c(I,J,K): %d,%d,%d\n",M,ci,cj,ck,cI,cJ,cK);
    copy_and_transpose(C + ci * M + cj,cI,cJ,CP,M,L2_BLOCK);
    copy_and_pack(B + cj * M + ck,cK,cJ,BP,M,L2_BLOCK);
    copy_and_transpose(A + ci * M + ck,cJ,cK,AP,M,L2_BLOCK);
    for(i = 0 ; i < sbi ; ++i){
         const int sbj = BLOCKSIN(L0_BLOCK,cJ);
         
         for(j = 0 ; j < sbj ; ++j){
              const int sbk = BLOCKSIN(L0_BLOCK,cK);

              for(k = 0 ; k < sbk ; ++k){
       		//printf("ci: %d cj: %d ck: %d i: %d j: %d k: %d\n",ci,cj,ck,i,j,k);
       		//printf("blocks: %d sbi: %d sbj: %d sbk: %d\n",blocks,sbi,sbj,sbk);
		//printf("Evaluating: %dx%dx%d\n",M0,N0,K0);
    		printf("i,j,k: %d,%d,%d\n",i,j,k);
       		basic_dgemm (L2_BLOCK, AP + i * L0_BLOCK * L2_BLOCK + k * L0_BLOCK,
       			BP + j * L0_BLOCK * L2_BLOCK + k * L0_BLOCK,
			CP + i * L0_BLOCK * L2_BLOCK + j * L0_BLOCK);
              }
         }
    }
    copy_and_transpose(CP,cI,cJ,C + ci * M + cj,L2_BLOCK,M);
   }
  }
 }
}

static inline void *mmmap(const size_t) __attribute__ ((malloc));

static inline void *
mmmap(const size_t s){
	void *ret;

	ret = mmap(NULL,s,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
	if(ret == MAP_FAILED){
		fprintf(stderr,"Couldn't mmap %zu\n",s); abort(); }
	return ret;
}

static double *get_packing_area(const size_t) __attribute__ ((malloc));

static double *
get_packing_area(const size_t objsize){
	size_t pgsize,s;
	double *ret;

	// See note following L2_BLOCK definition, assumption of 4KB pages.
	s = L2_BLOCK * L2_BLOCK * objsize;
	if((pgsize = sysconf(_SC_PAGE_SIZE)) != 4096 || s % pgsize){
		fprintf(stderr,"Bad assumptions for TLB!\n"); abort(); }
	ret = mmmap(s);
	return ret;
}

void square_dgemm(const int M, const double * restrict A,
			const double * restrict B, double * restrict C){
	static double * restrict AP = MAP_FAILED,* restrict BP = MAP_FAILED;
	static double * restrict CP = MAP_FAILED;

	//printf("here we go! size=%d, asize=%d\n",M,asize);
	if(AP == MAP_FAILED){
		AP = get_packing_area(sizeof(*A));
		BP = get_packing_area(sizeof(*B));
		CP = get_packing_area(sizeof(*C));
	}
	square_dgemm_pack(M,A,B,C,AP,BP,CP);
}
