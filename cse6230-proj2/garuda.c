#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include "garuda.h"

const char* dgemm_desc = "garuda dgemm.";

// Each row of A and C must be 16-byte aligned, since we use MOVAPD. B need
// only be 8-byte aligned.
static unsigned
resize_rows(const unsigned m,const double *a,const double *b,const double *c){
	if((unsigned long)a % 16){
		fprintf(stderr,"A wasn't properly aligned\n"); abort();
	}
	if((unsigned long)b % 8){
		fprintf(stderr,"B wasn't properly aligned\n"); abort();
	}
	if((unsigned long)c % 16){
		fprintf(stderr,"C wasn't properly aligned\n"); abort();
	}
	return m + m % 2;
}

static void
pad_rows(const double * restrict src,double * restrict dst,
		const unsigned oldc,const unsigned newc,const unsigned rows){
	size_t copylen = (oldc < newc ? oldc : newc) * sizeof(double);
	unsigned i;

	for(i = 0 ; i < rows ; ++i){
		memcpy(dst + (i * newc),src + (i * oldc),copylen);
	}
}

static double *movap_align(const double *,unsigned,unsigned,unsigned)
	__attribute__ ((malloc));

static double *
movap_align(const double * restrict mat,unsigned oldc,unsigned newc,unsigned rows){
	size_t s = rows * newc * sizeof(double),mask;
	double * restrict ret;

	if((mask = sysconf(_SC_PAGE_SIZE)) <= 0){ abort(); }
	if(s % mask){ size_t t = mask - 1; t ^= s; s += (t % mask) + 1; }
	if(s % mask){ abort(); }
	if((ret = mmap(NULL,s,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0)) == MAP_FAILED){ fprintf(stderr,"Couldn't mmap %zu\n",s); abort(); }
	pad_rows(mat,ret,oldc,newc,rows);
	return ret;
}

/*static double
dppd(const long double * restrict a,const long double * restrict b){
	double res,temp;

	//res = a[0] * b[0] + a[1] * b[1];
	asm (
		"movapd %2, %[res]\n\t"
		"movapd %3, %[temp]\n\t"
		"dppd $0x33, %[temp], %[res]\n\t"
		: [res] "=x" (res), [temp] "=x" (temp)
		: "m" (*a), "m" (*b));
	return res;
}

  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

// Multiplies a [1x2] by a [2x8] matrix, yielding a [1x8], using SSE4.1's DPPD
// instruction for massive throughput. We treat a [1x2] as the base case so as
// to store and load [A] and [C] using low-latency, high-throughput MOVAPD
// instructions; [A] and [C] thus must both be 16-byte aligned. We use a [2x8]
// [B] so as to block on the 16 XMM registers. Since [B] must be loaded using
// slower MOVSD/MOVHPD pairs, we only load each 2-row set once per [8x8] block.
/*
static void
simd12x28(const double * restrict A,const double * restrict B,
		double * restrict C,const int bstep){
	if((unsigned long)A % 16){
		fprintf(stderr,"A wasn't properly aligned\n");
		abort();
	}
	if((unsigned long)B % 8){
		fprintf(stderr,"C wasn't properly aligned\n");
		abort();
	}
	if((unsigned long)C % 16){
		fprintf(stderr,"C wasn't properly aligned\n");
		abort();
	}
	if(bstep <= 0){
		fprintf(stderr,"bstep was invalid\n");
		abort();
	}
	// FIXME
}

// Multiplies a [1x2] by a [2x8] matrix, yielding a [1x8]. The [2x8] matrix
// must already be set up in XMM[1,3,5,7,9,11,13,15], and will be preserved.
static void
resimd12x28(const double * restrict A,double * restrict C){
	if((unsigned long)A % 16){
		fprintf(stderr,"A wasn't properly aligned\n");
		abort();
	}
	if((unsigned long)C % 16){
		fprintf(stderr,"C wasn't properly aligned\n");
		abort();
	}
	// FIXME
}

// Multiplies an [8x2] by a [2x8] matrix, yielding an [8x8], keeping [B] in XMM
// registers throughout the operation ([A] and [C] can be loaded more quickly).
static void
simd82x28(const double * restrict A,const double * restrict B,
		double * restrict C,const int astep,const int bstep){
	simd12x28(A,B,C,bstep);
	resimd12x28(A + 1 * astep * sizeof(double),C + 1 * astep * sizeof(double));
	resimd12x28(A + 2 * astep * sizeof(double),C + 2 * astep * sizeof(double));
	resimd12x28(A + 3 * astep * sizeof(double),C + 3 * astep * sizeof(double));
	resimd12x28(A + 4 * astep * sizeof(double),C + 4 * astep * sizeof(double));
	resimd12x28(A + 5 * astep * sizeof(double),C + 5 * astep * sizeof(double));
	resimd12x28(A + 6 * astep * sizeof(double),C + 6 * astep * sizeof(double));
	resimd12x28(A + 7 * astep * sizeof(double),C + 7 * astep * sizeof(double));
}*/

// L0 valid for any SSE2+ machine (16 128-bit (16-byte) XMM registers), using
// various means to perform repeated SIMD dot products (DPPD in SSE4.1).
#define L0_BLOCK	16	// 16x16x8, 1Kb
// L1 valid for both my Core 2 Duo 6600 and Hogwarts's Xeon 5520's
#define L1_SIZE		32768
#define L1_LINE_SIZE	64
#define L1_ASSOC	8
#define L1_SETS		(L1_SIZE / (L1_ASSOC * L1_LINE_SIZE))	// 64
#define L1_BLOCK	32	// 32x32x8, 8Kb
// L2 valid only for Hogwarts's Xeon 5520's
#define L2_SIZE		(L1_SIZE * 8)				// I've got 4Mb
#define L2_LINE_SIZE	64
#define L2_ASSOC	8					// I've got 16
#define L2_SETS		(L2_SIZE / (L2_ASSOC * L2_LINE_SIZE))	// 512
#define L2_BLOCK	64	// 64x64x8, 32Kb (somewhat wasteful)
// L3 valid only for Hogwarts's Xeon 5520's (I haven't an L3)
#define L3_SIZE		(L2_SIZE * 32)
#define L3_BLOCK	512	// 512x512x8, 2Mb

static const int bsizes[] = { // Fractal blocking on N-hierarchy
#ifdef L3_BLOCK
	// L3_BLOCK,
#endif
	/*L2_BLOCK, L1_BLOCK, */L0_BLOCK };

// FIXME: this is accounting for (presented) column-major storage
//#define ADDRMAT(mat,row,col,step) ((mat) + ((col) * (step)) + (row))
#define ADDRMAT(mat,row,col,step) ((mat) + ((col) * (step)) + (row))

// Multiply two [8x8] matrices, placing the results in a new [8x8] matrix.
// Assuming every row in [A], [B] and [C] to be 64-byte-aligned, and a 64-byte
// cache line, and no L1 cache aliasing effects, everything will be in L1 cache
// by the end of the first simd82x28() call.
static void
simd8x8(const double * restrict A,const double * restrict B,
		double * restrict C,const int astep,const int bstep,
		const int rowa,const int colb,const int shared){
	int i,j,k;

	/*printf("a, b, c: %p %p %p\n",A,B,C);
	printf("astep, bstep: %d %d\n",astep,bstep);*/
	/*
	for(i = 0 ; i < L0_BLOCK ; ++i){ // row of A, first spec of c
		for(j = 0 ; j < L0_BLOCK ; ++j){ // col of B, second spec of C
			for(k = 0 ; k < L0_BLOCK ; ++k){ // row of B, col of A
				*ADDRMAT(C,i,j,astep) += 
				 *ADDRMAT(A,i,k,astep) * *ADDRMAT(B,k,j,bstep);
			}
		}
	}*/
				/*printf("A, B, C: %p %p %p\n",
						ADDRMAT(A,i,k,astep),
						ADDRMAT(B,k,j,bstep),
						ADDRMAT(C,i,j,astep));*/

	/*printf("rowa: %d colb: %d shared: %d\n",rowa,colb,shared);
	printf("I: %d J: %d\n",I,J);*/
	for(i = 0 ; i < rowa ; ++i){
		const double *Ai_ = A + i;

		for(j = 0 ; j < colb ; ++j){
			const double *B_j = B + j * bstep;
			double cij = *(C + j * astep + i);

			printf("calculating [%dx%d] (%p)\n",i,j,(C + j * astep + i));
			for(k = 0 ; k < shared ; ++k){
				cij += *(Ai_ + (k * astep)) * *(B_j + k);
			}
			*(C + j * astep + i) = cij;
		}
	}
	/*simd82x28(A,B,C,astep,bstep);
	simd82x28(A + 2,B + 2 * bstep,C + 2,astep,bstep);
	simd82x28(A + 4,B + 4 * bstep,C + 4,astep,bstep);
	simd82x28(A + 6,B + 6 * bstep,C + 6,astep,bstep);*/
}

/*static void
rsquare_dgemm_aligned(const int M,const int rowlen,
		const double * restrict A,const double * restrict B,
		double * restrict C,const int level,
		const int rowc,const int colc,const int shared){
	int r = 0; // row of A, row of C, [0..rowc)
	int rnow = bsizes[level];

	do{
		int rc = 0; // col of B, col of C, [0..colc)
		int rcnow = bsizes[level];

		if(r + rnow > rowc){
			rnow = rowc - r;
		}
		do{
			int c = 0; // row of B, col of A, [0..shared)
			int cnow = bsizes[level];

			if(rc + rcnow > colc){
				rcnow = colc - rc;
			}
			do{
				if(c + cnow > shared){
					cnow = shared - c;
				}
				if(level == sizeof(bsizes) / sizeof(*bsizes)){
					// printf("SIMD on [%p x %p]\n",A,B);
					simd8x8(A,B,C,rowlen,M);
				}else{
					// printf("blocking on [%d x %d]\n",rnow,cnow);
					rsquare_dgemm_aligned(M,rowlen,
							ADDRMAT(A,r,rc,rowlen),
							ADDRMAT(B,rc,c,M),
							ADDRMAT(C,r,c,rowlen),
							level + 1,rnow,rcnow,
							cnow);
				}
				c += cnow;
			}while(c != shared);
			rc += rcnow;
		}while(rc != colc);
		r += rnow;
	}while(r != rowc);
}*/

// Every row of A and C must be aligned to a 16-byte boundary (if necessary,
// by padding of rows; M is the number of actual data while rowlen is the
// number of logical data (rowlen >= M). Rows of B need be only 8-byte aligned.
// Recursion would be more natural to implement the multitiered blocking, but
// x86 processors hate it, so we simulate that by unwinding the bsizes "stack".
static void
square_dgemm_aligned_simp(const int M,const int rowlen,const double * restrict A,
			const double * restrict B,double * restrict C){
	const int bsize = L0_BLOCK;
	int i,j,k,wanti;

	wanti = bsize;
	for(i = 0 ; i < M ; i += bsize){ // row of A, row of B, row of C
		int wantj = bsize;

		if(M - i < wanti){
			wanti = M - i;
		}
		for(j = 0 ; j < M ; j += bsize){ // col of A
			if(M - j < wantj){
				wantj = M - j;
			}
			for(k = 0 ; k < M ; k += bsize){ // col of B, col of C
				// FIXME adjust pointers
				simd8x8(A,B,C,rowlen,M,wanti,wantj,bsize);
			}
		}
	}
}

// Every row of A and C must be aligned to a 16-byte boundary (if necessary,
// by padding of rows; M is the number of actual data while rowlen is the
// number of logical data (rowlen >= M). Rows of B need be only 8-byte aligned.
// Recursion would be more natural to implement the multitiered blocking, but
// x86 processors hate it, so we simulate that by unwinding the bsizes "stack".
/*static void
square_dgemm_aligned(const int M,const int rowlen,const double * restrict A,
			const double * restrict B,double * restrict C){
	struct { // emulate a recursion stack
		int axpos,aypos; // implies cxpos, cypos
		int bxpos,bypos;
		int axdst,aydst,bxdst,bydst;
	} bstack[sizeof(bsizes) / sizeof(*bsizes)];
	unsigned ld = 0,dims; // Hausdorffs, Minkowskis and Bouligands, oh my!

	// We might fit entirely within a given level of cache. If so, find the
	// smallest such level. We'll run at least one iteration there.
	while(ld < sizeof(bsizes) / sizeof(*bsizes) - 1 && M <= bsizes[ld + 1]){
		++ld;
	}
	dims = ld;
	bstack[ld].axpos = bstack[ld].aypos = 0;
       	bstack[ld].bxpos = bstack[ld].bypos = 0;
	bstack[ld].axdst = bstack[ld].aydst = M;
	bstack[ld].bxdst = bstack[ld].bydst = M;
	do{
		int byorig,axorig;

		axorig = bstack[dims].axpos;
		byorig = bstack[dims].bypos;
		do{
			do{
				// printf("DIMENSION: %d\n",dims); // FIXME currently FUBAR
				while(dims < sizeof(bsizes) / sizeof(*bsizes) - 1){
					// printf("DIMENSION: %d\n",dims); // FIXME currently FUBAR
					bstack[dims + 1].axpos = bstack[dims].axpos;
					bstack[dims + 1].aypos = bstack[dims].aypos;
					bstack[dims + 1].bxpos = bstack[dims].bxpos;
					bstack[dims + 1].bypos = bstack[dims].bypos;
					bstack[dims + 1].axdst = bstack[dims].axpos + bsizes[dims];
					bstack[dims + 1].aydst = bstack[dims].aypos + bsizes[dims];
					bstack[dims + 1].bxdst = bstack[dims].bxpos + bsizes[dims];
					bstack[dims + 1].bydst = bstack[dims].bypos + bsizes[dims];
					++dims;
				} // we're now blocked to 8x8 or smaller
				printf("a(x,y) b(x,y): %d,%d  %d,%d\n",bstack[dims].axpos,bstack[dims].aypos,
						bstack[dims].bxpos,bstack[dims].bypos);
				simd8x8(ADDRMAT(A,bstack[dims].axpos,bstack[dims].aypos,rowlen),
					ADDRMAT(B,bstack[dims].bxpos,bstack[dims].bypos,M),
					ADDRMAT(C,bstack[dims].axpos,bstack[dims].aypos,rowlen),
					rowlen,M);
				bstack[dims].axpos += bsizes[dims];
				bstack[dims].bypos += bsizes[dims];
			}while(bstack[dims].axpos < bstack[dims].axdst);
			bstack[dims].aypos += bsizes[dims];
			bstack[dims].bxpos += bsizes[dims];
			bstack[dims].bypos = byorig;
			bstack[dims].axpos = axorig;
		}while(bstack[dims].aypos < bstack[dims].aydst);
	}while(--dims > ld);
}*/

// B must be 8-byte aligned. While the restrict keyword is not used here, it is
// currently required that A, B and C not alias each other.
void square_dgemm(const int M, const double *A, const double *B, double *C){
	int rowlen;

	// printf("a, b, c: %p %p %p\n",A,B,C);
	if((rowlen = resize_rows(M,A,B,C)) != M){
		double * restrict newa = movap_align(A,M,rowlen,M);
		double * restrict newc = movap_align(C,M,rowlen,M);

		//square_dgemm_aligned(M,rowlen,newa,B,newc);
		//rsquare_dgemm_aligned(M,rowlen,newa,B,newc,0,M,M,M);
		square_dgemm_aligned_simp(M,rowlen,newa,B,newc);
		if(rowlen != M){
			pad_rows(newc,C,rowlen,M,M);
		}
		if(munmap(newc,M * rowlen * sizeof(double))){ abort(); }
		if(munmap(newa,M * rowlen * sizeof(double))){ abort(); }
	}else{
		//square_dgemm_aligned(M,rowlen,A,B,C);
		//rsquare_dgemm_aligned(M,rowlen,A,B,C,0,M,M,M);
		square_dgemm_aligned_simp(M,rowlen,A,B,C);
	}
}
