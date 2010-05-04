#ifndef HW2_KARMA
#define HW2_KARMA

#include <stdio.h>

// transposes A into AT. A is M*M, padded at orig * sizeof(*)-byte groups. AT
// is similarly M*M, but might enjoy different alignment (cur * sizeof(*)).
static inline void
transpose(const double * restrict A,double * restrict AT,const int orig,
		const int cur,const int M){
	int i;

	for(i = 0 ; i < M ; ++i){ // logical row of original
		int j;

		for(j = i ; j < M ; ++j){ // logical column of original
			typeof(*A) t = *(A + j * orig + i);

			*(AT + j * cur + i) = *(A + i * orig + j);
			*(AT + i * cur + j) = t;
		}
	}
}

// Takes two pointers to actual column-major matrix offsets, as opposed to
// two vectors -- ie, what we need to do for matmult. We're taking the row of
// 16 from A, and the column of 16 from B, to a single output in C. Since it's
// column-major, A is the one which skips by an offset. A ought be properly
// offset to the desired row, and B to the desired column. B's columns must
// each be 16-byte aligned for MOVAPD.
//
// The natural extension of matdp1x16x16x1() is to take the (expensively-
// loaded) row of A (expensively-loaded because it's stored as a column in
// DRAM, and thus requires scaler loads), and multiply it by more columns of
// B (which are stored as easily-loaded rows). This does mean we now affect
// multiple values of output matrix C with each operation, and thus cease
// returning a double (suitable for accumulation), instead writing to C
// ourselves. We're storing to N successive columns of C, thus a row, thus
// stored as a column, and thus also annoying to load (note we needn't load
// elements from C; simply store them back to memory directly, if it wins).

static inline void
matmul1x16x16x4(const double * restrict a,const double * restrict b,
		double * restrict c,const size_t off,const size_t clen){
	double t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15;
	const size_t skip = off * 2;

	//res = ((const double *)a)[0] * ((const double *)b)[0] + ((const double *)a)[1] * ((const double *)b)[1];
	asm (
		"movsd (%[a]), %[t15]\n\t"
		"movhps (%[a],%[off]), %[t15]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t1]\n\t"
		"movhps (%[a],%[off]), %[t1]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"dppd $0x31, %[t15], %[t0]\n\t"
		"dppd $0x31, %[t1], %[t2]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t3]\n\t"
		"movhps (%[a],%[off]), %[t3]\n\t"
		"movaps 0x20(%[b]), %[t4]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t5]\n\t"
		"movhps (%[a],%[off]), %[t5]\n\t"
		"movaps 0x30(%[b]), %[t6]\n\t"
		"dppd $0x31, %[t3], %[t4]\n\t"
		"dppd $0x31, %[t5], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t7]\n\t"
		"movhps (%[a],%[off]), %[t7]\n\t"
		"movaps 0x40(%[b]), %[t8]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t9]\n\t"
		"movhps (%[a],%[off]), %[t9]\n\t"
		"movaps 0x50(%[b]), %[t10]\n\t"
		"dppd $0x31, %[t7], %[t8]\n\t"
		"dppd $0x31, %[t9], %[t10]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t11]\n\t"
		"movhps (%[a],%[off]), %[t11]\n\t"
		"movaps 0x60(%[b]), %[t12]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t13]\n\t"
		"movhps (%[a],%[off]), %[t13]\n\t"
		"movaps 0x70(%[b]), %[t14]\n\t"
		"dppd $0x31, %[t11], %[t12]\n\t"
		"dppd $0x31, %[t13], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"addpd %[t14], %[t8]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addsd (%[c]), %[t0]\n\t"
		"movsd %[t0], (%[c])\n\t"

		// column [1]
		"add %[clen],%[b]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"dppd $0x31, %[t15], %[t0]\n\t"
		"dppd $0x31, %[t1], %[t2]\n\t"
		"movaps 0x20(%[b]), %[t4]\n\t"
		"movaps 0x30(%[b]), %[t6]\n\t"
		"dppd $0x31, %[t3], %[t4]\n\t"
		"dppd $0x31, %[t5], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"movaps 0x40(%[b]), %[t8]\n\t"
		"movaps 0x50(%[b]), %[t10]\n\t"
		"dppd $0x31, %[t7], %[t8]\n\t"
		"dppd $0x31, %[t9], %[t10]\n\t"
		"movaps 0x60(%[b]), %[t12]\n\t"
		"movaps 0x70(%[b]), %[t14]\n\t"
		"dppd $0x31, %[t11], %[t12]\n\t"
		"dppd $0x31, %[t13], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"addpd %[t14], %[t8]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addsd (%[c],%[off]), %[t0]\n\t"
		"movsd %[t0], (%[c],%[off])\n\t"

		"add %[skip],%[c]\n\t"

		// column [2]
		"add %[clen],%[b]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"dppd $0x31, %[t15], %[t0]\n\t"
		"dppd $0x31, %[t1], %[t2]\n\t"
		"movaps 0x20(%[b]), %[t4]\n\t"
		"movaps 0x30(%[b]), %[t6]\n\t"
		"dppd $0x31, %[t3], %[t4]\n\t"
		"dppd $0x31, %[t5], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"movaps 0x40(%[b]), %[t8]\n\t"
		"movaps 0x50(%[b]), %[t10]\n\t"
		"dppd $0x31, %[t7], %[t8]\n\t"
		"dppd $0x31, %[t9], %[t10]\n\t"
		"movaps 0x60(%[b]), %[t12]\n\t"
		"movaps 0x70(%[b]), %[t14]\n\t"
		"dppd $0x31, %[t11], %[t12]\n\t"
		"dppd $0x31, %[t13], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"addpd %[t14], %[t8]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addsd (%[c]), %[t0]\n\t"
		"movsd %[t0], (%[c])\n\t"

		// column [3]
		"add %[clen],%[b]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"dppd $0x31, %[t15], %[t0]\n\t"
		"dppd $0x31, %[t1], %[t2]\n\t"
		"movaps 0x20(%[b]), %[t4]\n\t"
		"movaps 0x30(%[b]), %[t6]\n\t"
		"dppd $0x31, %[t3], %[t4]\n\t"
		"dppd $0x31, %[t5], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"movaps 0x40(%[b]), %[t8]\n\t"
		"movaps 0x50(%[b]), %[t10]\n\t"
		"dppd $0x31, %[t7], %[t8]\n\t"
		"dppd $0x31, %[t9], %[t10]\n\t"
		"movaps 0x60(%[b]), %[t12]\n\t"
		"movaps 0x70(%[b]), %[t14]\n\t"
		"dppd $0x31, %[t11], %[t12]\n\t"
		"dppd $0x31, %[t13], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"addpd %[t14], %[t8]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addsd (%[c],%[off]), %[t0]\n\t"
		"movsd %[t0], (%[c],%[off])\n\t"
		
		//"add %[skip],%[c]\n\t"

		: [a] "=&g" (a), [c0] "+m" (*c), [b] "=&g" (b), [c] "+&g" (c), \
		 [t0] "=&x" (t0), [t1] "=&x" (t1), [t2] "=&x" (t2), [t3] "=&x" (t3), \
		 [t4] "=&x" (t4), [t5] "=&x" (t5), [t6] "=&x" (t6), [t7] "=&x" (t7), \
		 [t8] "=&x" (t8), [t9] "=&x" (t9), [t10] "=&x" (t10), [t11] "=&x" (t11), \
		 [t12] "=&x" (t12), [t13] "=&x" (t13), [t14] "=&x" (t14), [t15] "=&x" (t15)
		: "0" (a), "2" (b), [off] "g" (off), [skip] "g" (skip), [clen] "g" (clen)
	);
}

// version for transposed A and C (ie, they are row-major, B is still
// column-major)
static inline void
transmatmul1x16x16x4(const double * restrict a,const double * restrict b,
		double * restrict c,const size_t clen){
	double t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15;
	const size_t skip = clen * 2;

	//res = ((const double *)a)[0] * ((const double *)b)[0] + ((const double *)a)[1] * ((const double *)b)[1];
	asm (
		"movaps (%[a]), %[t1]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"movaps 0x40(%[a]), %[t9]\n\t"
		"movaps 0x40(%[b]), %[t8]\n\t"
		"mulpd %[t1], %[t0]\n\t" //"dppd $0x31, %[t1], %[t0]\n\t"
		"mulpd %[t9], %[t8]\n\t" //"dppd $0x31, %[t9], %[t8]\n\t"
		"movaps 0x10(%[a]), %[t3]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"movaps 0x20(%[a]), %[t5]\n\t"
		"movaps 0x20(%[b]), %[t4]\n\t"
		"mulpd %[t3], %[t2]\n\t" //"dppd $0x31, %[t3], %[t2]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"movaps 0x30(%[a]), %[t7]\n\t"
		"movaps 0x30(%[b]), %[t6]\n\t"
		"mulpd %[t5], %[t4]\n\t" //"dppd $0x31, %[t5], %[t4]\n\t"
		"mulpd %[t7], %[t6]\n\t" //"dppd $0x31, %[t7], %[t6]\n\t"
		"movaps 0x50(%[a]), %[t11]\n\t"
		"addpd %[t2], %[t4]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"movaps 0x50(%[b]), %[t10]\n\t"
		"movaps 0x60(%[a]), %[t13]\n\t"
		"mulpd %[t11], %[t10]\n\t" //"dppd $0x31, %[t11], %[t10]\n\t"
		"addpd %[t4], %[t0]\n\t"
		"movaps 0x60(%[b]), %[t12]\n\t"
		"mulpd %[t13], %[t12]\n\t" //"dppd $0x31, %[t13], %[t12]\n\t"
		"movaps 0x70(%[a]), %[t15]\n\t"
		"movaps 0x70(%[b]), %[t14]\n\t"
		"mulpd %[t15], %[t14]\n\t" //"dppd $0x31, %[t15], %[t14]\n\t"
		"addpd %[t10], %[t12]\n\t"
		"movaps 0x10(%[b],%[clen]), %[t2]\n\t"
		"mulpd %[t3], %[t2]\n\t" // "dppd $0x31, %[t3], %[t2]\n\t"
		"addpd %[t12], %[t0]\n\t"
		"movaps 0x40(%[b],%[clen]), %[t8]\n\t"
		"mulpd %[t9], %[t8]\n\t" //"dppd $0x31, %[t9], %[t8]\n\t"
		"addpd %[t14], %[t0]\n\t"
		"haddpd %[t0], %[t0]\n\t"
		"addsd (%[c]), %[t0]\n\t"
		"movsd %[t0], (%[c])\n\t"

		// column [1] ([t2], [t8] have already been done)
		"movaps (%[b],%[clen]), %[t0]\n\t"
		"mulpd %[t1], %[t0]\n\t" //"dppd $0x31, %[t1], %[t0]\n\t"
		"movaps 0x20(%[b],%[clen]), %[t4]\n\t"
		"mulpd %[t5], %[t4]\n\t" //"dppd $0x31, %[t5], %[t4]\n\t"
		"movaps 0x30(%[b],%[clen]), %[t6]\n\t"
		"mulpd %[t7], %[t6]\n\t" //"dppd $0x31, %[t7], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"
		"movaps 0x10(%[b],%[skip]), %[t2]\n\t"
		"movaps 0x50(%[b],%[clen]), %[t10]\n\t"
		"mulpd %[t11], %[t10]\n\t" //"dppd $0x31, %[t11], %[t10]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"movaps 0x60(%[b],%[clen]), %[t12]\n\t"
		"movaps 0x70(%[b],%[clen]), %[t14]\n\t"
		"mulpd %[t13], %[t12]\n\t" //"dppd $0x31, %[t13], %[t12]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"mulpd %[t15], %[t14]\n\t" //"dppd $0x31, %[t15], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"movaps 0x50(%[b],%[skip]), %[t10]\n\t"
		"mulpd %[t3], %[t2]\n\t" // "dppd $0x31, %[t3], %[t2]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"mulpd %[t11], %[t10]\n\t" //"dppd $0x31, %[t11], %[t10]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addpd %[t14], %[t0]\n\t"
		"haddpd %[t0], %[t0]\n\t"
		"addsd (%[c],%[clen]), %[t0]\n\t"
		"movsd %[t0], (%[c],%[clen])\n\t"

		"add %[skip],%[c]\n\t"

		// column [2] ([t2], [t10] have already been done)
		"movaps (%[b],%[skip]), %[t0]\n\t"
		"mulpd %[t1], %[t0]\n\t" //"dppd $0x31, %[t1], %[t0]\n\t"
		"movaps 0x40(%[b],%[skip]), %[t8]\n\t"
		"mulpd %[t9], %[t8]\n\t" //"dppd $0x31, %[t9], %[t8]\n\t"
		"movaps 0x20(%[b],%[skip]), %[t4]\n\t"
		"mulpd %[t5], %[t4]\n\t" //"dppd $0x31, %[t5], %[t4]\n\t"
		"movaps 0x30(%[b],%[skip]), %[t6]\n\t"
		"mulpd %[t7], %[t6]\n\t" //"dppd $0x31, %[t7], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"movaps 0x60(%[b],%[skip]), %[t12]\n\t"
		"movaps 0x70(%[b],%[skip]), %[t14]\n\t"
		"mulpd %[t13], %[t12]\n\t" //"dppd $0x31, %[t13], %[t12]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"mulpd %[t15], %[t14]\n\t" //"dppd $0x31, %[t15], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addpd %[t14], %[t0]\n\t"
		"haddpd %[t0], %[t0]\n\t"
		"addsd (%[c]), %[t0]\n\t"
		"movsd %[t0], (%[c])\n\t"
		
		"add %[skip],%[b]\n\t"
		
		// column [3]
		"movaps (%[b],%[clen]), %[t0]\n\t"
		"mulpd %[t1], %[t0]\n\t" //"dppd $0x31, %[t1], %[t0]\n\t"
		"movaps 0x40(%[b],%[clen]), %[t8]\n\t"
		"mulpd %[t9], %[t8]\n\t" //"dppd $0x31, %[t9], %[t8]\n\t"
		"movaps 0x10(%[b],%[clen]), %[t2]\n\t"
		"mulpd %[t3], %[t2]\n\t" // "dppd $0x31, %[t3], %[t2]\n\t"
		"movaps 0x20(%[b],%[clen]), %[t4]\n\t"
		"mulpd %[t5], %[t4]\n\t" //"dppd $0x31, %[t5], %[t4]\n\t"
		"movaps 0x30(%[b],%[clen]), %[t6]\n\t"
		"mulpd %[t7], %[t6]\n\t" //"dppd $0x31, %[t7], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"
		"movaps 0x50(%[b],%[clen]), %[t10]\n\t"
		"mulpd %[t11], %[t10]\n\t" //"dppd $0x31, %[t11], %[t10]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"movaps 0x60(%[b],%[clen]), %[t12]\n\t"
		"movaps 0x70(%[b],%[clen]), %[t14]\n\t"
		"mulpd %[t13], %[t12]\n\t" //"dppd $0x31, %[t13], %[t12]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"mulpd %[t15], %[t14]\n\t" //"dppd $0x31, %[t15], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addpd %[t14], %[t0]\n\t"
		"haddpd %[t0], %[t0]\n\t"
		"addsd (%[c],%[clen]), %[t0]\n\t"
		"movsd %[t0], (%[c],%[clen])\n\t"
		
		: [a] "=&g" (a), [c0] "+m" (*c), [b] "=&g" (b), [c] "+&g" (c), \
		 [t0] "=&x" (t0), [t1] "=&x" (t1), [t2] "=&x" (t2), [t3] "=&x" (t3), \
		 [t4] "=&x" (t4), [t5] "=&x" (t5), [t6] "=&x" (t6), [t7] "=&x" (t7), \
		 [t8] "=&x" (t8), [t9] "=&x" (t9), [t10] "=&x" (t10), [t11] "=&x" (t11), \
		 [t12] "=&x" (t12), [t13] "=&x" (t13), [t14] "=&x" (t14), [t15] "=&x" (t15)
		: "0" (a), "2" (b), [skip] "g" (skip), [clen] "g" (clen)
	);
}
static inline void
transmatmul1x16x16x2(const double * restrict a,const double * restrict b,
		double * restrict c,const size_t clen){
	double t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15;

	//res = ((const double *)a)[0] * ((const double *)b)[0] + ((const double *)a)[1] * ((const double *)b)[1];
	asm (
		"movaps (%[a]), %[t1]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"movaps 0x40(%[a]), %[t9]\n\t"
		"movaps 0x40(%[b]), %[t8]\n\t"
		"mulpd %[t1], %[t0]\n\t" //"dppd $0x31, %[t1], %[t0]\n\t"
		"mulpd %[t9], %[t8]\n\t" //"dppd $0x31, %[t9], %[t8]\n\t"
		"movaps 0x10(%[a]), %[t3]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"movaps 0x20(%[a]), %[t5]\n\t"
		"movaps 0x20(%[b]), %[t4]\n\t"
		"mulpd %[t3], %[t2]\n\t" //"dppd $0x31, %[t3], %[t2]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"movaps 0x30(%[a]), %[t7]\n\t"
		"movaps 0x30(%[b]), %[t6]\n\t"
		"mulpd %[t5], %[t4]\n\t" //"dppd $0x31, %[t5], %[t4]\n\t"
		"mulpd %[t7], %[t6]\n\t" //"dppd $0x31, %[t7], %[t6]\n\t"
		"movaps 0x50(%[a]), %[t11]\n\t"
		"addpd %[t2], %[t4]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"movaps 0x50(%[b]), %[t10]\n\t"
		"movaps 0x60(%[a]), %[t13]\n\t"
		"mulpd %[t11], %[t10]\n\t" //"dppd $0x31, %[t11], %[t10]\n\t"
		"addpd %[t4], %[t0]\n\t"
		"movaps 0x60(%[b]), %[t12]\n\t"
		"mulpd %[t13], %[t12]\n\t" //"dppd $0x31, %[t13], %[t12]\n\t"
		"movaps 0x70(%[a]), %[t15]\n\t"
		"movaps 0x70(%[b]), %[t14]\n\t"
		"mulpd %[t15], %[t14]\n\t" //"dppd $0x31, %[t15], %[t14]\n\t"
		"addpd %[t10], %[t12]\n\t"
		"addpd %[t12], %[t0]\n\t"
		"addpd %[t14], %[t0]\n\t"
		"haddpd %[t0], %[t0]\n\t"
		"addsd (%[c]), %[t0]\n\t"
		"movsd %[t0], (%[c])\n\t"

		"movaps (%[b],%[clen]), %[t0]\n\t"
		"mulpd %[t1], %[t0]\n\t" //"dppd $0x31, %[t1], %[t0]\n\t"
		"movaps 0x40(%[b],%[clen]), %[t8]\n\t"
		"mulpd %[t9], %[t8]\n\t" //"dppd $0x31, %[t9], %[t8]\n\t"
		"movaps 0x10(%[b],%[clen]), %[t2]\n\t"
		"mulpd %[t3], %[t2]\n\t" // "dppd $0x31, %[t3], %[t2]\n\t"
		"movaps 0x20(%[b],%[clen]), %[t4]\n\t"
		"mulpd %[t5], %[t4]\n\t" //"dppd $0x31, %[t5], %[t4]\n\t"
		"movaps 0x30(%[b],%[clen]), %[t6]\n\t"
		"mulpd %[t7], %[t6]\n\t" //"dppd $0x31, %[t7], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"

		"movaps 0x50(%[b],%[clen]), %[t10]\n\t"
		"mulpd %[t11], %[t10]\n\t" //"dppd $0x31, %[t11], %[t10]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"movaps 0x60(%[b],%[clen]), %[t12]\n\t"
		"movaps 0x70(%[b],%[clen]), %[t14]\n\t"
		"mulpd %[t13], %[t12]\n\t" //"dppd $0x31, %[t13], %[t12]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"mulpd %[t15], %[t14]\n\t" //"dppd $0x31, %[t15], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addpd %[t14], %[t0]\n\t"
		"haddpd %[t0], %[t0]\n\t"
		"addsd (%[c],%[clen]), %[t0]\n\t"
		"movsd %[t0], (%[c],%[clen])\n\t"
		: [a] "=&g" (a), [c0] "+m" (*c), [b] "=&g" (b), [c] "+&g" (c), \
		 [t0] "=&x" (t0), [t1] "=&x" (t1), [t2] "=&x" (t2), [t3] "=&x" (t3), \
		 [t4] "=&x" (t4), [t5] "=&x" (t5), [t6] "=&x" (t6), [t7] "=&x" (t7), \
		 [t8] "=&x" (t8), [t9] "=&x" (t9), [t10] "=&x" (t10), [t11] "=&x" (t11), \
		 [t12] "=&x" (t12), [t13] "=&x" (t13), [t14] "=&x" (t14), [t15] "=&x" (t15)
		: "0" (a), "2" (b), [clen] "g" (clen)
	);
}

static inline void
matmul1x16x16x2(const double * restrict a,const double * restrict b,
		double * restrict c,const size_t off,const size_t clen){
	double t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15;
	const size_t skip = off * 2;

	//res = ((const double *)a)[0] * ((const double *)b)[0] + ((const double *)a)[1] * ((const double *)b)[1];
	asm (
		"movsd (%[a]), %[t15]\n\t"
		"movhps (%[a],%[off]), %[t15]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t1]\n\t"
		"movhps (%[a],%[off]), %[t1]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"dppd $0x31, %[t15], %[t0]\n\t"
		"dppd $0x31, %[t1], %[t2]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t3]\n\t"
		"movhps (%[a],%[off]), %[t3]\n\t"
		"movaps 0x20(%[b]), %[t4]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t5]\n\t"
		"movhps (%[a],%[off]), %[t5]\n\t"
		"movaps 0x30(%[b]), %[t6]\n\t"
		"dppd $0x31, %[t3], %[t4]\n\t"
		"dppd $0x31, %[t5], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t7]\n\t"
		"movhps (%[a],%[off]), %[t7]\n\t"
		"movaps 0x40(%[b]), %[t8]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t9]\n\t"
		"movhps (%[a],%[off]), %[t9]\n\t"
		"movaps 0x50(%[b]), %[t10]\n\t"
		"dppd $0x31, %[t7], %[t8]\n\t"
		"dppd $0x31, %[t9], %[t10]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t11]\n\t"
		"movhps (%[a],%[off]), %[t11]\n\t"
		"movaps 0x60(%[b]), %[t12]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t13]\n\t"
		"movhps (%[a],%[off]), %[t13]\n\t"
		"movaps 0x70(%[b]), %[t14]\n\t"
		"dppd $0x31, %[t11], %[t12]\n\t"
		"dppd $0x31, %[t13], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"addpd %[t14], %[t8]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addsd %[c], %[t0]\n\t"
		"movsd %[t0], %[c]\n\t"

		"add %[clen],%[b]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"dppd $0x31, %[t15], %[t0]\n\t"
		"dppd $0x31, %[t1], %[t2]\n\t"
		"movaps 0x20(%[b]), %[t4]\n\t"
		"movaps 0x30(%[b]), %[t6]\n\t"
		"dppd $0x31, %[t3], %[t4]\n\t"
		"dppd $0x31, %[t5], %[t6]\n\t"
		"addpd %[t2], %[t0]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"addpd %[t6], %[t0]\n\t"
		"movaps 0x40(%[b]), %[t8]\n\t"
		"movaps 0x50(%[b]), %[t10]\n\t"
		"dppd $0x31, %[t7], %[t8]\n\t"
		"dppd $0x31, %[t9], %[t10]\n\t"
		"movaps 0x60(%[b]), %[t12]\n\t"
		"movaps 0x70(%[b]), %[t14]\n\t"
		"dppd $0x31, %[t11], %[t12]\n\t"
		"dppd $0x31, %[t13], %[t14]\n\t"
		"addpd %[t10], %[t8]\n\t"
		"addpd %[t12], %[t14]\n\t"
		"addpd %[t14], %[t8]\n\t"
		"addpd %[t8], %[t0]\n\t"
		"addsd %[c1], %[t0]\n\t"
		"movsd %[t0], %[c1]\n\t"
		: [a] "=&g" (a), [c] "+m" (*c), [b] "=&g" (b), [c1] "+m" (*(c + off / sizeof(*c))), \
		 [t0] "=&x" (t0), [t1] "=&x" (t1), [t2] "=&x" (t2), [t3] "=&x" (t3), \
		 [t4] "=&x" (t4), [t5] "=&x" (t5), [t6] "=&x" (t6), [t7] "=&x" (t7), \
		 [t8] "=&x" (t8), [t9] "=&x" (t9), [t10] "=&x" (t10), [t11] "=&x" (t11), \
		 [t12] "=&x" (t12), [t13] "=&x" (t13), [t14] "=&x" (t14), [t15] "=&x" (t15)
		: "0" (a), "2" (b), [off] "g" (off), [skip] "g" (skip), [clen] "g" (clen)
	);
}

static inline double
matdp1x16x16x1(const double * restrict a,const double * restrict b,
		const size_t off){
	double res,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14;
	const size_t skip = off * 2;

	//res = ((const double *)a)[0] * ((const double *)b)[0] + ((const double *)a)[1] * ((const double *)b)[1];
	asm (
		"movsd (%[a]), %[res]\n\t"
		"movhps (%[a],%[off]), %[res]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t1]\n\t"
		"movhps (%[a],%[off]), %[t1]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"dppd $0x31, %[t0], %[res]\n\t"
		"dppd $0x31, %[t1], %[t2]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t3]\n\t"
		"movhps (%[a],%[off]), %[t3]\n\t"
		"movaps 0x20(%[b]), %[t4]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t5]\n\t"
		"movhps (%[a],%[off]), %[t5]\n\t"
		"movaps 0x30(%[b]), %[t6]\n\t"
		"dppd $0x31, %[t3], %[t4]\n\t"
		"dppd $0x31, %[t5], %[t6]\n\t"
		"addpd %[t2], %[res]\n\t"
		"addpd %[t4], %[t6]\n\t"
		"addpd %[t6], %[res]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t7]\n\t"
		"movhps (%[a],%[off]), %[t7]\n\t"
		"movaps 0x40(%[b]), %[t8]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t9]\n\t"
		"movhps (%[a],%[off]), %[t9]\n\t"
		"movaps 0x50(%[b]), %[t10]\n\t"
		"dppd $0x31, %[t8], %[t7]\n\t"
		"dppd $0x31, %[t10], %[t9]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t11]\n\t"
		"movhps (%[a],%[off]), %[t11]\n\t"
		"movaps 0x60(%[b]), %[t12]\n\t"
		"add %[skip],%[a]\n\t"
		"movsd (%[a]), %[t13]\n\t"
		"movhps (%[a],%[off]), %[t13]\n\t"
		"movaps 0x70(%[b]), %[t14]\n\t"
		"dppd $0x31, %[t12], %[t11]\n\t"
		"dppd $0x31, %[t14], %[t13]\n\t"
		"addpd %[t9], %[t7]\n\t"
		"addpd %[t11], %[t13]\n\t"
		"addpd %[t13], %[t7]\n\t"
		"addpd %[t7], %[res]\n\t"
		: [a] "=&g" (a), [res] "=&x" (res), [t0] "=&x" (t0), [t1] "=&x" (t1), [t2] "=&x" (t2), \
			[t3] "=&x" (t3), [t4] "=&x" (t4), [t5] "=&x" (t5), [t6] "=&x" (t6), \
			[t7] "=&x" (t7), [t8] "=&x" (t8), [t9] "=&x" (t9), [t10] "=&x" (t10), \
			[t11] "=&x" (t11), [t12] "=&x" (t12), [t13] "=&x" (t13), [t14] "=&x" (t14)
		: "0" (a), [b] "r" (b), [off] "r" (off), [skip] "r" (skip));
	return res;
}

static inline double
matdp1x4x4x1(const double * restrict A,const double * restrict B,
		const size_t skip){
	double res,t0,t1,t2;

	asm (
		"movsd (%[a]), %[res]\n\t"
		"movhps (%[a],%[skip]), %[res]\n\t"
		"movaps (%[b]), %[t0]\n\t"
		"add %[skip],%[a]\n\t"
		"add %[skip],%[a]\n\t"
		"dppd $0x31, %[t0], %[res]\n\t"
		"movsd (%[a]), %[t1]\n\t"
		"movhps (%[a],%[skip]), %[t1]\n\t"
		"movaps 0x10(%[b]), %[t2]\n\t"
		"dppd $0x31, %[t1], %[t2]\n\t"
		"addpd %[t2], %[res]\n\t"
		: [res] "=x" (res), [t0] "=x" (t0), [t1] "=x" (t1), [t2] "=x" (t2)
		: [a] "r" (A), [b] "r" (B), [skip] "r" (skip));
	return res;
}

static inline double
matdp1x2x2x1(const double * restrict A,const double * restrict B,
		const size_t skip){
	double res,t0;

	asm (
		"movsd (%[A]), %[t0]\n\t"
		"movhps (%[A],%[skip]), %[t0]\n\t"
		"movaps (%[B]), %[res]\n\t"
		"dppd $0x31, %[t0], %[res]\n\t"
		: [res] "=&x" (res), [t0] "=&x" (t0)
		: [A] "r" (A), [B] "r" (B), [skip] "r" (skip)
	);
	return res;
}

#endif
