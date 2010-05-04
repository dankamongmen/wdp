#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <string.h>
#include "karma.h"
#include <sys/time.h>
#include <sys/signal.h>

static sigjmp_buf jmpbuf;

static void
invalid_opcode(int signum,struct siginfo *si __attribute__ ((unused)),
			void *v __attribute__ ((unused))){
	if(signum != SIGILL){
		exit(EXIT_FAILURE);
	}
	printf("Saw invalid opcode...");
	siglongjmp(jmpbuf,1);
}

static int
install_handler(int signum){
	struct sigaction sa;

	sa.sa_sigaction = invalid_opcode;
	if(sigaction(signum,&sa,NULL)){
		return -1;
	}
	return 0;
}

static int
benchmark_karma(unsigned s){
	double A[s*s],B[s*s],C[s*s],D[s*s];
	double E[s*s],F[s*s],G[s*s],H[s*s];
	double I[s*s];
	unsigned long delta,i;
	struct timeval t0,t1;
	double ret = 0.0;

	for(i = 0 ; i < sizeof(A) / sizeof(*A) ; ++i){
		A[i] = drand48();
		B[i] = drand48();
		C[i] = drand48();
		D[i] = drand48();
		E[i] = drand48();
		F[i] = drand48();
		G[i] = drand48();
		H[i] = drand48();
	}
	ret = 0.0;
	gettimeofday(&t0,NULL);
	for(i = 0 ; i < 1000000000 ; ++i){
		ret += A[0] * A[0]; ret += C[0] * C[0];
	       	ret += E[0] * E[0]; ret += G[0] * G[0];
		ret += B[0] * B[0]; ret += D[0] * D[0];
		ret += F[0] * F[0]; ret += H[0] * H[0];
	}
	gettimeofday(&t1,NULL);
	delta = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
	if(ret == 0 || delta == 0){ abort(); }
	printf("CMM: %lu usecs, %lu flops, %lu Mflops/s\n",
			delta,i * 8,i * 8 / delta);
	ret = 0.0;
	if(sigsetjmp(jmpbuf,1) == 0){
		printf("Benchmarking matmulX()...\n");
		memset(I,0,sizeof(I));
		gettimeofday(&t0,NULL);
		for(i = 0 ; i < 100000000 ; ++i){
			matmul1x16x16x2(A,B,I,sizeof(*A) * s,sizeof(*A) * s);
			matmul1x16x16x2(C,D,I,sizeof(*A) * s,sizeof(*A) * s);
			matmul1x16x16x2(E,F,I,sizeof(*A) * s,sizeof(*A) * s);
			matmul1x16x16x2(G,H,I,sizeof(*A) * s,sizeof(*A) * s);
		}
		gettimeofday(&t1,NULL);
		delta = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
		if(delta == 0){ abort(); }
		printf("MMult2: %lu usecs, %lu flops, %lu Mflops/s\n",
				delta,i * 128,i * 128 / delta);
		memset(I,0,sizeof(I));
		gettimeofday(&t0,NULL);
		for(i = 0 ; i < 100000000 ; ++i){
			matmul1x16x16x4(A,B,I,sizeof(*A) * s,sizeof(*A) * s);
			matmul1x16x16x4(F,E,I,sizeof(*A) * s,sizeof(*A) * s);
		}
		gettimeofday(&t1,NULL);
		delta = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
		if(delta == 0){ abort(); }
		printf("MMult4: %lu usecs, %lu flops, %lu Mflops/s\n",
				delta,i * 128,i * 128 / delta);
	}else{
		printf("This machine doesn't support DPPD!\n");
	}

	printf("Benchmarking transmatX()...\n");
	memset(I,0,sizeof(I));
	gettimeofday(&t0,NULL);
	for(i = 0 ; i < 100000000 ; ++i){
		transmatmul1x16x16x2(A,B,I,sizeof(*A) * s);
		transmatmul1x16x16x2(C,D,I,sizeof(*A) * s);
		transmatmul1x16x16x2(E,F,I,sizeof(*A) * s);
		transmatmul1x16x16x2(G,H,I,sizeof(*A) * s);
	}
	gettimeofday(&t1,NULL);
	delta = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
	if(delta == 0){ abort(); }
	printf("TMMult2: %lu usecs, %lu flops, %lu Mflops/s\n",
			delta,i * 128,i * 128 / delta);
	memset(I,0,sizeof(I));
	gettimeofday(&t0,NULL);
	for(i = 0 ; i < 100000000 ; ++i){
		transmatmul1x16x16x4(A,B,I,sizeof(*A) * s);
		transmatmul1x16x16x4(F,E,I,sizeof(*A) * s);
	}
	gettimeofday(&t1,NULL);
	delta = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
	if(delta == 0){ abort(); }
	printf("TMMult4: %lu usecs, %lu flops, %lu Mflops/s\n",
			delta,i * 128,i * 128 / delta);
	return 0;
}

int main(void){
	double A[16][16],B[16][16],C[16][16];
	double a[16][16],b[1][16];
	double e,f;
	unsigned i;

	if(install_handler(SIGILL)){
		return EXIT_FAILURE;
	}
	srand48(time(NULL));
	for(i = 0 ; i < sizeof(A) / sizeof(*A) ; ++i){
		unsigned j;

		a[i][0] = drand48();
		b[0][i] = drand48();
		for(j = 0 ; j < sizeof(A) / sizeof(*A) ; ++j){
			A[i][j] = drand48();
			B[j][i] = drand48();
		}
	}

	if(sigsetjmp(jmpbuf,1) == 0){
		printf("Testing matdpX()...\n");
		// We first test a, b, and dppd*()
		e = matdp1x2x2x1(*a,*b,sizeof(*a));
		f = b[0][0] * a[0][0] + a[1][0] * b[0][1];
		printf("2x Dot-products: %.20f %.20f\n",e,f);
		if(e != f){ // should be a precise match
			fprintf(stderr,"Dot products didn't match\n");
			return EXIT_FAILURE;
		}
		f = b[0][0] * a[0][0] + b[0][1] * a[1][0] + b[0][2] * a[2][0] + b[0][3] * a[3][0];
		e = matdp1x4x4x1(*a,*b,sizeof(*a));
		printf("4x Dot-products: %.20f %.20f\n",e,f);
		if((float)e != (float)f){
			fprintf(stderr,"Dot products didn't match\n");
			return EXIT_FAILURE;
		}
		e = matdp1x16x16x1(*a,*b,sizeof(*a));
		f = a[0][0] * b[0][0] + a[1][0] * b[0][1] + a[2][0] * b[0][2] + a[3][0] * b[0][3] +
			a[4][0] * b[0][4] + a[5][0] * b[0][5] + a[6][0] * b[0][6] + a[7][0] * b[0][7] +
			a[8][0] * b[0][8] + a[9][0] * b[0][9] + a[10][0] * b[0][10] + a[11][0] * b[0][11] +
			a[12][0] * b[0][12] + a[13][0] * b[0][13] + a[14][0] * b[0][14] + a[15][0] * b[0][15];
		printf("16x Dot-products: %.20f %.20f\n",e,f);
		if((float)e != (float)f){
			fprintf(stderr,"Dot products didn't match\n");
			return EXIT_FAILURE;
		}

		// Now we use A, B, and mat*()
		e = matdp1x2x2x1(*A,*B,sizeof(*A));
		f = A[0][0] * B[0][0] + A[1][0] * B[0][1];
		printf("2x matrix products: %.20f %.20f\n",e,f);
		if(e != f){ // should be a precise match
			fprintf(stderr,"Matrix products didn't match\n");
			return EXIT_FAILURE;
		}
		e = matdp1x4x4x1(*A,*B,sizeof(*A));
		f = A[0][0] * B[0][0] + A[1][0] * B[0][1] + A[2][0] * B[0][2] + A[3][0] * B[0][3];
		printf("4x matrix products: %.20f %.20f\n",e,f);
		if((float)e != (float)f){
			fprintf(stderr,"Matrix products didn't match\n");
			return EXIT_FAILURE;
		}
		// interpret A and B as column-major, get C[1x1]
		f = A[0][0] * B[0][0] + A[1][0] * B[0][1] + A[2][0] * B[0][2] + A[3][0] * B[0][3] +
			A[4][0] * B[0][4] + A[5][0] * B[0][5] + A[6][0] * B[0][6] + A[7][0] * B[0][7] +
			A[8][0] * B[0][8] + A[9][0] * B[0][9] + A[10][0] * B[0][10] + A[11][0] * B[0][11] +
			A[12][0] * B[0][12] + A[13][0] * B[0][13] + A[14][0] * B[0][14] + A[15][0] * B[0][15];
		e = matdp1x16x16x1(A[0],B[0],sizeof(*A));
		printf("16x matrix products: %.20f %.20f\n",e,f);
		if((float)e != (float)f){
			fprintf(stderr,"Matrix products didn't match\n");
			return EXIT_FAILURE;
		}

		printf("Testing matmulX()...\n");
		memset(C,0,sizeof(C));
		e = f;
		f = A[0][0] * B[1][0] + A[1][0] * B[1][1] + A[2][0] * B[1][2] + A[3][0] * B[1][3] +
			A[4][0] * B[1][4] + A[5][0] * B[1][5] + A[6][0] * B[1][6] + A[7][0] * B[1][7] +
			A[8][0] * B[1][8] + A[9][0] * B[1][9] + A[10][0] * B[1][10] + A[11][0] * B[1][11] +
			A[12][0] * B[1][12] + A[13][0] * B[1][13] + A[14][0] * B[1][14] + A[15][0] * B[1][15];
		matmul1x16x16x2(*A,*B,*C,sizeof(*A),sizeof(*B));
		printf("16x2 matrix multiply:\n\t%.15f %.15f\n\t%.15f %.15f\n",C[0][0],e,C[1][0],f);
		if((float)C[0][0] != (float)e || (float)C[1][0] != (float)f){
			fprintf(stderr,"Matrix multiply didn't match\n");
			return EXIT_FAILURE;
		}
		memset(C,0,sizeof(C));
		matmul1x16x16x4(*A,*B,*C,sizeof(*A),sizeof(*B)); // scale it 4 times
		matmul1x16x16x4(*A,*B,*C,sizeof(*A),sizeof(*B));
		matmul1x16x16x4(*A,*B,*C,sizeof(*A),sizeof(*B));
		matmul1x16x16x4(*A,*B,*C,sizeof(*A),sizeof(*B));
		e = A[0][0] * B[2][0] + A[1][0] * B[2][1] + A[2][0] * B[2][2] + A[3][0] * B[2][3] +
			A[4][0] * B[2][4] + A[5][0] * B[2][5] + A[6][0] * B[2][6] + A[7][0] * B[2][7] +
			A[8][0] * B[2][8] + A[9][0] * B[2][9] + A[10][0] * B[2][10] + A[11][0] * B[2][11] +
			A[12][0] * B[2][12] + A[13][0] * B[2][13] + A[14][0] * B[2][14] + A[15][0] * B[2][15];
		f = A[0][0] * B[3][0] + A[1][0] * B[3][1] + A[2][0] * B[3][2] + A[3][0] * B[3][3] +
			A[4][0] * B[3][4] + A[5][0] * B[3][5] + A[6][0] * B[3][6] + A[7][0] * B[3][7] +
			A[8][0] * B[3][8] + A[9][0] * B[3][9] + A[10][0] * B[3][10] + A[11][0] * B[3][11] +
			A[12][0] * B[3][12] + A[13][0] * B[3][13] + A[14][0] * B[3][14] + A[15][0] * B[3][15];
		e *= 4;
		f *= 4;
		printf("16x4 matrix multiply:\n\t%.15f %.15f\n\t%.15f %.15f\n",C[2][0],e,C[3][0],f);
		if((float)C[2][0] != (float)e || (float)C[3][0] != (float)f){
			fprintf(stderr,"Matrix multiply didn't match\n");
			return EXIT_FAILURE;
		}
	}else{
		printf("This machine doesn't support DPPD!\n");
	}

	// now test transmultX()...
	transpose(*A,*a,16,16,16);
	memset(C,0,sizeof(C));
	e = A[0][0] * B[0][0] + A[1][0] * B[0][1] + A[2][0] * B[0][2] + A[3][0] * B[0][3] +
		A[4][0] * B[0][4] + A[5][0] * B[0][5] + A[6][0] * B[0][6] + A[7][0] * B[0][7] +
		A[8][0] * B[0][8] + A[9][0] * B[0][9] + A[10][0] * B[0][10] + A[11][0] * B[0][11] +
		A[12][0] * B[0][12] + A[13][0] * B[0][13] + A[14][0] * B[0][14] + A[15][0] * B[0][15];
	f = A[0][0] * B[1][0] + A[1][0] * B[1][1] + A[2][0] * B[1][2] + A[3][0] * B[1][3] +
		A[4][0] * B[1][4] + A[5][0] * B[1][5] + A[6][0] * B[1][6] + A[7][0] * B[1][7] +
		A[8][0] * B[1][8] + A[9][0] * B[1][9] + A[10][0] * B[1][10] + A[11][0] * B[1][11] +
		A[12][0] * B[1][12] + A[13][0] * B[1][13] + A[14][0] * B[1][14] + A[15][0] * B[1][15];
	transmatmul1x16x16x2(*a,*B,*C,sizeof(*A));
	printf("16x2 Tmatrix multiply:\n\t%.15f %.15f\n\t%.15f %.15f\n",C[0][0],e,C[1][0],f);
	if((float)C[0][0] != (float)e || (float)C[1][0] != (float)f){
		fprintf(stderr,"Matrix multiply didn't match\n");
		return EXIT_FAILURE;
	}
	memset(C,0,sizeof(C));
	transmatmul1x16x16x4(*a,*B,*C,sizeof(*A)); // scale it 4 times
	transmatmul1x16x16x4(*a,*B,*C,sizeof(*A));
	transmatmul1x16x16x4(*a,*B,*C,sizeof(*A));
	transmatmul1x16x16x4(*a,*B,*C,sizeof(*A));
	e = A[0][0] * B[2][0] + A[1][0] * B[2][1] + A[2][0] * B[2][2] + A[3][0] * B[2][3] +
		A[4][0] * B[2][4] + A[5][0] * B[2][5] + A[6][0] * B[2][6] + A[7][0] * B[2][7] +
		A[8][0] * B[2][8] + A[9][0] * B[2][9] + A[10][0] * B[2][10] + A[11][0] * B[2][11] +
		A[12][0] * B[2][12] + A[13][0] * B[2][13] + A[14][0] * B[2][14] + A[15][0] * B[2][15];
	f = A[0][0] * B[3][0] + A[1][0] * B[3][1] + A[2][0] * B[3][2] + A[3][0] * B[3][3] +
		A[4][0] * B[3][4] + A[5][0] * B[3][5] + A[6][0] * B[3][6] + A[7][0] * B[3][7] +
		A[8][0] * B[3][8] + A[9][0] * B[3][9] + A[10][0] * B[3][10] + A[11][0] * B[3][11] +
		A[12][0] * B[3][12] + A[13][0] * B[3][13] + A[14][0] * B[3][14] + A[15][0] * B[3][15];
	e *= 4;
	f *= 4;
	printf("[0][0] %f [0][1] %f [0][2] %f [0][3] %f\n",C[0][0],C[0][1],C[0][2],C[0][3]);
	printf("[1][0] %f [1][1] %f [1][2] %f [1][3] %f\n",C[1][0],C[1][1],C[1][2],C[1][3]);
	printf("[2][2] %f [2][1] %f [2][2] %f [2][3] %f\n",C[2][0],C[2][1],C[2][2],C[2][3]);
	printf("[3][0] %f [3][3] %f [3][2] %f [3][3] %f\n",C[3][0],C[1][3],C[3][2],C[3][3]);
	printf("16x4 Tmatrix multiply:\n\t%.15f %.15f\n\t%.15f %.15f\n",C[2][0],e,C[3][0],f);
	if((float)C[2][0] != (float)e || (float)C[3][0] != (float)f){
		fprintf(stderr,"TMatrix multiply didn't match\n");
		return EXIT_FAILURE;
	}
	benchmark_karma(16);
	return EXIT_SUCCESS;
}
