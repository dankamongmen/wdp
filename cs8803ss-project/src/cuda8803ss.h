#ifndef CUDA8803SS
#define CUDA8803SS

#ifdef __cplusplus
extern "C" {
#endif

	// Result codes. _CUDAFAIL means that the CUDA kernel raised an
	// exception -- an expected mode of failure. _ERROR means some other
	// exception occurred (abort the binary search of the memory).
	enum {
		CUDARANGER_EXIT_SUCCESS,
		CUDARANGER_EXIT_ERROR,
		CUDARANGER_EXIT_CUDAFAIL,
	};

#ifdef __cplusplus
};
#endif

#endif
