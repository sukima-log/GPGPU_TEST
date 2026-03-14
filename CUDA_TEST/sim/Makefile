

matrix_gemm: ../src/matrix_gemm.cu
	nvcc ../src/matrix_gemm.cu -O2 -arch=sm_75 -o matrix_gemm

.PHONY: clean
clean:
	rm -f matrix_gemm
