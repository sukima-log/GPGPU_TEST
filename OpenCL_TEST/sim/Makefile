

matrix_gemm: ../src/matrix_gemm.cpp
	g++ ../src/matrix_gemm.cpp -O2 -lOpenCL -mcmodel=medium -o matrix_gemm

.PHONY: clean
clean:
	rm -f matrix_gemm
