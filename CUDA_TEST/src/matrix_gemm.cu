// ==============================================================
// CUDA Tensor Core GEMM
// 
// 1 Global Memory
// 2 Shared Memory Tiling
// 3 Warp Tiling
// 4 Tensor Core (WMMA)
// 5 CUDA Error Check
// 6 Kernel Timer
// 
// ==============================================================

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <chrono>

using namespace nvcuda;

// ==============================================================
// Matrix size
// ==============================================================
/* デバッグ用 */
// #define DEBUG

#ifdef DEBUG
#define ROW_A 2
#define COL_A 2
#define ROW_B COL_A
#define COL_B 2
#else
#define ROW_A 1000
#define COL_A 1000
#define ROW_B COL_A
#define COL_B 1000
#endif

#define TRY 3

// ==============================================================
// Block tile
// 
// block が担当する C の領域
// ==============================================================
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 16

// ==============================================================
// TensorCore tile
// ==============================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ==============================================================
// CUDA error check
// ==============================================================
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ==============================================================
// Shared Memory
// block内共有
// Global → Shared → TensorCore
// ==============================================================

__shared__ half As[BLOCK_M][BLOCK_K];
__shared__ half Bs[BLOCK_K][BLOCK_N];

// ==============================================================
// Kernel
// ==============================================================
__global__ void tensor_gemm(
    half *a,
    half *b,
    float *c) {

    /*
    thread / warp
    */

    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    /*
    block position
    */

    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;

    /*
    warp tile

    block 128x128
    warp 4個
    */

    int warpRow = (warpId / 2) * 64;
    int warpCol = (warpId % 2) * 64;

    /*
    TensorCore fragments
    */

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    /*
    K loop
    */

    for(int bk=0; bk<ROW_B; bk+=BLOCK_K)
    {

        /*
        ==========================================
        Global → Shared load
        ==========================================
        */

        for(int i=tid;i<BLOCK_M*BLOCK_K;i+=blockDim.x)
        {
            int r=i/BLOCK_K;
            int k=i%BLOCK_K;

            As[r][k]=a[(blockRow+r)*ROW_B + bk + k];
        }

        for(int i=tid;i<BLOCK_K*BLOCK_N;i+=blockDim.x)
        {
            int k=i/BLOCK_N;
            int col=i%BLOCK_N;

            Bs[k][col]=b[(bk+k)*COL_B + blockCol + col];
        }

        __syncthreads();

        /*
        ==========================================
        TensorCore compute
        ==========================================
        */

        for(int k=0;k<BLOCK_K;k+=WMMA_K)
        {

            wmma::load_matrix_sync(
                a_frag,
                &As[warpRow][k],
                BLOCK_K);

            wmma::load_matrix_sync(
                b_frag,
                &Bs[k][warpCol],
                BLOCK_N);

            wmma::mma_sync(
                c_frag,
                a_frag,
                b_frag,
                c_frag);
        }

        __syncthreads();
    }

    // ==========================================
    // Store result
    // ==========================================

    int cRow = blockRow + warpRow;
    int cCol = blockCol + warpCol;

    wmma::store_matrix_sync(
        c + cRow*COL_B + cCol,
        c_frag,
        COL_B,
        wmma::mem_row_major);
}


// ==============================================================
// Main
// ==============================================================
int main() {
    printf("CUDA Tensor Core GEMM\n");
    // ==========================================
    // memory size
    // ==========================================
    size_t sizeA=ROW_A*ROW_B*sizeof(half);
    size_t sizeB=ROW_B*COL_B*sizeof(half);
    size_t sizeC=ROW_A*COL_B*sizeof(float);

    // ==========================================
    // host memory
    // ==========================================
    half *hA=(half*)malloc(sizeA);
    half *hB=(half*)malloc(sizeB);
    float *hC=(float*)malloc(sizeC);

    // ==========================================
    // initialize
    // ==========================================
    srand(0);
    for(int i=0;i<ROW_A*ROW_B;i++)
        hA[i]=__float2half(rand()%5);
    for(int i=0;i<ROW_B*COL_B;i++)
        hB[i]=__float2half(rand()%5);

    // ==========================================
    // device memory
    // ==========================================
    half *dA;
    half *dB;
    float *dC;

    CUDA_CHECK(cudaMalloc(&dA,sizeA));
    CUDA_CHECK(cudaMalloc(&dB,sizeB));
    CUDA_CHECK(cudaMalloc(&dC,sizeC));

    // ==========================================
    // copy to GPU
    // ==========================================
    CUDA_CHECK(cudaMemcpy(dA,hA,sizeA,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,sizeB,cudaMemcpyHostToDevice));

    // ==========================================
    // kernel configuration
    // ==========================================
    dim3 threads(128);

    dim3 blocks(
        (COL_B+BLOCK_N-1)/BLOCK_N,
        (ROW_A+BLOCK_M-1)/BLOCK_M
    );

    // ==========================================
    // TRY 回の平均時間計測
    // ==========================================
    float total_ms=0.0f;

    for(int t=0;t<TRY;t++) {
        cudaEvent_t start,stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // ==========================================
        // kernel launch
        // ==========================================
        tensor_gemm<<<blocks,threads>>>(dA,dB,dC);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms,start,stop);
        total_ms+=ms;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("Average Kernel time : %f ms (over %d tries)\n",total_ms/TRY,TRY);

    // ==========================================
    // copy result
    // ==========================================
    CUDA_CHECK(cudaMemcpy(hC,dC,sizeC,cudaMemcpyDeviceToHost));

    // ==========================================
    // sample output
    // ==========================================
    printf("Result sample\n");

    for(int i=0;i<10;i++)
        printf("%f\n",hC[i]);

    // ==========================================
    // free memory
    // ==========================================

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hA);
    free(hB);
    free(hC);

}