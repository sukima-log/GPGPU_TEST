// ==============================================================
// OpenCL Tensor Core GEMM (近似)
//
// CUDA Tensor Core GEMM コードをベースに OpenCL に変換
// ==============================================================

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <assert.h>

// ==============================================================
// Matrix size
// ==============================================================

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
// ==============================================================

#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 16

// ==============================================================
// TensorCore tile (概念的)
// ==============================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ==============================================================
// OpenCL GEMM カーネル文字列
// ==============================================================

const char* kernelSource = R"CLC(

__kernel void tensor_gemm(
    __global float* a,
    __global float* b,
    __global float* c,
    const int ROW_A,
    const int COL_A,
    const int ROW_B,
    const int COL_B)
{
    // thread id
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);

    // bounds check
    if(gid_x >= ROW_A || gid_y >= COL_B) return;

    float sum = 0.0f;
    for(int k = 0; k < ROW_B; k++) {
        sum += a[gid_x * ROW_B + k] * b[k * COL_B + gid_y];
    }
    c[gid_x * COL_B + gid_y] = sum;
}
)CLC";

// ==============================================================
// Host
// ==============================================================

int main() {
    printf("OpenCL Tensor Core GEMM\n");

    size_t sizeA = ROW_A * ROW_B * sizeof(float);
    size_t sizeB = ROW_B * COL_B * sizeof(float);
    size_t sizeC = ROW_A * COL_B * sizeof(float);

    // host memory
    float *hA = (float*)aligned_alloc(64, ((sizeA+63)/64)*64); // 64 byte 境界に調整
    float *hB = (float*)aligned_alloc(64, ((sizeB+63)/64)*64);
    float *hC = (float*)aligned_alloc(64, ((sizeC+63)/64)*64);

    srand(0);
    for(int i=0;i<ROW_A*ROW_B;i++) hA[i] = rand() % 5;
    for(int i=0;i<ROW_B*COL_B;i++) hB[i] = rand() % 5;

    // 1. OpenCL プラットフォーム取得
    cl_platform_id platform;
    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL); if(err != CL_SUCCESS){printf("clGetPlatformIDs failed\n"); return -1;}

    // 2. デバイス取得
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); if(err != CL_SUCCESS){printf("clGetDeviceIDs failed\n"); return -1;}

    // 3. コンテキスト作成
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); if(err != CL_SUCCESS){printf("clCreateContext failed\n"); return -1;}

    // 4. コマンドキュー
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err); if(err != CL_SUCCESS){printf("clCreateCommandQueueWithProperties failed\n"); return -1;}

    // 5. バッファ作成
    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeA, hA, &err);
    cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeB, hB, &err);
    cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeC, NULL, &err);

    // 6. プログラム作成
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(err != CL_SUCCESS){printf("clBuildProgram failed\n"); return -1;}

    // 7. カーネル作成
    cl_kernel kernel = clCreateKernel(program, "tensor_gemm", &err);

    // 8. カーネル引数
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);

    int row_a = ROW_A;
    int col_a = COL_A;
    int row_b = ROW_B;
    int col_b = COL_B;

    clSetKernelArg(kernel, 3, sizeof(int), &row_a);
    clSetKernelArg(kernel, 4, sizeof(int), &col_a);
    clSetKernelArg(kernel, 5, sizeof(int), &row_b);
    clSetKernelArg(kernel, 6, sizeof(int), &col_b);

    // NDRange 設定 (local size 明示)
    size_t local[2]  = {16, 16};
    size_t global[2] = {
        ((ROW_A + local[0]-1)/local[0])*local[0],
        ((COL_B + local[1]-1)/local[1])*local[1]
    };

    double total_host_ms = 0.0;
    double total_device_ms = 0.0;

    for(int t=0;t<TRY;t++){
        cl_event event;
        auto start = std::chrono::high_resolution_clock::now();

        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
        if(err != CL_SUCCESS){printf("clEnqueueNDRangeKernel failed\n"); return -1;}
        clFinish(queue);

        auto end = std::chrono::high_resolution_clock::now();
        total_host_ms += std::chrono::duration<double, std::milli>(end - start).count();

        cl_ulong start_time, end_time;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
        total_device_ms += (end_time - start_time) * 1.0e-6;
        clReleaseEvent(event);
    }

    printf("Average Kernel time (GPU profiling) : %lf ms (over %d tries)\n", total_device_ms/TRY, TRY);
    printf("Average Kernel time (host)          : %lf ms (over %d tries)\n", total_host_ms/TRY, TRY);

    clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, sizeC, hC, 0, NULL, NULL);

    printf("Result sample\n");
    for(int i=0;i<10;i++) printf("%f\n", hC[i]);

    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(hA);
    free(hB);
    free(hC);
}