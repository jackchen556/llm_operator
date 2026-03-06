#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cmath>
#include <cstdlib>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << ": code " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 预热函数：使用 cublasGemmStridedBatchedEx 和 多流 cublasGemmEx
void warmup(cublasHandle_t handle,
            float* d_A, float* d_B,
            float* d_C_batch, float* d_C_single,
            int M, int N, int K, int batchCount,
            int num_streams) {

    const float alpha = 1.0f, beta = 0.0f;
    cudaStream_t warmup_stream;
    CHECK_CUDA(cudaStreamCreate(&warmup_stream));
    CHECK_CUBLAS(cublasSetStream(handle, warmup_stream));

    // --- Batched GEMM Warmup ---
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_A, CUDA_R_32F, M, M * K,
        d_B, CUDA_R_32F, K, K * N,
        &beta,
        d_C_batch, CUDA_R_32F, M, M * N,
        batchCount,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

    // --- Multi-stream Single GEMM Warmup ---
    cudaStream_t streams;
    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams));
    }

    for (int i = 0; i < batchCount; ++i) {
        CHECK_CUBLAS(cublasSetStream(handle, streams));

        float *d_A_i = d_A + i * M * K;
        float *d_B_i = d_B + i * K * N;
        float *d_C_i = d_C_single + i * M * N;

        CHECK_CUBLAS(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A_i, CUDA_R_32F, M,
            d_B_i, CUDA_R_32F, K,
            &beta,
            d_C_i, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
    }

    cudaDeviceSynchronize();

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(warmup_stream));
    CHECK_CUDA(cudaStreamDestroy(streams));
}

int main() {
    const int M = 1;
    const int N = 3584;
    const int K = 18944;
    //const int M = 1;
    //const int N = 256;
    //const int K = 256;
    const int batchCount = 8;        // 可改为 1 或 10 测试
    const int num_streams = 8;        // 多流数量

    // Host memory
    std::vector<float> h_A(M * K * batchCount);
    std::vector<float> h_B(K * N * batchCount);
    std::vector<float> h_C_batch(M * N * batchCount);
    std::vector<float> h_C_single(M * N * batchCount);

    // 初始化数据（列主序）
    for (int batch = 0; batch < batchCount; ++batch) {
        for (int k = 0; k < K; ++k) {
            for (int m = 0; m < M; ++m) {
                h_A[batch * M * K + k * M + m] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                h_B[batch * K * N + n * K + k] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }

    // Device memory
    float *d_A, *d_B, *d_C_batch, *d_C_single;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * batchCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * batchCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_batch, M * N * batchCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_single, M * N * batchCount * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * batchCount * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * batchCount * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Warmup
    std::cout << "Running warmup..." << std::endl;
    warmup(handle, d_A, d_B, d_C_batch, d_C_single, M, N, K, batchCount, num_streams);
    std::cout << "Warmup completed." << std::endl;

    const float alpha = 1.0f, beta = 0.0f;

    // --- Batched GEMM: 使用 cublasGemmStridedBatchedEx ---
    cudaStream_t stream_batch;
    CHECK_CUDA(cudaStreamCreate(&stream_batch));
    CHECK_CUBLAS(cublasSetStream(handle, stream_batch));

    cudaDeviceSynchronize();
    auto start_batch = std::chrono::high_resolution_clock::now();

    CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_A, CUDA_R_32F, M, M * K,           // A: stride = M*K
        d_B, CUDA_R_32F, K, K * N,           // B: stride = K*N
        &beta,
        d_C_batch, CUDA_R_32F, M, M * N,     // C: stride = M*N
        batchCount,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

    cudaDeviceSynchronize();
    auto end_batch = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_batch = end_batch - start_batch;

    // --- Multi-stream Single GEMM ---
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    cudaDeviceSynchronize();
    auto start_single = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < batchCount; ++i) {
        int stream_idx = i % num_streams;
        CHECK_CUBLAS(cublasSetStream(handle, streams[stream_idx]));

        float *d_A_i = d_A + i * M * K;
        float *d_B_i = d_B + i * K * N;
        float *d_C_i = d_C_single + i * M * N;

        CHECK_CUBLAS(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A_i, CUDA_R_32F, M,
            d_B_i, CUDA_R_32F, K,
            &beta,
            d_C_i, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
    }

    cudaDeviceSynchronize();
    auto end_single = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_single = end_single - start_single;

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_C_batch.data(), d_C_batch, M * N * batchCount * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_single.data(), d_C_single, M * N * batchCount * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证结果
    bool passed = true;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N * batchCount; ++i) {
        float diff = std::abs(h_C_batch[i] - h_C_single[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) {
            passed = false;
        }
    }

    // 输出结果
    std::cout << "\n--- Performance Results (cublasGemmStridedBatchedEx) ---" << std::endl;
    std::cout << "Batched GEMM time: " << elapsed_batch.count() * 1000 << " ms" << std::endl;
    std::cout << "Multi-stream GEMM time: " << elapsed_single.count() * 1000 << " ms" << std::endl;
    std::cout << "Speedup (multi-stream / batched): " << elapsed_batch.count() / elapsed_single.count() << "x" << std::endl;
    std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << std::endl;
    if (!passed) {
        std::cout << "Max difference: " << max_diff << std::endl;
    }

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream_batch));
    for (int i = 0; i < num_streams; ++i) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_batch));
    CHECK_CUDA(cudaFree(d_C_single));

    return 0;
}


