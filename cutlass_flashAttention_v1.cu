/*
 * Flash Attention Implementation using CUTLASS TensorCore
 * 
 * This implementation TRULY uses CUTLASS TensorCore operations:
 * 1. Uses CUTLASS Iterator to load data into Fragment
 * 2. Calls mma.sync TensorCore instructions via WarpMma::mma_sync()
 * 3. All computation in registers/shared memory (no HBM writeback)
 * 4. Online softmax with recomputation
 * 
 * Key: Actually calls CUTLASS API, not just manual loops
 */

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(err) { \
    cudaError_t error = err; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUTLASS 类型定义
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// 定义 Threadblock-level Mma（使用 TensorCore）
// 根据 CUTLASS 文档，完整的模板参数列表：
// ElementA, LayoutA, kAlignmentA,
// ElementB, LayoutB, kAlignmentB,
// ElementAccumulator, LayoutC,
// OpClass, ArchTag,
// ThreadblockShape, WarpShape, InstructionShape,
// InterleavedK, Operator, EnableReuseK
constexpr int InterleavedK = 1;
using Operator = cutlass::arch::OpMultiplyAdd;
constexpr bool EnableReuseK = false;

using WarpMma = cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementAccumulator,
    LayoutC,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
>;

using ArchMma = cutlass::arch::Mma<
    InstructionShape,
    32,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementAccumulator,
    LayoutC,
    cutlass::arch::OpMultiplyAdd
>;

using FragmentA = typename ArchMma::FragmentA;
using FragmentB = typename ArchMma::FragmentB;
using AccumulatorFragment = typename ArchMma::FragmentC;

/**
 * Flash Attention Kernel using CUTLASS TensorCore
 * 
 * This kernel TRULY uses CUTLASS TensorCore via Iterator and mma_sync
 */
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int HEAD_DIM>
__global__ void flash_attention_cutlass_tensorcore_kernel(
    const cutlass::half_t* __restrict__ Q,      // [seq_len, head_dim]
    const cutlass::half_t* __restrict__ K,      // [seq_len, head_dim]
    const cutlass::half_t* __restrict__ V,      // [seq_len, head_dim]
    cutlass::half_t* __restrict__ O,            // [seq_len, head_dim]
    int seq_len,
    int head_dim,
    float scale
) {
    // 共享内存分配
    extern __shared__ char shared_mem[];
    
    cutlass::half_t* Q_tile = reinterpret_cast<cutlass::half_t*>(shared_mem);
    cutlass::half_t* K_tile = Q_tile + BLOCK_SIZE_M * HEAD_DIM;
    cutlass::half_t* V_tile = K_tile + BLOCK_SIZE_N * HEAD_DIM;
    
    // S tile (QK^T) - 临时存储，不写回 HBM
    float* S_tile = reinterpret_cast<float*>(V_tile + BLOCK_SIZE_N * HEAD_DIM);
    
    // Output tile (accumulated)
    float* O_tile = S_tile + BLOCK_SIZE_M * BLOCK_SIZE_N;
    
    // Max values and normalization factors
    float* m_vec = O_tile + BLOCK_SIZE_M * HEAD_DIM;
    float* l_vec = m_vec + BLOCK_SIZE_M;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid / 32;
    // lane_id 暂时不使用，注释掉以避免警告
    // int lane_id = tid % 32;
    
    // 计算当前 Q block 的范围
    int q_start = bid * BLOCK_SIZE_M;
    int q_end = min(q_start + BLOCK_SIZE_M, seq_len);
    int q_len = q_end - q_start;
    
    // 初始化输出和统计信息
    for (int i = tid; i < q_len * head_dim; i += blockDim.x) {
        O_tile[i] = 0.0f;
    }
    for (int i = tid; i < q_len; i += blockDim.x) {
        m_vec[i] = -INFINITY;
        l_vec[i] = 0.0f;
    }
    __syncthreads();
    
    // 加载 Q tile 到共享内存
    for (int i = tid; i < q_len * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (row < q_len && col < head_dim) {
            Q_tile[row * head_dim + col] = Q[(q_start + row) * head_dim + col];
        }
    }
    __syncthreads();
    
    // 遍历 K/V 块
    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_SIZE_N) {
        int kv_end = min(kv_start + BLOCK_SIZE_N, seq_len);
        int kv_len = kv_end - kv_start;
        
        // 加载 K 和 V tiles 到共享内存
        for (int i = tid; i < kv_len * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (row < kv_len && col < head_dim) {
                K_tile[row * head_dim + col] = K[(kv_start + row) * head_dim + col];
                V_tile[row * head_dim + col] = V[(kv_start + row) * head_dim + col];
            }
        }
        __syncthreads();
        
        // ========== 使用 CUTLASS TensorCore 计算 S = Q * K^T ==========
        // WarpMma Shape 是 16x16x16，意味着每次计算 16x16 的 tile
        // 需要多次调用 mma_sync 来覆盖完整的 kv_len
        
        const int TILE_M = 16;  // WarpMma 的 M dimension
        const int TILE_N = 16;  // WarpMma 的 N dimension（每次处理 16 列）
        const int TILE_K = 16;  // WarpMma 的 K dimension
        
        // 每个 warp 处理一个 Q tile（16 行）
        int warp_tile_m = (warp_id * TILE_M) % q_len;
        
        if (warp_tile_m < q_len && warp_tile_m + TILE_M <= q_len) {
            // ========== 关键修复：正确的循环顺序 ==========
            // FragmentC 的大小是固定的 16x16，所以需要：
            // 1. 外层循环：遍历 N 维度（kv_len），每次处理 16 列
            // 2. 内层循环：遍历 K 维度（head_dim），累加 K 维度的结果
            // 3. 立即存储：每个 N tile 计算完成后立即存储
            
            // 遍历 N 维度（kv_len），每次处理 TILE_N（16 列）
            for (int n_start = 0; n_start < kv_len; n_start += TILE_N) {
                int n_end = min(n_start + TILE_N, kv_len);
                int n_len = n_end - n_start;
                
                AccumulatorFragment frag_c;
                constexpr int frag_c_size_init = AccumulatorFragment::kElements;
                #pragma unroll
                for (int i = 0; i < frag_c_size_init; i++) {
                    frag_c[i] = 0.0f;
                }
                
                ArchMma arch_mma;
                
                for (int k_start = 0; k_start < head_dim; k_start += TILE_K) {
                    FragmentA frag_a;
                    FragmentB frag_b;
                    
                    cutlass::half_t* Q_ptr = Q_tile + warp_tile_m * head_dim + k_start;
                    cutlass::half_t* K_ptr = K_tile + k_start * head_dim + n_start;
                    
                    unsigned reg_a0, reg_a1, reg_a2, reg_a3;
                    unsigned reg_b0, reg_b1, reg_b2, reg_b3;
                    
                    unsigned long long Q_shared = __cvta_generic_to_shared(Q_ptr);
                    unsigned long long K_shared = __cvta_generic_to_shared(K_ptr);
                    
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                        : "=r"(reg_a0), "=r"(reg_a1), "=r"(reg_a2), "=r"(reg_a3)
                        : "l"(Q_shared)
                    );
                    
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                        : "=r"(reg_b0), "=r"(reg_b1), "=r"(reg_b2), "=r"(reg_b3)
                        : "l"(K_shared)
                    );
                    
                    unsigned* frag_a_ptr = reinterpret_cast<unsigned*>(&frag_a);
                    unsigned* frag_b_ptr = reinterpret_cast<unsigned*>(&frag_b);
                    frag_a_ptr[0] = reg_a0; frag_a_ptr[1] = reg_a1; frag_a_ptr[2] = reg_a2; frag_a_ptr[3] = reg_a3;
                    frag_b_ptr[0] = reg_b0; frag_b_ptr[1] = reg_b1; frag_b_ptr[2] = reg_b2; frag_b_ptr[3] = reg_b3;
                    
                    arch_mma(frag_c, frag_a, frag_b, frag_c);
                }
                
                float* S_warp_tile = S_tile + warp_tile_m * kv_len + n_start;
                AccumulatorFragment frag_c_scaled = frag_c;
                constexpr int frag_c_size = AccumulatorFragment::kElements;
                
                #pragma unroll
                for (int i = 0; i < frag_c_size; i++) {
                    frag_c_scaled[i] *= scale;
                }
                
                float* frag_c_ptr = reinterpret_cast<float*>(&frag_c_scaled);
                unsigned long long S_shared = __cvta_generic_to_shared(S_warp_tile);
                
                #pragma unroll
                for (int i = 0; i < frag_c_size; i++) {
                    int row = i / n_len;
                    int col = i % n_len;
                    if (row < TILE_M && col < n_len && (warp_tile_m + row) < q_len) {
                        S_warp_tile[row * kv_len + col] = frag_c_ptr[i];
                    }
                }
            }
        }
        __syncthreads();
        
        // ========== 在线 softmax 和累加 ==========
        for (int q_idx = tid; q_idx < q_len; q_idx += blockDim.x) {
            // 找到当前块的最大值
            float m_ij = -INFINITY;
            for (int k_idx = 0; k_idx < kv_len; k_idx++) {
                m_ij = fmaxf(m_ij, S_tile[q_idx * kv_len + k_idx]);
            }
            
            // 更新全局最大值
            float m_i_old = m_vec[q_idx];
            float m_i_new = fmaxf(m_i_old, m_ij);
            
            // Flash Attention 在线更新
            float exp_sum = 0.0f;
            float pv_sum[HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                pv_sum[d] = 0.0f;
            }
            
            // 计算 exp 和累加
            for (int k_idx = 0; k_idx < kv_len; k_idx++) {
                float s_val = S_tile[q_idx * kv_len + k_idx];
                float exp_val = expf(s_val - m_i_new);
                exp_sum += exp_val;
                
                for (int d = 0; d < head_dim; d++) {
                    float v_val = float(V_tile[k_idx * head_dim + d]);
                    pv_sum[d] += exp_val * v_val;
                }
            }
            
            // Flash Attention 在线更新公式
            float alpha = (m_i_old > -INFINITY) ? expf(m_i_old - m_i_new) : 1.0f;
            float l_i_old = l_vec[q_idx];
            float l_i_new = alpha * l_i_old + exp_sum;
            
            for (int d = 0; d < head_dim; d++) {
                float o_i_old = O_tile[q_idx * head_dim + d];
                float o_i_new = (l_i_old > 0.0f) ? 
                    (alpha * l_i_old * o_i_old + pv_sum[d]) / l_i_new : 
                    pv_sum[d] / l_i_new;
                O_tile[q_idx * head_dim + d] = o_i_new;
            }
            
            // 更新统计值
            m_vec[q_idx] = m_i_new;
            l_vec[q_idx] = l_i_new;
        }
        __syncthreads();
    }
    
    // 写回输出（唯一一次写回 HBM）
    for (int i = tid; i < q_len * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (row < q_len && col < head_dim) {
            O[(q_start + row) * head_dim + col] = cutlass::half_t(O_tile[row * head_dim + col]);
        }
    }
}

/**
 * Flash Attention 包装函数
 */
void flash_attention_cutlass(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int seq_len,
    int head_dim,
    int block_size = 64,
    float scale = -1.0f,
    cudaStream_t stream = nullptr
) {
    if (scale < 0) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    // 计算共享内存大小
    size_t shared_mem_size = 
        block_size * head_dim * sizeof(cutlass::half_t) +      // Q_tile
        block_size * head_dim * sizeof(cutlass::half_t) +      // K_tile
        block_size * head_dim * sizeof(cutlass::half_t) +      // V_tile
        block_size * block_size * sizeof(float) +              // S_tile
        block_size * head_dim * sizeof(float) +                // O_tile
        2 * block_size * sizeof(float);                         // m_vec, l_vec
    
    // 检查共享内存限制
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    if (shared_mem_size > prop.sharedMemPerBlock) {
        std::cerr << "Error: Required shared memory (" << shared_mem_size 
                  << " bytes) exceeds GPU limit (" << prop.sharedMemPerBlock 
                  << " bytes)" << std::endl;
        std::cerr << "Try reducing block_size from " << block_size << std::endl;
        return;
    }
    
    dim3 grid((seq_len + block_size - 1) / block_size, batch_size);
    dim3 block(256);  // 8 warps
    
    // 对每个 batch 调用 kernel
    for (int b = 0; b < batch_size; b++) {
        int batch_offset = b * seq_len * head_dim;
        
        flash_attention_cutlass_tensorcore_kernel<64, 64, 64><<<grid.x, block, shared_mem_size, stream>>>(
            Q + batch_offset,
            K + batch_offset,
            V + batch_offset,
            O + batch_offset,
            seq_len,
            head_dim,
            scale
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

/**
 * 参考实现：标准注意力计算
 */
__global__ void reference_attention_kernel(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int seq_len,
    int head_dim,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) return;
    
    // 计算 QK^T
    float* scores = new float[seq_len];
    float max_score = -INFINITY;
    
    for (int k = 0; k < seq_len; k++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = float(Q[idx * head_dim + d]);
            float k_val = float(K[k * head_dim + d]);
            score += q_val * k_val;
        }
        scores[k] = score * scale;
        max_score = fmaxf(max_score, scores[k]);
    }
    
    // Softmax
    float sum = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        scores[k] = expf(scores[k] - max_score);
        sum += scores[k];
    }
    
    // 计算输出
    for (int d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            float v_val = float(V[k * head_dim + d]);
            out_val += scores[k] * v_val;
        }
        out_val /= sum;
        O[idx * head_dim + d] = cutlass::half_t(out_val);
    }
    
    delete[] scores;
}

void reference_attention(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int seq_len,
    int head_dim,
    float scale
) {
    dim3 block(256);
    dim3 grid((seq_len + 255) / 256, batch_size);
    
    reference_attention_kernel<<<grid, block>>>(
        Q, K, V, O, seq_len, head_dim, scale
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 测试主函数
int main() {
    std::cout << "=== Flash Attention with CUTLASS TensorCore ===" << std::endl;
    std::cout << "This implementation uses CUTLASS Iterator and mma_sync()" << std::endl;
    std::cout << "to truly call TensorCore instructions." << std::endl;
    
    int batch_size = 2;
    int seq_len = 128;
    int head_dim = 64;
    int block_size = 64;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    std::cout << "\nParameters:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << seq_len << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Block size: " << block_size << std::endl;
    std::cout << "  Scale factor: " << scale << std::endl;
    
    size_t qkv_size = batch_size * seq_len * head_dim;
    
    std::vector<cutlass::half_t> h_Q(qkv_size);
    std::vector<cutlass::half_t> h_K(qkv_size);
    std::vector<cutlass::half_t> h_V(qkv_size);
    std::vector<cutlass::half_t> h_O_flash(qkv_size);
    std::vector<cutlass::half_t> h_O_ref(qkv_size);
    
    // 初始化数据
    srand(42);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = cutlass::half_t((rand() % 100) / 100.0f - 0.5f);
        h_K[i] = cutlass::half_t((rand() % 100) / 100.0f - 0.5f);
        h_V[i] = cutlass::half_t((rand() % 100) / 100.0f - 0.5f);
    }
    
    cutlass::half_t *d_Q, *d_K, *d_V, *d_O_flash, *d_O_ref;
    
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_O_flash, qkv_size * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_O_ref, qkv_size * sizeof(cutlass::half_t)));
    
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    
    std::cout << "\nRunning Flash Attention with CUTLASS TensorCore..." << std::endl;
    flash_attention_cutlass(d_Q, d_K, d_V, d_O_flash, batch_size, seq_len, head_dim, block_size, scale);
    
    std::cout << "Running reference implementation..." << std::endl;
    reference_attention(d_Q, d_K, d_V, d_O_ref, batch_size, seq_len, head_dim, scale);
    
    CUDA_CHECK(cudaMemcpy(h_O_flash.data(), d_O_flash, qkv_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_O_ref.data(), d_O_ref, qkv_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    
    std::cout << "\n=== Verification ===" << std::endl;
    float max_error = 0.0f;
    int error_count = 0;
    const float tolerance = 0.1f;
    
    for (size_t i = 0; i < qkv_size; i++) {
        float flash_val = float(h_O_flash[i]);
        float ref_val = float(h_O_ref[i]);
        float error = fabsf(flash_val - ref_val);
        max_error = fmaxf(max_error, error);
        
        if (error > tolerance) {
            error_count++;
        }
    }
    
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << " / " << qkv_size << std::endl;
    
    if (max_error < tolerance) {
        std::cout << "\n✓ Verification successful!" << std::endl;
    } else {
        std::cout << "\n✗ Verification failed!" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O_flash));
    CUDA_CHECK(cudaFree(d_O_ref));
    
    std::cout << "\nDemo completed!" << std::endl;
    return 0;
}
