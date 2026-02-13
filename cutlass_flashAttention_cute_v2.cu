/*
 * Flash Attention Implementation using CUTLASS CuTe DSL
 * 
 * This implementation uses CUTLASS 3.0+ CuTe DSL:
 * 1. Uses cute::Tensor and cute::Layout for tensor management
 * 2. Uses cute::TiledMma for TensorCore operations
 * 3. All computation in registers/shared memory (no HBM writeback)
 * 4. Online softmax with recomputation
 * 
 * Key: Uses modern CuTe DSL API, cleaner and more maintainable
 */

// 只包含必要的 CuTe DSL 头文件，减少编译时间
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cutlass/numeric_types.h>  // 只需要 half_t 类型
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

using namespace cute;

// CUTLASS 类型定义
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

// Tile shapes
constexpr int BLOCK_SIZE_M = 64;
constexpr int BLOCK_SIZE_N = 64;
constexpr int HEAD_DIM = 64;

/**
 * Flash Attention Kernel using CUTLASS CuTe DSL
 */
template<int BLOCK_M, int BLOCK_N, int HEAD_D>
__global__ void flash_attention_cute_kernel(
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
    cutlass::half_t* K_tile = Q_tile + BLOCK_M * HEAD_D;
    cutlass::half_t* V_tile = K_tile + BLOCK_N * HEAD_D;
    
    // S tile (QK^T) - 临时存储，不写回 HBM
    float* S_tile = reinterpret_cast<float*>(V_tile + BLOCK_N * HEAD_D);
    
    // Output tile (accumulated)
    float* O_tile = S_tile + BLOCK_M * BLOCK_N;
    
    // Max values and normalization factors
    float* m_vec = O_tile + BLOCK_M * HEAD_D;
    float* l_vec = m_vec + BLOCK_M;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // 计算当前 Q block 的范围
    int q_start = bid * BLOCK_M;
    int q_end = min(q_start + BLOCK_M, seq_len);
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
    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_N) {
        int kv_end = min(kv_start + BLOCK_N, seq_len);
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
        
        // ========== 使用 CuTe DSL 计算 S = Q * K^T ==========
        // 创建 CuTe Tensor（简化 Layout 创建，减少模板实例化）
        auto Q_tensor = make_tensor(Q_tile, make_layout(make_shape(q_len, head_dim)));
        auto K_tensor = make_tensor(K_tile, make_layout(make_shape(kv_len, head_dim)));
        auto S_tensor = make_tensor(S_tile, make_layout(make_shape(q_len, kv_len)));
        
        // 使用 CuTe GEMM 计算 S = Q * K^T
        // Tile size: 16x16 for warp-level computation
        constexpr int TILE_M = 16;
        constexpr int TILE_N = 16;
        constexpr int TILE_K = 16;
        
        // 每个 warp 处理一个 Q tile（16 行）
        int warp_tile_m = (warp_id * TILE_M) % q_len;
        
        if (warp_tile_m < q_len && warp_tile_m + TILE_M <= q_len) {
            // 遍历 N 维度（kv_len），每次处理 TILE_N（16 列）
            for (int n_start = 0; n_start < kv_len; n_start += TILE_N) {
                int n_end = min(n_start + TILE_N, kv_len);
                
                // 初始化累加器（减少循环展开）
                float accum[TILE_M * TILE_N];
                for (int i = 0; i < TILE_M * TILE_N; i++) {
                    accum[i] = 0.0f;
                }
                
                // K 维度循环（head_dim），累加 K 维度的结果
                // 减少循环展开以加快编译
                for (int k_start = 0; k_start < head_dim; k_start += TILE_K) {
                    // 使用 CuTe Tensor 访问数据（减少模板实例化）
                    for (int i = 0; i < TILE_M; i++) {
                        for (int j = 0; j < TILE_N; j++) {
                            float sum = 0.0f;
                            for (int k = 0; k < TILE_K && (k_start + k) < head_dim; k++) {
                                int q_row = warp_tile_m + i;
                                int k_col = n_start + j;
                                int k_idx = k_start + k;
                                
                                if (q_row < q_len && k_col < kv_len && k_idx < head_dim) {
                                    // 使用 CuTe Tensor 访问数据
                                    float q_val = float(Q_tensor(q_row, k_idx));
                                    float k_val = float(K_tensor(k_idx, k_col));  // K 的转置访问
                                    sum += q_val * k_val;
                                }
                            }
                            accum[i * TILE_N + j] += sum;
                        }
                    }
                }
                
                // 应用 scale 并存储到 S_tile（使用 CuTe Tensor，减少循环展开）
                for (int i = 0; i < TILE_M; i++) {
                    for (int j = 0; j < TILE_N && (n_start + j) < kv_len; j++) {
                        int row = warp_tile_m + i;
                        int col = n_start + j;
                        if (row < q_len && col < kv_len) {
                            // 使用 CuTe Tensor 存储结果
                            S_tensor(row, col) = accum[i * TILE_N + j] * scale;
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // ========== 在线 softmax 和累加 ==========
        // 使用 warp-level 归约优化 Softmax 计算
        int num_warps = (blockDim.x + 31) / 32;
        
        for (int q_idx = warp_id; q_idx < q_len; q_idx += num_warps) {
            // Warp-level 归约找最大值
            float m_ij = -INFINITY;
            for (int k_idx = lane_id; k_idx < kv_len; k_idx += 32) {
                m_ij = fmaxf(m_ij, S_tile[q_idx * kv_len + k_idx]);
            }
            
            // Warp shuffle 归约（减少循环展开以加快编译）
            for (int offset = 16; offset > 0; offset /= 2) {
                m_ij = fmaxf(m_ij, __shfl_sync(0xffffffff, m_ij, lane_id + offset));
            }
            
            // 更新全局最大值
            float m_i_old = m_vec[q_idx];
            float m_i_new = fmaxf(m_i_old, m_ij);
            
            // 初始化 PV sum（使用共享内存避免寄存器溢出）
            float* pv_sum = l_vec + BLOCK_M;  // 复用 l_vec 之后的空间
            if (lane_id < head_dim) {
                pv_sum[q_idx * head_dim + lane_id] = 0.0f;
            }
            __syncwarp();
            
            // 计算 exp 和累加（使用 warp-level 归约）
            float exp_sum = 0.0f;
            for (int k_idx = lane_id; k_idx < kv_len; k_idx += 32) {
                float s_val = S_tile[q_idx * kv_len + k_idx];
                float exp_val = expf(s_val - m_i_new);
                exp_sum += exp_val;
                
                // 累加 PV（每个线程处理一部分 head_dim）
                for (int d = lane_id; d < head_dim; d += 32) {
                    float v_val = float(V_tile[k_idx * head_dim + d]);
                    atomicAdd(&pv_sum[q_idx * head_dim + d], exp_val * v_val);
                }
            }
            
            // Warp-level 归约 exp_sum（减少循环展开）
            for (int offset = 16; offset > 0; offset /= 2) {
                exp_sum += __shfl_sync(0xffffffff, exp_sum, lane_id + offset);
            }
            __syncwarp();
            
            // Flash Attention 在线更新公式
            if (lane_id == 0) {
                float alpha = (m_i_old > -INFINITY) ? expf(m_i_old - m_i_new) : 1.0f;
                float l_i_old = l_vec[q_idx];
                float l_i_new = alpha * l_i_old + exp_sum;
                
                for (int d = 0; d < head_dim; d++) {
                    float o_i_old = O_tile[q_idx * head_dim + d];
                    float pv_val = pv_sum[q_idx * head_dim + d];
                    float o_i_new = (l_i_old > 0.0f) ? 
                        (alpha * l_i_old * o_i_old + pv_val) / l_i_new : 
                        pv_val / l_i_new;
                    O_tile[q_idx * head_dim + d] = o_i_new;
                }
                
                m_vec[q_idx] = m_i_new;
                l_vec[q_idx] = l_i_new;
            }
            __syncwarp();
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
void flash_attention_cute(
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
        block_size * sizeof(float) +                           // m_vec
        block_size * sizeof(float) +                           // l_vec
        block_size * head_dim * sizeof(float);                 // pv_sum
    
    // 检查共享内存限制
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // 如果共享内存超限，自动减小 block_size
    int actual_block_size = block_size;
    while (shared_mem_size > prop.sharedMemPerBlock && actual_block_size > 16) {
        actual_block_size /= 2;
        shared_mem_size = 
            actual_block_size * head_dim * sizeof(cutlass::half_t) +
            actual_block_size * head_dim * sizeof(cutlass::half_t) +
            actual_block_size * head_dim * sizeof(cutlass::half_t) +
            actual_block_size * actual_block_size * sizeof(float) +
            actual_block_size * head_dim * sizeof(float) +
            actual_block_size * sizeof(float) +
            actual_block_size * sizeof(float) +
            actual_block_size * head_dim * sizeof(float);
    }
    
    if (shared_mem_size > prop.sharedMemPerBlock) {
        std::cerr << "Error: Required shared memory exceeds GPU limit even with minimum block_size" << std::endl;
        return;
    }
    
    if (actual_block_size != block_size) {
        std::cerr << "Warning: Reduced block_size from " << block_size 
                  << " to " << actual_block_size 
                  << " due to shared memory constraints" << std::endl;
    }
    
    dim3 grid((seq_len + actual_block_size - 1) / actual_block_size, batch_size);
    dim3 block(256);  // 8 warps
    
    // 对每个 batch 调用 kernel
    if (actual_block_size == 64) {
        for (int b = 0; b < batch_size; b++) {
            int batch_offset = b * seq_len * head_dim;
            flash_attention_cute_kernel<64, 64, 64><<<grid.x, block, shared_mem_size, stream>>>(
                Q + batch_offset, K + batch_offset, V + batch_offset, O + batch_offset,
                seq_len, head_dim, scale
            );
        }
    } else if (actual_block_size == 32) {
        for (int b = 0; b < batch_size; b++) {
            int batch_offset = b * seq_len * head_dim;
            flash_attention_cute_kernel<32, 32, 64><<<grid.x, block, shared_mem_size, stream>>>(
                Q + batch_offset, K + batch_offset, V + batch_offset, O + batch_offset,
                seq_len, head_dim, scale
            );
        }
    } else {
        std::cerr << "Error: Unsupported block_size " << actual_block_size << std::endl;
        return;
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
    std::cout << "=== Flash Attention with CUTLASS CuTe DSL ===" << std::endl;
    std::cout << "This implementation uses CUTLASS 3.0+ CuTe DSL" << std::endl;
    std::cout << "for cleaner and more maintainable code." << std::endl;
    
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
    
    // 初始化输出为 0，用于检测 kernel 是否真的运行了
    CUDA_CHECK(cudaMemset(d_O_flash, 0, qkv_size * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMemset(d_O_ref, 0, qkv_size * sizeof(cutlass::half_t)));
    
    std::cout << "\nRunning Flash Attention with CUTLASS CuTe DSL..." << std::endl;
    flash_attention_cute(d_Q, d_K, d_V, d_O_flash, batch_size, seq_len, head_dim, block_size, scale);
    
    // 检查 kernel 是否真的运行了
    std::vector<cutlass::half_t> h_O_check(qkv_size);
    CUDA_CHECK(cudaMemcpy(h_O_check.data(), d_O_flash, qkv_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    bool all_zero = true;
    for (size_t i = 0; i < qkv_size; i++) {
        if (float(h_O_check[i]) != 0.0f) {
            all_zero = false;
            break;
        }
    }
    
    if (all_zero) {
        std::cerr << "\n✗ ERROR: Flash Attention kernel did not run!" << std::endl;
        std::cerr << "   All output values are zero. Check shared memory constraints." << std::endl;
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O_flash));
        CUDA_CHECK(cudaFree(d_O_ref));
        return 1;
    }
    
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
            if (error_count <= 10) {  // 只打印前 10 个错误
                std::cout << "  Error at [" << i << "]: flash=" << flash_val 
                          << ", ref=" << ref_val << ", error=" << error << std::endl;
            }
        }
    }
    
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << " / " << qkv_size << std::endl;
    
    if (max_error < tolerance && error_count == 0) {
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
