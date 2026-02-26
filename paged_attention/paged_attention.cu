/*
 * Paged Attention Implementation
 * 
 * 分为两部分：
 * 1. Python 端：逻辑管理（页表管理、物理块分配）
 * 2. CUDA 端：数值计算（根据页表从物理内存读取数据并计算注意力）
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cassert>

#define CUDA_CHECK(err) { \
    cudaError_t error = err; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * Paged Attention CUDA Kernel
 * 
 * 根据页表从物理内存读取数据并计算注意力
 * 
 * @param Q_physical: 物理内存中的 Q 矩阵 [num_physical_blocks, block_size, head_dim]
 * @param K_physical: 物理内存中的 K 矩阵 [num_physical_blocks, block_size, head_dim]
 * @param V_physical: 物理内存中的 V 矩阵 [num_physical_blocks, block_size, head_dim]
 * @param O: 输出矩阵 [batch_size, seq_len, head_dim]
 * @param page_table: 页表 [batch_size, num_logical_blocks] - 存储物理块索引
 * @param block_size: 每个块的大小
 * @param head_dim: 头维度
 * @param num_logical_blocks: 逻辑块数量
 * @param scale: 缩放因子
 */
__global__ void paged_attention_kernel(
    const __half* __restrict__ Q_physical,      // [num_physical_blocks, block_size, head_dim]
    const __half* __restrict__ K_physical,      // [num_physical_blocks, block_size, head_dim]
    const __half* __restrict__ V_physical,      // [num_physical_blocks, block_size, head_dim]
    __half* __restrict__ O,                     // [batch_size, seq_len, head_dim]
    const int* __restrict__ page_table,         // [batch_size, num_logical_blocks]
    int block_size,
    int head_dim,
    int num_logical_blocks,
    float scale
) {
    int batch_idx = blockIdx.x;
    int seq_idx = threadIdx.x + blockIdx.y * blockDim.x;
    int seq_len = num_logical_blocks * block_size;
    
    if (seq_idx >= seq_len) return;
    
    // 计算当前序列位置对应的逻辑块和块内偏移
    int logical_block_idx = seq_idx / block_size;
    int offset_in_block = seq_idx % block_size;
    
    // 根据页表查找物理块索引（Python 端管理的映射）
    int physical_block_idx = page_table[batch_idx * num_logical_blocks + logical_block_idx];
    
    // 从物理内存读取 Q（根据页表）
    float q_vec[64];  // 假设 head_dim <= 64
    for (int d = 0; d < head_dim; d++) {
        int q_offset = physical_block_idx * block_size * head_dim + 
                       offset_in_block * head_dim + d;
        q_vec[d] = __half2float(Q_physical[q_offset]);
    }
    
    // 计算注意力分数和输出
    float max_score = -1e30f;
    float scores[256];  // 假设 seq_len <= 256
    float sum_exp = 0.0f;
    
    // 遍历所有逻辑块，计算 QK^T
    for (int k_block = 0; k_block < num_logical_blocks; k_block++) {
        int k_physical_block = page_table[batch_idx * num_logical_blocks + k_block];
        
        for (int k_offset = 0; k_offset < block_size; k_offset++) {
            int k_seq_idx = k_block * block_size + k_offset;
            if (k_seq_idx >= seq_len) break;
            
            // 计算 QK^T
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                int k_addr = k_physical_block * block_size * head_dim + 
                            k_offset * head_dim + d;
                float k_val = __half2float(K_physical[k_addr]);
                score += q_vec[d] * k_val;
            }
            score *= scale;
            scores[k_seq_idx] = score;
            max_score = fmaxf(max_score, score);
        }
    }
    
    // Softmax
    for (int k = 0; k < seq_len; k++) {
        scores[k] = expf(scores[k] - max_score);
        sum_exp += scores[k];
    }
    
    // 计算输出：加权求和 V
    for (int d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        
        for (int k_block = 0; k_block < num_logical_blocks; k_block++) {
            int k_physical_block = page_table[batch_idx * num_logical_blocks + k_block];
            
            for (int k_offset = 0; k_offset < block_size; k_offset++) {
                int k_seq_idx = k_block * block_size + k_offset;
                if (k_seq_idx >= seq_len) break;
                
                int v_addr = k_physical_block * block_size * head_dim + 
                            k_offset * head_dim + d;
                float v_val = __half2float(V_physical[v_addr]);
                out_val += scores[k_seq_idx] * v_val;
            }
        }
        
        out_val /= sum_exp;
        int out_addr = batch_idx * seq_len * head_dim + seq_idx * head_dim + d;
        O[out_addr] = __float2half(out_val);
    }
}

/**
 * CUDA 包装函数
 */
extern "C" {
    void paged_attention_cuda(
        const __half* Q_physical,
        const __half* K_physical,
        const __half* V_physical,
        __half* O,
        const int* page_table,
        int batch_size,
        int block_size,
        int head_dim,
        int num_logical_blocks,
        float scale
    ) {
        int seq_len = num_logical_blocks * block_size;
        dim3 grid(batch_size, (seq_len + 255) / 256);
        dim3 block(256);
        
        paged_attention_kernel<<<grid, block>>>(
            Q_physical, K_physical, V_physical, O,
            page_table, block_size, head_dim, num_logical_blocks, scale
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

