/*
 * Paged Attention — GQA：Q 多头稠密，KV 少头分页（同一 KV 头可被多个 Q 头复用）
 */

#include <cuda_runtime.h>
#include <math.h>

#ifndef MAX_HEAD_DIM
#define MAX_HEAD_DIM 128
#endif
#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 1024
#endif

#define CUDA_CHECK(err) { \
    cudaError_t error = err; \
    if (error != cudaSuccess) { \
    } \
}

/**
 * Q:  [batch_size, seq_len, num_q_heads, head_dim]
 * K/V:[num_physical_blocks, block_size, num_kv_heads, head_dim]
 * O:  [batch_size, seq_len, num_q_heads, head_dim]
 * kv_head = q_head / (num_q_heads / num_kv_heads)
 */
__global__ void paged_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_physical,
    const float* __restrict__ V_physical,
    float* __restrict__ O,
    const int* __restrict__ page_table,
    int block_size,
    int head_dim,
    int num_q_heads,
    int num_kv_heads,
    int num_logical_blocks,
    float scale
) {
    int batch_idx = blockIdx.x;
    int q_head_idx = blockIdx.y;
    int seq_idx = threadIdx.x + blockIdx.z * blockDim.x;
    int seq_len = num_logical_blocks * block_size;

    if (q_head_idx >= num_q_heads) return;
    if (seq_idx >= seq_len) return;
    if (head_dim > MAX_HEAD_DIM) return;
    if (seq_len > MAX_SEQ_LEN) return;

    int queries_per_kv = num_q_heads / num_kv_heads;
    int kv_head_idx = q_head_idx / queries_per_kv;

    long long q_row = (long long)batch_idx * seq_len * num_q_heads * head_dim
                    + (long long)seq_idx * num_q_heads * head_dim
                    + (long long)q_head_idx * head_dim;

    float q_vec[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        q_vec[d] = Q[q_row + d];
    }

    float max_score = -1e30f;
    float scores[MAX_SEQ_LEN];
    float sum_exp = 0.0f;

    int k_stride_block = block_size * num_kv_heads * head_dim;

    for (int k_block = 0; k_block < num_logical_blocks; k_block++) {
        int k_physical_block = page_table[batch_idx * num_logical_blocks + k_block];

        for (int k_offset = 0; k_offset < block_size; k_offset++) {
            int k_seq_idx = k_block * block_size + k_offset;
            if (k_seq_idx >= seq_len) break;

            float score = 0.0f;
            int k_base = k_physical_block * k_stride_block
                       + k_offset * num_kv_heads * head_dim
                       + kv_head_idx * head_dim;
            for (int d = 0; d < head_dim; d++) {
                score += q_vec[d] * K_physical[k_base + d];
            }
            score *= scale;
            scores[k_seq_idx] = score;
            max_score = fmaxf(max_score, score);
        }
    }

    for (int k = 0; k < seq_len; k++) {
        scores[k] = expf(scores[k] - max_score);
        sum_exp += scores[k];
    }

    for (int d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (int k_block = 0; k_block < num_logical_blocks; k_block++) {
            int k_physical_block = page_table[batch_idx * num_logical_blocks + k_block];

            for (int k_offset = 0; k_offset < block_size; k_offset++) {
                int k_seq_idx = k_block * block_size + k_offset;
                if (k_seq_idx >= seq_len) break;

                int v_base = k_physical_block * k_stride_block
                           + k_offset * num_kv_heads * head_dim
                           + kv_head_idx * head_dim;
                out_val += scores[k_seq_idx] * V_physical[v_base + d];
            }
        }
        out_val /= sum_exp;
        long long out_addr = (long long)batch_idx * seq_len * num_q_heads * head_dim
                           + (long long)seq_idx * num_q_heads * head_dim
                           + (long long)q_head_idx * head_dim + d;
        O[out_addr] = out_val;
    }
}

extern "C" void paged_attention_cuda(
    const float* Q,
    const float* K_physical,
    const float* V_physical,
    float* O,
    const int* page_table,
    int batch_size,
    int block_size,
    int head_dim,
    int num_q_heads,
    int num_kv_heads,
    int num_logical_blocks,
    float scale
) {
    int seq_len = num_logical_blocks * block_size;

    dim3 block(256, 1, 1);
    dim3 grid(batch_size, num_q_heads, (seq_len + 255) / 256);

    paged_attention_kernel<<<grid, block>>>(
        Q, K_physical, V_physical, O,
        page_table,
        block_size, head_dim, num_q_heads, num_kv_heads,
        num_logical_blocks, scale
    );
    cudaDeviceSynchronize();
}

