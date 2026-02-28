/*
 * Chunked Prefill CUDA Implementation
 *
 * Chunked Prefill 将长序列分成固定大小的 chunk，逐块处理以控制峰值显存。
 * 每个 chunk 的 Q 会 attend 到：之前 chunk 的 KV cache + 当前 chunk 的 KV（因果注意力）
 *
 * 分为两部分：
 * 1. Python 端：chunk 划分、KV cache 累积、调度
 * 2. CUDA 端：单 chunk 的注意力计算（Q_chunk @ K_full^T -> softmax -> V_full）
 */

#include <cuda_runtime.h>
#include <math.h>

#define MAX_HEAD_DIM 128
#define MAX_CONTEXT_LEN 512   /* 单 chunk 最大 context，demo 用 64 足够 */

/**
 * Chunked Prefill Attention Kernel
 *
 * 对当前 chunk 计算因果注意力：
 * - Q_chunk[q] attends to K[0 : chunk_start + q + 1], V[0 : chunk_start + q + 1]
 *
 * @param Q: 当前 chunk 的 Q [chunk_len, head_dim]
 * @param K: 完整 context 的 K [context_len, head_dim]，含 cached + current chunk
 * @param V: 完整 context 的 V [context_len, head_dim]
 * @param O: 输出 [chunk_len, head_dim]
 * @param chunk_start: 当前 chunk 在 context 中的起始位置
 * @param chunk_len: 当前 chunk 的长度
 * @param head_dim: 头维度
 * @param scale: 缩放因子 1/sqrt(head_dim)
 */
__global__ void chunked_prefill_kernel(
    const float* __restrict__ Q,       // [chunk_len, head_dim]
    const float* __restrict__ K,        // [context_len, head_dim]
    const float* __restrict__ V,        // [context_len, head_dim]
    float* __restrict__ O,              // [chunk_len, head_dim]
    int chunk_start,
    int chunk_len,
    int head_dim,
    float scale
) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= chunk_len) return;

    // 当前 Q 对应的 context 长度（因果：attend to [0, chunk_start + q_idx]）
    int context_len = chunk_start + q_idx + 1;

    // 读取 Q 向量
    float q_vec[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        q_vec[d] = Q[q_idx * head_dim + d];
    }

    // 计算 QK^T 分数
    float max_score = -1e30f;
    float scores[MAX_CONTEXT_LEN];
    for (int k = 0; k < context_len; k++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * K[k * head_dim + d];
        }
        score *= scale;
        scores[k] = score;
        max_score = fmaxf(max_score, score);
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int k = 0; k < context_len; k++) {
        scores[k] = expf(scores[k] - max_score);
        sum_exp += scores[k];
    }

    // 加权求和 V
    for (int d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (int k = 0; k < context_len; k++) {
            out_val += scores[k] * V[k * head_dim + d];
        }
        O[q_idx * head_dim + d] = out_val / sum_exp;
    }
}

/**
 * 批量 Chunked Prefill Kernel（支持 batch）
 * 每个 batch 独立处理，共享相同的 chunk 参数
 */
__global__ void chunked_prefill_batch_kernel(
    const float* __restrict__ Q,       // [batch, chunk_len, head_dim]
    const float* __restrict__ K,        // [batch, context_len, head_dim]
    const float* __restrict__ V,        // [batch, context_len, head_dim]
    float* __restrict__ O,              // [batch, chunk_len, head_dim]
    int batch_size,
    int chunk_start,
    int chunk_len,
    int head_dim,
    float scale
) {
    int batch_idx = blockIdx.z;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || q_idx >= chunk_len) return;

    int batch_offset_q = batch_idx * chunk_len * head_dim;
    int batch_offset_kv = batch_idx * (chunk_start + chunk_len) * head_dim;

    const float* Q_b = Q + batch_offset_q;
    const float* K_b = K + batch_offset_kv;
    const float* V_b = V + batch_offset_kv;
    float* O_b = O + batch_offset_q;

    int context_len = chunk_start + q_idx + 1;

    float q_vec[MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        q_vec[d] = Q_b[q_idx * head_dim + d];
    }

    float max_score = -1e30f;
    float scores[MAX_CONTEXT_LEN];
    for (int k = 0; k < context_len; k++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * K_b[k * head_dim + d];
        }
        score *= scale;
        scores[k] = score;
        max_score = fmaxf(max_score, score);
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < context_len; k++) {
        scores[k] = expf(scores[k] - max_score);
        sum_exp += scores[k];
    }

    for (int d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (int k = 0; k < context_len; k++) {
            out_val += scores[k] * V_b[k * head_dim + d];
        }
        O_b[q_idx * head_dim + d] = out_val / sum_exp;
    }
}

