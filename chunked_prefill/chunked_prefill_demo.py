"""
Chunked Prefill 完整示例

展示 Python（chunk 划分、KV cache 管理）和 CUDA（分块注意力计算）的结合
"""

import numpy as np
import os

try:
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("Warning: PyCUDA not available, using CPU fallback")


def demo_with_cuda():
    """使用 CUDA 的完整示例"""
    if not PYCUDA_AVAILABLE:
        print("PyCUDA not available, skipping CUDA demo")
        return

    print("=== Chunked Prefill with CUDA Demo ===\n")

    # 配置
    seq_len = 64
    chunk_size = 16
    head_dim = 64
    scale = 1.0 / np.sqrt(head_dim)

    print("1. Python 端：生成测试数据")
    Q_full = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    K_full = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    V_full = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    print(f"   Q/K/V 形状: {Q_full.shape}")
    print()

    print("2. Python 端：Chunk 划分与 KV cache 累积")
    O_chunked = np.zeros((seq_len, head_dim), dtype=np.float32)
    kv_cache_k = []
    kv_cache_v = []

    cuda_file = os.path.join(os.path.dirname(__file__), "chunked_prefill.cu")
    with open(cuda_file, "r") as f:
        cuda_code = f.read()
    mod = SourceModule(cuda_code)
    kernel = mod.get_function("chunked_prefill_kernel")

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        curr_chunk_len = chunk_end - chunk_start

        Q_chunk = Q_full[chunk_start:chunk_end]
        K_chunk = K_full[chunk_start:chunk_end]
        V_chunk = V_full[chunk_start:chunk_end]

        kv_cache_k.append(K_chunk)
        kv_cache_v.append(V_chunk)
        K_context = np.concatenate(kv_cache_k, axis=0)
        V_context = np.concatenate(kv_cache_v, axis=0)
        context_len = K_context.shape[0]

        print(f"   Chunk [{chunk_start}:{chunk_end}] | context_len={context_len}")

        # 3. CUDA 端：传输数据到 GPU
        Q_gpu = gpuarray.to_gpu(Q_chunk.astype(np.float32))
        K_gpu = gpuarray.to_gpu(K_context.astype(np.float32))
        V_gpu = gpuarray.to_gpu(V_context.astype(np.float32))
        O_gpu = gpuarray.zeros((curr_chunk_len, head_dim), dtype=np.float32)

        # 4. CUDA 端：调用 kernel 计算
        kernel(
            Q_gpu, K_gpu, V_gpu, O_gpu,
            np.int32(chunk_start),
            np.int32(curr_chunk_len),
            np.int32(head_dim),
            np.float32(scale),
            block=(256, 1, 1),
            grid=((curr_chunk_len + 255) // 256, 1, 1),
        )
        O_chunked[chunk_start:chunk_end] = O_gpu.get()

    print()
    print("5. 结果验证（与 CPU 完整因果注意力对比）")
    # CPU 参考：完整因果注意力
    O_ref = np.zeros_like(Q_full)
    for q_idx in range(seq_len):
        ctx_len = q_idx + 1
        scores = (Q_full[q_idx] @ K_full[:ctx_len].T) * scale
        probs = np.exp(scores - np.max(scores))
        probs /= np.sum(probs)
        O_ref[q_idx] = probs @ V_full[:ctx_len]

    max_diff = np.max(np.abs(O_chunked - O_ref))
    mean_diff = np.mean(np.abs(O_chunked - O_ref))
    print(f"   输出形状: {O_chunked.shape}")
    print(f"   与参考最大误差: {max_diff:.2e}")
    print(f"   与参考平均误差: {mean_diff:.2e}")
    print(f"   输出示例（前3个token）:")
    print(O_chunked[:3, :5])
    if max_diff < 1e-4:
        print("   ✓ 验证通过：Chunked Prefill 与完整注意力结果一致")
    else:
        print("   ⚠ 误差较大，请检查实现")
    print()

    print("=== Demo 完成 ===")
    print("\n总结:")
    print("✓ Python 端：将序列划分为 chunk，管理 KV cache 累积")
    print("✓ Python 端：每 chunk 调度 CUDA kernel")
    print("✓ CUDA 端：单 chunk 因果注意力（Q attend to cached + current KV）")
    print("✓ 峰值显存由 chunk_size 控制，而非完整 seq_len")


def demo_without_cuda():
    """不使用 CUDA 的简化示例（纯 Python）"""
    print("=== Chunked Prefill (CPU) Demo ===\n")

    seq_len = 32
    chunk_size = 8
    head_dim = 64
    scale = 1.0 / np.sqrt(head_dim)

    Q_full = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    K_full = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1
    V_full = np.random.randn(seq_len, head_dim).astype(np.float32) * 0.1

    O_chunked = np.zeros_like(Q_full)
    kv_k, kv_v = [], []

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        curr_len = chunk_end - chunk_start

        Q_c = Q_full[chunk_start:chunk_end]
        K_c = K_full[chunk_start:chunk_end]
        V_c = V_full[chunk_start:chunk_end]
        kv_k.append(K_c)
        kv_v.append(V_c)
        K_ctx = np.concatenate(kv_k, axis=0)
        V_ctx = np.concatenate(kv_v, axis=0)

        for q_idx in range(curr_len):
            ctx_len = chunk_start + q_idx + 1
            scores = (Q_c[q_idx] @ K_ctx[:ctx_len].T) * scale
            probs = np.exp(scores - np.max(scores))
            probs /= np.sum(probs)
            O_chunked[chunk_start + q_idx] = probs @ V_ctx[:ctx_len]

        print(f"Chunk [{chunk_start}:{chunk_end}] 完成")

    print(f"\n输出形状: {O_chunked.shape}")
    print("输出示例:", O_chunked[:3, :3])


if __name__ == "__main__":
    if PYCUDA_AVAILABLE:
        demo_with_cuda()
    else:
        demo_without_cuda()

