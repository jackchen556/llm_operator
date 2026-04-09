import numpy as np
import os

from paged_attention_manager import PagedAttentionManager

try:
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("Warning: PyCUDA not available, using CPU fallback")


def cpu_paged_attention_gqa_batch(
    Q,
    K_phys,
    V_phys,
    page_table,
    block_size,
    head_dim,
    num_q_heads,
    num_kv_heads,
    num_logical_blocks,
    scale,
):
    """
    与 paged_attention.cu 一致的 GQA 全序列注意力（float32），用于校验 GPU。
    Q: [seq_len, num_q_heads, head_dim]
    K_phys / V_phys: [num_physical_blocks, block_size, num_kv_heads, head_dim]
    page_table: [num_logical_blocks]
    """
    seq_len = num_logical_blocks * block_size
    queries_per_kv = num_q_heads // num_kv_heads
    O = np.zeros((seq_len, num_q_heads, head_dim), dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    K_phys = np.asarray(K_phys, dtype=np.float32)
    V_phys = np.asarray(V_phys, dtype=np.float32)

    for seq_idx in range(seq_len):
        for qh in range(num_q_heads):
            kv_h = qh // queries_per_kv
            q_vec = Q[seq_idx, qh]
            scores = np.zeros(seq_len, dtype=np.float32)
            for k_block in range(num_logical_blocks):
                pb = page_table[k_block]
                for k_offset in range(block_size):
                    k_seq_idx = k_block * block_size + k_offset
                    if k_seq_idx >= seq_len:
                        break
                    k_vec = K_phys[pb, k_offset, kv_h]
                    scores[k_seq_idx] = float(np.dot(q_vec, k_vec) * scale)
            max_s = np.max(scores)
            exp_s = np.exp(scores - max_s)
            sum_e = np.sum(exp_s)
            w = exp_s / sum_e
            for d in range(head_dim):
                acc = 0.0
                for k_block in range(num_logical_blocks):
                    pb = page_table[k_block]
                    for k_offset in range(block_size):
                        k_seq_idx = k_block * block_size + k_offset
                        if k_seq_idx >= seq_len:
                            break
                        acc += w[k_seq_idx] * V_phys[pb, k_offset, kv_h, d]
                O[seq_idx, qh, d] = acc
    return O


def demo_with_cuda():
    if not PYCUDA_AVAILABLE:
        print("PyCUDA not available, skipping CUDA demo")
        return

    print("=== Paged Attention with CUDA Demo (GQA) ===\n")

    manager = PagedAttentionManager(
        block_size=16,
        head_dim=128,
        num_q_heads=28,
        num_kv_heads=4,
        num_physical_blocks=100,
    )

    batch_size = 2
    num_logical_blocks = 4
    seq_len = num_logical_blocks * manager.block_size

    Q_logical = np.random.randn(
        batch_size, seq_len, manager.num_q_heads, manager.head_dim
    ).astype(np.float32)
    K_logical = np.random.randn(
        batch_size, seq_len, manager.num_kv_heads, manager.head_dim
    ).astype(np.float32)
    V_logical = np.random.randn(
        batch_size, seq_len, manager.num_kv_heads, manager.head_dim
    ).astype(np.float32)

    print("1. Python 端：创建页表（逻辑管理）")
    page_tables = []
    for batch_id in range(batch_size):
        page_table = manager.create_page_table(batch_id, num_logical_blocks)
        page_tables.append(page_table)
        print(f"   Batch {batch_id} 页表: {page_table}")
    print()

    print("2. 仅将 K/V 映射到物理内存 [phys, block, num_kv_heads, head_dim]")

    K_physical = np.zeros(
        (
            manager.num_physical_blocks,
            manager.block_size,
            manager.num_kv_heads,
            manager.head_dim,
        ),
        dtype=np.float32,
    )
    V_physical = np.zeros_like(K_physical)

    for batch_id in range(batch_size):
        page_table = page_tables[batch_id]
        for logical_idx in range(num_logical_blocks):
            physical_idx = page_table[logical_idx]
            sl = slice(
                logical_idx * manager.block_size,
                (logical_idx + 1) * manager.block_size,
            )
            K_physical[physical_idx] = K_logical[batch_id, sl]
            V_physical[physical_idx] = V_logical[batch_id, sl]

    print(f"   K/V 物理池形状: {K_physical.shape}")
    print()

    print("3. 传输到 GPU")
    K_physical_gpu = gpuarray.to_gpu(K_physical.ravel())
    V_physical_gpu = gpuarray.to_gpu(V_physical.ravel())

    Q_gpu = gpuarray.to_gpu(np.ascontiguousarray(Q_logical).ravel())
    O_gpu = gpuarray.zeros(
        (batch_size, seq_len, manager.num_q_heads, manager.head_dim),
        dtype=np.float32,
    )

    page_table_flat = np.concatenate(page_tables).astype(np.int32)
    page_table_gpu = gpuarray.to_gpu(page_table_flat)
    print("   已上传 Q / K / V / 页表（页表按 batch 展平拼接）")
    print()

    print("4. 调用 kernel（grid: batch × num_q_heads × seq 分块）")

    cuda_file = os.path.join(os.path.dirname(__file__), "paged_attention.cu")
    with open(cuda_file, "r", encoding="utf-8") as f:
        cuda_code = f.read()

    mod = SourceModule(cuda_code)
    kernel = mod.get_function("paged_attention_kernel")

    scale = 1.0 / np.sqrt(manager.head_dim)

    kernel(
        Q_gpu,
        K_physical_gpu,
        V_physical_gpu,
        O_gpu,
        page_table_gpu,
        np.int32(manager.block_size),
        np.int32(manager.head_dim),
        np.int32(manager.num_q_heads),
        np.int32(manager.num_kv_heads),
        np.int32(num_logical_blocks),
        np.float32(scale),
        block=(256, 1, 1),
        grid=(
            batch_size,
            manager.num_q_heads,
            (seq_len + 255) // 256,
        ),
    )

    O_cuda = O_gpu.get()
    print(f"   GPU 输出形状: {O_cuda.shape} (float32)")
    print()

    print("5. CPU 参考实现对比（与 GPU 同一份 Q / K_phys / V_phys / 页表）")
    rtol, atol = 1e-4, 1e-5
    for b in range(batch_size):
        O_cpu = cpu_paged_attention_gqa_batch(
            Q_logical[b],
            K_physical,
            V_physical,
            page_tables[b],
            manager.block_size,
            manager.head_dim,
            manager.num_q_heads,
            manager.num_kv_heads,
            num_logical_blocks,
            scale,
        )
        diff = np.abs(O_cuda[b] - O_cpu)
        max_err = float(np.max(diff))
        mean_err = float(np.mean(diff))
        ok = np.allclose(O_cuda[b], O_cpu, rtol=rtol, atol=atol)
        print(f"   Batch {b}: max|GPU-CPU|={max_err:.6e}, mean={mean_err:.6e}, "
              f"allclose(rtol={rtol}, atol={atol})={'PASS' if ok else 'FAIL'}")
    print()

    O = O_cuda.astype(np.float16)
    print("6. 结果示例（Batch 0, token 0, Q 头 0 前 5 维, float16 展示）:")
    print(O[0, 0, 0, :5])
    print()

    print("=== Demo 完成 ===")
    print("GQA: 28 个 Q 头共享 4 个 KV 头（每 KV 头对应 7 个 Q 头）")


def demo_without_cuda():
    print("=== Paged Attention (CPU) Demo ===\n")

    manager = PagedAttentionManager(
        block_size=16,
        head_dim=128,
        num_q_heads=28,
        num_kv_heads=4,
        num_physical_blocks=100,
    )

    num_logical_blocks = 4
    seq_len = num_logical_blocks * manager.block_size

    Q_logical = np.random.randn(
        seq_len, manager.num_q_heads, manager.head_dim
    ).astype(np.float16)
    K_logical = np.random.randn(
        seq_len, manager.num_kv_heads, manager.head_dim
    ).astype(np.float16)
    V_logical = np.random.randn(
        seq_len, manager.num_kv_heads, manager.head_dim
    ).astype(np.float16)

    batch_id = 0
    page_table = manager.create_page_table(batch_id, num_logical_blocks)
    print("页表（逻辑块 -> 物理块）:")
    for i, p in enumerate(page_table):
        print(f"  逻辑块 {i} -> 物理块 {p}")
    print()

    K_reshaped = K_logical.reshape(
        num_logical_blocks, manager.block_size, manager.num_kv_heads, manager.head_dim
    )
    V_reshaped = V_logical.reshape(
        num_logical_blocks, manager.block_size, manager.num_kv_heads, manager.head_dim
    )

    manager.load_data_to_physical(K_reshaped, V_reshaped, page_table)

    O = manager.compute_attention(batch_id, Q_logical)

    print(f"输出形状: {O.shape}")
    print("输出示例 O[0, 0, :5]:")
    print(O[0, 0, :5])


if __name__ == "__main__":
    if PYCUDA_AVAILABLE:
        demo_with_cuda()
    else:
        demo_without_cuda()

