"""
Paged Attention 完整示例

展示 Python（逻辑管理）和 CUDA（数值计算）的结合
"""

import numpy as np
import sys
import os

# 导入重命名后的 Python 模块，避免与可能的 .so 文件冲突
from paged_attention_manager import PagedAttentionManager

try:
    import pycuda.autoinit
    import pycuda.driver as cuda_driver
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
    
    print("=== Paged Attention with CUDA Demo ===\n")
    
    # 1. Python 端：初始化管理器（逻辑管理）
    manager = PagedAttentionManager(block_size=16, head_dim=64, num_physical_blocks=100)
    
    batch_size = 2
    num_logical_blocks = 4
    seq_len = num_logical_blocks * manager.block_size
    
    # 生成测试数据（使用 float32，因为 PyCUDA 的 gpuarray 不支持 float16）
    Q_logical = np.random.randn(batch_size, seq_len, manager.head_dim).astype(np.float32)
    K_logical = np.random.randn(batch_size, seq_len, manager.head_dim).astype(np.float32)
    V_logical = np.random.randn(batch_size, seq_len, manager.head_dim).astype(np.float32)
    
    print("1. Python 端：创建页表（逻辑管理）")
    page_tables = []
    for batch_id in range(batch_size):
        page_table = manager.create_page_table(batch_id, num_logical_blocks)
        page_tables.append(page_table)
        print(f"   Batch {batch_id} 页表: {page_table}")
    print()
    
    # 2. Python 端：将数据加载到物理内存（根据页表）
    print("2. Python 端：将数据映射到物理内存")
    # 这里简化处理，实际应该为每个 batch 分别管理
    # 使用 float32，因为 PyCUDA 的 gpuarray 不支持 float16
    Q_physical = np.zeros((manager.num_physical_blocks, manager.block_size, manager.head_dim), 
                          dtype=np.float32)
    K_physical = np.zeros((manager.num_physical_blocks, manager.block_size, manager.head_dim), 
                          dtype=np.float32)
    V_physical = np.zeros((manager.num_physical_blocks, manager.block_size, manager.head_dim), 
                          dtype=np.float32)
    
    for batch_id in range(batch_size):
        page_table = page_tables[batch_id]
        for logical_idx in range(num_logical_blocks):
            physical_idx = page_table[logical_idx]
            Q_physical[physical_idx] = Q_logical[batch_id, 
                                                  logical_idx * manager.block_size:
                                                  (logical_idx + 1) * manager.block_size]
            K_physical[physical_idx] = K_logical[batch_id,
                                                  logical_idx * manager.block_size:
                                                  (logical_idx + 1) * manager.block_size]
            V_physical[physical_idx] = V_logical[batch_id,
                                                  logical_idx * manager.block_size:
                                                  (logical_idx + 1) * manager.block_size]
    print(f"   物理内存形状: {Q_physical.shape}")
    print()
    
    # 3. CUDA 端：将数据传输到 GPU
    print("3. CUDA 端：传输数据到 GPU")
    # PyCUDA 的 gpuarray 不支持 float16，使用 float32
    Q_physical_float32 = Q_physical.astype(np.float32).flatten()
    K_physical_float32 = K_physical.astype(np.float32).flatten()
    V_physical_float32 = V_physical.astype(np.float32).flatten()
    
    Q_physical_gpu = gpuarray.to_gpu(Q_physical_float32)
    K_physical_gpu = gpuarray.to_gpu(K_physical_float32)
    V_physical_gpu = gpuarray.to_gpu(V_physical_float32)
    
    # 将页表传输到 GPU
    page_table_gpu = gpuarray.to_gpu(np.array(page_tables[0], dtype=np.int32))
    
    # 分配输出内存（使用 float32，PyCUDA 不支持 float16）
    O_gpu = gpuarray.zeros((batch_size, seq_len, manager.head_dim), dtype=np.float32)
    print("   数据已传输到 GPU")
    print()
    
    # 4. CUDA 端：调用 kernel 计算（数值计算）
    print("4. CUDA 端：调用 kernel 计算注意力")
    print("   （根据页表从物理内存读取数据）")
    
    # 读取 CUDA kernel 代码
    cuda_file = os.path.join(os.path.dirname(__file__), "paged_attention.cu")
    with open(cuda_file, "r") as f:
        cuda_code = f.read()
    
    # 编译并调用
    mod = SourceModule(cuda_code)
    kernel = mod.get_function("paged_attention_kernel")
    
    scale = 1.0 / np.sqrt(manager.head_dim)
    
    for batch_id in range(batch_size):
        page_table_gpu = gpuarray.to_gpu(np.array(page_tables[batch_id], dtype=np.int32))
        
        kernel(
            Q_physical_gpu, K_physical_gpu, V_physical_gpu,
            O_gpu[batch_id:batch_id+1],
            page_table_gpu,
            np.int32(manager.block_size),
            np.int32(manager.head_dim),
            np.int32(num_logical_blocks),
            np.float32(scale),
            block=(256, 1, 1),
            grid=(1, (seq_len + 255) // 256, 1)
        )
    
    # 5. 将结果传回 CPU
    O_float32 = O_gpu.get()
    # 转换回 float16（如果需要）
    O = O_float32.astype(np.float16)
    print(f"   输出形状: {O.shape}")
    print()
    
    print("5. 结果验证")
    print(f"   输出示例（Batch 0, 前3个token）:")
    print(O[0, :3, :5])
    print()
    
    print("=== Demo 完成 ===")
    print("\n总结:")
    print("✓ Python 端：管理页表（逻辑块 -> 物理块映射）")
    print("✓ Python 端：管理物理块分配和释放")
    print("✓ CUDA 端：根据页表从物理内存读取数据")
    print("✓ CUDA 端：执行注意力计算")


def demo_without_cuda():
    """不使用 CUDA 的简化示例（纯 Python）"""
    print("=== Paged Attention (CPU) Demo ===\n")
    
    manager = PagedAttentionManager(block_size=16, head_dim=64, num_physical_blocks=100)
    
    num_logical_blocks = 4
    seq_len = num_logical_blocks * manager.block_size
    
    Q_logical = np.random.randn(seq_len, manager.head_dim).astype(np.float16)
    K_logical = np.random.randn(seq_len, manager.head_dim).astype(np.float16)
    V_logical = np.random.randn(seq_len, manager.head_dim).astype(np.float16)
    
    # Python 端：创建页表
    batch_id = 0
    page_table = manager.create_page_table(batch_id, num_logical_blocks)
    print("页表（逻辑块 -> 物理块）:")
    for i, p in enumerate(page_table):
        print(f"  逻辑块 {i} -> 物理块 {p}")
    print()
    
    # 加载数据到物理内存
    Q_reshaped = Q_logical.reshape(num_logical_blocks, manager.block_size, manager.head_dim)
    K_reshaped = K_logical.reshape(num_logical_blocks, manager.block_size, manager.head_dim)
    V_reshaped = V_logical.reshape(num_logical_blocks, manager.block_size, manager.head_dim)
    
    manager.load_data_to_physical(Q_reshaped, K_reshaped, V_reshaped, page_table)
    
    # 计算注意力
    O = manager.compute_attention(batch_id)
    
    print(f"输出形状: {O.shape}")
    print(f"输出示例:")
    print(O[:3, :5])


if __name__ == "__main__":
    #if PYCUDA_AVAILABLE:
    demo_with_cuda()
    #else:
    #    demo_without_cuda()

