"""
Paged Attention Implementation - Python 端（逻辑管理）

Python 端负责：
1. 页表管理：记录逻辑块到物理块的映射
2. 物理块分配：管理物理内存块的分配和释放
3. 调用 CUDA kernel：传递页表给 CUDA 进行计算
"""

import numpy as np
import ctypes
from ctypes import c_int, c_float, POINTER, c_void_p
import os

# 尝试加载 CUDA kernel 库（可选，如果使用 PyCUDA SourceModule 则不需要）
CUDA_LIB_AVAILABLE = False
try:
    # 假设 CUDA kernel 已编译为 .so 文件
    lib_path = os.path.join(os.path.dirname(__file__), "paged_attention.so")
    if os.path.exists(lib_path):
        lib = ctypes.CDLL(lib_path)
        
        # 定义函数签名
        lib.paged_attention_cuda.argtypes = [
            c_void_p,  # Q_physical
            c_void_p,  # K_physical
            c_void_p,  # V_physical
            c_void_p,  # O
            c_void_p,  # page_table
            c_int,     # batch_size
            c_int,     # block_size
            c_int,     # head_dim
            c_int,     # num_logical_blocks
            c_float    # scale
        ]
        lib.paged_attention_cuda.restype = None
        
        CUDA_LIB_AVAILABLE = True
except Exception:
    # 静默失败，使用 PyCUDA SourceModule 代替
    CUDA_LIB_AVAILABLE = False


class PagedAttentionManager:
    """
    Paged Attention 管理器
    
    负责逻辑块到物理块的映射管理（页表）
    """
    
    def __init__(self, block_size=16, head_dim=64, num_physical_blocks=1024):
        """
        初始化 Paged Attention 管理器
        
        Args:
            block_size: 每个块的大小（token 数）
            head_dim: 注意力头的维度
            num_physical_blocks: 物理内存块的总数
        """
        self.block_size = block_size
        self.head_dim = head_dim
        self.num_physical_blocks = num_physical_blocks
        
        # 页表：逻辑块 -> 物理块映射
        # page_table[batch_idx][logical_block_idx] = physical_block_idx
        self.page_tables = {}  # batch_id -> page_table (numpy array)
        
        # 物理块分配状态
        self.physical_block_allocated = np.zeros(num_physical_blocks, dtype=bool)
        self.next_free_block = 0
        
        # 物理内存中的数据（模拟）
        # 实际应该存储在 GPU 显存中
        self.Q_physical = None
        self.K_physical = None
        self.V_physical = None
        
    def allocate_physical_block(self):
        """分配一个物理块"""
        for i in range(self.num_physical_blocks):
            idx = (self.next_free_block + i) % self.num_physical_blocks
            if not self.physical_block_allocated[idx]:
                self.physical_block_allocated[idx] = True
                self.next_free_block = (idx + 1) % self.num_physical_blocks
                return idx
        raise RuntimeError("No free physical blocks available")
    
    def free_physical_block(self, physical_block_idx):
        """释放一个物理块"""
        if 0 <= physical_block_idx < self.num_physical_blocks:
            self.physical_block_allocated[physical_block_idx] = False
    
    def create_page_table(self, batch_id, num_logical_blocks):
        """
        为某个 batch 创建页表
        
        Args:
            batch_id: batch 标识符
            num_logical_blocks: 逻辑块数量
            
        Returns:
            page_table: [num_logical_blocks] 数组，存储物理块索引
        """
        page_table = np.zeros(num_logical_blocks, dtype=np.int32)
        
        # 为每个逻辑块分配物理块
        for logical_idx in range(num_logical_blocks):
            physical_idx = self.allocate_physical_block()
            page_table[logical_idx] = physical_idx
        
        self.page_tables[batch_id] = page_table
        return page_table
    
    def get_page_table(self, batch_id):
        """获取页表"""
        return self.page_tables.get(batch_id)
    
    def load_data_to_physical(self, Q, K, V, page_table):
        """
        将数据加载到物理内存
        
        Args:
            Q, K, V: 逻辑视图的数据 [num_logical_blocks, block_size, head_dim]
            page_table: 页表 [num_logical_blocks]
        """
        num_logical_blocks = len(page_table)
        
        # 分配物理内存（实际应该使用 cudaMalloc）
        if self.Q_physical is None:
            total_size = self.num_physical_blocks * self.block_size * self.head_dim
            self.Q_physical = np.zeros((total_size,), dtype=np.float16)
            self.K_physical = np.zeros((total_size,), dtype=np.float16)
            self.V_physical = np.zeros((total_size,), dtype=np.float16)
        
        # 根据页表将逻辑数据映射到物理内存
        # Q/K/V 形状: [num_logical_blocks, block_size, head_dim]，按逻辑块索引
        for logical_idx in range(num_logical_blocks):
            physical_idx = page_table[logical_idx]
            
            p_start = physical_idx * self.block_size * self.head_dim
            p_end = p_start + self.block_size * self.head_dim
            
            # 将逻辑块的数据复制到对应的物理块（每个逻辑块形状为 block_size * head_dim）
            self.Q_physical[p_start:p_end] = Q[logical_idx].flatten()
            self.K_physical[p_start:p_end] = K[logical_idx].flatten()
            self.V_physical[p_start:p_end] = V[logical_idx].flatten()
    
    def compute_attention(self, batch_id, scale=None):
        """
        计算注意力（调用 CUDA kernel）
        
        Args:
            batch_id: batch 标识符
            scale: 缩放因子
            
        Returns:
            O: 输出 [seq_len, head_dim]
        """
        page_table = self.get_page_table(batch_id)
        if page_table is None:
            raise ValueError(f"No page table found for batch {batch_id}")
        
        num_logical_blocks = len(page_table)
        seq_len = num_logical_blocks * self.block_size
        
        if scale is None:
            scale = 1.0 / np.sqrt(self.head_dim)
        
        # 这里应该调用 CUDA kernel
        # 实际实现需要：
        # 1. 将 page_table 传输到 GPU
        # 2. 调用 paged_attention_kernel
        # 3. 将结果传回 CPU
        
        # 简化版本：CPU 实现（用于演示）
        O = np.zeros((seq_len, self.head_dim), dtype=np.float16)
        
        # 模拟 CUDA kernel 的计算逻辑
        for seq_idx in range(seq_len):
            logical_block_idx = seq_idx // self.block_size
            offset_in_block = seq_idx % self.block_size
            physical_block_idx = page_table[logical_block_idx]
            
            # 读取 Q
            q_start = physical_block_idx * self.block_size * self.head_dim + \
                     offset_in_block * self.head_dim
            q_vec = self.Q_physical[q_start:q_start + self.head_dim]
            
            # 计算 QK^T
            scores = np.zeros(seq_len, dtype=np.float32)
            for k_block in range(num_logical_blocks):
                k_physical_block = page_table[k_block]
                for k_offset in range(self.block_size):
                    k_seq_idx = k_block * self.block_size + k_offset
                    if k_seq_idx >= seq_len:
                        break
                    
                    k_start = k_physical_block * self.block_size * self.head_dim + \
                             k_offset * self.head_dim
                    k_vec = self.K_physical[k_start:k_start + self.head_dim]
                    
                    score = np.dot(q_vec.astype(np.float32), k_vec.astype(np.float32)) * scale
                    scores[k_seq_idx] = score
            
            # Softmax
            max_score = np.max(scores)
            exp_scores = np.exp(scores - max_score)
            sum_exp = np.sum(exp_scores)
            
            # 计算输出
            for d in range(self.head_dim):
                out_val = 0.0
                for k_block in range(num_logical_blocks):
                    k_physical_block = page_table[k_block]
                    for k_offset in range(self.block_size):
                        k_seq_idx = k_block * self.block_size + k_offset
                        if k_seq_idx >= seq_len:
                            break
                        
                        v_start = k_physical_block * self.block_size * self.head_dim + \
                                 k_offset * self.head_dim
                        v_val = self.V_physical[v_start + d]
                        out_val += exp_scores[k_seq_idx] * float(v_val)
                
                O[seq_idx, d] = np.float16(out_val / sum_exp)
        
        return O


def demo_paged_attention():
    """演示 Paged Attention 的使用"""
    print("=== Paged Attention Demo ===\n")
    
    # 初始化管理器
    manager = PagedAttentionManager(block_size=16, head_dim=64, num_physical_blocks=100)
    
    # 创建一些逻辑数据
    num_logical_blocks = 4
    seq_len = num_logical_blocks * manager.block_size
    
    # 生成随机 Q, K, V
    Q_logical = np.random.randn(seq_len, manager.head_dim).astype(np.float16)
    K_logical = np.random.randn(seq_len, manager.head_dim).astype(np.float16)
    V_logical = np.random.randn(seq_len, manager.head_dim).astype(np.float16)
    
    print(f"逻辑数据形状:")
    print(f"  Q: {Q_logical.shape}")
    print(f"  K: {K_logical.shape}")
    print(f"  V: {V_logical.shape}")
    print(f"  逻辑块数: {num_logical_blocks}")
    print(f"  每块大小: {manager.block_size}")
    print()
    
    # 创建页表（Python 端：逻辑管理）
    batch_id = 0
    page_table = manager.create_page_table(batch_id, num_logical_blocks)
    print(f"页表（逻辑块 -> 物理块映射）:")
    for i, physical_idx in enumerate(page_table):
        print(f"  逻辑块 {i} -> 物理块 {physical_idx}")
    print()
    
    # 将数据加载到物理内存（根据页表）
    Q_reshaped = Q_logical.reshape(num_logical_blocks, manager.block_size, manager.head_dim)
    K_reshaped = K_logical.reshape(num_logical_blocks, manager.block_size, manager.head_dim)
    V_reshaped = V_logical.reshape(num_logical_blocks, manager.block_size, manager.head_dim)
    
    manager.load_data_to_physical(Q_reshaped, K_reshaped, V_reshaped, page_table)
    print("数据已加载到物理内存（根据页表映射）")
    print()
    
    # 计算注意力（CUDA 端：数值计算）
    print("计算注意力（使用页表访问物理内存）...")
    O = manager.compute_attention(batch_id)
    
    print(f"输出形状: {O.shape}")
    print(f"输出示例（前5个token的前3个维度）:")
    print(O[:5, :3])
    print()
    
    print("=== Demo 完成 ===")
    print("\n关键点:")
    print("1. Python 端管理页表：逻辑块 -> 物理块映射")
    print("2. CUDA 端根据页表访问物理内存进行计算")
    print("3. 物理块可以动态分配和释放，实现内存复用")


if __name__ == "__main__":
    demo_paged_attention()

