"""
Paged Attention Implementation - Python 端（逻辑管理）

GQA：num_q_heads 个 Q 头共享 num_kv_heads 个 KV 头（kv_head = q_head // (num_q_heads // num_kv_heads)）。
"""

import numpy as np
import ctypes
from ctypes import c_int, c_float, c_void_p
import os

CUDA_LIB_AVAILABLE = False
try:
    lib_path = os.path.join(os.path.dirname(__file__), "paged_attention.so")
    if os.path.exists(lib_path):
        lib = ctypes.CDLL(lib_path)

        lib.paged_attention_cuda.argtypes = [
            c_void_p,
            c_void_p,
            c_void_p,
            c_void_p,
            c_void_p,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_float,
        ]
        lib.paged_attention_cuda.restype = None

        CUDA_LIB_AVAILABLE = True
except Exception:
    CUDA_LIB_AVAILABLE = False


class PagedAttentionManager:
    def __init__(
        self,
        block_size=16,
        head_dim=128,
        num_q_heads=28,
        num_kv_heads=4,
        num_physical_blocks=1024,
    ):
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads 必须能被 num_kv_heads 整除（GQA）")

        self.block_size = block_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        self.num_physical_blocks = num_physical_blocks

        self.page_tables = {}
        self.physical_block_allocated = np.zeros(num_physical_blocks, dtype=bool)
        self.next_free_block = 0

        self.K_physical = None
        self.V_physical = None

    def allocate_physical_block(self):
        for i in range(self.num_physical_blocks):
            idx = (self.next_free_block + i) % self.num_physical_blocks
            if not self.physical_block_allocated[idx]:
                self.physical_block_allocated[idx] = True
                self.next_free_block = (idx + 1) % self.num_physical_blocks
                return idx
        raise RuntimeError("No free physical blocks available")

    def free_physical_block(self, physical_block_idx):
        if 0 <= physical_block_idx < self.num_physical_blocks:
            self.physical_block_allocated[physical_block_idx] = False

    def create_page_table(self, batch_id, num_logical_blocks):
        page_table = np.zeros(num_logical_blocks, dtype=np.int32)
        for logical_idx in range(num_logical_blocks):
            physical_idx = self.allocate_physical_block()
            page_table[logical_idx] = physical_idx
        self.page_tables[batch_id] = page_table
        return page_table

    def get_page_table(self, batch_id):
        return self.page_tables.get(batch_id)

    def load_data_to_physical(self, K, V, page_table):
        """
        K, V: [num_logical_blocks, block_size, num_kv_heads, head_dim]
        """
        num_logical_blocks = len(page_table)
        kv_el = self.block_size * self.num_kv_heads * self.head_dim

        if self.K_physical is None:
            total_size = self.num_physical_blocks * kv_el
            self.K_physical = np.zeros((total_size,), dtype=np.float16)
            self.V_physical = np.zeros((total_size,), dtype=np.float16)

        for logical_idx in range(num_logical_blocks):
            physical_idx = page_table[logical_idx]
            p_start = physical_idx * kv_el
            p_end = p_start + kv_el
            self.K_physical[p_start:p_end] = K[logical_idx].reshape(-1)
            self.V_physical[p_start:p_end] = V[logical_idx].reshape(-1)

    def compute_attention(self, batch_id, Q_dense, scale=None):
        """
        Q_dense: [seq_len, num_q_heads, head_dim]
        返回 O: [seq_len, num_q_heads, head_dim]
        """
        page_table = self.get_page_table(batch_id)
        if page_table is None:
            raise ValueError(f"No page table found for batch {batch_id}")

        num_logical_blocks = len(page_table)
        seq_len = num_logical_blocks * self.block_size

        Q_dense = np.asarray(Q_dense)
        if Q_dense.shape != (seq_len, self.num_q_heads, self.head_dim):
            raise ValueError(
                f"Q_dense 形状应为 ({seq_len}, {self.num_q_heads}, {self.head_dim})，"
                f"实际为 {Q_dense.shape}"
            )

        if scale is None:
            scale = 1.0 / np.sqrt(self.head_dim)

        O = np.zeros((seq_len, self.num_q_heads, self.head_dim), dtype=np.float16)

        for seq_idx in range(seq_len):
            for qh in range(self.num_q_heads):
                kv_h = qh // self.num_queries_per_kv
                q_vec = Q_dense[seq_idx, qh].astype(np.float32)

                scores = np.zeros(seq_len, dtype=np.float32)
                for k_block in range(num_logical_blocks):
                    k_physical_block = page_table[k_block]
                    for k_offset in range(self.block_size):
                        k_seq_idx = k_block * self.block_size + k_offset
                        if k_seq_idx >= seq_len:
                            break
                        k_start = (
                            k_physical_block * self.block_size * self.num_kv_heads
                            * self.head_dim
                            + k_offset * self.num_kv_heads * self.head_dim
                            + kv_h * self.head_dim
                        )
                        k_end = k_start + self.head_dim
                        k_vec = self.K_physical[k_start:k_end].astype(np.float32)
                        scores[k_seq_idx] = float(np.dot(q_vec, k_vec) * scale)

                max_score = np.max(scores)
                exp_scores = np.exp(scores - max_score)
                sum_exp = np.sum(exp_scores)

                for d in range(self.head_dim):
                    out_val = 0.0
                    for k_block in range(num_logical_blocks):
                        k_physical_block = page_table[k_block]
                        for k_offset in range(self.block_size):
                            k_seq_idx = k_block * self.block_size + k_offset
                            if k_seq_idx >= seq_len:
                                break
                            v_idx = (
                                k_physical_block * self.block_size * self.num_kv_heads
                                * self.head_dim
                                + k_offset * self.num_kv_heads * self.head_dim
                                + kv_h * self.head_dim
                                + d
                            )
                            v_val = float(self.V_physical[v_idx])
                            out_val += exp_scores[k_seq_idx] * v_val
                    O[seq_idx, qh, d] = np.float16(out_val / sum_exp)

        return O


def demo_paged_attention():
    print("=== Paged Attention Demo (GQA) ===\n")

    manager = PagedAttentionManager(
        block_size=16,
        head_dim=128,
        num_q_heads=28,
        num_kv_heads=4,
        num_physical_blocks=100,
    )

    num_logical_blocks = 4
    seq_len = num_logical_blocks * manager.block_size

    Q_logical = np.random.randn(seq_len, manager.num_q_heads, manager.head_dim).astype(
        np.float16
    )
    K_logical = np.random.randn(seq_len, manager.num_kv_heads, manager.head_dim).astype(
        np.float16
    )
    V_logical = np.random.randn(seq_len, manager.num_kv_heads, manager.head_dim).astype(
        np.float16
    )

    print(f"  Q: {Q_logical.shape}  K/V: {K_logical.shape} / {V_logical.shape}")
    print()

    batch_id = 0
    page_table = manager.create_page_table(batch_id, num_logical_blocks)
    print("页表（逻辑块 -> 物理块）:")
    for i, physical_idx in enumerate(page_table):
        print(f"  逻辑块 {i} -> 物理块 {physical_idx}")
    print()

    K_reshaped = K_logical.reshape(
        num_logical_blocks, manager.block_size, manager.num_kv_heads, manager.head_dim
    )
    V_reshaped = V_logical.reshape(
        num_logical_blocks, manager.block_size, manager.num_kv_heads, manager.head_dim
    )

    manager.load_data_to_physical(K_reshaped, V_reshaped, page_table)
    print("K/V 已加载到物理内存；Q 为稠密 [seq, num_q_heads, head_dim]")
    print()

    print("计算注意力...")
    O = manager.compute_attention(batch_id, Q_logical)

    print(f"输出形状: {O.shape}")
    print(f"示例 O[0, 0, :5]: {O[0, 0, :5]}")
    print()

    print("=== Demo 完成 ===")


if __name__ == "__main__":
    demo_paged_attention()

