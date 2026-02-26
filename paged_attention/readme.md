# Paged Attention 实现示例

这个示例展示了如何实现 Paged Attention，分为两部分：

## 1. Python 端：逻辑管理（记账）

**文件：`paged_attention.py`**

负责：
- **页表管理**：记录逻辑块到物理块的映射
  ```python
  page_table[logical_block_idx] = physical_block_idx
  ```
- **物理块分配**：管理物理内存块的分配和释放
- **数据映射**：根据页表将逻辑数据映射到物理内存

**关键类：`PagedAttentionManager`**
- `create_page_table()`: 创建页表
- `allocate_physical_block()`: 分配物理块
- `load_data_to_physical()`: 将数据加载到物理内存

## 2. CUDA 端：数值计算（算账）

**文件：`paged_attention.cu`**

负责：
- **根据页表访问物理内存**：使用页表查找物理块索引
- **计算注意力**：从物理内存读取 Q、K、V 并计算注意力

**关键函数：`paged_attention_kernel`**
```cuda
// 根据页表查找物理块
int physical_block_idx = page_table[batch_idx * num_logical_blocks + logical_block_idx];

// 从物理内存读取数据
int q_offset = physical_block_idx * block_size * head_dim + offset_in_block * head_dim + d;
float q_val = __half2float(Q_physical[q_offset]);
```

## 工作流程

```
1. Python 端：创建页表
   └─> 逻辑块 0 -> 物理块 5
   └─> 逻辑块 1 -> 物理块 2
   └─> 逻辑块 2 -> 物理块 8
   └─> ...

2. Python 端：将数据映射到物理内存
   └─> 逻辑块 0 的数据 -> 物理块 5
   └─> 逻辑块 1 的数据 -> 物理块 2
   └─> ...

3. CUDA 端：根据页表计算注意力
   └─> 需要逻辑块 0 的数据？
   └─> 查页表：逻辑块 0 -> 物理块 5
   └─> 从物理块 5 读取数据
   └─> 计算注意力
```

## 编译和运行

### 编译 CUDA kernel

```bash
nvcc -shared -Xcompiler -fPIC -o paged_attention.so paged_attention.cu -lcudart  -arch=sm_80
```

### 运行 Python 示例

```bash
# 需要 PyCUDA
pip install pycuda

# 运行示例
python paged_attention_demo.py
```

## 关键优势

1. **内存复用**：物理块可以动态分配和释放，实现内存复用
2. **灵活性**：逻辑块和物理块的映射可以动态改变
3. **效率**：CUDA kernel 直接根据页表访问物理内存，无需额外的数据重组

## 文件说明

- `paged_attention.cu`: CUDA kernel 实现
- `paged_attention.py`: Python 端页表管理
- `paged_attention_wrapper.py`: Python-CUDA 接口包装
- `paged_attention_demo.py`: 完整示例

