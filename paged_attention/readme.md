# Paged Attention Implementation Example

This example demonstrates how to implement Paged Attention, divided into two parts:

## 1. Python Side: Logic Management (Bookkeeping)

**File: `paged_attention.py`**

Responsibilities:
- **Page table management**: Records the mapping from logical blocks to physical blocks
  ```python
  page_table[logical_block_idx] = physical_block_idx
  ```
- **Physical block allocation**: Manages allocation and deallocation of physical memory blocks
- **Data mapping**: Maps logical data to physical memory according to the page table

**Key class: `PagedAttentionManager`**
- `create_page_table()`: Create page table
- `allocate_physical_block()`: Allocate physical block
- `load_data_to_physical()`: Load data to physical memory

## 2. CUDA Side: Numerical Computation (Calculation)

**File: `paged_attention.cu`**

Responsibilities:
- **Access physical memory via page table**: Uses page table to look up physical block indices
- **Compute attention**: Reads Q, K, V from physical memory and computes attention

**Key function: `paged_attention_kernel`**
```cuda
// Look up physical block via page table
int physical_block_idx = page_table[batch_idx * num_logical_blocks + logical_block_idx];

// Read data from physical memory
int q_offset = physical_block_idx * block_size * head_dim + offset_in_block * head_dim + d;
float q_val = __half2float(Q_physical[q_offset]);
```

## Workflow

```
1. Python side: Create page table
   └─> Logical block 0 -> Physical block 5
   └─> Logical block 1 -> Physical block 2
   └─> Logical block 2 -> Physical block 8
   └─> ...

2. Python side: Map data to physical memory
   └─> Logical block 0 data -> Physical block 5
   └─> Logical block 1 data -> Physical block 2
   └─> ...

3. CUDA side: Compute attention via page table
   └─> Need data from logical block 0?
   └─> Look up page table: Logical block 0 -> Physical block 5
   └─> Read from physical block 5
   └─> Compute attention
```

## Build and Run

### Option 1: Using PyCUDA SourceModule (Recommended, no compilation needed)

PyCUDA automatically compiles the CUDA kernel; no manual compilation required:

```bash
# PyCUDA is required
pip install pycuda

# Run the example directly
python paged_attention_demo.py
```

### Option 2: Compile to .so file (Optional)

If you need to compile to a standalone library:

```bash
nvcc -shared -o paged_attention.so paged_attention.cu -lcudart -arch=sm_80  -Xcompiler -fPIC
```

Note: When compiling to .so, keep the `extern "C"` wrapper functions.

## Key Advantages

1. **Memory reuse**: Physical blocks can be dynamically allocated and freed for memory reuse
2. **Flexibility**: The mapping between logical and physical blocks can be changed dynamically
3. **Efficiency**: CUDA kernel directly accesses physical memory via the page table without extra data reorganization

## File Description

- `paged_attention.cu`: CUDA kernel implementation
- `paged_attention.py`: Python-side page table management
- `paged_attention_wrapper.py`: Python-CUDA interface wrapper
- `paged_attention_demo.py`: Complete example

