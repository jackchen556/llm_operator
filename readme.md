
---

**`cutlass_flashAttention_cute_v2.cu`**、**`cutlass_flashAttention_v1.cu`**、**`tensorcore-via-register.cu`**、**`gemm.cu`**

编辑 **`CMakeLists.txt`**（将目标源文件改成你需要的那一个），再用 CMake 构建：

```bash
mkdir build && cd build
cmake ..
cmake --build .
```


