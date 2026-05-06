
## `gemm.cu`
```bash
nvcc gemm.cu -o gemm -lcublas
```
生成可执行文件 `gemm`。

---

## `cutlass_flashAttention_cute_v2.cu` / `cutlass_flashAttention_v1.cu` / `tensorcore-via-register.cu`

**编辑 `CMakeLists.txt`**（例如将目标源文件改为你需要的那一个），然后使用 CMake 构建

```bash
mkdir build && cd build
cmake ..
cmake --build .
```



