
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include <fstream>
#include "NvInfer.h"
#include <string>
#include <vector>
#include <cassert>

#define SEGMENTS 128

namespace ptextlxlink {

  typedef volatile uint32_t vu32;
  typedef struct {
      vu32 magic;             // 0x00
      vu32 control;           // 0x04
      vu32 control_w1s;       // 0x08
      vu32 control_w1c;       // 0x0C
      vu32 reserved1[12];     // 0x10 - 0x3C
      vu32 status;            // 0x40
      vu32 status_rc;         // 0x44
      vu32 desc_done;         // 0x48
      vu32 alignments;        // 0x4C
      vu32 reserved2[16];     // 0x50 - 0x8C
      vu32 irq_enmask;        // 0x90
      vu32 irq_enmask_w1s;    // 0x94
      vu32 irq_enmask_w1c;    // 0x98
      vu32 reserved3[9];      // 0x9C - 0xBC
      vu32 perf_control;      // 0xC0
      vu32 perf_cycle;        // 0xC4
      vu32 perf_cyc_max;      // 0xC8
      vu32 perf_cyc_data;     // 0xCC
      vu32 perf_cyc_data_max; // 0xD0
  } xdma_reg_t;
  
  // 32位版本的 volatile 全局内存读取指令
  // 使用 ld.volatile.global.u32 指令强制从全局内存重新读取32位数据
  // 避免编译器优化和缓存，确保读取到最新的内存值
  __device__  __forceinline__ uint32_t ld_volatile_global_u32(const uint32_t *ptr) {
      uint32_t ret;
      asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
      return ret;
  }
  
  __device__ __forceinline__ void st_na_release(const uint32_t *ptr, uint32_t val) {
      asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
  }


  __device__  __forceinline__ int64_t ld_volatile_global(const uint64_t *ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}


__global__ void unify_kernel_package_kv3_quant_badperformance(
    __half2* __restrict__ inptrk,
    __half2* __restrict__ inptrv,
    volatile char* __restrict__ pinptr,
    int align_fpga,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 1;
    static constexpr int vec = 2;

    int offvec_c = 64/tsz/vec;
    int offvec_t = 64/tsz/vec;
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _Loopid = blockIdx.z;
    int _Tks = blockDim.x;
    int _Lps = (blockDim.x + pack_num - 1)/ pack_num; // TODO: % == 0
    int _tid = threadIdx.x % pack_num;
    int _pid = threadIdx.x / pack_num;
    int _Kvid = threadIdx.y;

    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;
    int _tokenid = _Loopid*_Tks+_pid*pack_num+_tid;

    int _loop = (seq_size + pack_num - 1)/pack_num;
    int _Kv_offset = _Kvid == 0 ? 0 : _loop*(offvec_c + pack_num*head_dim/vec + offvec_t);
    int _htype = _Kvid == 0 ? htype : htype + 1;
    __half2* inptr = _Kvid == 0 ? inptrk : inptrv;

    // if (_tokenid >= seq_size) return;
    
    __half2* h_ptr = inptr + (_tokenid*_Nh + _headid)*head_dim/vec;
    volatile char2* p_ptr = reinterpret_cast<volatile char2*>(pinptr + _Npid*align_fpga/tsz);

    int idx_l = _Kv_offset + (_Loopid*_Lps+_pid)*(offvec_c + pack_num*head_dim/vec + offvec_t);

    if (_tid == 0)
    {
        int iidy_c = (idx_l + 0) / interleave; 
        int iidx_c = (idx_l + 0) % interleave;
        int idx_c = iidy_c * _Ncg * interleave + _Ncgid * interleave + iidx_c; 
        volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + idx_c);
        *c_ptr = uint16_t(((_htype&0x1)<<14) | ((tkidx+_Loopid*_Lps+_pid)&0x3FFF));
    }

    float local_max= 0.f;
    for (int i = 0; i < head_dim/vec; i++)
    {
        local_max = fmaxf(local_max, fabsf(__half2float(h_ptr[i].x)));
        local_max = fmaxf(local_max, fabsf(__half2float(h_ptr[i].y)));
    }
    float h_scale = 1.f / (local_max + 1e-6f);
    for (int i = 0; i < head_dim/vec; i++)
    {
        int iidy_b = (idx_l + offvec_c + _tid*head_dim/vec + i) / interleave; 
        int iidx_b = (idx_l + offvec_c + _tid*head_dim/vec + i) % interleave;
        int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
        p_ptr[idx_b].x = char(__float2int_rn(__half2float(h_ptr[i].x) * h_scale * 127.f));
        p_ptr[idx_b].y = char(__float2int_rn(__half2float(h_ptr[i].y) * h_scale * 127.f));
    }

    int iidy_t = (idx_l + offvec_c + pack_num*head_dim/vec + _tid) / interleave; 
    int iidx_t = (idx_l + offvec_c + pack_num*head_dim/vec + _tid) % interleave;
    int idx_t = iidy_t * _Ncg * interleave + _Ncgid * interleave + iidx_t; 
    volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
    *t_ptr = __float2half_rn(local_max);
}

#define FINAL_MASK 0xFFFFFFFF

template <typename T, int NUM>
__inline__ __device__ T warpReduceMaxV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1){
            val[i] = fmaxf(val[i], __shfl_xor_sync(FINAL_MASK, val[i], mask, 32));
        } 
    }
    return (T) (0.0f);
}

// TODO 单播
__global__ void unify_kernel_package_kv4_quant(
    __half2* __restrict__ inptrk,
    __half2* __restrict__ inptrv,
    volatile char* __restrict__ pinptr,
    int align_fpga,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int NcgNhy,
    int NcgNty,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 1;
    static constexpr int vec = 2;

    int offvec_c = 64/tsz/vec;
    int offvec_t = 64/tsz/vec;
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _Loopid = blockIdx.z;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Kvid = threadIdx.z;

    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid / NcgNty;
    int _Ntyid = _Ncgid % NcgNty;
    int _Nhz = _Np;
    int _Nhy = NcgNhy;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _pkg_num = ((seq_size + NcgNty - 1)/NcgNty + pack_num - 1)/pack_num; // TODO: % == 0
    int _Kv_offset = _Kvid == 0 ? 0 : _pkg_num*(offvec_c + pack_num*head_dim/vec + offvec_t);  
    int _htype = _Kvid == 0 ? htype : htype + 1;
    __half2* inptr = _Kvid == 0 ? inptrk : inptrv;
    
    int _loop = (SEGMENTS + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++) 
    {
        int _lidx = _Loopid * _loop + l;

        // if ((_Ntyid * _pkg_num + _lidx)*pack_num+_pid >= seq_size) return;

        __half2* h_ptr = inptr + ((_Ntyid * _pkg_num + _lidx)*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;
        volatile char2* p_ptr = reinterpret_cast<volatile char2*>(pinptr + _Npid*align_fpga/tsz);

        int idx_l = _Kv_offset + _lidx*(offvec_c + pack_num*head_dim/vec + offvec_t);

        if (_tid == 0 && _pid == 0)
        {
            int iidy_c = (idx_l + 0) / interleave; 
            int iidx_c = (idx_l + 0) % interleave;
            int idx_c = iidy_c * _Ncg * interleave + _Ncgid * interleave + iidx_c; 
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + idx_c);
            *c_ptr = uint16_t(((_htype&0x1)<<14) | ((tkidx/NcgNty+_lidx+_pid)&0x3FFF)); // TODO: tkidx % NcgNty==0
        }

        float local_max[1] = {0.f};
        for (int i = _tid; i < head_dim/vec; i+=blockDim.x) {
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].x)));
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].y)));
        }
        warpReduceMaxV2<float, 1>(local_max);
        float h_scale = 1.f / (local_max[0] + 1e-6f);
        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (idx_l + offvec_c + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (idx_l + offvec_c + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
            p_ptr[idx_b].x = char(__float2int_rn(__half2float(h_ptr[i].x) * h_scale * 127.f));
            p_ptr[idx_b].y = char(__float2int_rn(__half2float(h_ptr[i].y) * h_scale * 127.f));
        }

        if (_tid == 0)
        {
            int iidy_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid) / interleave; 
            int iidx_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid) % interleave;
            int idx_t = iidy_t * _Ncg * interleave + _Ncgid * interleave + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
            *t_ptr = __float2half_rn(local_max[0]);
        }
    }
}

__global__ void unify_kernel_package_kv3_quant(
    __half2* __restrict__ inptrk,
    __half2* __restrict__ inptrv,
    volatile char* __restrict__ pinptr,
    int align_fpga,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 1;
    static constexpr int vec = 2;

    int offvec_c = 64/tsz/vec;
    int offvec_t = 64/tsz/vec;
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _Loopid = blockIdx.z;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Kvid = threadIdx.z;

    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _Kv_offset = _Kvid == 0 ? 0 : ((seq_size + pack_num - 1)/pack_num)*(offvec_c + pack_num*head_dim/vec + offvec_t);
    int _htype = _Kvid == 0 ? htype : htype + 1;
    __half2* inptr = _Kvid == 0 ? inptrk : inptrv;
    
    int _loop = (SEGMENTS + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++) 
    {
        int _lidx = _Loopid * _loop + l;

        // if (_lidx*pack_num+_pid >= seq_size) return;

        __half2* h_ptr = inptr + (_lidx*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;
        volatile char2* p_ptr = reinterpret_cast<volatile char2*>(pinptr + _Npid*align_fpga/tsz);

        int idx_l = _Kv_offset + _lidx*(offvec_c + pack_num*head_dim/vec + offvec_t);

        if (_tid == 0 && _pid == 0)
        {
            int iidy_c = (idx_l + 0) / interleave; 
            int iidx_c = (idx_l + 0) % interleave;
            int idx_c = iidy_c * _Ncg * interleave + _Ncgid * interleave + iidx_c; 
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + idx_c);
            *c_ptr = uint16_t(((_htype&0x1)<<14) | ((tkidx+_lidx+_pid)&0x3FFF));
        }

        float local_max[1] = {0.f};
        for (int i = _tid; i < head_dim/vec; i+=blockDim.x) {
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].x)));
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].y)));
        }
        warpReduceMaxV2<float, 1>(local_max);
        float h_scale = 1.f / (local_max[0] + 1e-6f);
        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (idx_l + offvec_c + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (idx_l + offvec_c + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
            p_ptr[idx_b].x = char(__float2int_rn(__half2float(h_ptr[i].x) * h_scale * 127.f));
            p_ptr[idx_b].y = char(__float2int_rn(__half2float(h_ptr[i].y) * h_scale * 127.f));
        }

        if (_tid == 0)
        {
            int iidy_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid) / interleave; 
            int iidx_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid) % interleave;
            int idx_t = iidy_t * _Ncg * interleave + _Ncgid * interleave + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
            *t_ptr = __float2half_rn(local_max[0]);
        }
    }
}

__global__ void unify_kernel_package_q3_quant(
    __half2* __restrict__ inptr,
    volatile char* __restrict__ pinptr,
    int align_fpga,
    int align_pack,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 1;
    static constexpr int vec = 2;
    static constexpr int offvec = 32; // 64/tsz/vec
    
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Lp = gridDim.z;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _Loopid = blockIdx.z;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _loop = (SEGMENTS + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++) // TODO: _loop循环转移到grid.z维度，性能优化
    {
        int _lidx = _Loopid * _loop + l;

        // if (_lidx*pack_num+_pid >= seq_size) return;

        __half2* h_ptr = inptr + (_lidx*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;
        volatile char2* p_ptr = reinterpret_cast<volatile char2*>(pinptr + _Npid*align_fpga/tsz + _lidx*_Ncg*align_pack/tsz);
            
        if (_tid == 0 && _pid==0)
        {
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + _Ncgid * interleave);
            *c_ptr = uint16_t(((htype&0x3)<<14) | ((tkidx+_lidx+_pid)&0x3FFF));
        } 

        float local_max[1] = {0.f};
        for (int i = _tid; i < head_dim/vec; i+=blockDim.x) {
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].x)));
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].y)));
        }
        warpReduceMaxV2<float, 1>(local_max);
        float h_scale = 1.f / (local_max[0] + 1e-6f);
        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (offvec + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (offvec + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
            p_ptr[idx_b].x = char(__float2int_rn(__half2float(h_ptr[i].x) * h_scale * 127.f));
            p_ptr[idx_b].y = char(__float2int_rn(__half2float(h_ptr[i].y) * h_scale * 127.f));
        }

        if (_tid == 0)
        {
            int iidy_t = (offvec + pack_num*head_dim/vec + _pid) / interleave; 
            int iidx_t = (offvec + pack_num*head_dim/vec + _pid) % interleave;
            int idx_t = iidy_t * _Ncg * interleave + _Ncgid * interleave + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
            *t_ptr = __float2half_rn(local_max[0]);
        }
    }

    if (_Loopid == _Lp - 1 && _tid == 0 && _pid==0)
    {
        volatile char2* p_ptr = reinterpret_cast<volatile char2*>(pinptr + _Npid*align_fpga/tsz + (_Lp*_loop)*_Ncg*align_pack/tsz);
        volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + _Ncgid * interleave);
        *c_ptr = uint16_t(((htype&0x3)<<14) | ((tkidx+(_Lp*_loop)+_pid)&0x3FFF));
    }

}

__global__ void unify_kernel_3_unpackage(
    volatile uint16_t* __restrict__ pinptr,
    __half2* __restrict__ outptr,
    int align_fpga,
    int seq_size,
    int pack_num,
    int head_dim)
{
    static constexpr int tsz = 2;
    static constexpr int vec = 2;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _Loopid = blockIdx.z;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;

    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _loop = (SEGMENTS + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++)
    {
        int _lidx = _Loopid * _loop + l;

        // if (_lidx*pack_num+_pid >= seq_size) return;

        volatile __half2* p_ptr = reinterpret_cast<volatile __half2*>(pinptr + _Npid*align_fpga/tsz + (_lidx*_Ncg+_Ncgid)*pack_num*head_dim + _pid*head_dim);
        __half2* o_ptr = outptr + (_lidx*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;

        for(int i = _tid; i < head_dim/vec; i += blockDim.x){
            o_ptr[i].x = p_ptr[i].x;
            o_ptr[i].y = p_ptr[i].y;
        }
    }
}

__global__ void unify_kernel_3_unpackage_quant(
    volatile uint16_t* __restrict__ pinptr,
    __half2* __restrict__ outptr,
    int align_fpga,
    int seq_size,
    int pack_num,
    int head_dim)
{
    static constexpr int tsz = 2;
    static constexpr int vec = 2;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _Loopid = blockIdx.z;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;

    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _dummy_offset = 1;

    float inv_scale = 1.0f / (1 << 9);

    int _loop = (SEGMENTS + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++)
    {
        int _lidx = _Loopid * _loop + l;

        // if (_lidx*pack_num+_pid >= seq_size) return;

        volatile short2* p_ptr = reinterpret_cast<volatile short2*>(pinptr + _Npid*align_fpga/tsz + ((_dummy_offset+_lidx)*_Ncg+_Ncgid)*pack_num*head_dim + _pid*head_dim);
        __half2* o_ptr = outptr + (_lidx*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;

        float2 fval;
        short2 val;
        for(int i = _tid; i < head_dim/vec; i += blockDim.x){
            asm volatile("ld.global.v2.s16 {%0, %1}, [%2];" 
                        : "=h"(val.x), "=h"(val.y) 
                        : "l"(&p_ptr[i]));
            fval.x = static_cast<float>(val.x) * inv_scale;
            fval.y = static_cast<float>(val.y) * inv_scale;
            o_ptr[i] = __float22half2_rn(fval);
        }
    }
}

// ==============================================================

__global__ void unify_kernel_package_kv2(
    __half2* __restrict__ inptrk,
    __half2* __restrict__ inptrv,
    volatile uint16_t* __restrict__ pinptr,
    int align_fpga,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 2;
    static constexpr int vec = 2;

    int offvec_c = 64/tsz/vec;
    int offvec_t = 64/tsz/vec; // align to 64
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Kv = gridDim.z;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _Kvid = blockIdx.z;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _loop = (seq_size + pack_num - 1)/pack_num;
    int _Kv_offset = _Kvid == 0 ? 0 : _loop*(offvec_c + pack_num*head_dim/vec + offvec_t);
    int _htype = _Kvid == 0 ? htype : htype + 1;
    __half2* inptr = _Kvid == 0 ? inptrk : inptrv;
    for (int l = 0; l < _loop; l++) // TODO: _loop循环转移到grid.z维度，性能优化
    {
        if (l*pack_num+_pid >= seq_size) continue; // TODO: pinptr映射区域置0，边界处理

        __half2* h_ptr = inptr + (l*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;
        volatile __half2* p_ptr = reinterpret_cast<volatile __half2*>(pinptr + _Npid*align_fpga/tsz);

        int idx_l = _Kv_offset + l*(offvec_c + pack_num*head_dim/vec + offvec_t);

        if (_tid == 0 && _pid==0)
        {
            int iidy_c = (idx_l + 0) / interleave; 
            int iidx_c = (idx_l + 0) % interleave;
            int idx_c = iidy_c * _Ncg * interleave + _Ncgid * interleave + iidx_c; 
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + idx_c);
            *c_ptr = uint16_t(((_htype&0x1)<<14) | ((tkidx+l+_pid)&0x3FFF));
        }

        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (idx_l + offvec_c + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (idx_l + offvec_c + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
            p_ptr[idx_b].x = h_ptr[i].x;
            p_ptr[idx_b].y = h_ptr[i].y;
        }

        if (_tid == 0)
        {
            int iidy_t = (idx_l*vec + offvec_c*vec + pack_num*head_dim + _pid) / (interleave*vec); 
            int iidx_t = (idx_l*vec + offvec_c*vec + pack_num*head_dim + _pid) % (interleave*vec);
            int idx_t = iidy_t * _Ncg * interleave * vec + _Ncgid * interleave * vec + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
            *t_ptr = __float2half_rn(0.0f);
        }
    }
}

__global__ void unify_kernel_package_kv2_quant(
    __half2* __restrict__ inptrk,
    __half2* __restrict__ inptrv,
    volatile char* __restrict__ pinptr,
    int align_fpga,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 1;
    static constexpr int vec = 2;

    int offvec_c = 64/tsz/vec;
    int offvec_t = 64/tsz/vec;
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Kv = gridDim.z;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _Kvid = blockIdx.z;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _loop = (seq_size + pack_num - 1)/pack_num;
    int _Kv_offset = _Kvid == 0 ? 0 : _loop*(offvec_c + pack_num*head_dim/vec + offvec_t);
    int _htype = _Kvid == 0 ? htype : htype + 1;
    __half2* inptr = _Kvid == 0 ? inptrk : inptrv;
    for (int l = 0; l < _loop; l++) // TODO: _loop循环转移到grid.z维度，性能优化
    {
        if (l*pack_num+_pid >= seq_size) continue; // TODO: pinptr映射区域置0，边界处理

        __half2* h_ptr = inptr + (l*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;
        volatile char2* p_ptr = reinterpret_cast<volatile char2*>(pinptr + _Npid*align_fpga/tsz);

        int idx_l = _Kv_offset + l*(offvec_c + pack_num*head_dim/vec + offvec_t);

        if (_tid == 0 && _pid == 0)
        {
            int iidy_c = (idx_l + 0) / interleave; 
            int iidx_c = (idx_l + 0) % interleave;
            int idx_c = iidy_c * _Ncg * interleave + _Ncgid * interleave + iidx_c; 
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + idx_c);
            *c_ptr = uint16_t(((_htype&0x1)<<14) | ((tkidx+l+_pid)&0x3FFF));
        }

        float local_max[1] = {0.f};
        for (int i = _tid; i < head_dim/vec; i+=blockDim.x) {
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].x)));
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].y)));
        }
        warpReduceMaxV2<float, 1>(local_max);
        
        float h_scale = 1.f / (local_max[0] + 1e-6f);
        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (idx_l + offvec_c + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (idx_l + offvec_c + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
            p_ptr[idx_b].x = char(__float2int_rn(__half2float(h_ptr[i].x) * h_scale * 127.f));
            p_ptr[idx_b].y = char(__float2int_rn(__half2float(h_ptr[i].y) * h_scale * 127.f));
        }

        if (_tid == 0)
        {
            int iidy_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid) / interleave; 
            int iidx_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid) % interleave;
            int idx_t = iidy_t * _Ncg * interleave + _Ncgid * interleave + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
            *t_ptr = __float2half_rn(local_max[0]);
        }
    }
}

__global__ void unify_kernel_package_kv(
    __half2* __restrict__ inptr,
    volatile uint16_t* __restrict__ pinptr,
    int align_fpga,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 2;
    static constexpr int vec = 2;

    int offvec_c = 64/tsz/vec;
    int offvec_t = 64/tsz/vec;
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _loop = (seq_size + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++) // TODO: _loop循环转移到grid.z维度，性能优化
    {
        if (l*pack_num+_pid >= seq_size) continue; // TODO: pinptr映射区域置0，边界处理

        __half2* h_ptr = inptr + (l*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;
        volatile __half2* p_ptr = reinterpret_cast<volatile __half2*>(pinptr + _Npid*align_fpga/tsz);

        int idx_l = l*(offvec_c + pack_num*head_dim/vec + offvec_t);

        if (_tid == 0 && _pid==0)
        {
            int iidy_c = (idx_l + 0) / interleave; 
            int iidx_c = (idx_l + 0) % interleave;
            int idx_c = iidy_c * _Ncg * interleave + _Ncgid * interleave + iidx_c; 
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + idx_c);
            *c_ptr = uint16_t(((htype&0x3)<<14) | ((tkidx+l+_pid)&0x3FFF));
        }

        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (idx_l + offvec_c + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (idx_l + offvec_c + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
            p_ptr[idx_b].x = h_ptr[i].x;
            p_ptr[idx_b].y = h_ptr[i].y;
        }

        if (_tid == 0)
        {
            int iidy_t = (idx_l*vec + offvec_c*vec + pack_num*head_dim + _pid) / (interleave*vec); 
            int iidx_t = (idx_l*vec + offvec_c*vec + pack_num*head_dim + _pid) % (interleave*vec);
            int idx_t = iidy_t * _Ncg * interleave * vec + _Ncgid * interleave * vec + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
            *t_ptr = __float2half_rn(0.0f);
        }
    }
}

__global__ void unify_kernel_package_kv_quant(
    __half2* __restrict__ inptr,
    volatile char* __restrict__ pinptr,
    int align_fpga,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 1;
    static constexpr int vec = 2;

    int offvec_c = 64/tsz/vec;
    int offvec_t = 64/tsz/vec;
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _loop = (seq_size + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++) // TODO: _loop循环转移到grid.z维度，性能优化
    {
        if (l*pack_num+_pid >= seq_size) continue; // TODO: pinptr映射区域置0，边界处理

        __half2* h_ptr = inptr + (l*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;
        volatile char2* p_ptr = reinterpret_cast<volatile char2*>(pinptr + _Npid*align_fpga/tsz);

        int idx_l = l*(offvec_c + pack_num*head_dim/vec + offvec_t);

        if (_tid == 0 && _pid == 0)
        {
            int iidy_c = (idx_l + 0) / interleave; 
            int iidx_c = (idx_l + 0) % interleave;
            int idx_c = iidy_c * _Ncg * interleave + _Ncgid * interleave + iidx_c; 
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + idx_c);
            *c_ptr = uint16_t(((htype&0x3)<<14) | ((tkidx+l+_pid)&0x3FFF));
        }

        float local_max[1] = {0.f};
        for (int i = _tid; i < head_dim/vec; i+=blockDim.x) {
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].x)));
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].y)));
        }
        warpReduceMaxV2<float, 1>(local_max);
        
        float h_scale = 1.f / (local_max[0] + 1e-6f);
        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (idx_l + offvec_c + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (idx_l + offvec_c + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
            p_ptr[idx_b].x = char(__float2int_rn(__half2float(h_ptr[i].x) * h_scale * 127.f));
            p_ptr[idx_b].y = char(__float2int_rn(__half2float(h_ptr[i].y) * h_scale * 127.f));
        }

        if (_tid == 0)
        {
            int iidy_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid) / interleave; 
            int iidx_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid) % interleave;
            int idx_t = iidy_t * _Ncg * interleave + _Ncgid * interleave + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
            *t_ptr = __float2half_rn(local_max[0]);
        }
    }
}

__global__ void unify_kernel_package_q(
    __half2* __restrict__ inptr,
    volatile uint16_t* __restrict__ pinptr,
    int align_fpga,
    int align_pack,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 2;
    static constexpr int vec = 2;
    static constexpr int offvec = 16; // 64/tsz/vec
    
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _loop = (seq_size + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++) // TODO: _loop循环转移到grid.z维度，性能优化
    {
        if (l*pack_num+_pid >= seq_size) continue; // TODO: pinptr映射区域置0，边界处理

        __half2* h_ptr = inptr + (l*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;
        volatile __half2* p_ptr = reinterpret_cast<volatile __half2*>(pinptr + _Npid*align_fpga/tsz + l*_Ncg*align_pack/tsz);

        if (_tid == 0 && _pid==0)
        {
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + _Ncgid * interleave);
            *c_ptr = uint16_t(((htype&0x3)<<14) | ((tkidx+l+_pid)&0x3FFF));
        }

        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (offvec + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (offvec + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
            p_ptr[idx_b].x = h_ptr[i].x;
            p_ptr[idx_b].y = h_ptr[i].y;
        }

        if (_tid == 0)
        {
            int iidy_t = (offvec*vec + pack_num*head_dim + _pid) / (interleave*vec); 
            int iidx_t = (offvec*vec + pack_num*head_dim + _pid) % (interleave*vec);
            int idx_t = iidy_t * _Ncg * interleave * vec + _Ncgid * interleave * vec + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
            *t_ptr = __float2half_rn(0.0f);
        }
    }
}

__global__ void unify_kernel_package_q_quant(
    __half2* __restrict__ inptr,
    volatile char* __restrict__ pinptr,
    int align_fpga,
    int align_pack,
    int align_block,
    int seq_size,
    int pack_num,
    int head_dim,
    int tkidx,
    int htype)
{
    static constexpr int tsz = 1;
    static constexpr int vec = 2;
    static constexpr int offvec = 32; // 64/tsz/vec
    
    int interleave = align_block/tsz/vec;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _loop = (seq_size + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++) // TODO: _loop循环转移到grid.z维度，性能优化
    {
        if (l*pack_num+_pid >= seq_size) continue; // TODO: pinptr映射区域置0，边界处理

        __half2* h_ptr = inptr + (l*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;
        volatile char2* p_ptr = reinterpret_cast<volatile char2*>(pinptr + _Npid*align_fpga/tsz + l*_Ncg*align_pack/tsz);
        
        if (_tid == 0 && _pid==0)
        {
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + _Ncgid * interleave);
            *c_ptr = uint16_t(((htype&0x3)<<14) | ((tkidx+l+_pid)&0x3FFF));
        } 

        float local_max[1] = {0.f};
        for (int i = _tid; i < head_dim/vec; i+=blockDim.x) {
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].x)));
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].y)));
        }
        warpReduceMaxV2<float, 1>(local_max);

        float h_scale = 1.f / (local_max[0] + 1e-6f);
        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (offvec + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (offvec + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * _Ncg * interleave + _Ncgid * interleave + iidx_b; 
            p_ptr[idx_b].x = char(__float2int_rn(__half2float(h_ptr[i].x) * h_scale * 127.f));
            p_ptr[idx_b].y = char(__float2int_rn(__half2float(h_ptr[i].y) * h_scale * 127.f));
        }

        if (_tid == 0)
        {
            int iidy_t = (offvec + pack_num*head_dim/vec + _pid) / interleave; 
            int iidx_t = (offvec + pack_num*head_dim/vec + _pid) % interleave;
            int idx_t = iidy_t * _Ncg * interleave + _Ncgid * interleave + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr) + idx_t;
            *t_ptr = __float2half_rn(local_max[0]);
        }
    }
}

__global__ void unify_kernel_unpackage(
    volatile uint16_t* __restrict__ pinptr,
    __half2* __restrict__ outptr,
    int align_fpga,
    int seq_size,
    int pack_num,
    int head_dim)
{
    static constexpr int tsz = 2;
    static constexpr int vec = 2;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    int _loop = (seq_size + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++) // TODO: _loop循环转移到grid.z维度，性能优化
    {
        if (l*pack_num+_pid >= seq_size) continue;

        volatile __half2* p_ptr = reinterpret_cast<volatile __half2*>(pinptr + _Npid*align_fpga/tsz + (l*_Ncg+_Ncgid)*pack_num*head_dim + _pid*head_dim);
        __half2* o_ptr = outptr + (l*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;

        for(int i = _tid; i < head_dim/vec; i += blockDim.x){
            o_ptr[i].x = p_ptr[i].x;
            o_ptr[i].y = p_ptr[i].y;
        }
    }
}

__global__ void unify_kernel_unpackage_quant(
    volatile uint16_t* __restrict__ pinptr,
    __half2* __restrict__ outptr,
    int align_fpga,
    int seq_size,
    int pack_num,
    int head_dim)
{
    static constexpr int tsz = 2;
    static constexpr int vec = 2;

    int _Ncg = gridDim.x;
    int _Np = gridDim.y;
    int _Ncgid = blockIdx.x;
    int _Npid  = blockIdx.y;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _Nhzid = _Npid;
    int _Nhyid = _Ncgid;
    int _Nhz = _Np;
    int _Nhy = _Ncg;
    int _Nh = _Nhz * _Nhy;
    int _headid = _Nhzid * _Nhy + _Nhyid;

    float inv_scale = 1.0f / (1 << 9);

    int _loop = (seq_size + pack_num - 1)/pack_num;
    for (int l = 0; l < _loop; l++) // TODO: _loop循环转移到grid.z维度，性能优化
    {
        if (l*pack_num+_pid >= seq_size) continue;

        volatile short2* p_ptr = reinterpret_cast<volatile short2*>(pinptr + _Npid*align_fpga/tsz + (l*_Ncg+_Ncgid)*pack_num*head_dim + _pid*head_dim);
        __half2* o_ptr = outptr + (l*pack_num+_pid)*_Nh*head_dim/vec + _headid*head_dim/vec;

        float2 fval;
        short2 val;
        for(int i = _tid; i < head_dim/vec; i += blockDim.x){
            asm volatile("ld.global.v2.s16 {%0, %1}, [%2];" 
                        : "=h"(val.x), "=h"(val.y) 
                        : "l"(&p_ptr[i]));
            fval.x = static_cast<float>(val.x) * inv_scale;
            fval.y = static_cast<float>(val.y) * inv_scale;
            o_ptr[i] = __float22half2_rn(fval);
        }
    }
}

__global__ void unify_kernel_triggerw(char** regs_base, uint32_t desc_num)
{
    int idx = threadIdx.x;
    uint32_t status = 0;
    uint32_t descs = 0;
    xdma_reg_t* regs = reinterpret_cast<xdma_reg_t*>(regs_base[idx]);

    st_na_release((uint32_t*)&regs->control_w1s, 0x1);
    // __threadfence();

    // printf("****triggerw s**** regs->status: %d\n", status);
    // printf("****triggerw s**** regs->desc_done: %d\n", descs);
    while (true) {
        status = ld_volatile_global_u32((const uint32_t*)&regs->status);
        descs = ld_volatile_global_u32((const uint32_t*)&regs->desc_done);
        // __threadfence();
        if ((status & 0x4) || (descs == desc_num)) {
            break;
        }
    }
    // printf("****triggerw e**** regs->status: %d\n", status);
    // printf("****triggerw e**** regs->desc_done: %d\n", descs);

    st_na_release((uint32_t*)&regs->control_w1c, 0x1);
    // __threadfence();
}

__global__ void unify_kernel_triggerwr(char** regs_in_base, char** regs_out_base, uint32_t desc_num)
{
    int idx = threadIdx.x;
    uint32_t status = 0;
    uint32_t descs = 0;
    xdma_reg_t* regs_in = reinterpret_cast<xdma_reg_t*>(regs_in_base[idx]);
    xdma_reg_t* regs_out = reinterpret_cast<xdma_reg_t*>(regs_out_base[idx]);

    st_na_release((uint32_t*)&regs_in->control_w1s, 0x1);
    st_na_release((uint32_t*)&regs_out->control_w1s, 0x1);
    // __threadfence();

    // printf("****triggerwr s**** regs_out->status: %d\n", status);
    // printf("****triggerwr s**** regs_out->desc_done: %d\n", descs);
    while (true) {
        status = ld_volatile_global_u32((const uint32_t*)&regs_out->status);
        descs = ld_volatile_global_u32((const uint32_t*)&regs_out->desc_done);
        // __threadfence();
        if ((status & 0x4) || (descs == desc_num)) {
            break;
        }
    }
    // printf("****triggerwr e**** regs_out->status: %d\n", status);
    // printf("****triggerwr s**** regs_out->desc_done: %d\n", descs);

    st_na_release((uint32_t*)&regs_in->control_w1c, 0x1);
    st_na_release((uint32_t*)&regs_out->control_w1c, 0x1);
    // __threadfence();
}

void packagekv(void* inptrk, void* inptrv, void* pinptr, int align_fpga, int align_block,
    int seq_size, int pack_num, int head_dim, int NpNhz, int NcgNhy, int NcgNty, int InMode, int tkidx, int htype, cudaStream_t & stream)
{
    if(0 == InMode){
        dim3 grid(NcgNhy*NcgNty, NpNhz, 2);
        dim3 block(32, pack_num, 1);
        unify_kernel_package_kv2<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half2*>(inptrk),
            reinterpret_cast<__half2*>(inptrv),
            reinterpret_cast<volatile uint16_t*>(pinptr),
            align_fpga,
            align_block,
            seq_size,
            pack_num,
            head_dim,
            tkidx,
            htype
        );
    }else{
        dim3 grid(NcgNhy*NcgNty, NpNhz, ((seq_size + NcgNty - 1)/NcgNty + SEGMENTS - 1)/SEGMENTS);
        dim3 block(32, pack_num, 2);
        unify_kernel_package_kv4_quant<<<grid, block, 0, stream>>>(
            reinterpret_cast<__half2*>(inptrk),
            reinterpret_cast<__half2*>(inptrv),
            reinterpret_cast<volatile char*>(pinptr),
            align_fpga,
            align_block,
            seq_size,
            pack_num,
            head_dim,
            NcgNhy,
            NcgNty,
            tkidx,
            htype
        );
        // dim3 grid(NcgNhy*NcgNty, NpNhz, (seq_size + SEGMENTS - 1)/SEGMENTS);
        // dim3 block(32, pack_num, 2);
        // unify_kernel_package_kv3_quant<<<grid, block, 0, stream>>>(
        //     reinterpret_cast<__half2*>(inptrk),
        //     reinterpret_cast<__half2*>(inptrv),
        //     reinterpret_cast<volatile char*>(pinptr),
        //     align_fpga,
        //     align_block,
        //     seq_size,
        //     pack_num,
        //     head_dim,
        //     tkidx,
        //     htype
        // );
    }
}

void package(void* inptr, void* pinptr, int align_fpga, int align_pack, int align_block,
    int seq_size, int pack_num, int head_dim, int NpNhz, int NcgNhy, int InMode, int tkidx, int htype, cudaStream_t & stream)
{
    if(0 == InMode){
        if(2 == htype){
            dim3 grid(NcgNhy, NpNhz, 1);
            dim3 block(32, pack_num, 1);
            unify_kernel_package_q<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half2*>(inptr),
                reinterpret_cast<volatile uint16_t*>(pinptr),
                align_fpga,
                align_pack,
                align_block,
                seq_size,
                pack_num,
                head_dim,
                tkidx,
                htype
            );
        }else{
            dim3 grid(NcgNhy, NpNhz, 1);
            dim3 block(32, pack_num, 1);
            unify_kernel_package_kv<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half2*>(inptr),
                reinterpret_cast<volatile uint16_t*>(pinptr),
                align_fpga,
                align_block,
                seq_size,
                pack_num,
                head_dim,
                tkidx,
                htype
            );
        }
    }else{
        if(2 == htype){
            dim3 grid(NcgNhy, NpNhz, (seq_size + SEGMENTS - 1)/SEGMENTS);
            dim3 block(32, pack_num, 1);
            unify_kernel_package_q3_quant<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half2*>(inptr),
                reinterpret_cast<volatile char*>(pinptr),
                align_fpga,
                align_pack,
                align_block,
                seq_size,
                pack_num,
                head_dim,
                tkidx,
                htype
            );
        }else{
            dim3 grid(NcgNhy, NpNhz, 1);
            dim3 block(32, pack_num, 1);
            unify_kernel_package_kv_quant<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half2*>(inptr),
                reinterpret_cast<volatile char*>(pinptr),
                align_fpga,
                align_block,
                seq_size,
                pack_num,
                head_dim,
                tkidx,
                htype
            );
        }
    }
}

void unpackage(void* pinptr, void* outptr, int align_fpga,
    int seq_size, int pack_num, int head_dim, int NpNhz, int NcgNhy, int OuMode, cudaStream_t & stream)
{
    if(0 == OuMode){
        dim3 grid(NcgNhy, NpNhz, (seq_size + SEGMENTS - 1)/SEGMENTS);
        dim3 block(32, pack_num, 1);
        unify_kernel_3_unpackage<<<grid, block, 0, stream>>>(
            reinterpret_cast<volatile uint16_t*>(pinptr),
            reinterpret_cast<__half2*>(outptr),
            align_fpga,
            seq_size,
            pack_num,
            head_dim
        );
    }else{
        dim3 grid(NcgNhy, NpNhz, (seq_size + SEGMENTS - 1)/SEGMENTS);
        dim3 block(32, pack_num, 1);
        unify_kernel_3_unpackage_quant<<<grid, block, 0, stream>>>(
            reinterpret_cast<volatile uint16_t*>(pinptr),
            reinterpret_cast<__half2*>(outptr),
            align_fpga,
            seq_size,
            pack_num,
            head_dim
        );
    }
}

void triggerw(char** dDBAddrVec, int NpNhz, int desc_num, cudaStream_t & stream)
{
    if (dDBAddrVec == nullptr || NpNhz <= 0) {
        return;
    }
    dim3 grid(1, 1, 1);
    dim3 block(NpNhz, 1, 1);
    unify_kernel_triggerw<<<grid, block, 0, stream>>>(
        dDBAddrVec,
        desc_num
    );
}

void triggerwr(char** dDBAddrInVec, char** dDBAddrOutVec, int NpNhz, int desc_num, cudaStream_t & stream)
{
    if (dDBAddrInVec == nullptr || dDBAddrOutVec == nullptr || NpNhz <= 0) {
        return;
    }
    dim3 grid(1, 1, 1);
    dim3 block(NpNhz, 1, 1);
    unify_kernel_triggerwr<<<grid, block, 0, stream>>>(
        dDBAddrInVec,
        dDBAddrOutVec,
        desc_num
    );
}

__global__ void unify_kernel_packageqkv(
    __half2* __restrict__ inptrq,
    __half2* __restrict__ inptrk,
    __half2* __restrict__ inptrv,
    volatile char* __restrict__ pinptr,
    int align_fpga,
    int align_pack,
    int loop_bank,
    int seq_size,
    int pack_num,
    int Np,
    int Nh,
    int Nm,
    int Nn,
    int tkidx)
{
    static constexpr int tsz = 1;
    static constexpr int vec = 2;

    int offvec_c = 64/tsz/vec;
    int offvec_t = 64/tsz/vec;
    int interleave = 512/tsz/vec;
    int head_dim = 128;

    int _Npid  = blockIdx.x/(Nh*Nm);
    int _Nhid  = (blockIdx.x%(Nh*Nm))/Nm;
    int _Nmid  = (blockIdx.x%(Nh*Nm))%Nm;
    int _Qkvid = blockIdx.y;
    int _Lbankid = blockIdx.z;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _headid = _Npid * Nh + _Nhid;

    __half2* inptr = _Qkvid == 0 ? inptrk : _Qkvid == 1 ? inptrv : inptrq;
    
    for (int lb = 0; lb < loop_bank; lb++) 
    {
        int _lidx = _Lbankid * loop_bank + lb;

        __half2* h_ptr = inptr + ((_lidx*Nm +_Nmid)*pack_num+_pid)*(Np*Nh)*head_dim/vec + _headid*head_dim/vec;
        volatile char2* p_ptr = reinterpret_cast<volatile char2*>(pinptr + _Npid*align_fpga/tsz);
        
        int idx_l = (_lidx*(3*Nm) + 2*Nm + _Nmid)*align_pack/tsz/vec;
        if (_Qkvid == 0 || _Qkvid == 1) idx_l = (_lidx*(3*Nm) + _Nmid*2 + _Qkvid)*align_pack/tsz/vec;

        if (_tid == 0 && _pid == 0)
        {
            int iidy_c = (idx_l + 0) / interleave; 
            int iidx_c = (idx_l + 0) % interleave;
            int idx_c = iidy_c * Nh * interleave + _Nhid * interleave + iidx_c; 
            volatile uint16_t* c_ptr = reinterpret_cast<volatile uint16_t*>(p_ptr + idx_c);
            *c_ptr = uint16_t(((_Qkvid&0x3)<<14) | ((tkidx+_lidx)&0x3FFF));
        }

        float local_max[1] = {0.f};
        for (int i = _tid; i < head_dim/vec; i+=blockDim.x) {
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].x)));
            local_max[0] = fmaxf(local_max[0], fabsf(__half2float(h_ptr[i].y)));
        }
        warpReduceMaxV2<float, 1>(local_max);
        float h_scale = 1.f / (local_max[0] + 1e-6f);
        for(int i = _tid; i < head_dim/vec; i += blockDim.x)
        {
            int iidy_b = (idx_l + offvec_c + _pid*head_dim/vec + i) / interleave; 
            int iidx_b = (idx_l + offvec_c + _pid*head_dim/vec + i) % interleave;
            int idx_b = iidy_b * Nh * interleave + _Nhid * interleave + iidx_b; 
            p_ptr[idx_b].x = char(__float2int_rn(__half2float(h_ptr[i].x) * h_scale * 127.f));
            p_ptr[idx_b].y = char(__float2int_rn(__half2float(h_ptr[i].y) * h_scale * 127.f));
        }

        if (_tid == 0)
        {
            int iidy_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid*2/tsz/vec) / interleave; 
            int iidx_t = (idx_l + offvec_c + pack_num*head_dim/vec + _pid*2/tsz/vec) % interleave;
            int idx_t = iidy_t * Nh * interleave + _Nhid * interleave + iidx_t; 
            volatile __half* t_ptr = reinterpret_cast<volatile __half*>(p_ptr + idx_t);
            *t_ptr = __float2half_rn(local_max[0]);
        }
    }
}

__global__ void unify_kernel_unpackageqkv(
    volatile uint16_t* __restrict__ pinptr,
    __half2* __restrict__ outptr,
    int align_fpga,
    int loop_bank,
    int seq_size,
    int pack_num,
    int Np,
    int Nh,
    int Nn)
{
    static constexpr int tsz = 2;
    static constexpr int vec = 2;

    int head_dim = 128;

    int _Nnid = blockIdx.x;
    int _Npid  = blockIdx.y/Nh;
    int _Nhid  = blockIdx.y%Nh;
    int _Lbankid = blockIdx.z;
    int _tid = threadIdx.x;
    int _pid = threadIdx.y;
    int _headid = _Npid * Nh + _Nhid;

    float inv_scale = 1.0f / (1 << 9);

    for (int lb = 0; lb < loop_bank; lb++)
    {
        int _lidx = _Lbankid * loop_bank + lb;

        volatile short2* p_ptr = reinterpret_cast<volatile short2*>(pinptr + _Npid*align_fpga/tsz + ((_lidx*Nn+_Nnid)*Nh+_Nhid)*pack_num*head_dim + _pid*head_dim);
        __half2* o_ptr = outptr + ((_lidx*Nn+_Nnid)*pack_num+_pid)*(Np*Nh)*head_dim/vec + _headid*head_dim/vec;

        float2 fval;
        short2 val;
        for(int i = _tid; i < head_dim/vec; i += blockDim.x){
            asm volatile("ld.global.v2.s16 {%0, %1}, [%2];" 
                        : "=h"(val.x), "=h"(val.y) 
                        : "l"(&p_ptr[i]));
            fval.x = static_cast<float>(val.x) * inv_scale;
            fval.y = static_cast<float>(val.y) * inv_scale;
            o_ptr[i] = __float22half2_rn(fval);
        }
    }
}

void packageqkv(void* inptrq, void* inptrk, void* inptrv, void* pinptr, int align_fpga, int align_pack, int loop_bank,
    int seq_size, int pack_num, int Np, int Nh, int Nm, int Nn, int InMode, int tkidx, cudaStream_t & stream)
{
    assert(pack_num <= 32);
    assert(Nm == Nn);
    assert(InMode == 1);

    // hp232x fpga(noInterleave) vs lxlink fpga(Interleave)
    // loop2 * loop1 * loop
    // loop2 * (loop1a*[h]*loop1b) * ((m*loopm+n*loopn)*looppkg)
    // layer * ((75K/seq)*[h]*fifo_buffer::(seq/(16*4))) * ((kv::4*2+q::4*1)*16pkg_align)

    dim3 grid(Np*Nh*Nm, 3, seq_size/(Nm*pack_num)/loop_bank);
    dim3 block(32, pack_num, 1);
    unify_kernel_packageqkv<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half2*>(inptrq),
        reinterpret_cast<__half2*>(inptrk),
        reinterpret_cast<__half2*>(inptrv),
        reinterpret_cast<volatile char*>(pinptr),
        align_fpga,
        align_pack,
        loop_bank,
        seq_size,
        pack_num,
        Np,
        Nh,
        Nm,
        Nn,
        tkidx
    );
}

void unpackageqkv(void* pinptr, void* outptr, int align_fpga, int loop_bank,
    int seq_size, int pack_num, int Np, int Nh, int Nn, int OuMode, cudaStream_t & stream)
{
    assert(pack_num <= 32);
    assert(OuMode == 1);

    // hp232x fpga(fifo) vs lxlink fpga(64KB)
    // loop2 * loop1 * loop
    // loop2 * (loop1a*loop1b*Nn) * ([h]*loop)
    // layer * ((75K/seq)*fifo_buffer::((seq/(16*4))*(o::4))) * ([h]*16pkg)

    dim3 grid(Nn, Np*Nh, seq_size/(Nn*pack_num)/loop_bank);
    dim3 block(32, pack_num, 1);
    unify_kernel_unpackageqkv<<<grid, block, 0, stream>>>(
        reinterpret_cast<volatile uint16_t*>(pinptr),
        reinterpret_cast<__half2*>(outptr),
        align_fpga,
        loop_bank,
        seq_size,
        pack_num,
        Np,
        Nh,
        Nn
    );
}

} // namespace ptextlxlink
