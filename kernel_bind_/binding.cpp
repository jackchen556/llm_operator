
#include <memory>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include <fstream>
#include "NvInfer.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mutex>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// lynxlink sdk 相关枚举定义
typedef enum {
  LYNX_STRATEGY_SEQUENTIAL = 0,   // 顺序策略
  LYNX_STRATEGY_BALANCED = 1,     // 均分策略
  LYNX_STRATEGY_CONCENTRATED = 2  // 集中策略
} lynxStrategy_t;

typedef enum {
  P2P_DIR_GPU_TO_LYNXLINK = 0,
  P2P_DIR_LYNXLINK_TO_GPU = 1
} lynxP2PDirection_t;

typedef enum {
  XDMA_P2P_UNICAST = 0,  // unicast interleave
  XDMA_P2P_MULTICAST,    // multicast interleave
  XDMA_P2P_BROADCAST,    // broadcast interleave
  XDMA_P2P_MAX,
} lynxP2PType_t;

namespace py = pybind11;

// kernels are defined in kernels.cu and exported via the same module;
// here we call them through the Python-visible symbols by declaring them again.
namespace ptextlxlink {
  void package(void* inptr, void* pinptr, int align_fpga, int align_pack, int align_block,
     int seq_size, int pack_num, int head_dim, int NpNhz, int NcgNhy, int InMode, int tkidx, int htype, cudaStream_t & stream);
  void unpackage(void* pinptr, void* outptr, int align_fpga,
     int seq_size, int pack_num, int head_dim, int NpNhz, int NcgNhy, int OuMode, cudaStream_t & stream);
  void triggerw(char** dDBAddrVec, int NpNhz, int desc_num, cudaStream_t & stream);
  void triggerwr(char** dDBAddrInVec, char** dDBAddrOutVec, int NpNhz, int desc_num, cudaStream_t & stream);
  void packagekv(void* inptrk, void* inptrv, void* pinptr, int align_fpga, int align_block,
     int seq_size, int pack_num, int head_dim, int NpNhz, int NcgNhy, int NcgNty, int InMode, int tkidx, int htype, cudaStream_t & stream);
  
  void packageqkv(void* inptrq, void* inptrk, void* inptrv, void* pinptr, int align_fpga, int align_pack, int loop_bank,
     int seq_size, int pack_num, int Np, int Nh, int Nm, int Nn, int InMode, int tkidx, cudaStream_t & stream);
  void unpackageqkv(void* pinptr, void* outptr, int align_fpga, int loop_bank,
     int seq_size, int pack_num, int Np, int Nh, int Nn, int OuMode, cudaStream_t & stream);
} // namespace ptextlxlink


#define CHECK_DRV(call) \
do { \
    CUresult result = (call); \
    if (result != CUDA_SUCCESS) { \
        const char* msg; \
        cuGetErrorName(result, &msg); \
        std::cerr << "CUDA Driver error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << msg << " (" << result << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define GPU_PAGE_SIZE    0x10000 // 64KB

#define PAGE_ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

typedef struct gpuMemHandle 
{
    CUdeviceptr ptr; // aligned ptr if requested; otherwise, the same as unaligned_ptr.
    union {
        CUdeviceptr unaligned_ptr; // for tracking original ptr; may be unaligned.
        // VMM with GDR support is available from CUDA 11.0
        CUmemGenericAllocationHandle handle;
    };
    size_t size;
    size_t allocated_size;
} gpu_mem_handle_t;

CUresult gpu_vmm_alloc(gpu_mem_handle_t *handle, const size_t size, bool aligned_mapping, bool set_sync_memops)
{
    int version;
    CHECK_DRV(cuDriverGetVersion(&version));

    if (version < 11000) {
        printf("VMM with RDMA is not supported in this CUDA version.\n");
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    CUdevice gpu_dev;
    CHECK_DRV(cuCtxGetDevice(&gpu_dev));

    int RDMASupported = 0;
    CHECK_DRV(cuDeviceGetAttribute(&RDMASupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, gpu_dev));

    if (!RDMASupported) {
        printf("GPUDirect RDMA is not supported on this GPU.\n");
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    CUmemAllocationProp mprop;
    memset(&mprop, 0, sizeof(CUmemAllocationProp));
    mprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    mprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    mprop.location.id = gpu_dev;
    mprop.allocFlags.gpuDirectRDMACapable = 1;

    size_t gran;
    CHECK_DRV(cuMemGetAllocationGranularity(&gran, &mprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    // In case gran is smaller than GPU_PAGE_SIZE
    size_t granularity = PAGE_ROUND_UP(gran, GPU_PAGE_SIZE);

    size_t rounded_size = PAGE_ROUND_UP(size, granularity);
    CUdeviceptr ptr;
    CHECK_DRV(cuMemAddressReserve(&ptr, rounded_size, granularity, 0, 0));

    CUmemGenericAllocationHandle mem_handle;
    CHECK_DRV(cuMemCreate(&mem_handle, rounded_size, &mprop, 0));
    
    CHECK_DRV(cuMemMap(ptr, rounded_size, 0, mem_handle, 0));

    CUmemAccessDesc access;
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = gpu_dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CHECK_DRV(cuMemSetAccess(ptr, rounded_size, &access, 1));

    // cuMemAddressReserve always returns aligned ptr
    handle->ptr = ptr;
    handle->handle = mem_handle;
    handle->size = size;
    handle->allocated_size = rounded_size;

    return CUDA_SUCCESS;
}

CUresult gpu_vmm_free(gpu_mem_handle_t *handle)
{
    if (!handle || !handle->ptr)
        return CUDA_ERROR_INVALID_VALUE;

    CHECK_DRV(cuMemUnmap(handle->ptr, handle->allocated_size));
    CHECK_DRV(cuMemRelease(handle->handle));
    CHECK_DRV(cuMemAddressFree(handle->ptr, handle->allocated_size));

    memset(handle, 0, sizeof(gpu_mem_handle_t));

    return CUDA_SUCCESS;
}

void* regist_host_mem_to_gpu(void* host_mem_ptr, int32_t size) {
  cudaError_t err = cudaGetLastError();  // 清除之前的错误
  if (err != cudaSuccess) {
      printf("regist_host_mem_to_gpu: CUDA in error state before operation: %s", cudaGetErrorString(err));
      exit(1);
  }
  err = cudaHostRegister(host_mem_ptr, size, cudaHostRegisterIoMemory);
  if(err != cudaSuccess) {
      printf("cudaHostRegister failed: %s\n", cudaGetErrorString(err));
      exit(1);
  }
  void* gpu_mem_ptr = nullptr;
  err = cudaHostGetDevicePointer(&gpu_mem_ptr, host_mem_ptr, 0);
  if(err != cudaSuccess) {
      printf("cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
      exit(1);
  }
  return gpu_mem_ptr;
}

class PackageKVImpl 
{
 public:
  PackageKVImpl(std::vector<int64_t> fpga_ids, int ch, int mSeqSize, int mPackNum, int NpNhz, int NcgNhy, int NcgNty)
      : fpga_ids_(std::move(fpga_ids)),
        m_ch(ch),
        mSeqSize(mSeqSize),
        mPackNum(mPackNum),
        m_NpNhz(NpNhz),
        m_NcgNhy(NcgNhy),
        m_NcgNty(NcgNty)
  {
    // For now we only allocate on current device; multi-card is modeled as "channels" (multiple buffers)
    int device = at::cuda::current_device();
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mInitialized) {
        initializeSharedResources(mSeqSize, mPackNum, m_head_dim, m_NpNhz, m_NcgNhy, m_NcgNty, m_InMode);
        mInitialized = true;
    }
  }

  ~PackageKVImpl() 
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mInitialized) {
        releaseSharedResources();
        mInitialized = false;
    }
  }

  void forward(torch::Tensor& k, torch::Tensor& v, int32_t tkidx, int32_t htype)
  {

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ptextlxlink::packagekv(reinterpret_cast<void*>(k.data_ptr()), reinterpret_cast<void*>(v.data_ptr()), (void*)mData.ptr, 
      mDataAlign, m_align_block, mSeqSize, mPackNum, m_head_dim, m_NpNhz, m_NcgNhy, m_NcgNty, m_InMode, tkidx, htype, stream);

    return;
  }

  int32_t data_size() const
  {
    return mP2PDataSize;
  }

private:
  void initializeSharedResources(int seq_size, int pack_num, int head_dim, int NpNhz, int NcgNhy, int NcgNty, int InMode)
  {
    mLibHandle = dlopen("/data/shiquan.zhang/git/fusioninfer/pyExtlxlink/pyextlxlink/libLYNXLINK.so", RTLD_LAZY);
    if (!mLibHandle) {
        printf("dlopen failed: %s\n", dlerror());
        return;
    }

    // 加载新的lynx接口函数
    lynxInitP2PEx = (lynxInitP2PExFunc)dlsym(mLibHandle, "lynxInitP2PEx");

    if (!lynxInitP2PEx) {
        printf("Failed to load lynx symbols: %s\n", dlerror());
        dlclose(mLibHandle);
        mLibHandle = nullptr;
        return;
    }

    int type_size = sizeof(half);
    int raw_size = (64 + pack_num * head_dim * type_size + 64) * ((((seq_size + NcgNty - 1) / NcgNty) + pack_num - 1) / pack_num) * 2;
    int align_size = ((raw_size + 511) / 512) * 512;
    int data_size = NcgNhy * NcgNty * align_size;

    // int8 input
    if(1 == InMode){
      type_size = sizeof(int8_t);
      raw_size = (64 + pack_num * head_dim * type_size + 64) * ((((seq_size + NcgNty - 1) / NcgNty) + pack_num - 1) / pack_num) * 2;
      align_size = ((raw_size + 511) / 512) * 512;
      data_size = NcgNhy * NcgNty * align_size;
    }

    int data_size_align = (data_size + GPU_PAGE_SIZE - 1) & (~(GPU_PAGE_SIZE - 1));

    int gpu_id = 0;
    cudaGetDevice(&gpu_id);
    cudaSetDeviceFlags(cudaDeviceMapHost);

    printf("****packageKVImpl**** seq_size: %d, pack_num: %d, head_dim: %d, NpNhz: %d, NcgNhy: %d, NcgNty: %d, InMode: %d, gpu_id: %d\n", 
      seq_size, pack_num, head_dim, NpNhz, NcgNhy, NcgNty, InMode, gpu_id);

    mP2PDataSize = data_size;
    mDataAlign = data_size_align;
    mDataSize = data_size_align * NpNhz;
    CHECK_DRV(gpu_vmm_alloc(&mData, mDataSize, true, true));

    assert(fpga_ids_.size() == NpNhz);
    for (int i = 0; i < NpNhz; i++)
    {
        uint32_t lynxlink_id = fpga_ids_[i];
        int err = 0;

        void* data_addr = (char*)mData.ptr + i * data_size_align;
        err = lynxInitP2PEx(lynxlink_id, m_ch, data_addr, data_size, P2P_DIR_GPU_TO_LYNXLINK, XDMA_P2P_UNICAST);
        if (err != 0) {
            printf("Failed to init P2P for input data on device %d\n", i);
        }
        printf("****packageKVImpl**** lynxlink_id: %d, data_addr: %lx, data_size: %d\n", lynxlink_id, (uint64_t)data_addr, data_size);
    }
  }

  void releaseSharedResources()
  {
    if (mLibHandle) 
    {
        dlclose(mLibHandle);
        mLibHandle = nullptr;
    }
    CHECK_DRV(gpu_vmm_free(&mData));
  }

  std::mutex mMutex;
  bool mInitialized = false;
  void* mLibHandle = nullptr;

  // 更新的函数指针定义 - 根据新的API接口
  using lynxInitP2PExFunc = int (*)(uint32_t lynxId, uint32_t dmaId, void *addr, uint32_t size, lynxP2PDirection_t direction, lynxP2PType_t type);

  lynxInitP2PExFunc lynxInitP2PEx = nullptr;

  // 共享资源
  int mSeqSize;
  int mPackNum;
  int mDataAlign = 0;
  int mDataSize = 0;
  int32_t mP2PDataSize = 0;
  gpu_mem_handle_t mData;

 private:
  std::vector<int64_t> fpga_ids_;
  int m_ch;
  int m_NpNhz;
  int m_NcgNhy;
  int m_NcgNty;
  int m_align_block = 512;
  int m_head_dim = 128;
  int m_InMode = 1;
};

class PackageImpl 
{
 public:
  PackageImpl(std::vector<int64_t> fpga_ids, int ch, int mSeqSize, int mPackNum, int NpNhz, int NcgNhy)
      : fpga_ids_(std::move(fpga_ids)), m_ch(ch), mSeqSize(mSeqSize), mPackNum(mPackNum), m_NpNhz(NpNhz), m_NcgNhy(NcgNhy)
  {
    // For now we only allocate on current device; multi-card is modeled as "channels" (multiple buffers)
    int device = at::cuda::current_device();
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mInitialized) {
        initializeSharedResources(mSeqSize, mPackNum, m_head_dim, m_NpNhz, m_NcgNhy, m_InMode);
        mInitialized = true;
    }
  }

  ~PackageImpl() 
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mInitialized) {
        releaseSharedResources();
        mInitialized = false;
    }
  }

  void forward(torch::Tensor& h, int32_t tkidx, int32_t htype)
  {

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ptextlxlink::package(reinterpret_cast<void*>(h.data_ptr()), (void*)mData.ptr, mDataAlign, mPackAlign, m_align_block, mSeqSize, mPackNum, m_head_dim, m_NpNhz, m_NcgNhy, m_InMode, tkidx, htype, stream);

    // {
    //     size_t file_size = mDataSize;
    //     void* src_ptr = (void*)mData.ptr;
    //     std::string htype_str;
    //     switch (htype) {
    //         case 0: htype_str = "k"; break;
    //         case 1: htype_str = "v"; break;
    //         case 2: htype_str = "q"; break;
    //         default: htype_str = "unknown"; break;
    //     }
    //     std::string filename = "package_dump_" + htype_str + ".bin";
    //     std::vector<uint8_t> hbuf(file_size);
    //     cudaMemcpy(hbuf.data(), src_ptr, file_size, cudaMemcpyDeviceToHost);
    //     std::ofstream ofs(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    //     if (ofs.is_open()) {
    //         ofs.write(reinterpret_cast<const char*>(hbuf.data()), file_size);
    //         ofs.close();
    //         printf("[PackageImpl] Dumped %zu bytes to %s\n", file_size, filename.c_str());
    //     } else {
    //         printf("[PackageImpl] Failed to open %s for writing\n", filename.c_str());
    //     }
    // }

    return;
  }

  int32_t data_size() const
  {
    return mP2PDataSize;
  }

private:
  void initializeSharedResources(int seq_size, int pack_num, int head_dim, int NpNhz, int NcgNhy, int InMode)
  {
    mLibHandle = dlopen("/data/shiquan.zhang/git/fusioninfer/pyExtlxlink/pyextlxlink/libLYNXLINK.so", RTLD_LAZY);
    if (!mLibHandle) {
        printf("dlopen failed: %s\n", dlerror());
        return;
    }

    // 加载新的lynx接口函数
    lynxInitP2PEx = (lynxInitP2PExFunc)dlsym(mLibHandle, "lynxInitP2PEx");

    if (!lynxInitP2PEx) {
        printf("Failed to load lynx symbols: %s\n", dlerror());
        dlclose(mLibHandle);
        mLibHandle = nullptr;
        return;
    }

    int dummy_size = 1;
    int type_size = sizeof(half);
    int raw_size = (64 + pack_num * head_dim * type_size + 64);
    int align_size = ((raw_size + 511) / 512) * 512;
    int data_size = NcgNhy * (align_size * ((seq_size + pack_num - 1) / pack_num + dummy_size));

    // int8 input
    if(1 == InMode){
      type_size = sizeof(int8_t);
      raw_size = (64 + pack_num * head_dim * type_size + 64);
      align_size = ((raw_size + 511) / 512) * 512;
      data_size = NcgNhy * (align_size * ((seq_size + pack_num - 1) / pack_num + dummy_size));
    }

    int data_size_align = (data_size + GPU_PAGE_SIZE - 1) & (~(GPU_PAGE_SIZE - 1));

    int gpu_id = 0;
    cudaGetDevice(&gpu_id);
    cudaSetDeviceFlags(cudaDeviceMapHost);

    printf("****packageImpl**** seq_size: %d, pack_num: %d, head_dim: %d, NpNhz: %d, NcgNhy: %d, InMode: %d, gpu_id: %d\n", 
      seq_size, pack_num, head_dim, NpNhz, NcgNhy, InMode, gpu_id);

    mP2PDataSize = data_size;
    mPackAlign = align_size;
    mDataAlign = data_size_align;
    mDataSize = data_size_align * NpNhz;
    CHECK_DRV(gpu_vmm_alloc(&mData, mDataSize, true, true));

    assert(fpga_ids_.size() == NpNhz);
    for (int i = 0; i < NpNhz; i++)
    {
        uint32_t lynxlink_id = fpga_ids_[i];
        int err = 0;

        void* data_addr = (char*)mData.ptr + i * data_size_align;
        err = lynxInitP2PEx(lynxlink_id, m_ch, data_addr, data_size, P2P_DIR_GPU_TO_LYNXLINK, XDMA_P2P_MULTICAST);
        if (err != 0) {
            printf("Failed to init P2P for input data on device %d\n", i);
        }
        printf("****PackageImpl**** lynxlink_id: %d, data_addr: %lx, data_size: %d\n", lynxlink_id, (uint64_t)data_addr, data_size);
    }
  }

  void releaseSharedResources()
  {
    if (mLibHandle) 
    {
        dlclose(mLibHandle);
        mLibHandle = nullptr;
    }
    CHECK_DRV(gpu_vmm_free(&mData));
  }

  std::mutex mMutex;
  bool mInitialized = false;
  void* mLibHandle = nullptr;

  // 更新的函数指针定义 - 根据新的API接口
  using lynxInitP2PExFunc = int (*)(uint32_t lynxId, uint32_t dmaId, void *addr, uint32_t size, lynxP2PDirection_t direction, lynxP2PType_t type);

  lynxInitP2PExFunc lynxInitP2PEx = nullptr;

  // 共享资源
  int mSeqSize;
  int mPackNum;
  int mPackAlign = 0;
  int mDataAlign = 0;
  int mDataSize = 0;
  int32_t mP2PDataSize = 0;
  gpu_mem_handle_t mData;

 private:
  std::vector<int64_t> fpga_ids_;
  int m_ch;
  int m_NpNhz;
  int m_NcgNhy;
  int m_align_block = 512;
  int m_head_dim = 128;
  int m_InMode = 1;
};

class UnpackageImpl
{
 public:
  UnpackageImpl(std::vector<int64_t> fpga_ids, int ch, int mSeqSize, int mPackNum, int NpNhz, int NcgNhy)
      : fpga_ids_(std::move(fpga_ids)), m_ch(ch), mSeqSize(mSeqSize), mPackNum(mPackNum), m_NpNhz(NpNhz), m_NcgNhy(NcgNhy)
  {
    int device = at::cuda::current_device();
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mInitialized) {
        initializeSharedResources(mSeqSize, mPackNum, m_head_dim, m_NpNhz, m_NcgNhy);
        mInitialized = true;
    }
  }

  ~UnpackageImpl()
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mInitialized) {
        releaseSharedResources();
        mInitialized = false;
    }
  }

  void forward(torch::Tensor& o)
  {

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ptextlxlink::unpackage((void*)mData.ptr, reinterpret_cast<void*>(o.data_ptr()), mDataAlign, mSeqSize, mPackNum, m_head_dim, m_NpNhz, m_NcgNhy, m_OuMode, stream);

    return;
  }

  int32_t data_size() const
  {
    return mP2PDataSize;
  }

 private:
  void initializeSharedResources(int seq_size, int pack_num, int head_dim, int NpNhz, int NcgNhy)
  {
    mLibHandle = dlopen("/data/shiquan.zhang/git/fusioninfer/pyExtlxlink/pyextlxlink/libLYNXLINK.so", RTLD_LAZY);
    if (!mLibHandle) {
        printf("dlopen failed: %s\n", dlerror());
        return;
    }

    // 加载新的lynx接口函数
    lynxInitP2PEx = (lynxInitP2PExFunc)dlsym(mLibHandle, "lynxInitP2PEx");

    if (!lynxInitP2PEx) {
        printf("Failed to load lynx symbols: %s\n", dlerror());
        dlclose(mLibHandle);
        mLibHandle = nullptr;
        return;
    }

    int dummy_size = 1;
    int type_size = sizeof(half);
    int data_size = NcgNhy  * (pack_num * head_dim * type_size) * ((seq_size + pack_num - 1) / pack_num + dummy_size);
    int data_size_align = (data_size + GPU_PAGE_SIZE - 1) & (~(GPU_PAGE_SIZE - 1));

    int gpu_id = 0;
    cudaGetDevice(&gpu_id);
    cudaSetDeviceFlags(cudaDeviceMapHost);

    printf("****UnpackageImpl**** seq_size: %d, pack_num: %d, head_dim: %d, NpNhz: %d, NcgNhy: %d, OuMode: %d, gpu_id: %d\n", 
      seq_size, pack_num, head_dim, NpNhz, NcgNhy, m_OuMode, gpu_id);

    mP2PDataSize = data_size;
    mDataAlign = data_size_align;
    mDataSize = data_size_align * NpNhz;
    CHECK_DRV(gpu_vmm_alloc(&mData, mDataSize, true, true));


    assert(fpga_ids_.size() == NpNhz);
    for (int i = 0; i < NpNhz; i++)
    {
        uint32_t lynxlink_id = fpga_ids_[i];
        int err = 0;

        void* data_addr = (char*)mData.ptr + i * data_size_align;
        err = lynxInitP2PEx(lynxlink_id, m_ch, data_addr, data_size, P2P_DIR_LYNXLINK_TO_GPU, XDMA_P2P_MULTICAST);
        if (err != 0) {
            printf("Failed to init P2P for output data on device %d\n", i);
        }
        printf("****UnpackageImpl**** lynxlink_id: %d, data_addr: %lx, data_size: %d\n", lynxlink_id, (uint64_t)data_addr, data_size);
    }
  }

  void releaseSharedResources()
  {
    if (mLibHandle) 
    {
        dlclose(mLibHandle);
        mLibHandle = nullptr;
    }
    CHECK_DRV(gpu_vmm_free(&mData));
  }

  std::mutex mMutex;
  bool mInitialized = false;
  void* mLibHandle = nullptr;

  // 更新的函数指针定义 - 根据新的API接口
  using lynxInitP2PExFunc = int (*)(uint32_t lynxId, uint32_t dmaId, void *addr, uint32_t size, lynxP2PDirection_t direction, lynxP2PType_t type);

  lynxInitP2PExFunc lynxInitP2PEx = nullptr;

  // 共享资源
  int mSeqSize;
  int mPackNum;
  int mDataAlign = 0;
  int mDataSize = 0;
  int32_t mP2PDataSize = 0;
  gpu_mem_handle_t mData;

 private:
  std::vector<int64_t> fpga_ids_;
  int m_ch;
  int m_NpNhz;
  int m_NcgNhy;
  int m_head_dim = 128;
  int m_OuMode = 1;
};


class TriggerImpl
{
 public:
  TriggerImpl(std::vector<int64_t> fpga_ids, int ch, int NpNhz)
      : fpga_ids_(std::move(fpga_ids)), m_ch(ch), m_NpNhz(NpNhz)
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mInitialized) {
        initializeSharedResources(m_NpNhz);
        mInitialized = true;
    }
  }

  ~TriggerImpl()
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mInitialized) {
        releaseSharedResources();
        mInitialized = false;
    }
  }

  void forward(int mode, int desc_num)
  {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (mode == 0) {
      ptextlxlink::triggerw(dDBAddrInVec, m_NpNhz, desc_num, stream);
    } else {
      assert(mode == 1);
      ptextlxlink::triggerwr(dDBAddrInVec, dDBAddrOutVec, m_NpNhz, desc_num, stream);
    }
    return;
  }

 private:
  void initializeSharedResources(int NpNhz)
  {
    mLibHandle = dlopen("/data/shiquan.zhang/git/fusioninfer/pyExtlxlink/pyextlxlink/libLYNXLINK.so", RTLD_LAZY);
    if (!mLibHandle) {
        printf("dlopen failed: %s\n", dlerror());
        return;
    }

    lynxGetP2PDBAddrEx = (lynxGetP2PDBAddrExFunc)dlsym(mLibHandle, "lynxGetP2PDBAddrEx");
    if (!lynxGetP2PDBAddrEx) {
        printf("Failed to load lynx symbols: %s\n", dlerror());
        dlclose(mLibHandle);
        mLibHandle = nullptr;
        return;
    }

    assert(fpga_ids_.size() == NpNhz);
    std::vector<void*> DBAddrInVec;
    DBAddrInVec.reserve(NpNhz);
    std::vector<void*> DBAddrOutVec;
    DBAddrOutVec.reserve(NpNhz);
    for (int i = 0; i < NpNhz; i++) {
        uint32_t lynxlink_id = fpga_ids_[i];

        void* db_addr_in = nullptr;
        int err = lynxGetP2PDBAddrEx(lynxlink_id, m_ch,P2P_DIR_GPU_TO_LYNXLINK, &db_addr_in);
        if (err != 0 || db_addr_in == nullptr) {
            printf("Failed to get trigger DB address for device %d\n", i);
            continue;
        }
        void* gpu_db_addr_in = regist_host_mem_to_gpu(db_addr_in, 0x100);
        DBAddrInVec.push_back(gpu_db_addr_in);

        void* db_addr_ou = nullptr;
        err = lynxGetP2PDBAddrEx(lynxlink_id, m_ch, P2P_DIR_LYNXLINK_TO_GPU, &db_addr_ou);
        if (err != 0 || db_addr_ou == nullptr) {
            printf("Failed to get trigger DB address for device %d\n", i);
            continue;
        }
        void* gpu_db_addr_ou = regist_host_mem_to_gpu(db_addr_ou, 0x100);
        DBAddrOutVec.push_back(gpu_db_addr_ou);
    }

    cudaMalloc(&dDBAddrInVec, sizeof(char*) * NpNhz);
    cudaMemcpy(dDBAddrInVec, DBAddrInVec.data(), sizeof(char*) * NpNhz, cudaMemcpyHostToDevice);
    cudaMalloc(&dDBAddrOutVec, sizeof(char*) * NpNhz);
    cudaMemcpy(dDBAddrOutVec, DBAddrOutVec.data(), sizeof(char*) * NpNhz, cudaMemcpyHostToDevice);
  }

  void releaseSharedResources()
  {
    if (mLibHandle) {
        dlclose(mLibHandle);
        mLibHandle = nullptr;
    }
    if (dDBAddrInVec) {
        cudaFree(dDBAddrInVec);
        dDBAddrInVec = nullptr;
    }
    if (dDBAddrOutVec) {
        cudaFree(dDBAddrOutVec);
        dDBAddrOutVec = nullptr;
    }
  }

  std::mutex mMutex;
  bool mInitialized = false;
  void* mLibHandle = nullptr;

  using lynxGetP2PDBAddrExFunc = int (*)(uint32_t lynxId, uint32_t dmaId, lynxP2PDirection_t direction, void **addr);
  lynxGetP2PDBAddrExFunc lynxGetP2PDBAddrEx = nullptr;

  char** dDBAddrInVec = nullptr;
  char** dDBAddrOutVec = nullptr;

private:
  std::vector<int64_t> fpga_ids_;
  int m_ch;
  int m_NpNhz;
};

class PackageQKVImpl 
{
 public:
 PackageQKVImpl(std::vector<int64_t> fpga_ids, int ch, int mSeqSize, int mPackNum, int Np, int Nh, int Nm, int Nn, int InMode)
      : fpga_ids_(std::move(fpga_ids)),
        m_ch(ch),
        m_SeqSize(mSeqSize),
        m_PackNum(mPackNum),
        m_Np(Np),
        m_Nh(Nh),
        m_Nm(Nm),
        m_Nn(Nn),
        m_InMode(InMode)
  {
    // For now we only allocate on current device; multi-card is modeled as "channels" (multiple buffers)
    int device = at::cuda::current_device();
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (!m_Initialized) {
        initializeSharedResources(m_SeqSize, m_PackNum, m_Np, m_Nh, m_Nm, m_Nn, m_InMode);
        m_Initialized = true;
    }
  }

  ~PackageQKVImpl() 
  {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_Initialized) {
        releaseSharedResources();
        m_Initialized = false;
    }
  }

  void forward(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, int32_t tkidx)
  {

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ptextlxlink::packageqkv(reinterpret_cast<void*>(q.data_ptr()), reinterpret_cast<void*>(k.data_ptr()), reinterpret_cast<void*>(v.data_ptr()), (void*)m_Data.ptr, 
      m_align_fpga, m_align_pack, m_loop_bank, m_SeqSize, m_PackNum, m_Np, m_Nh, m_Nm, m_Nn, m_InMode, tkidx, stream);

    return;
  }

  int p2p_data_size() const
  {
    return m_P2PDataSize;
  }

private:
  void initializeSharedResources(int seq_size, int pack_num, int Np, int Nh, int Nm, int Nn, int InMode)
  {
    m_LibHandle = dlopen("libLYNXLINK.so", RTLD_LAZY);
    if (!m_LibHandle) {
        printf("dlopen failed: %s\n", dlerror());
        return;
    }

    // 加载新的lynx接口函数
    lynxInitP2PEx = (lynxInitP2PExFunc)dlsym(m_LibHandle, "lynxInitP2PEx");

    if (!lynxInitP2PEx) {
        printf("Failed to load lynx symbols: %s\n", dlerror());
        dlclose(m_LibHandle);
        m_LibHandle = nullptr;
        return;
    }

    // hp232x fpga(noInterleave) vs lxlink fpga(Interleave)
    // loop2 * loop1 * loop
    // loop2 * (loop1a*[h]*loop1b) * ((m*loopm+n*loopn)*looppkg)
    // layer * ((75K/seq)*[h]*fifo_buffer::(seq/(16*4))) * ((kv::4*2+q::4*1)*16pkg_align)

    assert(InMode == 1 && "InMode只支持int8模式");
    assert(Nm == Nn && "Nm必须等于Nn");
    assert(seq_size % (Nm*pack_num) == 0 && "seq_size必须能被Nm*pack_num整除");

    int loop1b = seq_size / (Nm*pack_num);
    int type_size = sizeof(int8_t);
    int raw_size = 64 + pack_num * 128 * type_size + 64;
    int align_size = ((raw_size + 511) / 512) * 512;
    int data_size =  Nh * loop1b * ((Nm*2 + Nn) * align_size);
    int data_size_align = (data_size + GPU_PAGE_SIZE - 1) & (~(GPU_PAGE_SIZE - 1));

    int loop1bx = 1;
    for (int factor = 2; factor * factor <= loop1b; ++factor) {
        if (loop1b % factor == 0) {
            int candidate1 = factor;
            int candidate2 = loop1b / factor;
            if (candidate1 < candidate2) {
                loop1bx = candidate1;
            }
        }
    }

    int gpu_id = 0;
    cudaGetDevice(&gpu_id);
    cudaSetDeviceFlags(cudaDeviceMapHost);

    printf("****packageQKVImpl**** seq_size: %d, pack_num: %d, Np: %d, Nh: %d, Nm: %d, Nn: %d, InMode: %d, gpu_id: %d\n", 
      seq_size, pack_num, Np, Nh, Nm, Nn, InMode, gpu_id);

    m_loop_bank = loop1bx;
    m_align_pack = align_size;
    m_P2PDataSize = data_size;
    m_align_fpga = data_size_align;
    CHECK_DRV(gpu_vmm_alloc(&m_Data, data_size_align*Np, true, true));

    assert(fpga_ids_.size() == Np);
    for (int i = 0; i < Np; i++)
    {
        uint32_t lynxlink_id = fpga_ids_[i];
        int err = 0;

        void* data_addr = (char*)m_Data.ptr + i * data_size_align;
        err = lynxInitP2PEx(lynxlink_id, m_ch, data_addr, data_size, P2P_DIR_GPU_TO_LYNXLINK, XDMA_P2P_UNICAST);
        if (err != 0) {
            printf("Failed to init P2P for input data on device %d\n", i);
        }
        printf("****packageQKVImpl**** lynxlink_id: %d, data_addr: %lx, data_size: %d\n", lynxlink_id, (uint64_t)data_addr, data_size);
    }
  }

  void releaseSharedResources()
  {
    if (m_LibHandle) 
    {
        dlclose(m_LibHandle);
        m_LibHandle = nullptr;
    }
    CHECK_DRV(gpu_vmm_free(&m_Data));
  }

  std::mutex m_Mutex;
  bool m_Initialized = false;
  void* m_LibHandle = nullptr;
  using lynxInitP2PExFunc = int (*)(uint32_t lynxId, uint32_t dmaId, void *addr, uint32_t size, lynxP2PDirection_t direction, lynxP2PType_t type);
  lynxInitP2PExFunc lynxInitP2PEx = nullptr;
  int m_loop_bank;
  int m_align_fpga;
  int m_align_pack;
  int m_P2PDataSize;
  gpu_mem_handle_t m_Data;

 private:
  std::vector<int64_t> fpga_ids_;
  int m_ch;
  int m_SeqSize;
  int m_PackNum;
  int m_Np;
  int m_Nh;
  int m_Nm;
  int m_Nn;
  int m_InMode;
};

class UnpackageQKVImpl
{
 public:
 UnpackageQKVImpl(std::vector<int64_t> fpga_ids, int ch, int mSeqSize, int mPackNum, int Np, int Nh, int Nm, int Nn, int OuMode)
      : fpga_ids_(std::move(fpga_ids)), m_ch(ch), m_SeqSize(mSeqSize), m_PackNum(mPackNum), m_Np(Np), m_Nh(Nh), m_Nm(Nm), m_Nn(Nn), m_OuMode(OuMode)
  {
    int device = at::cuda::current_device();
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (!m_Initialized) {
        initializeSharedResources(m_SeqSize, m_PackNum, m_Np, m_Nh, m_Nn, m_OuMode);
        m_Initialized = true;
    }
  }

  ~UnpackageQKVImpl()
  {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_Initialized) {
        releaseSharedResources();
        m_Initialized = false;
    }
  }

  void forward(torch::Tensor& o)
  {

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ptextlxlink::unpackageqkv((void*)m_Data.ptr, reinterpret_cast<void*>(o.data_ptr()),
        m_align_fpga, m_loop_bank, m_SeqSize, m_PackNum, m_Np, m_Nh, m_Nn, m_OuMode, stream);

    return;
  }

  int p2p_data_size() const
  {
    return m_P2PDataSize;
  }

 private:
  void initializeSharedResources(int seq_size, int pack_num, int Np, int Nh, int Nn, int OuMode)
  {
    m_LibHandle = dlopen("libLYNXLINK.so", RTLD_LAZY);
    if (!m_LibHandle) {
        printf("dlopen failed: %s\n", dlerror());
        return;
    }

    lynxInitP2PEx = (lynxInitP2PExFunc)dlsym(m_LibHandle, "lynxInitP2PEx");

    if (!lynxInitP2PEx) {
        printf("Failed to load lynx symbols: %s\n", dlerror());
        dlclose(m_LibHandle);
        m_LibHandle = nullptr;
        return;
    }

    // hp232x fpga(fifo) vs lxlink fpga(64KB)
    // loop2 * loop1 * loop
    // loop2 * (loop1a*loop1b*Nn) * ([h]*loop)
    // layer * ((75K/seq)*fifo_buffer::((seq/(16*4))*(o::4))) * ([h]*16pkg)

    assert(OuMode == 1 && "OuMode只支持定点数模式");
    int loop1b = seq_size / (Nn*pack_num);
    int type_size = sizeof(half);
    int data_size = loop1b * Nn * Nh * (pack_num * 128 * type_size);
    int data_size_align = (data_size + GPU_PAGE_SIZE - 1) & (~(GPU_PAGE_SIZE - 1));

    int loop1bx = 1;
    for (int factor = 2; factor * factor <= loop1b; ++factor) {
        if (loop1b % factor == 0) {
            int candidate1 = factor;
            int candidate2 = loop1b / factor;
            if (candidate1 < candidate2) {
                loop1bx = candidate1;
            }
        }
    }

    int gpu_id = 0;
    cudaGetDevice(&gpu_id);
    cudaSetDeviceFlags(cudaDeviceMapHost);

    printf("****UnpackageQKVImpl**** seq_size: %d, pack_num: %d, Np: %d, Nh: %d, Nn: %d, OuMode: %d, gpu_id: %d\n", 
      seq_size, pack_num, Np, Nh, Nn, OuMode, gpu_id);

    m_loop_bank = loop1bx;
    m_P2PDataSize = data_size;
    m_align_fpga = data_size_align;
    CHECK_DRV(gpu_vmm_alloc(&m_Data, data_size_align*Np, true, true));

    assert(fpga_ids_.size() == Np);
    for (int i = 0; i < Np; i++)
    {
        uint32_t lynxlink_id = fpga_ids_[i];
        int err = 0;

        void* data_addr = (char*)m_Data.ptr + i * data_size_align;
        err = lynxInitP2PEx(lynxlink_id, m_ch, data_addr, data_size, P2P_DIR_LYNXLINK_TO_GPU, XDMA_P2P_MULTICAST);
        if (err != 0) {
            printf("Failed to init P2P for output data on device %d\n", i);
        }
        printf("****UnpackageQKVImpl**** lynxlink_id: %d, data_addr: %lx, data_size: %d\n", lynxlink_id, (uint64_t)data_addr, data_size);
    }
  }

  void releaseSharedResources()
  {
    if (m_LibHandle) 
    {
        dlclose(m_LibHandle);
        m_LibHandle = nullptr;
    }
    CHECK_DRV(gpu_vmm_free(&m_Data));
  }

  std::mutex m_Mutex;
  bool m_Initialized = false;
  void* m_LibHandle = nullptr;
  using lynxInitP2PExFunc = int (*)(uint32_t lynxId, uint32_t dmaId, void *addr, uint32_t size, lynxP2PDirection_t direction, lynxP2PType_t type);
  lynxInitP2PExFunc lynxInitP2PEx = nullptr;
  int m_loop_bank;
  int m_align_fpga;
  int m_P2PDataSize;
  gpu_mem_handle_t m_Data;

 private:
  std::vector<int64_t> fpga_ids_;
  int m_ch;
  int m_SeqSize;
  int m_PackNum;
  int m_Np;
  int m_Nh;
  int m_Nm;
  int m_Nn;
  int m_OuMode;
};


// PYBIND11_MODULE 是 pybind11 提供的宏，它定义了一个 Python 扩展模块的初始化函数。
// TORCH_EXTENSION_NAME 由 PyTorch JIT 自动定义（一般为 _C），决定生成的 Python 可导入模块名。
// m 是 pybind11::module_，用于在此模块下注册 C++ 类和函数，使其暴露到 Python 层。
// 通俗来说，这一行声明了“C++ 代码要生成哪个 Python 包”，而模块名一般就是 _C；
// Python 侧在 package.py / unpackage.py 用 `from . import _C` 导入的就是这里定义的模块。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PackageImpl>(m, "PackageImpl")
      .def(py::init<std::vector<int64_t>, int, int, int, int, int>(),
           py::arg("fpga_ids"),
           py::arg("ch"),
           py::arg("mSeqSize"),
           py::arg("mPackNum"),
           py::arg("NpNhz"),
           py::arg("NcgNhy"))
      .def("forward", &PackageImpl::forward, py::arg("h"), py::arg("tkidx") = 0, py::arg("htype") = 0)
      .def("data_size", &PackageImpl::data_size);

  py::class_<UnpackageImpl>(m, "UnpackageImpl")
      .def(py::init<std::vector<int64_t>, int, int, int, int, int>(),
           py::arg("fpga_ids"),
           py::arg("ch"),
           py::arg("mSeqSize"),
           py::arg("mPackNum"),
           py::arg("NpNhz"),
           py::arg("NcgNhy"))
      .def("forward", &UnpackageImpl::forward)
      .def("data_size", &UnpackageImpl::data_size);

  py::class_<TriggerImpl>(m, "TriggerImpl")
      .def(py::init<std::vector<int64_t>, int, int>(),
           py::arg("fpga_ids"),
           py::arg("ch") = 0,
           py::arg("NpNhz"))
      .def("forward", &TriggerImpl::forward, py::arg("mode") = 1, py::arg("desc_num") = 0);

  py::class_<PackageKVImpl>(m, "PackageKVImpl")
      .def(py::init<std::vector<int64_t>, int, int, int, int, int, int>(),
           py::arg("fpga_ids"),
           py::arg("ch"),
           py::arg("mSeqSize"),
           py::arg("mPackNum"),
           py::arg("NpNhz"),
           py::arg("NcgNhy"),
           py::arg("NcgNty"))
      .def("forward", &PackageKVImpl::forward, py::arg("k"), py::arg("v"), py::arg("tkidx") = 0, py::arg("htype") = 0)
      .def("data_size", &PackageKVImpl::data_size);

  py::class_<PackageQKVImpl>(m, "PackageQKVImpl")
      .def(py::init<std::vector<int64_t>, int, int, int, int, int, int, int, int>(),
           py::arg("fpga_ids"),
           py::arg("ch"),
           py::arg("mSeqSize"),
           py::arg("mPackNum"),
           py::arg("Np"),
           py::arg("Nh"),
           py::arg("Nm"),
           py::arg("Nn"),
           py::arg("InMode"))
      .def("forward", &PackageQKVImpl::forward, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("tkidx") = 0)
      .def("p2p_data_size", &PackageQKVImpl::p2p_data_size);

  py::class_<UnpackageQKVImpl>(m, "UnpackageQKVImpl")
      .def(py::init<std::vector<int64_t>, int, int, int, int, int, int, int, int>(),
           py::arg("fpga_ids"),
           py::arg("ch"),
           py::arg("mSeqSize"),
           py::arg("mPackNum"),
           py::arg("Np"),
           py::arg("Nh"),
           py::arg("Nm"),
           py::arg("Nn"),
           py::arg("OuMode"))
      .def("forward", &UnpackageQKVImpl::forward, py::arg("o"))
      .def("p2p_data_size", &UnpackageQKVImpl::p2p_data_size);
}

