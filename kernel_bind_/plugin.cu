#include "p2pPlugin.h"
#include <cassert>
#include <cstring>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <sstream>
#include <vector>
#include <cstdlib>

using namespace nvinfer1;

namespace {
	const char* P2P_PLUGIN_VERSION{ "1" };
	const char* P2P_PLUGIN_NAME{ "P2P_Attention" };  //名称要和onnx中对应的一致
}

static uint16_t cfg = 0;
static int token_num = 0;
static int token_id = 0;


std::vector<int> parseXdmaIdEnv() {
    std::vector<int> ids;
    const char* env = std::getenv("P2P_PLUGIN_XDMA_ID");
    if (!env) {
        printf("[P2PPlugin] Env P2P_PLUGIN_XDMA_ID not set. Default xdma_id = 0\n");
        ids.push_back(0);
        return ids;
    }

    std::stringstream ss(env);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            ids.push_back(std::stoi(item));
        } catch (...) {
            printf("[P2PPlugin] Invalid xdma_id entry in env: %s\n", item.c_str());
        }
    }

    if (ids.empty()) {
        ids.push_back(0);  // fallback
    }

    return ids;
}


__global__ void kernel_trigger(
    volatile uint16_t* data_in_ptr,
    volatile uint16_t* data_ou_ptr,
    char* regs_in_ptr,
    char* regs_ou_ptr,
    int bytes_size_in,
    int bytes_size_ou,
    volatile uint16_t* wait_out_ptr,
    int trigger_channel,
    int wait_channel,
    uint16_t cfg
)
{
    if (trigger_channel != -1) {
        int loops = bytes_size_in / 2560;
        for(int i = 0; i < loops; i++) {
            data_in_ptr[i * 1280] = cfg;
        }
        __threadfence();

        int *t_in = (int *)(regs_in_ptr + 0x0108);
        int *c_in = (int *)(regs_in_ptr + 0x010C);
        int *t_ou = (int *)(regs_ou_ptr + 0x1108);
        int *c_ou = (int *)(regs_ou_ptr + 0x110C);

        data_ou_ptr[(bytes_size_ou >> 1) - 1] = 0xFFFF;
        c_in[0] = 0x1;
        c_ou[0] = 0x1;
        __threadfence();
        t_in[0] = 0x1;
        t_ou[0] = 0x1;
    }
    
    // 等待另一路完成
    if (wait_channel != -1) {
        while (wait_out_ptr[(bytes_size_ou >> 1) - 1] == 0xFFFF);
    }
    else {
        // 否则初始化该 buffer（清空）
        int arr_size = bytes_size_in >> 1;
        for (int i = 0; i < arr_size; ++i) {
            wait_out_ptr[i] = 0;
        }
    }
    return;
}

__global__ void qkv_pack_kernel(
    const half* __restrict__ qkv_ptr, // [b, 2048]
    half* __restrict__ in_ptr,        // [b, h, 1280]
    int b, int h,
    int dq, int dk, int dv,
    int off, int t  // t = 1280
) {
    int batch_id = blockIdx.x;
    int head_id = threadIdx.y;
    int tid = threadIdx.x;

    if (batch_id >= b || head_id >= h) return;

    int total_head_size = h * (dq + dk + dv);  // 每个 batch 占有的数据量（单位：half）
    const half* q_ptr = qkv_ptr + batch_id * total_head_size + head_id * dq;
    const half* k_ptr = qkv_ptr + batch_id * total_head_size + h * dq + head_id * dk;
    const half* v_ptr = qkv_ptr + batch_id * total_head_size + h * (dq + dk) + head_id * dv;

    half* out_ptr = in_ptr + batch_id * h * t + head_id * t;

    // Copy K
    for (int i = tid; i < dk; i += blockDim.x) {
        out_ptr[off + i] = k_ptr[i];
    }
    // Copy V
    for (int i = tid; i < dv; i += blockDim.x) {
        out_ptr[off + dk + i] = v_ptr[i];
    }
    // Copy Q
    for (int i = tid; i < dq; i += blockDim.x) {
        out_ptr[off + dk + dv + i] = q_ptr[i];
    }
}


P2PPlugin::P2PPlugin(int layer_idx, int num_heads, int num_kv_heads, int head_dims, int batch_size, int ping_status, int pong_status,
                     void* data_in_ptr,
                     void* data_out_ptr,
                     void* wait_data_ptr,
                     std::vector<void*> regs_in_ptr,
                     std::vector<void*> regs_out_ptr) :
    mLayerIdx(layer_idx), mNumHeads(num_heads), mNumKvHeads(num_kv_heads), mHeadDims(head_dims), 
    mBatchSize(batch_size), mPingStatus(ping_status), mPongStatus(pong_status),
    mDataInPtr(data_in_ptr), mDataOutPtr(data_out_ptr), mWaitDataPtr(wait_data_ptr),
    mRegsInVec(regs_in_ptr), mRegsOutVec(regs_out_ptr), 
    mNamespace("") { }

P2PPlugin::P2PPlugin(const void* data, size_t length) {
    assert(length == sizeof(int) *  7);
    const char* d = reinterpret_cast<const char*>(data);
    mLayerIdx = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mNumHeads = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mNumKvHeads = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mHeadDims = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mBatchSize = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mPingStatus = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mPongStatus = *reinterpret_cast<const int*>(d);
}

P2PPlugin::P2PPlugin(const void* data, size_t length,
                     void* data_in_ptr,
                     void* data_out_ptr,
                     void* wait_data_ptr,
                     std::vector<void*> regs_in_ptr,
                     std::vector<void*> regs_out_ptr)
    : mDataInPtr(data_in_ptr), mDataOutPtr(data_out_ptr), mWaitDataPtr(wait_data_ptr),
      mRegsInVec(regs_in_ptr), mRegsOutVec(regs_out_ptr) {
    assert(length == sizeof(int) * 7);
    const char* d = reinterpret_cast<const char*>(data);
    mLayerIdx = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mNumHeads = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mNumKvHeads = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mHeadDims = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mBatchSize = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mPingStatus = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mPongStatus = *reinterpret_cast<const int*>(d); d += sizeof(int);
}

P2PPlugin::~P2PPlugin() {}

int P2PPlugin::getNbOutputs() const noexcept { return 1; }

int P2PPlugin::initialize() noexcept { return 0; }

void P2PPlugin::terminate() noexcept {}

size_t P2PPlugin::getSerializationSize() const noexcept {
    return sizeof(int) * 7;
}

void P2PPlugin::serialize(void* buffer) const noexcept {
    char* d = reinterpret_cast<char*>(buffer);
    *reinterpret_cast<int*>(d) = mLayerIdx;        d += sizeof(int);
    *reinterpret_cast<int*>(d) = mNumHeads;        d += sizeof(int);
    *reinterpret_cast<int*>(d) = mNumKvHeads;      d += sizeof(int);
    *reinterpret_cast<int*>(d) = mHeadDims;        d += sizeof(int);
    *reinterpret_cast<int*>(d) = mBatchSize;       d += sizeof(int);
    *reinterpret_cast<int*>(d) = mPingStatus;      d += sizeof(int);
    *reinterpret_cast<int*>(d) = mPongStatus;
}

void P2PPlugin::destroy() noexcept {
    delete this;
}

const char* P2PPlugin::getPluginType() const noexcept {
    return P2P_PLUGIN_NAME;
}

const char* P2PPlugin::getPluginVersion() const noexcept {
    return P2P_PLUGIN_VERSION;
}

void P2PPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* P2PPlugin::getPluginNamespace() const noexcept {
    return mNamespace;
}

nvinfer1::IPluginV2DynamicExt* P2PPlugin::clone() const noexcept {
    return new P2PPlugin(mLayerIdx, mNumHeads, mNumKvHeads, mHeadDims, mBatchSize, mPingStatus, mPongStatus,
                         mDataInPtr, mDataOutPtr, mWaitDataPtr, mRegsInVec, mRegsOutVec);
}

nvinfer1::DimsExprs P2PPlugin::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
                                        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs outputDims;
    outputDims.nbDims = 2;
    outputDims.d[0] = inputs[0].d[0];
    outputDims.d[1] = exprBuilder.constant(mNumHeads * mHeadDims);
    return outputDims;
}

bool P2PPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                               int nbInputs, int nbOutputs) noexcept {

    if (pos == 0) {
        // 主输入：qkv，half+linear
        return inOut[pos].type == DataType::kHALF &&
               inOut[pos].format == TensorFormat::kLINEAR;
    } else if (pos == 1) {
        // 输出 feature, out, half+linear
        return inOut[pos].type == DataType::kHALF &&
               inOut[pos].format == TensorFormat::kLINEAR;
    } else if (pos == 2) {
        // 输出 feature, out, half+linear
        return inOut[pos].type == DataType::kHALF &&
               inOut[pos].format == TensorFormat::kLINEAR;
    }
    return false;
}

void P2PPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {}

size_t P2PPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                       const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    return 0;
}

int P2PPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                           const void* const* inputs, void* const* outputs,
                           void* workspace, cudaStream_t stream) noexcept {
    char* qkv_ptr = reinterpret_cast<char*>(const_cast<void*>(inputs[0]));
    char* output_ptr = reinterpret_cast<char*>(const_cast<void*>(outputs[0]));

    char* in_ptr = reinterpret_cast<char*>(const_cast<void*>(mDataInPtr));
    char* ou_ptr = reinterpret_cast<char*>(const_cast<void*>(mDataOutPtr));
    char* wait_ptr = reinterpret_cast<char*>(const_cast<void*>(mWaitDataPtr));

    cudaError_t err;
    int ousize = 1;
    for (int i = 0; i < outputDesc[0].dims.nbDims; ++i) {
        ousize *= outputDesc[0].dims.d[i];
    }

    int xdma_num = mRegsInVec.size();
    int b = inputDesc[0].dims.d[0], h = mNumKvHeads;
    int dq = mNumHeads / mNumKvHeads * mHeadDims, dk = mHeadDims, dv = mHeadDims;
    int off = 32;
    int raw_size = mHeadDims * 2 * (mNumHeads / mNumKvHeads + 2) + 64;
    int t = ((raw_size + 511) / 512) * 512 / 2;
    size_t bytes_size_in = b*h*t*sizeof(half) / xdma_num;
    size_t bytes_size_ou = ousize * sizeof(half) / xdma_num;

    uint16_t cfg;
    if (mLayerIdx == 0) token_id = token_num++;
    int cur_token_num = token_id / 2;
    cfg = (cur_token_num == 0) ? (0x0040 + mLayerIdx) : (0x0000 + mLayerIdx);

    int trigger_channel = (mPingStatus == 1) ? 0 :
                          (mPongStatus == 1) ? 1 : -1;
    int wait_channel = (mPingStatus == 2) ? 0 :
                       (mPongStatus == 2) ? 1 : -1;
    if (wait_channel == trigger_channel)  wait_channel = -1;

    // QKV pack kernel
    dim3 grid(b);           // 每个 batch 一个 block
    dim3 block(256, h);     // 每个 head 一个 thread group，blockDim.y = h
    qkv_pack_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(qkv_ptr),
        reinterpret_cast<half*>(in_ptr),
        b, h, dq, dk, dv, off, t
    );

    for (int i = 0; i < xdma_num; i++) {
        volatile uint16_t* data_in_ptr = reinterpret_cast<volatile uint16_t*>(in_ptr + i * bytes_size_in);
        volatile uint16_t* data_ou_ptr = reinterpret_cast<volatile uint16_t*>(ou_ptr + i * bytes_size_ou);
        volatile uint16_t* data_wait_ptr = reinterpret_cast<volatile uint16_t*>(wait_ptr + i * bytes_size_ou);
        char* regs_in_ptr = reinterpret_cast<char*>(const_cast<void*>(mRegsInVec[i]));
        char* regs_ou_ptr = reinterpret_cast<char*>(const_cast<void*>(mRegsOutVec[i]));
        // Launch kernel
        kernel_trigger<<<1, 1, 0, stream>>>(
            data_in_ptr,
            data_ou_ptr,
            regs_in_ptr,
            regs_ou_ptr,
            bytes_size_in,
            bytes_size_ou,
            data_wait_ptr,
            trigger_channel,
            wait_channel,
            cfg
        );
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    // 将 ou_ptr 拷贝到 output
    err = cudaMemcpyAsync(output_ptr, wait_ptr, ousize * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) printf("Output memcpy failed: %s\n", cudaGetErrorString(err));
    return 0;
}



nvinfer1::DataType P2PPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                    int nbInputs) const noexcept {
    return inputTypes[0];
}


// -------------------- P2PPluginCreator Implementation --------------------
PluginFieldCollection P2PPluginCreator::mFC{};
std::vector<PluginField> P2PPluginCreator::mPluginAttributes;

P2PPluginCreator::P2PPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("layer_idx", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("head_dims", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("batch_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("ping_status", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("pong_status", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

P2PPluginCreator::~P2PPluginCreator() {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mInitialized) {
        releaseSharedResources();
        mInitialized = false;
    }
}

void P2PPluginCreator::initializeSharedResources(int num_heads, int num_kv_heads, int head_dims, int batch_size) {
    mLibHandle = dlopen("/datas/qiang.wu/fusioninfer/plugins/lib/libgpux640_resources.so", RTLD_LAZY);
    if (!mLibHandle) {
        printf("dlopen failed: %s\n", dlerror());
        return;
    }

    std::vector<int> xdma_ids = parseXdmaIdEnv();
    for (int i = 0; i < xdma_ids.size(); i++) {
        printf("xdma_ids[%d]: %d\n", i, xdma_ids[i]);
    }

    pcie_p2p_dma_create = (CreateFunc)dlsym(mLibHandle, "pcie_p2p_dma_create");
    pcie_p2p_dma_init = (InitFunc)dlsym(mLibHandle, "pcie_p2p_dma_init");
    pcie_p2p_cuda_init = (CudaInitFunc)dlsym(mLibHandle, "pcie_p2p_cuda_init");
    pcie_p2p_dma_get_in_base_reg = (GetRegFunc)dlsym(mLibHandle, "pcie_p2p_dma_get_in_base_reg");
    pcie_p2p_dma_get_ou_base_reg = (GetRegFunc)dlsym(mLibHandle, "pcie_p2p_dma_get_ou_base_reg");
    pcie_p2p_dma_free = (FreeFunc)dlsym(mLibHandle, "pcie_p2p_dma_free");

    if (!pcie_p2p_dma_create || !pcie_p2p_dma_init || !pcie_p2p_cuda_init || !pcie_p2p_dma_get_in_base_reg || !pcie_p2p_dma_get_ou_base_reg) {
        printf("Failed to load symbols: %s\n", dlerror());
        dlclose(mLibHandle);
        mLibHandle = nullptr;
        return;
    }

    // 这里得batch_size是编译时传入的, 实际需要从运行时的输入Tensor获取, 这里取巧, 根据xdma的数目来确定batch_size
    // 外部需要确保xdma_ids的正确性, 比如配置了8个xdma, 那么就是64bacth
    batch_size = xdma_ids.size() * 32 / num_kv_heads;

    constexpr int NUM_CHANNELS = 2;
    int batch_pre_xdma = 32 / num_kv_heads; // TODO：Qwen 7B
    int xmda_num_per_channel = batch_size / batch_pre_xdma / NUM_CHANNELS;   // 外部确保batch_size是batch_pre_xdma的整数倍
    printf("batch_size: %d, batch_pre_xdma: %d, xmda_num_per_channel: %d\n", batch_size, batch_pre_xdma, xmda_num_per_channel);
    assert(xmda_num_per_channel * NUM_CHANNELS == xdma_ids.size());

    int raw_size = head_dims * 2 * (num_heads / num_kv_heads + 2) + 64;
    int align_dim = ((raw_size + 511) / 512) * 512 / 2;
    int type_size = sizeof(half);

    int data_size_in = batch_pre_xdma * num_kv_heads * align_dim * type_size;
    int data_size_out = batch_pre_xdma * num_heads * head_dims * type_size;

    int gpu_id = 0;
    cudaGetDevice(&gpu_id);  // 获取当前使用的GPU设备ID
    for (int i = 0; i < NUM_CHANNELS; ++i) {
        mDataIn[i] = nullptr;
        mDataOut[i] = nullptr;
        cudaMalloc(&mDataIn[i], data_size_in * xmda_num_per_channel);
        cudaMalloc(&mDataOut[i], data_size_out * xmda_num_per_channel);

        // 通道号 i 作为最后一个参数传给 init
        for (int j = 0; j < xmda_num_per_channel; j++) {

            int handle = pcie_p2p_dma_create();
            int idx =  i * xmda_num_per_channel + j;
            assert(idx < xdma_ids.size());
            pcie_p2p_dma_init(handle, (uint64_t)mDataIn[i] + j * data_size_in, data_size_in, (uint64_t)mDataOut[i] + j * data_size_out, data_size_out, xdma_ids[idx]);
            pcie_p2p_cuda_init(handle, gpu_id);
            mRegsInVec2D[i].push_back(pcie_p2p_dma_get_in_base_reg(handle));
            mRegsOutVec2D[i].push_back(pcie_p2p_dma_get_ou_base_reg(handle));
            mHandles.push_back(handle);
        }
    }
}

void P2PPluginCreator::releaseSharedResources() {
    if (mLibHandle) {
        for (auto handle : mHandles) {
            pcie_p2p_dma_free(handle);
        }
        dlclose(mLibHandle);
        mLibHandle = nullptr;
    }
    
    if (mDataIn[0]) cudaFree(mDataIn[0]);
    if (mDataIn[1]) cudaFree(mDataIn[1]);
    if (mDataOut[0]) cudaFree(mDataOut[0]);
    if (mDataOut[1]) cudaFree(mDataOut[1]);
}

const char* P2PPluginCreator::getPluginName() const noexcept { return P2P_PLUGIN_NAME; }
const char* P2PPluginCreator::getPluginVersion() const noexcept { return P2P_PLUGIN_VERSION; }
const PluginFieldCollection* P2PPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* P2PPluginCreator::createPlugin(const char* name,
                                          const PluginFieldCollection* fc) noexcept
{
    int layer_idx = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dims = 0;
    int batch_size = 0;
    int ping_status = 0;
    int pong_status = 0;

    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fields[i].name;

        if (strcmp(attrName, "layer_idx") == 0) {
            layer_idx = *static_cast<const int*>(fields[i].data);
        } else if (strcmp(attrName, "num_heads") == 0) {
            num_heads = *static_cast<const int*>(fields[i].data);
        } else if (strcmp(attrName, "num_kv_heads") == 0) {
            num_kv_heads = *static_cast<const int*>(fields[i].data);
        } else if (strcmp(attrName, "head_dims") == 0) {
            head_dims = *static_cast<const int*>(fields[i].data);
        } else if (strcmp(attrName, "batch_size") == 0) {
            batch_size = *static_cast<const int*>(fields[i].data);
        } else if (strcmp(attrName, "ping_status") == 0) {
            ping_status = *static_cast<const int*>(fields[i].data);
        } else if (strcmp(attrName, "pong_status") == 0) {
            pong_status = *static_cast<const int*>(fields[i].data);
        }
    }

    // status : 0 null, 1 trigger, 2 wait
    int trigger_xdma = 0;
    int wait_xdma = 1;
    if (ping_status == 1 && pong_status != 1){
        trigger_xdma = 0;
    } else if (pong_status == 1 && ping_status != 1){
        trigger_xdma = 1;
    }

    if (ping_status == 2 && pong_status != 2){
        wait_xdma = 0;
    } else if (pong_status == 2 && ping_status != 2){
        wait_xdma = 1;
    }

    auto* plugin = new P2PPlugin(layer_idx, num_heads, num_kv_heads, head_dims, batch_size, ping_status, pong_status,
                                 mDataIn[trigger_xdma], mDataOut[trigger_xdma], mDataOut[wait_xdma], mRegsInVec2D[trigger_xdma], mRegsOutVec2D[trigger_xdma]);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* P2PPluginCreator::deserializePlugin(const char* name,
                                               const void* serialData,
                                               size_t serialLength) noexcept
{
    assert(serialLength == sizeof(int) * 7);
    const char* d = reinterpret_cast<const char*>(serialData);
    int layer_idx = *reinterpret_cast<const int*>(d); d += sizeof(int);
    int num_heads = *reinterpret_cast<const int*>(d); d += sizeof(int);
    int num_kv_heads = *reinterpret_cast<const int*>(d); d += sizeof(int);
    int head_dims = *reinterpret_cast<const int*>(d); d += sizeof(int);
    int batch_size = *reinterpret_cast<const int*>(d); d += sizeof(int);
    int ping_status = *reinterpret_cast<const int*>(d); d += sizeof(int);
    int pong_status = *reinterpret_cast<const int*>(d);

    std::lock_guard<std::mutex> lock(mMutex);
    if (!mInitialized) {
        initializeSharedResources(num_heads, num_kv_heads, head_dims, batch_size);
        mInitialized = true;
    }

    // status : 0 null, 1 trigger, 2 wait
    int trigger_xdma = 0;
    int wait_xdma = 1;
    if (ping_status == 1 && pong_status != 1){
        trigger_xdma = 0;
    } else if (pong_status == 1 && ping_status != 1){
        trigger_xdma = 1;
    }

    if (ping_status == 2 && pong_status != 2){
        wait_xdma = 0;
    } else if (pong_status == 2 && ping_status != 2){
        wait_xdma = 1;
    }

    auto* plugin = new P2PPlugin(serialData, serialLength, mDataIn[trigger_xdma], mDataOut[trigger_xdma], mDataOut[wait_xdma], mRegsInVec2D[trigger_xdma], mRegsOutVec2D[trigger_xdma]);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void P2PPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* P2PPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        return;
    }
};

class LoggerFinder : public nvinfer1::ILoggerFinder
{
public:
    LoggerFinder(Logger& logger) : mLogger(logger) {}

    nvinfer1::ILogger* findLogger() override
    {
        return &mLogger;
    }

private:
    Logger& mLogger;
};

REGISTER_TENSORRT_PLUGIN(P2PPluginCreator);

extern "C" {
    void * getCreators(){
    	P2PPluginCreator creator;
        return &creator;
    }
}


extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder& loggerFinder)
{
    // 设置全局的 LoggerFinder
    //gLoggerFinder = &loggerFinder;
    return;
}



extern "C" {
  nvinfer1::IPluginCreator* const*  getPluginCreators(int32_t& nb) {
    nb = 1;
    static  P2PPluginCreator creator{};
    static  nvinfer1::IPluginCreator* const pluginCreatorList[] = {&creator};
    return pluginCreatorList;
}
}

// Register plugin creator
// REGISTER_TENSORRT_PLUGIN(P2PPluginCreator);
