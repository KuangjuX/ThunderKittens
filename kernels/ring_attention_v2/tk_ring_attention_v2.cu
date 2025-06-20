#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
namespace py = pybind11;

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "kittens.cuh"
#include "pyutils/torch_helpers.cuh"

constexpr int NUM_DEVICES         = 8;
constexpr int CONSUMER_WARPGROUPS = 3; 
constexpr int PRODUCER_WARPGROUPS = 1; 
constexpr int NUM_WARPGROUPS      = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS; 
constexpr int NUM_WORKERS         = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;

using namespace kittens;

template<int D> struct fwd_tile_dims {};
template<> struct fwd_tile_dims<64> {
    constexpr static int tile_width = 64;
    constexpr static int QO_height  = 4 * 16;
    constexpr static int KV_height  = 8 * 16;
    constexpr static int stages     = 4; 
};
template<> struct fwd_tile_dims<128> {
    constexpr static int tile_width = 128;
    constexpr static int QO_height  = 4 * 16;
    constexpr static int KV_height  = 8 * 16;
    constexpr static int stages     = 2;
};
template<int D> struct fwd_pglobals {
    using Q_tile = st_bf<fwd_tile_dims<D>::QO_height, fwd_tile_dims<D>::tile_width>;
    using K_tile = st_bf<fwd_tile_dims<D>::KV_height, fwd_tile_dims<D>::tile_width>;
    using V_tile = st_bf<fwd_tile_dims<D>::KV_height, fwd_tile_dims<D>::tile_width>;
    using O_tile = st_bf<fwd_tile_dims<D>::QO_height, fwd_tile_dims<D>::tile_width>;

    using Q_gl = gl<bf16, -1, -1, -1, -1, Q_tile>;
    using K_gl = gl<bf16, -1, -1, -1, -1, K_tile>;
    using V_gl = gl<bf16, -1, -1, -1, -1, V_tile>;
    using O_gl = gl<bf16, -1, -1, -1, -1, O_tile>;

    Q_gl Q[NUM_DEVICES];
    K_gl K[NUM_DEVICES];
    V_gl V[NUM_DEVICES];
    O_gl O[NUM_DEVICES];

    const int N_per_dev; // sequence length per device

    fwd_pglobals(Q_gl* _Q, K_gl* _K, V_gl* _V, O_gl* _O, const int _N_per_dev) : 
        fwd_pglobals(std::make_index_sequence<NUM_DEVICES>{}, _Q, _K, _V, _O, _N_per_dev) {}
    template<std::size_t... I>
    fwd_pglobals(std::index_sequence<I...>, Q_gl* _Q, K_gl* _K, V_gl* _V, O_gl* _O, const int _N_per_dev) :
        Q{_Q[I]...}, K{_K[I]...}, V{_V[I]...}, O{_O[I]...}, N_per_dev(_N_per_dev) {}
};

template<int D, bool is_causal>
__global__  __launch_bounds__(NUM_WORKERS * kittens::WARP_THREADS, 1)
void blockwise_attn_kernel(const __grid_constant__ fwd_pglobals<D> p_G, const __grid_constant__ int dev_idx) {
    // ==================== 内存分配和初始化 ====================
    extern __shared__ int __shm[];  // 共享内存声明
    tma_swizzle_allocator al((int*)&__shm[0]);  // TMA内存分配器，用于管理共享内存
    int warpid = kittens::warpid();  // 当前warp ID
    int warpgroupid = warpid / kittens::WARPGROUP_WARPS;  // warp group ID

    // 定义tile类型别名
    using K = fwd_tile_dims<D>;
    using q_tile = fwd_pglobals<D>::Q_tile;
    using k_tile = fwd_pglobals<D>::K_tile;
    using v_tile = fwd_pglobals<D>::V_tile;
    using o_tile = fwd_pglobals<D>::O_tile;
    
    // ==================== 共享内存分配 ====================
    // 分配Q tiles的共享内存：每个consumer warpgroup一个Q tile
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    // 分配K tiles的共享内存：使用stages进行流水线处理
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    // 分配V tiles的共享内存：使用stages进行流水线处理
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    // O tiles复用Q tiles的共享内存空间
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);

    // ==================== 索引计算 ====================
    // 计算每张卡的KV block数量
    int kv_blocks_per_dev = p_G.N_per_dev / (K::KV_height);
    // 计算所有卡的KV block总数
    int kv_blocks_total = kv_blocks_per_dev * NUM_DEVICES;
    // 计算当前卡的KV block起始索引
    int kv_block_idx_start = kv_blocks_per_dev * dev_idx;
    // 当前处理的head索引
    int kv_head_idx = blockIdx.y;
    // 当前处理的序列索引
    int seq_idx     = blockIdx.x * CONSUMER_WARPGROUPS; 

    // ==================== 信号量初始化 ====================
    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) { 
        // 初始化Q内存加载完成的信号量
        init_semaphore(qsmem_semaphore, 0, 1); 
        for(int j = 0; j < K::stages; j++) {
            // 初始化K内存加载完成的信号量
            init_semaphore(k_smem_arrived[j], 0, 1); 
            // 初始化V内存加载完成的信号量
            init_semaphore(v_smem_arrived[j], 0, 1); 
            // 初始化计算完成的信号量
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS, 0); 
        }

        // 设置TMA期望的字节数
        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        // ==================== 第一阶段：加载Q tiles ====================
        // 为每个consumer warpgroup加载对应的Q tile
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            // 异步加载Q tile到共享内存，使用信号量同步
            tma::load_async(q_smem[wg], p_G.Q[dev_idx], q_tile_idx, qsmem_semaphore);
        }

        // ==================== 第二阶段：预加载K/V tiles ====================
        // 预加载前K::stages-1个K/V tiles，为流水线做准备
        for (int j = 0; j < K::stages - 1; j++) {
            // 计算当前要加载的KV block的全局索引（环形访问）
            int kv_blk_idx = (j + kv_block_idx_start) % kv_blocks_total;
            // 计算KV block所在的设备ID（跨卡通信的关键）
            int kv_blk_dev_idx = kv_blk_idx / kv_blocks_per_dev;
            // 计算KV block在对应设备上的本地索引
            int kv_blk_local_idx = kv_blk_idx % kv_blocks_per_dev;
            coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_blk_local_idx, 0};
            
            // 异步加载K tile，可能来自其他设备（跨卡通信）
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async(k_smem[j], p_G.K[kv_blk_dev_idx], kv_tile_idx, k_smem_arrived[j]);
            
            // 异步加载V tile，可能来自其他设备（跨卡通信）
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async(v_smem[j], p_G.V[kv_blk_dev_idx], kv_tile_idx, v_smem_arrived[j]);
        }
    }
    __syncthreads();  // 同步所有线程，确保初始化完成

    int pipe_idx = K::stages - 1; 

    // ==================== Producer Warpgroup：持续加载K/V tiles ====================
    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();  // 减少寄存器使用，为加载任务腾出空间
        int kv_iters = kv_blocks_total - 2;
        if(warpid == NUM_WORKERS-4) {
            // 持续加载剩余的K/V tiles，维持流水线
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                // 计算下一个要加载的KV block索引（环形访问）
                int kv_blk_idx = (kv_idx + 1 + kv_block_idx_start) % kv_blocks_total;
                // 计算KV block所在的设备ID（跨卡通信）
                int kv_blk_dev_idx = kv_blk_idx / kv_blocks_per_dev;
                // 计算KV block在对应设备上的本地索引
                int kv_blk_local_idx = kv_blk_idx % kv_blocks_per_dev;
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_blk_local_idx, 0};
                
                // 异步加载K tile到流水线缓冲区
                tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                tma::load_async(k_smem[(kv_idx+1)%K::stages], p_G.K[kv_blk_dev_idx], kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                
                // 异步加载V tile到流水线缓冲区
                tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                tma::load_async(v_smem[(kv_idx+1)%K::stages], p_G.V[kv_blk_dev_idx], kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);
                
                // 等待当前stage的计算完成，确保流水线同步
                wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            }
        }
    }
    // ==================== Consumer Warpgroups：执行attention计算 ====================
    else {
        warpgroup::increase_registers<160>();  // 增加寄存器使用，为计算任务分配更多寄存器

        // ==================== 寄存器分配 ====================
        // attention block：存储Q*K^T的结果
        rt_fl<16, K::KV_height>  att_block;
        // attention block for MMA：用于矩阵乘法的attention block
        rt_bf<16, K::KV_height>  att_block_mma;
        // output register：存储最终的输出结果
        rt_fl<16, K::tile_width> o_reg;
        
        // 用于softmax计算的向量
        col_vec<rt_fl<16, K::KV_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
        
        // ==================== 初始化计算状态 ====================
        neg_infty(max_vec);  // 初始化max向量为负无穷
        zero(norm_vec);      // 初始化norm向量为0
        zero(o_reg);         // 初始化输出寄存器为0

        int kv_iters = kv_blocks_total - 1;
        // 等待Q tiles加载完成
        wait(qsmem_semaphore, 0);

        // ==================== 主计算循环：处理所有KV blocks ====================
        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
            // 等待当前K tile加载完成
            wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            
            // ==================== Q*K^T 计算 ====================
            // 执行Q和K的矩阵乘法：att_block = Q * K^T
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
            
            // ==================== Softmax 计算 ====================
            // 保存上一次的max值用于数值稳定性
            copy(max_vec_last_scaled, max_vec);
            // 缩放上一次的max值
            mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f);
            // 等待矩阵乘法完成
            warpgroup::mma_async_wait();
            
            // 计算当前attention block的每行最大值
            row_max(max_vec, att_block, max_vec);
            // 缩放attention block
            mul(att_block, att_block,    1.44269504089f*0.125f);
            // 缩放max向量
            mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            // 从attention block中减去max值（数值稳定性）
            sub_row(att_block, att_block, max_vec_scaled);
            // 计算exp(attention)
            exp2(att_block, att_block);
            // 计算exp(max_last - max_current)
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled,       max_vec_last_scaled);
            // 更新norm向量
            mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            // 计算attention block的每行和
            row_sum(norm_vec,  att_block, norm_vec);
            
            // ==================== 准备V计算 ====================
            add(att_block, att_block, 0.f);  // 确保数据格式正确
            copy(att_block_mma, att_block);  // 复制到MMA版本
            // 缩放输出寄存器
            mul_row(o_reg, o_reg, max_vec_last_scaled); 
            
            // 等待当前V tile加载完成
            wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 
            
            // ==================== Attention*V 计算 ====================
            // 执行attention和V的矩阵乘法：o_reg += attention * V
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
            warpgroup::mma_async_wait();
            
            // 标记当前stage的计算完成
            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        // ==================== 最终输出处理 ====================
        // 归一化输出：o_reg = o_reg / norm_vec
        div_row(o_reg, o_reg, norm_vec);
        // 将结果存储到共享内存
        warpgroup::store(o_smem[warpgroupid], o_reg); 
        // 同步warpgroup
        warpgroup::sync(warpgroupid + 4);

        // ==================== 异步存储到全局内存 ====================
        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + warpgroupid, 0};
            // 异步存储结果到全局内存
            tma::store_async(p_G.O[dev_idx], o_smem[warpgroupid], o_tile_idx);
        }
    
        // ==================== 等待存储完成 ====================
        warpgroup::sync(warpgroupid+4);
        tma::store_async_wait();
    }
}

// #ifdef TORCH_COMPILE

template <int I, int SIZE> struct CHECK_INPUTS {
    static inline void apply(const int64_t B,
                             const int64_t H_qo,
                             const int64_t H_kv,
                             const int64_t N,
                             const int64_t D_h,
                             const std::vector<torch::Tensor>& Qs,
                             const std::vector<torch::Tensor>& Ks,
                             const std::vector<torch::Tensor>& Vs) {
        CHECK_INPUT(Qs[I]);
        CHECK_INPUT(Ks[I]);
        CHECK_INPUT(Vs[I]);

        TORCH_CHECK(Qs[I].size(0) == B, "Q batch dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Ks[I].size(0) == B, "K batch dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Vs[I].size(0) == B, "V batch dimension (device ", I, ") does not match with other inputs");

        TORCH_CHECK(Qs[I].size(1) == H_qo, "QO head dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Ks[I].size(1) == H_kv, "KV head dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Vs[I].size(1) == H_kv, "KV head dimension (device ", I, ") does not match with other inputs");

        TORCH_CHECK(Qs[I].size(2) == N, "Q sequence length dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Ks[I].size(2) == N, "K sequence length dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Vs[I].size(2) == N, "V sequence length dimension (device ", I, ") does not match with other inputs");

        TORCH_CHECK(Qs[I].size(3) == D_h, "Q head dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Ks[I].size(3) == D_h, "K head dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Vs[I].size(3) == D_h, "V head dimension (device ", I, ") does not match with other inputs");
        
        CHECK_INPUTS<I + 1, SIZE>::apply(B, H_qo, H_kv, N, D_h, Qs, Ks, Vs);  
    }
};
template <int SIZE> struct CHECK_INPUTS<SIZE, SIZE> {
    static inline void apply(const int64_t B,
                             const int64_t H_qo,
                             const int64_t H_kv,
                             const int64_t N,
                             const int64_t D_h,
                             const std::vector<torch::Tensor>&, 
                             const std::vector<torch::Tensor>&, 
                             const std::vector<torch::Tensor>&) {}
};

torch::Tensor pgl_tensor(
    const std::vector<int64_t> &sizes,
    const at::ScalarType dtype,
    const std::vector<int> &device_ids,
    const int device_id,
    const bool requires_grad
);
torch::Tensor pgl_tensor(
    const std::vector<int64_t> &sizes,
    const at::ScalarType dtype,
    const int *device_ids,
    const int device_id,
    const bool requires_grad
);
torch::Tensor pgl_tensor(
    const torch::Tensor &other, 
    const std::vector<int> &device_ids, 
    const int device_id
);

std::vector<torch::Tensor> ring_attention_forward(
    const std::vector<torch::Tensor> &Qs, 
    const std::vector<torch::Tensor> &Ks, 
    const std::vector<torch::Tensor> &Vs, 
    bool causal
) {
    // Input checking (up to CHECK_INPUTS<...>) takes about 3us 
    TORCH_CHECK(Qs.size() == NUM_DEVICES, "Qs must be of size ", NUM_DEVICES);
    TORCH_CHECK(Ks.size() == NUM_DEVICES, "Ks must be of size ", NUM_DEVICES);
    TORCH_CHECK(Vs.size() == NUM_DEVICES, "Vs must be of size ", NUM_DEVICES);

    int64_t B    = Qs[0].size(0);
    int64_t H_qo = Qs[0].size(1);
    int64_t H_kv = Ks[0].size(1);
    int64_t N    = Qs[0].size(2); // per-block sequence length
    int64_t D_h  = Qs[0].size(3);

    TORCH_CHECK(H_qo >= H_kv, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(H_qo % H_kv == 0, "QO heads must be divisible by KV heads");

    CHECK_INPUTS<0, NUM_DEVICES>::apply(B, H_qo, H_kv, N, D_h, Qs, Ks, Vs);

    // TODO: support different head sizes
    TORCH_CHECK(H_qo == H_kv, "For now, different head sizes not supported");
    // TODO: support different head dims
    TORCH_CHECK(D_h == 64, "For now, head dim must be 64");
    // TODO: support causal attention
    TORCH_CHECK(!causal, "Causal attention not supported yet");

    // Initialize the KC threadpool
    int device_ids[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
    KittensClub club(device_ids, NUM_DEVICES);

    // Initialize output tensor, device pointers, and streams
    std::vector<torch::Tensor> Os(NUM_DEVICES);
    bf16 *d_Q[NUM_DEVICES];
    bf16 *d_K[NUM_DEVICES];
    bf16 *d_V[NUM_DEVICES];
    bf16 *d_O[NUM_DEVICES];
    cudaStream_t streams[NUM_DEVICES];
    club.execute([&](int i) {
        Os[i] = torch::empty({B, H_qo, N, D_h}, Vs[i].options());
        d_Q[i] = reinterpret_cast<bf16*>(Qs[i].data_ptr<c10::BFloat16>());
        d_K[i] = reinterpret_cast<bf16*>(Ks[i].data_ptr<c10::BFloat16>());
        d_V[i] = reinterpret_cast<bf16*>(Vs[i].data_ptr<c10::BFloat16>());
        d_O[i] = reinterpret_cast<bf16*>(Os[i].data_ptr<c10::BFloat16>());
        streams[i] = at::cuda::getCurrentCUDAStream().stream();
    });

    // Initialize the parallel global layouts
    using pglobals = fwd_pglobals<64>;
    using Q_gl = typename fwd_pglobals<64>::Q_gl;
    using K_gl = typename fwd_pglobals<64>::K_gl;
    using V_gl = typename fwd_pglobals<64>::V_gl;
    using O_gl = typename fwd_pglobals<64>::O_gl;
    std::vector<Q_gl> g_Q;
    std::vector<K_gl> g_K;
    std::vector<V_gl> g_V;
    std::vector<O_gl> g_O;
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        g_Q.push_back(Q_gl(d_Q[dev_idx], B, H_qo, N, D_h));
        g_K.push_back(K_gl(d_K[dev_idx], B, H_kv, N, D_h));
        g_V.push_back(V_gl(d_V[dev_idx], B, H_kv, N, D_h));
        g_O.push_back(O_gl(d_O[dev_idx], B, H_qo, N, D_h));
    }
    pglobals p_G{g_Q.data(), g_K.data(), g_V.data(), g_O.data(), static_cast<int>(N)};

    // Initialize and run the kernel
    TORCH_CHECK(N % (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4) == 0, "sequence length must be divisible by 192");
    dim3 grid(N / (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4), H_qo, B);
    constexpr int smem = kittens::MAX_SHARED_MEMORY;

    club.execute([&](int i) {
        cudaFuncSetAttribute(blockwise_attn_ker<64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        blockwise_attn_ker<64, false><<<grid, NUM_WORKERS * kittens::WARP_THREADS, smem, streams[i]>>>(p_G, i);
        cudaStreamSynchronize(streams[i]);
        CHECK_CUDA_ERROR(cudaGetLastError());
    });

    return Os;
}

std::vector<torch::Tensor> ring_attention_qkvpacked_forward(
    const std::vector<torch::Tensor> &QKVs, 
    bool causal
) {
    // Input checking
    TORCH_CHECK(QKVs.size() == NUM_DEVICES, "QKVs must be of size ", NUM_DEVICES);

    int64_t B    = QKVs[0].size(0);  // batch_size
    int64_t N    = QKVs[0].size(1);  // seqlen
    int64_t H_qo = QKVs[0].size(3);  // nheads
    int64_t D_h  = QKVs[0].size(4);  // d

    TORCH_CHECK(QKVs[0].size(2) == 3, "QKV tensor must have dimension 2 equal to 3 (Q, K, V)");
    TORCH_CHECK(D_h == 64, "For now, head dim must be 64");
    TORCH_CHECK(!causal, "Causal attention not supported yet");

    // Check all QKV tensors have the same shape
    for (int i = 1; i < NUM_DEVICES; ++i) {
        TORCH_CHECK(QKVs[i].size(0) == B, "QKV batch dimension (device ", i, ") does not match");
        TORCH_CHECK(QKVs[i].size(1) == N, "QKV sequence length dimension (device ", i, ") does not match");
        TORCH_CHECK(QKVs[i].size(2) == 3, "QKV QKV dimension (device ", i, ") does not match");
        TORCH_CHECK(QKVs[i].size(3) == H_qo, "QKV head dimension (device ", i, ") does not match");
        TORCH_CHECK(QKVs[i].size(4) == D_h, "QKV feature dimension (device ", i, ") does not match");
    }

    // Split QKV packed tensors into separate Q, K, V tensors using slicing
    // QKV layout: [batch_size, seqlen, 3, nheads, d]
    // Extract Q: qkv[:, :, 0, :, :], K: qkv[:, :, 1, :, :], V: qkv[:, :, 2, :, :]
    std::vector<torch::Tensor> Qs(NUM_DEVICES);
    std::vector<torch::Tensor> Ks(NUM_DEVICES);
    std::vector<torch::Tensor> Vs(NUM_DEVICES);
    
    for (int i = 0; i < NUM_DEVICES; ++i) {
        // Q: qkv[:, :, 0, :, :]
        Qs[i] = QKVs[i].select(2, 0);
        // K: qkv[:, :, 1, :, :]  
        Ks[i] = QKVs[i].select(2, 1);
        // V: qkv[:, :, 2, :, :]
        Vs[i] = QKVs[i].select(2, 2);
    }

    // Initialize the KC threadpool
    int device_ids[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
    KittensClub club(device_ids, NUM_DEVICES);

    // Initialize output tensor, device pointers, and streams
    std::vector<torch::Tensor> Os(NUM_DEVICES);
    bf16 *d_Q[NUM_DEVICES];
    bf16 *d_K[NUM_DEVICES];
    bf16 *d_V[NUM_DEVICES];
    bf16 *d_O[NUM_DEVICES];
    cudaStream_t streams[NUM_DEVICES];
    club.execute([&](int i) {
        Os[i] = torch::empty({B, H_qo, N, D_h}, Vs[i].options());
        d_Q[i] = reinterpret_cast<bf16*>(Qs[i].data_ptr<c10::BFloat16>());
        d_K[i] = reinterpret_cast<bf16*>(Ks[i].data_ptr<c10::BFloat16>());
        d_V[i] = reinterpret_cast<bf16*>(Vs[i].data_ptr<c10::BFloat16>());
        d_O[i] = reinterpret_cast<bf16*>(Os[i].data_ptr<c10::BFloat16>());
        streams[i] = at::cuda::getCurrentCUDAStream().stream();
    });

    // Initialize the parallel global layouts
    using pglobals = fwd_pglobals<64>;
    using Q_gl = typename fwd_pglobals<64>::Q_gl;
    using K_gl = typename fwd_pglobals<64>::K_gl;
    using V_gl = typename fwd_pglobals<64>::V_gl;
    using O_gl = typename fwd_pglobals<64>::O_gl;
    std::vector<Q_gl> g_Q;
    std::vector<K_gl> g_K;
    std::vector<V_gl> g_V;
    std::vector<O_gl> g_O;
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        g_Q.push_back(Q_gl(d_Q[dev_idx], B, H_qo, N, D_h));
        g_K.push_back(K_gl(d_K[dev_idx], B, H_qo, N, D_h));
        g_V.push_back(V_gl(d_V[dev_idx], B, H_qo, N, D_h));
        g_O.push_back(O_gl(d_O[dev_idx], B, H_qo, N, D_h));
    }
    pglobals p_G{g_Q.data(), g_K.data(), g_V.data(), g_O.data(), static_cast<int>(N)};

    // Initialize and run the kernel
    TORCH_CHECK(N % (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4) == 0, "sequence length must be divisible by 192");
    dim3 grid(N / (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4), H_qo, B);
    constexpr int smem = kittens::MAX_SHARED_MEMORY;

    club.execute([&](int i) {
        cudaFuncSetAttribute(blockwise_attn_kernel<64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        blockwise_attn_kernel<64, false><<<grid, NUM_WORKERS * kittens::WARP_THREADS, smem, streams[i]>>>(p_G, i);
        cudaStreamSynchronize(streams[i]);
        CHECK_CUDA_ERROR(cudaGetLastError());
    });

    return Os;
}

std::vector<torch::Tensor> ring_attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
    torch::Tensor l_vec, torch::Tensor og, bool causal
) {
    TORCH_CHECK(false, "Backward ring attention not implemented");
    return {q, k, v, o, l_vec, og};
}

struct pgl_tensor_context {
    int device_id;
    void *raw_ptr;
    size_t size;
};

void _pgl_tensor_deleter(void* ptr) {
    pgl_tensor_context *ctx = static_cast<pgl_tensor_context*>(ptr);
    pglCudaFree(ctx->device_id, ctx->raw_ptr, ctx->size);
    free(ctx);
}

torch::Tensor pgl_tensor(
    const std::vector<int64_t> &sizes,
    const at::ScalarType dtype,
    const int *device_ids,
    const int device_id,
    const bool requires_grad
) {
    TORCH_CHECK(device_id >= 0 && device_id < NUM_DEVICES, "Invalid device ID");

    // Calculate number of elements and bytes
    int64_t numel = 1;
    for (auto s : sizes) {
        TORCH_CHECK(s > 0, "Size dimensions must be positive");
        numel *= s;
    }

    // Allocate CUDA memory
    pgl_tensor_context *ctx = new pgl_tensor_context;
    ctx->device_id = device_id;
    ctx->raw_ptr = nullptr;
    ctx->size = numel * c10::elementSize(dtype);
    pglCudaMalloc<true>(NUM_DEVICES, const_cast<int*>(device_ids), device_id, &ctx->raw_ptr, ctx->size);

    // Construct Tensor
    c10::DataPtr data_ptr(ctx->raw_ptr, ctx, _pgl_tensor_deleter,
        c10::Device(c10::DeviceType::CUDA, device_id));
    at::TensorOptions options = at::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
    at::Storage storage = at::Storage({}, ctx->size, std::move(data_ptr), nullptr, false);
    torch::Tensor tensor = at::empty(0, options).set_(storage, 0, at::IntArrayRef(sizes.data(), sizes.size()), {});
    tensor.set_requires_grad(requires_grad);

    // Sanity check. Can be removed in production code
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    return tensor;
}

torch::Tensor pgl_tensor(
    const std::vector<int64_t> &sizes,
    const at::ScalarType dtype,
    const std::vector<int> &device_ids,
    const int device_id,
    const bool requires_grad
) {
    TORCH_CHECK(device_id >= 0 && device_id < static_cast<int>(device_ids.size()), "Invalid device ID");
    return pgl_tensor(sizes, dtype, device_ids.data(), device_id, requires_grad);
}

torch::Tensor pgl_tensor(
    const torch::Tensor &other, 
    const std::vector<int> &device_ids, 
    const int device_id
) {
    TORCH_CHECK(device_id >= 0 && device_id < static_cast<int>(device_ids.size()), "Invalid device ID");

    bool on_gpu = other.device().is_cuda();
    if (on_gpu) {
        std::cerr << "WARNING (pgl_tensor): the given tensor is already on GPU. "
                  << "This will result in a redundant memory allocation and copy.\n";
    }
    
    // Allocate CUDA memory
    pgl_tensor_context *ctx = new pgl_tensor_context;
    ctx->device_id = device_id;
    ctx->raw_ptr = nullptr;
    ctx->size = other.nbytes();
    pglCudaMalloc<true>(NUM_DEVICES, const_cast<int*>(device_ids.data()), device_id, &ctx->raw_ptr, ctx->size);

    // Copy data
    cudaMemcpyKind copy_kind = on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpy(ctx->raw_ptr, other.data_ptr(), ctx->size, copy_kind);

    // Construct Tensor (this is required because data_ptr is a smart pointer)
    c10::DataPtr data_ptr(ctx->raw_ptr, ctx, _pgl_tensor_deleter,
        c10::Device(c10::DeviceType::CUDA, device_id));
    at::TensorOptions options = other.options().device(torch::kCUDA, device_id); // includes dtype, device, layout
    at::Storage storage = at::Storage({}, ctx->size, std::move(data_ptr), nullptr, false);
    torch::Tensor tensor = at::empty(0, options).set_(storage, 0, other.sizes(), {});
    if (other.requires_grad()) tensor.set_requires_grad(true);

    // Sanity check. Can be removed in production code
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ThunderKittens Ring Attention Kernels";
    m.def(
        "ring_mha_forward",  
        torch::wrap_pybind_function(ring_attention_forward),
        "Forward ring MHA"
    );
    m.def(
        "ring_mha_qkvpacked_forward",  
        torch::wrap_pybind_function(ring_attention_qkvpacked_forward),
        "Forward ring MHA with QKV packed tensors"
    );
    m.def(
        "ring_mha_backward", 
        torch::wrap_pybind_function(ring_attention_backward), 
        "Backward ring MHA"
    );
    m.def(
        "pgl_tensor", 
        static_cast<torch::Tensor(*)(const torch::Tensor&, const std::vector<int>&, const int)>(&pgl_tensor),
        "Create a PGL tensor from existing tensor"
    );
    m.def(
        "pgl_tensor", 
        static_cast<torch::Tensor(*)(const std::vector<int64_t>&, const at::ScalarType, const std::vector<int>&, const int, const bool)>(&pgl_tensor),
        "Create a new PGL tensor from sizes and dtype"
    );
}

// #else

// #endif
