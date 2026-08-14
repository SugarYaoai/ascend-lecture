## Part 3：TensorOJ C++ API 实战

本节仍完成同一个 Easy Version Add：输入为两个长度 `172032` 的 `float32` 向量，输出为逐元素相加结果。

计算划分为 `16` 个 Block，每个 Block 处理 `10752` 个元素。每个 Block 的三个临时张量共占用：

$$
3 \times 10752 \times 4\ \text{B} = 129024\ \text{B} = 126\ \text{KB}
$$

因此三个临时张量可以同时放入一个 AI Core 的 UB 中。

### C API 与 C++ API 的分工对应

| 计算步骤 | C API 写法 | C++ API 写法 |
| --- | --- | --- |
| 访问 Global Memory | `__gm__ float*` | `AscendC::GlobalTensor<float>` |
| 在 UB 中申请临时空间 | `__ubuf__ float local[...]` | `TPipe` + `TQue` + `AllocTensor<float>()` |
| GM 到 UB 搬运 | `asc_copy_gm2ub` | `AscendC::DataCopy` |
| 向量逐元素加法 | `asc_add` | `AscendC::Add` |
| UB 到 GM 搬运 | `asc_copy_ub2gm` | `AscendC::DataCopy` |
| 归还临时张量 | 静态数组自动结束生命周期 | `FreeTensor` |

C API 直接操作指针和 UB 数组，代码短，适合先看清一次 Add 的数据搬运与计算。C++ API 将 Global Memory、UB 缓冲区和队列封装成对象；代码多了一层组织，但在后续引入 Tile、双缓冲、多个计算阶段时更容易扩展。

### 1. 将 Global Memory 绑定为 GlobalTensor

每个 Block 都要先计算自己的全局起始下标。`AscendC::GetBlockIdx()` 取得当前 Block 编号；`SetGlobalBuffer` 将输入输出地址和当前 Block 的长度绑定到三个 `GlobalTensor` 对象。

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    uint32_t offset = AscendC::GetBlockIdx() * BLOCK_LENGTH;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, BLOCK_LENGTH);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + offset, BLOCK_LENGTH);
    zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + offset, BLOCK_LENGTH);
}
```

这样，后续 `DataCopy(xLocal, xGm, BLOCK_LENGTH)` 就表示把当前 Block 对应的一段 `x` 搬到 UB，不需要再手动书写指针偏移。

### 2. 用 TPipe 和 TQue 划分 UB 缓冲区

`TPipe` 管理本 Block 的 UB 空间，三个 `TQue` 分别对应输入 `x`、输入 `y` 和输出 `z`。队列深度设为 `1`，因此这是单缓冲版本：一次只处理一个 Block 内的数据块。

```cpp
pipe.InitBuffer(xQueue, 1, BLOCK_LENGTH * sizeof(float));
pipe.InitBuffer(yQueue, 1, BLOCK_LENGTH * sizeof(float));
pipe.InitBuffer(zQueue, 1, BLOCK_LENGTH * sizeof(float));
```

`InitBuffer` 只负责预留 UB 空间；真正取得可读写的 UB 张量时，使用 `AllocTensor<float>()`。完成使用后，必须通过 `FreeTensor` 归还张量。

### 3. CopyIn：从 GM 搬运两个输入

```cpp
__aicore__ inline void CopyIn()
{
    AscendC::LocalTensor<float> xLocal = xQueue.AllocTensor<float>();
    AscendC::LocalTensor<float> yLocal = yQueue.AllocTensor<float>();

    AscendC::DataCopy(xLocal, xGm, BLOCK_LENGTH);
    AscendC::DataCopy(yLocal, yGm, BLOCK_LENGTH);

    xQueue.EnQue(xLocal);
    yQueue.EnQue(yLocal);
}
```

`DataCopy` 负责 Global Memory 与 UB 之间的数据搬运。`EnQue` 表示这个张量已经准备完毕，可以交给下一个阶段读取。

### 4. Compute：由 AI Core 的向量单元执行 Add

```cpp
__aicore__ inline void Compute()
{
    AscendC::LocalTensor<float> xLocal = xQueue.DeQue<float>();
    AscendC::LocalTensor<float> yLocal = yQueue.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = zQueue.AllocTensor<float>();

    AscendC::Add(zLocal, xLocal, yLocal, BLOCK_LENGTH);

    xQueue.FreeTensor(xLocal);
    yQueue.FreeTensor(yLocal);
    zQueue.EnQue(zLocal);
}
```

`AscendC::Add` 对 UB 中的 `float32` 数据执行向量逐元素加法。AI Core 的向量计算单元会以向量指令批量处理连续元素；这里不需要为每个元素手写循环。

### 5. CopyOut：将结果写回 GM

```cpp
__aicore__ inline void CopyOut()
{
    AscendC::LocalTensor<float> zLocal = zQueue.DeQue<float>();
    AscendC::DataCopy(zGm, zLocal, BLOCK_LENGTH);
    zQueue.FreeTensor(zLocal);
}
```

至此，一个 Block 的路径为：`GM x/y -> UB x/y -> 向量 Add -> UB z -> GM z`。16 个 Block 使用相同逻辑，分别处理完整向量的 16 段。

### 完整 kernel.asc

```cpp
#include <cstdint>
#include "kernel_operator.h"

constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;

class KernelAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        uint32_t offset = AscendC::GetBlockIdx() * BLOCK_LENGTH;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, BLOCK_LENGTH);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + offset, BLOCK_LENGTH);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + offset, BLOCK_LENGTH);

        pipe.InitBuffer(xQueue, 1, BLOCK_LENGTH * sizeof(float));
        pipe.InitBuffer(yQueue, 1, BLOCK_LENGTH * sizeof(float));
        pipe.InitBuffer(zQueue, 1, BLOCK_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> xLocal = xQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = yQueue.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm, BLOCK_LENGTH);
        AscendC::DataCopy(yLocal, yGm, BLOCK_LENGTH);
        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = xQueue.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = yQueue.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = zQueue.AllocTensor<float>();

        AscendC::Add(zLocal, xLocal, yLocal, BLOCK_LENGTH);

        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
        zQueue.EnQue(zLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> zLocal = zQueue.DeQue<float>();
        AscendC::DataCopy(zGm, zLocal, BLOCK_LENGTH);
        zQueue.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> yQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> zQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
};

__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}

extern "C" void run_kernel(GM_ADDR x, const TensorGroupInfo& info_x,
                           GM_ADDR y, const TensorGroupInfo& info_y,
                           GM_ADDR z, const TensorGroupInfo& info_z,
                           int64_t availableCoreNum, aclrtStream stream)
{
    if (info_x.numTensors != 1 || info_y.numTensors != 1 ||
        info_z.numTensors != 1 || info_x.tensors[0].dtype != 0 ||
        info_y.tensors[0].dtype != 0 || info_z.tensors[0].dtype != 0 ||
        info_x.tensors[0].shape[0] != TOTAL_LENGTH ||
        info_y.tensors[0].shape[0] != TOTAL_LENGTH ||
        info_z.tensors[0].shape[0] != TOTAL_LENGTH ||
        availableCoreNum <= 0) {
        return;
    }

    add_custom<<<NUM_BLOCKS, nullptr, stream>>>(x, y, z);
}
```

这个版本与 Part 2 的 C API 版本完成相同的计算和相同的 Block 划分。差别在于：C++ API 将 UB 的分配、传递和释放明确表达为队列操作，为下一阶段的 Tile 分块、双缓冲与多阶段流水线提供了结构。
