### 第四节 TensorOJ 实战：基于 C++ API 实现 Add 算子

本节继续使用 TensorOJ 的 [Add Simple](https://cannjudge.cn/pku-tensor/education/add-simple/submit) 题目。第三节已经用 C API 完成了同一份 Add：16 个 Block 各自处理一段 `float32` 数据，将片段从 GM 搬入 UB，完成向量加法后再写回 GM。

本节不改变 Add 的数学规则、Block 切分或 UB 容量规划，而是改用 Ascend C 的 C++ Tensor API 表达同一份执行计划。它的价值不在于自动省略数据搬运，而在于用带类型的张量对象表达 GM 视图、UB 工作区和数据流，使 Kernel 变长后仍能保持清晰的资源边界。

#### 一、同一执行计划的两种代码表达

C API 与 C++ API 最根本的差异是内存和同步的表达方式，而不是硬件路径。两者执行的仍是同一条数据流：`GM -> UB -> Vector -> UB -> GM`。

| 任务 | C API | 本节的 C++ API |
| --- | --- | --- |
| 初始化设备侧状态 | `asc_init()` | `AscendC::InitSocState()` |
| 描述当前 Block 的 GM 数据 | `__gm__ float*` | `AscendC::GlobalTensor<float>` |
| 申请 UB 局部空间 | `__ubuf__ float local[...]` | `LocalMemAllocator<UB>` + `LocalTensor<float>` |
| GM 到 UB | `asc_copy_gm2ub` | `AscendC::DataCopy` |
| UB 内向量加法 | `asc_add` | `AscendC::Add` |
| UB 到 GM | `asc_copy_ub2gm` | `AscendC::DataCopy` |
| 核内流水线依赖 | `asc_sync()` | `AscendC::PipeBarrier<PIPE_ALL>()` |

这里最容易产生的误解是：“使用 Tensor 对象后，数据会自动从 GM 进入 UB。”事实并非如此。`GlobalTensor<float>` 只是当前 Block 对 GM 中一段 `float32` 数据的视图；`LocalTensor<float>` 只是 UB 中一段局部空间的视图。真正触发数据移动的仍然是 `DataCopy`，同步依赖仍然需要 `PipeBarrier` 显式表达。

#### 二、`add_custom` Kernel 算子实现

##### （一）保持不变的物理执行计划

C++ API 没有改变第三节确定的物理约束：总长度仍为 `172032`，启动 `16` 个 Block，每个 Block 处理 `10752` 个 `float32` 元素。每个 Block 的 `x`、`y`、`z` 三段局部数据共占 `126 KB`，仍在单个 AI Core 的 UB 资源预算内。

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;
```

与 C API 示例不同的是，C++ API 将 `blockLength` 作为 Kernel 模板参数传入。这样 `ubAllocator.Alloc<float, blockLength>()` 可以在编译期知道需要申请的局部张量长度；启动 Kernel 时再通过 `add_custom<BLOCK_LENGTH>` 将固定值实例化进去。

##### （二）用 `GlobalTensor` 绑定当前 Block 的 GM 区间

每个 Block 仍从 `block_idx` 获得自己的逻辑编号，并用 `offset = block_idx × blockLength` 定位输入、输出片段。区别在于，C++ API 不把这三个地址一直以裸指针形式传递，而是将它们绑定为三个带元素类型和长度的 `GlobalTensor<float>`：

```cpp
AscendC::GlobalTensor<float> xGm;
AscendC::GlobalTensor<float> yGm;
AscendC::GlobalTensor<float> zGm;

const uint32_t offset = block_idx * blockLength;
xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, blockLength);
yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + offset, blockLength);
zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + offset, blockLength);
```

`SetGlobalBuffer` 不会复制任何数据。它只完成“当前张量对象对应 GM 的哪一段数据”的绑定：后续 `DataCopy(xLocal, xGm, blockLength)` 会读取这一段 `xGm`，而不是重新计算地址偏移。

##### （三）用 `LocalTensor` 表达 UB 工作区

当前 Block 需要同时保存 `x`、`y` 与 `z` 三段局部数据。`LocalMemAllocator<AscendC::Hardware::UB>` 从当前 AI Core 的 UB 中分配局部张量，`LocalTensor<float>` 则以带类型的对象保存这些分配结果：

```cpp
AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;

AscendC::LocalTensor<float> xLocal = ubAllocator.Alloc<float, blockLength>();
AscendC::LocalTensor<float> yLocal = ubAllocator.Alloc<float, blockLength>();
AscendC::LocalTensor<float> zLocal = ubAllocator.Alloc<float, blockLength>();
```

三块 `LocalTensor` 与 C API 的 `xLocal`、`yLocal`、`zLocal` 一一对应，合计占用：

$$
3 \times 10752 \times 4\text{ B} = 129024\text{ B} = 126\text{ KB}
$$

这里使用的是 `LocalMemAllocator` 的直接分配方式。它把“UB 中有哪些局部对象、每个对象的元素类型和长度”显式写进代码，但没有引入队列或双缓冲机制。

##### （四）由 `DataCopy`、`Add` 与 `PipeBarrier` 组织数据依赖

GM 视图和 UB 工作区准备好后，数据仍按三阶段流动：先将 `x`、`y` 搬入 UB，再在 UB 中相加，最后将 `z` 写回 GM。

```cpp
AscendC::DataCopy(xLocal, xGm, blockLength);
AscendC::DataCopy(yLocal, yGm, blockLength);
AscendC::PipeBarrier<PIPE_ALL>();

AscendC::Add(zLocal, xLocal, yLocal, blockLength);
AscendC::PipeBarrier<PIPE_ALL>();

AscendC::DataCopy(zGm, zLocal, blockLength);
AscendC::PipeBarrier<PIPE_ALL>();
```

`DataCopy` 的源和目的都是带类型的 Tensor，因此长度 `blockLength` 按 `float` 元素个数解释；这与 C API 的搬运接口需要显式传入字节数不同。`PipeBarrier<PIPE_ALL>()` 是当前 Block、当前 AI Core 内部的流水同步：它保证输入搬运完成后再开始向量计算，也保证向量计算完成后再写回结果。这里每类局部张量只有一块，因此三阶段按依赖顺序推进。

#### 三、`run_kernel` 调用接口实现

##### （一）复用固定规格的参数校验

题目模板传入的 `GM_ADDR`、张量元信息和 Stream 接口与第三节完全相同。由于这一份 C++ API Kernel 仍只支持长度为 `172032` 的单个 `float32` 向量，入口校验也保持相同：

```cpp
if (info_x.numTensors != 1 || info_y.numTensors != 1 ||
    info_z.numTensors != 1 || info_x.tensors[0].dtype != 0 ||
    info_y.tensors[0].dtype != 0 || info_z.tensors[0].dtype != 0 ||
    info_x.tensors[0].shape[0] != TOTAL_LENGTH ||
    info_y.tensors[0].shape[0] != TOTAL_LENGTH ||
    info_z.tensors[0].shape[0] != TOTAL_LENGTH ||
    availableCoreNum <= 0) {
    return;
}
```

##### （二）实例化模板 Kernel 并挂载到 Stream

校验通过后，Host 侧入口需要同时确定 Block 数与 Kernel 模板参数：

```cpp
add_custom<BLOCK_LENGTH><<<NUM_BLOCKS, nullptr, stream>>>(x, y, z);
```

尖括号前的 `<BLOCK_LENGTH>` 是 C++ 模板实参，它在编译期确定 `add_custom` 中局部张量的长度；三尖括号 `<<<NUM_BLOCKS, nullptr, stream>>>` 是运行时执行配置，分别指定启动 16 个 Block、不使用动态 UB，并将任务加入模板传入的 Stream。圆括号中的 `x`、`y`、`z` 则是本次调用的 GM 地址。

#### 四、完整交付代码解析：kernel.asc

下面代码可以直接放入 TensorOJ 的 `kernel.asc`。它没有改变第三节的内存流向和运行时接口，只将 C API 的裸指针和 UB 数组替换为 C++ API 的 `GlobalTensor`、`LocalTensor`、`LocalMemAllocator` 与 `DataCopy`。

```cpp
#include <cstdint>
#include "kernel_operator.h"

constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;

template <uint32_t blockLength>
__vector__ __global__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    AscendC::InitSocState();

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;

    const uint32_t offset = block_idx * blockLength;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, blockLength);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + offset, blockLength);
    zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + offset, blockLength);

    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    AscendC::LocalTensor<float> xLocal = ubAllocator.Alloc<float, blockLength>();
    AscendC::LocalTensor<float> yLocal = ubAllocator.Alloc<float, blockLength>();
    AscendC::LocalTensor<float> zLocal = ubAllocator.Alloc<float, blockLength>();

    AscendC::DataCopy(xLocal, xGm, blockLength);
    AscendC::DataCopy(yLocal, yGm, blockLength);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Add(zLocal, xLocal, yLocal, blockLength);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopy(zGm, zLocal, blockLength);
    AscendC::PipeBarrier<PIPE_ALL>();
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

    add_custom<BLOCK_LENGTH><<<NUM_BLOCKS, nullptr, stream>>>(x, y, z);
}
```
