### 第四节 TensorOJ 实战：基于 C++ API 实现 Add 算子

第三节的 C API 允许开发者以最贴近硬件的形式操作 GM 裸地址、UB 数组和字节长度。它适合建立对数据搬运与向量计算的直接认识；但当算子需要处理更多输入、临时张量与多阶段数据流时，开发者必须同时维护地址偏移、元素类型、字节长度、UB 资源和流水线依赖。这些约束一旦彼此脱节，错误往往不会表现为普通的 C++ 类型报错，而会在设备侧以错误的数据范围、错误的搬运长度或资源分配失败的形式出现。

从 C API 迁移到 C++ Tensor API，主要解决两类工程问题：

- **类型与范围的结构化表达**：C API 中，GM 地址、元素类型和长度由开发者分散维护；DMA 搬运按字节计数，Vector 计算按元素计数，需要手动完成换算。`GlobalTensor<float>` 与 `LocalTensor<float>` 将元素类型和当前数据范围组织为 Tensor 视图，`DataCopy`、`Add` 等接口据此按元素表达长度，减少指针算术与单位换算混杂的机会。
- **片上资源与流水线的可演进组织**：C 风格 `__ubuf__` 数组可以手动实现多缓冲，但每增加一块 Buffer 或一个 Tile 阶段，都需要开发者自行安排 UB 占用与同步关系。`LocalMemAllocator` 先将 UB 局部资源组织为显式对象；后续的 `TPipe`、`TQue` 则在此基础上进一步表达缓冲区复用、入队出队与双缓冲流水线。

因此，C++ Tensor API 并不是替开发者自动完成 GM/UB 搬运，也不能替代容量、对齐和边界检查；它的作用是让这些硬件约束在代码结构中拥有明确的类型、范围和资源归属。本节将在不改变物理执行计划的前提下，将同一份 Add 重写为带类型的 GM/UB 张量视图与结构化局部内存分配。

#### 一、C API 与 C++ Tensor API 的范式差异

从 C API 迁移到 C++ Tensor API，核心并不是把 `asc_add` 换成 `AscendC::Add`，而是改变了内存资源与数据范围在代码中的表达方式：

| 维度 | C API | C++ Tensor API | 对开发者的意义 |
| --- | --- | --- | --- |
| GM 数据范围 | `__gm__ float*` 裸指针与手写偏移 | `GlobalTensor<float>` | 将当前 Block 的元素类型、起始地址和长度组织为一个 GM 视图。 |
| UB 局部空间 | `__ubuf__ float local[...]` | `LocalMemAllocator<UB>` + `LocalTensor<float>` | 将 UB 中的局部对象、元素类型和长度显式表达为结构化资源。 |
| 数据搬运 | 传入地址与字节数 | 在两个带类型 Tensor 间 `DataCopy` | 元素类型由 Tensor 携带，长度参数按元素个数表达。 |
| 向量计算 | `asc_add` | `AscendC::Add` | 对 UB 中带类型的局部张量进行向量运算。 |
| 核内依赖 | `asc_sync()` | `AscendC::PipeBarrier<PIPE_ALL>()` | 显式约束搬运流水线与向量流水线之间的数据依赖。 |

两种 API 的硬件数据路径完全一致：`GM -> UB -> Vector -> UB -> GM`。`GlobalTensor` 与 `LocalTensor` 都是内存视图，不会自行复制数据；GM 与 UB 之间的物理搬运仍由 `DataCopy` 发起，阶段间依赖仍由 `PipeBarrier` 保证。

#### 二、物理视图绑定与片上内存的结构化分配

算子继续复用第三节的静态执行参数：

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;
```

在 C++ API 中，`blockLength` 被写成 Kernel 模板参数。这样 `Alloc<float, blockLength>()` 能在编译期获知局部张量长度；Host 侧随后用 `add_custom<BLOCK_LENGTH>` 实例化对应的 Kernel。

每个 Block 仍以 `block_idx` 计算数据偏移。区别在于，裸 GM 地址会先绑定为三个 `GlobalTensor<float>`，让后续代码直接围绕“当前 Block 的 x、y、z 视图”组织：

```cpp
AscendC::GlobalTensor<float> xGm;
AscendC::GlobalTensor<float> yGm;
AscendC::GlobalTensor<float> zGm;

const uint32_t offset = block_idx * blockLength;
xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, blockLength);
yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + offset, blockLength);
zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + offset, blockLength);
```

`SetGlobalBuffer` 只绑定地址和长度，不触发数据搬运。`reinterpret_cast<__gm__ float*>` 仍然是必要的：题目接口传入的是通用 GM 地址，转换后 `+ offset` 才按 `float32` 元素而非字节进行寻址。

UB 侧不再使用三个 C 风格数组，而是通过 `LocalMemAllocator` 分配三个 `LocalTensor<float>`：

```cpp
AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;

AscendC::LocalTensor<float> xLocal = ubAllocator.Alloc<float, blockLength>();
AscendC::LocalTensor<float> yLocal = ubAllocator.Alloc<float, blockLength>();
AscendC::LocalTensor<float> zLocal = ubAllocator.Alloc<float, blockLength>();
```

它们仍对应三段 `10752` 元素的 UB 空间：

$$
3 \times 10752 \times 4\text{ B} = 129024\text{ B} = 126\text{ KB}
$$

`LocalMemAllocator` 的直接分配方式清楚地列出了 UB 中有哪些局部资源及其规格，同时没有引入队列或双缓冲。后续引入 TPipe/TQue 时，改变的是这些局部资源的组织与复用方式，而不是 Add 的 GM/UB 基本路径。

#### 三、带类型的 DataCopy 与流水线屏障控制

GM 视图与 UB 工作区建立后，数据按三个依赖阶段执行：搬入 `x`、`y`，在 UB 中计算 `z`，再写回 `z`。

```cpp
AscendC::DataCopy(xLocal, xGm, blockLength);
AscendC::DataCopy(yLocal, yGm, blockLength);
AscendC::PipeBarrier<PIPE_ALL>();

AscendC::Add(zLocal, xLocal, yLocal, blockLength);
AscendC::PipeBarrier<PIPE_ALL>();

AscendC::DataCopy(zGm, zLocal, blockLength);
AscendC::PipeBarrier<PIPE_ALL>();
```

由于 `DataCopy` 的源和目的都是 `float` Tensor，`blockLength` 表示元素个数；C++ API 已从 Tensor 类型中得知单元素大小，因此不需要像 C API 那样手动写 `sizeof(float)`。`PipeBarrier<PIPE_ALL>()` 是当前 Block、当前 AI Core 内部的流水线屏障：它保证输入片段写入 UB 后再被 Vector 单元读取，也保证结果生成后再写回 GM。此处每种局部张量只有一块，三个阶段按依赖顺序执行。

#### 四、完整交付代码解析：kernel.asc

`run_kernel` 的题目接口与第三节相同：它校验本题固定的输入规格后，使用 `add_custom<BLOCK_LENGTH><<<NUM_BLOCKS, nullptr, stream>>>(x, y, z)` 实例化模板 Kernel，并将 16 Block 的任务挂载到模板传入的 Stream。下面是完整的 `kernel.asc`：

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
