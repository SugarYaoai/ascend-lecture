### 第四节 TensorOJ 实战：基于 C++ API 实现 Add 算子

第三节用 C API 写 Add 时，开发者亲自管理 GM 裸指针、UB 数组以及字节和元素两套长度单位。这种方式能直接看到硬件如何工作，但真实算子中很容易遇到两类问题：

- **单位混淆**：DMA 搬运接口需要 `10752 * sizeof(float)` 这样的字节数，向量加法却只需要 `10752` 个元素。漏写一次 `sizeof(float)` 往往仍能通过编译，却会让搬运范围错误，进而导致结果异常或设备侧访问失败。
- **复杂流水线难以手工维护**：直接声明 `__ubuf__` 数组适合单次搬入、计算和写回；当一个 Block 需要多个 Tile、多个 Buffer 或双缓冲时，开发者必须自己计算每块 UB 的位置、复用时机和同步状态，代码很容易失去可读性。

C++ API 的作用就是降低这两类错误的概率：`GlobalTensor<float>`、`LocalTensor<float>` 让代码始终带着元素类型和长度来描述数据；`LocalMemAllocator` 先把 UB 中的局部数据组织为明确的对象，后续可以自然升级到用 `TPipe`、`TQue` 管理缓冲区复用与入队出队。

C++ API 不会自动完成任何 GM/UB 搬运，也不会替代容量、对齐和边界检查。它只是给裸地址和 UB 数组加上更清楚的类型与长度信息，并为后续的流水线优化提供更容易维护的代码结构。

#### 一、GM 地址绑定与 UB 内存分配

算子继续复用第三节的静态执行参数：

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;
```

在 C++ API 中，`blockLength` 被写成 Kernel 模板参数。这样 `Alloc<float, blockLength>()` 能在编译期获知局部张量长度；Host 侧随后用 `add_custom<BLOCK_LENGTH>` 实例化对应的 Kernel。

每个 Block 仍以 `block_idx` 计算数据偏移。区别在于，裸 GM 地址会先绑定为三个 `GlobalTensor<float>`，让后续代码直接围绕当前 Block 的 `x`、`y`、`z` 数据范围组织：

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

#### 二、带类型的 DataCopy 与流水线屏障控制

GM 地址和 UB 工作区建立后，数据按三个依赖阶段执行：搬入 `x`、`y`，在 UB 中计算 `z`，再写回 `z`。

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

#### 三、C API 与 C++ API 的区别

完成同一份 `add_custom` 后，可以更具体地比较两种 API 的差异。核心并不是函数名替换，而是内存资源、数据范围与依赖关系在代码中的表达方式：

| 维度 | C API | C++ API | 对开发者的意义 |
| --- | --- | --- | --- |
| GM 数据范围 | `__gm__ float*` 裸指针与手写偏移 | `GlobalTensor<float>` | 将当前 Block 的元素类型、起始地址和长度放在同一个对象中。 |
| UB 局部空间 | `__ubuf__ float local[...]` | `LocalMemAllocator<UB>` + `LocalTensor<float>` | 用对象记录 UB 中局部数据的类型和长度。 |
| 数据搬运 | 传入地址与字节数 | 在两个带类型 Tensor 间 `DataCopy` | 元素类型由 Tensor 携带，长度参数按元素个数表达。 |
| 向量计算 | `asc_add` | `AscendC::Add` | 对 UB 中带类型的局部张量进行向量运算。 |
| 核内依赖 | `asc_sync()` | `AscendC::PipeBarrier<PIPE_ALL>()` | 显式约束搬运流水线与向量流水线之间的数据依赖。 |

两种 API 的硬件数据路径完全一致：`GM -> UB -> Vector -> UB -> GM`。`GlobalTensor` 与 `LocalTensor` 只描述数据位于哪里、长度是多少，不会自行复制数据；GM 与 UB 之间的物理搬运仍由 `DataCopy` 发起，阶段间依赖仍由 `PipeBarrier` 保证。

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
