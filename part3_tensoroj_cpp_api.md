### 第四节 TensorOJ 实战：基于 C++ API 实现 Add 算子

第三节已经用 C API 通过了同一个 Add：直接写 GM/UB 指针时，数据位置非常直观。随着 Kernel 变长，地址偏移、局部缓冲区和同步关系也会越来越多；C++ API 的价值是把这些对象显式表达出来，而不是改变 Add 的数学规则或运行方式。

本节仍完成 Add Simple：两个长度为 `172032` 的 `float32` 向量逐元素相加。计算参数保持不变：

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;
```

这是一种表达方式的切换：C API 用指针描述内存，C++ API 用张量对象描述 GM 与 UB 中的一段连续数据。

#### 一、C API 与 C++ API 的对应

| 任务 | C API | 本节的 C++ API |
| --- | --- | --- |
| 初始化设备侧状态 | `asc_init()` | `AscendC::InitSocState()` |
| 描述 GM 中的一段数据 | `__gm__ float*` | `AscendC::GlobalTensor<float>` |
| 申请 UB 临时空间 | `__ubuf__ float local[...]` | `LocalMemAllocator<UB>` + `LocalTensor<float>` |
| GM 到 UB | `asc_copy_gm2ub` | `AscendC::DataCopy` |
| UB 内 Add | `asc_add` | `AscendC::Add` |
| UB 到 GM | `asc_copy_ub2gm` | `AscendC::DataCopy` |
| 阶段依赖 | `asc_sync()` | `AscendC::PipeBarrier<PIPE_ALL>()` |

C++ API 没有改变 Add 的数据路径，只是把“地址”“片上张量”“搬运”“计算”表达成更明确的对象和函数调用。`GlobalTensor` 是对 GM 中一段连续元素的视图，不会自行搬运数据；`LocalTensor` 则表示 UB 中一段局部缓冲区。两者之间仍然必须通过 `DataCopy` 搬运。

#### 二、将当前 Block 的 GM 地址绑定为 GlobalTensor

`GM_ADDR` 是 TensorOJ 运行时传入的设备地址。当前 Block 先根据 `block_idx` 计算自己的起始位置，再把这段地址绑定到三个 `GlobalTensor<float>`：

```cpp
const uint32_t offset = block_idx * BLOCK_LENGTH;

xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + offset, BLOCK_LENGTH);
yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + offset, BLOCK_LENGTH);
zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + offset, BLOCK_LENGTH);
```

之后 `xGm`、`yGm`、`zGm` 分别代表当前 Block 要读写的三段 Global Memory，不再需要在每个数据搬运调用中手写地址偏移。

#### 三、用 LocalMemAllocator 直接申请三块 UB

`LocalMemAllocator<AscendC::Hardware::UB>` 是 UB 的局部内存分配器。它直接从当前 AI Core 的 UB 中申请三块局部张量，分别保存 `x`、`y` 和 `z`：

```cpp
AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;

AscendC::LocalTensor<float> xLocal =
    ubAllocator.Alloc<float, BLOCK_LENGTH>();
AscendC::LocalTensor<float> yLocal =
    ubAllocator.Alloc<float, BLOCK_LENGTH>();
AscendC::LocalTensor<float> zLocal =
    ubAllocator.Alloc<float, BLOCK_LENGTH>();
```

三块 `LocalTensor` 与 C API 的 `xLocal`、`yLocal`、`zLocal` 对应。它们占用：

$$
3 \times 10752 \times 4\ \text{B} = 126\ \text{KB}
$$

局部张量只服务当前 Kernel 的这一次固定长度计算。

#### 四、按顺序搬入、计算和写回

```cpp
AscendC::DataCopy(xLocal, xGm, BLOCK_LENGTH);
AscendC::DataCopy(yLocal, yGm, BLOCK_LENGTH);
AscendC::PipeBarrier<PIPE_ALL>();

AscendC::Add(zLocal, xLocal, yLocal, BLOCK_LENGTH);
AscendC::PipeBarrier<PIPE_ALL>();

AscendC::DataCopy(zGm, zLocal, BLOCK_LENGTH);
AscendC::PipeBarrier<PIPE_ALL>();
```

`PipeBarrier<PIPE_ALL>()` 与 C API 的 `asc_sync()` 作用相同：前一阶段完成前，后一阶段不能读取其结果。这里的**单缓冲**，指 `xLocal`、`yLocal`、`zLocal` 各只有一块 UB 空间；每个阶段都完成后才开始下一个阶段，因此这是串行的执行流程。

#### 五、完整 kernel.asc

下面代码可以直接放入 TensorOJ 的 `kernel.asc`。它采用 C++ API 的 `GlobalTensor`、`LocalTensor` 和 `DataCopy`。

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
