### 第四节 TensorOJ 实战：基于 C++ API 实现 Add 算子

第三节用 C API 写 Add 时，开发者亲自管理 GM 裸指针、UB 数组以及字节和元素两套长度单位。这种方式能直接看到硬件如何工作，但真实算子中很容易遇到两类问题：

- **单位混淆**：DMA 搬运接口需要 `10752 * sizeof(float)` 这样的字节数，向量加法却只需要 `10752` 个元素。漏写一次 `sizeof(float)` 往往仍能通过编译，却会让搬运范围错误，进而导致结果异常或设备侧访问失败。
- **复杂流水线难以手工维护**：直接声明 `__ubuf__` 数组适合单次搬入、计算和写回；当一个 Block 需要多个 Tile、多个 Buffer 或双缓冲时，开发者必须自己计算每块 UB 的位置、复用时机和同步状态，代码很容易失去可读性。

C++ API 的作用就是降低这两类错误的概率：`GlobalTensor<float>`、`LocalTensor<float>` 让代码始终带着元素类型和长度来描述数据；`LocalMemAllocator` 先把 UB 中的局部数据组织为明确的对象，后续可以自然升级到用 `TPipe`、`TQue` 管理缓冲区复用与入队出队。

C++ API 不会自动完成任何 GM/UB 搬运，也不会替代容量、对齐和边界检查。它只是给裸地址和 UB 数组加上更清楚的类型与长度信息，并为后续的流水线优化提供更容易维护的代码结构。

#### 一、从 C API 到 C++ API

##### （一）编译期模板参数与 GM 视图绑定

算子继续复用第三节的静态执行参数：

`LocalMemAllocator` 的 `Alloc<float, blockLength>()` 需要在编译期获知局部张量长度，才能为 UB 规划固定空间。因此 `blockLength` 不是普通的运行时变量，而是 `add_custom` 的 C++ 模板参数；Host 侧通过 `add_custom<BLOCK_LENGTH>` 实例化长度为 `10752` 的 Kernel：

<div class="api-compare">
  <div class="api-compare-side c-api">
    <span class="api-compare-label">第三节：C API</span>
    <pre><code class="language-cpp">constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH =
    NUM_BLOCKS * BLOCK_LENGTH;</code></pre>
  </div>
  <div class="api-compare-side cpp-api">
    <span class="api-compare-label">本节：C++ API</span>
    <pre><code class="language-cpp">template &lt;uint32_t blockLength&gt;
__vector__ __global__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    // blockLength 用于 UB 分配。
}</code></pre>
  </div>
</div>

题目接口传入的 `GM_ADDR` 是通用 GM 字节地址。当前 Block 的偏移却以 `float32` 元素个数计算，因此要先用 `reinterpret_cast<__gm__ float*>` 将基地址按 `float` 解释，再加上 `offset`。这样 `+ offset` 的步进单位才是 `4 B`，而不是 `1 B`。随后用 `SetGlobalBuffer` 绑定当前 Block 的 GM 地址和长度；这个操作只建立地址映射，不触发数据搬运：

<div class="api-compare">
  <div class="api-compare-side c-api">
    <span class="api-compare-label">第三节：C API</span>
    <pre><code class="language-cpp">const uint32_t offset =
    block_idx * BLOCK_LENGTH;
__gm__ float* xGm =
    reinterpret_cast&lt;__gm__ float*&gt;(x) + offset;
__gm__ float* yGm =
    reinterpret_cast&lt;__gm__ float*&gt;(y) + offset;
__gm__ float* zGm =
    reinterpret_cast&lt;__gm__ float*&gt;(z) + offset;</code></pre>
  </div>
  <div class="api-compare-side cpp-api">
    <span class="api-compare-label">本节：C++ API</span>
    <pre><code class="language-cpp">AscendC::GlobalTensor&lt;float&gt; xGm, yGm, zGm;
const uint32_t offset = block_idx * blockLength;
xGm.SetGlobalBuffer(
    reinterpret_cast&lt;__gm__ float*&gt;(x) + offset, blockLength);
yGm.SetGlobalBuffer(
    reinterpret_cast&lt;__gm__ float*&gt;(y) + offset, blockLength);
zGm.SetGlobalBuffer(
    reinterpret_cast&lt;__gm__ float*&gt;(z) + offset, blockLength);</code></pre>
  </div>
</div>

##### （二）UB 结构化分配

UB 侧不再使用三个 C 风格数组，而是通过 `LocalMemAllocator` 分配三个 `LocalTensor<float>`。这三块张量分别保存当前 Block 的 `x`、`y` 和 `z` 片段：

<div class="api-compare">
  <div class="api-compare-side c-api">
    <span class="api-compare-label">第三节：C API</span>
    <pre><code class="language-cpp">__ubuf__ float xLocal[BLOCK_LENGTH];
__ubuf__ float yLocal[BLOCK_LENGTH];
__ubuf__ float zLocal[BLOCK_LENGTH];</code></pre>
  </div>
  <div class="api-compare-side cpp-api">
    <span class="api-compare-label">本节：C++ API</span>
    <pre><code class="language-cpp">AscendC::LocalMemAllocator&lt;AscendC::Hardware::UB&gt;
    ubAllocator;
auto xLocal = ubAllocator.Alloc&lt;float, blockLength&gt;();
auto yLocal = ubAllocator.Alloc&lt;float, blockLength&gt;();
auto zLocal = ubAllocator.Alloc&lt;float, blockLength&gt;();</code></pre>
  </div>
</div>

三块 `LocalTensor` 对应三段 `10752` 元素的 UB 空间：

$$
3 \times 10752 \times 4\text{ B} = 129024\text{ B} = 126\text{ KB}
$$

这段代码直接给出了当前 AI Core 的 UB 工作区：三段 `float32` 张量，共 `126 KB`。局部资源的类型、长度和分配位置集中在同一处，后续阅读 `DataCopy` 与 `Add` 时不必再回溯数组地址和大小。

##### （三）带类型的 DataCopy 与流水线屏障控制

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

#### 二、C API 与 C++ API 的架构范式对比

| 架构维度 | C API | C++ API | 工程价值 |
| --- | --- | --- | --- |
| **GM 地址、类型与范围** | `__gm__ float*` 配合手写指针偏移 | `GlobalTensor<float>` 绑定地址、元素类型和当前 Block 长度 | 将原本分散在指针、偏移量和长度常量中的约束集中起来，减少类型不匹配和偏移范围写错的机会。 |
| **搬运长度与步进单位** | 搬运写字节数：`BLOCK_LENGTH * sizeof(float)`；计算写元素数 | `DataCopy`、`Add` 统一传入 `blockLength` 个元素 | Tensor 类型提供元素宽度，开发者无需在每次搬运时手动换算 `sizeof(float)`，避免字节数与元素数混用。 |
| **UB 资源与流水线演进** | `__ubuf__` 静态数组直接占用 UB | `LocalMemAllocator` + `LocalTensor` 描述 UB 局部资源 | 当前能集中管理 UB 对象的类型和长度；当数据流扩展为多 Tile、多 Buffer 时，可进一步过渡到 `TPipe`、`TQue` 的缓冲区复用与队列调度。 |

底层执行模型并未改变：

$$
\text{GM} \rightarrow \text{UB} \rightarrow \text{Vector} \rightarrow \text{UB} \rightarrow \text{GM}
$$

`GlobalTensor` 和 `LocalTensor` 用于绑定这条路径上的数据范围，`DataCopy` 负责实际搬运，`PipeBarrier` 负责当前 AI Core 内搬运与计算阶段的依赖同步。C++ API 提升的是代码对硬件资源、单位和生命周期的表达能力，而不是替代这些硬件步骤。

#### 三、完整交付代码解析：kernel.asc

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
