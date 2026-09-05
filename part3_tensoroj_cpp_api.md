### 1.4 TensorOJ 实战：基于 C++ API 实现 Add 算子

第三节用 C API 写 Add 时，开发者需要亲自管理 GM 裸指针、UB 静态数组，还要时刻切换“字节数”与“元素个数”两套单位。这种底层写法能帮助我们理解硬件原理，但在实际算子开发中极易引发两类问题：

- **单位混淆的隐蔽 Bug**：DMA 搬运接口需要传入 `10752 * sizeof(float)`，而向量加法只需要传入 `10752`。一旦漏写 `sizeof(float)`，编译器不会报错，却会导致数据搬运截断或越界访存。
- **复杂流水线难以手工维护**：直接声明 `__ubuf__` 数组只适合简单的“单次搬入、计算、写回”。当后续面临 Tile 切分、多 Buffer 循环复用或流水线掩盖时，手工计算每块 UB 的偏移量与生命周期极易出错，代码可读性也会急剧下降。

C++ API 的核心使命就是利用类型系统解决这些痛点。它引入 `GlobalTensor<float>` 与 `LocalTensor<float>`，让数据携带确切的类型与长度信息；同时通过 `LocalMemAllocator` 结构化地管理 UB 片上空间，为后续升级到 `TPipe`、`TQue` 队列化流水线奠定基础。

C++ API 并没有改变底层的物理搬运规则。它只是在裸地址与硬件 SRAM 之上套了一层强类型外衣，在编译时帮助开发者拦截大部分低级错误。

#### 1.4.1 从 C API 到 C++ API 的重构三步法

##### 1.4.1.1 编译时模板参数与 GM 视图绑定

算子继续复用第三节的静态执行参数：

`LocalMemAllocator` 的 `Alloc<float, blockLength>()` 需要在编译时获知局部张量长度，才能为 UB 规划固定空间。因此 `blockLength` 不是普通的运行时变量，而是 `add_custom` 的 C++ 模板参数；Host 侧通过 `add_custom<BLOCK_LENGTH>` 实例化长度为 `10752` 的 Kernel：

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

题目接口传入的 `GM_ADDR` 是通用 GM 字节地址。当前 Block 的偏移却以 `float32` 元素个数计算，因此要先用 `reinterpret_cast<__gm__ float*>` 将基地址按 `float` 解释，再加上 `offset`。这样 `+ offset` 的步长才是 `4 B`，而不是 `1 B`。随后用 `SetGlobalBuffer` 绑定当前 Block 的 GM 地址和长度；这个操作只建立地址映射，不触发数据搬运：

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

##### 1.4.1.2 UB 结构化分配

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

##### 1.4.1.3 强类型 DataCopy 与流水线屏障

GM 视图与 UB 工作区建立后，数据流仍然遵循“搬入 -> 计算 -> 写回”三个阶段；改变的是长度的表达方式。C API 需要为搬运接口手动换算字节数，C++ API 则由 Tensor 的 `float` 类型推导单元素宽度：

<div class="api-compare">
  <div class="api-compare-side c-api">
    <span class="api-compare-label">第三节：C API</span>
    <pre><code class="language-cpp">asc_copy_gm2ub(xLocal, xGm,
    BLOCK_LENGTH * sizeof(float));
asc_copy_gm2ub(yLocal, yGm,
    BLOCK_LENGTH * sizeof(float));
asc_sync();

asc_add(zLocal, xLocal, yLocal, BLOCK_LENGTH);
asc_sync();

asc_copy_ub2gm(zGm, zLocal,
    BLOCK_LENGTH * sizeof(float));
asc_sync();</code></pre>
  </div>
  <div class="api-compare-side cpp-api">
    <span class="api-compare-label">本节：C++ API</span>
    <pre><code class="language-cpp">AscendC::DataCopy(xLocal, xGm, blockLength);
AscendC::DataCopy(yLocal, yGm, blockLength);
AscendC::PipeBarrier&lt;PIPE_ALL&gt;();

AscendC::Add(zLocal, xLocal, yLocal, blockLength);
AscendC::PipeBarrier&lt;PIPE_ALL&gt;();

AscendC::DataCopy(zGm, zLocal, blockLength);
AscendC::PipeBarrier&lt;PIPE_ALL&gt;();</code></pre>
  </div>
</div>

`DataCopy` 的源和目的都已是 `float` Tensor，因此 `blockLength` 直接表示元素个数，不必手写 `sizeof(float)`。`PipeBarrier<PIPE_ALL>()` 是当前 Block、当前 AI Core 内的流水线屏障：它保证输入写入 UB 后再被 Vector 单元读取，也保证结果生成后再写回 GM。

#### 1.4.2 C API 与 C++ API 的代码组织差异

| 架构维度 | C API | C++ API | 工程价值 |
| --- | --- | --- | --- |
| **GM 地址、类型与范围** | `__gm__ float*` 配合手写指针偏移 | `GlobalTensor<float>` 绑定地址、元素类型和当前 Block 长度 | 将原本分散在指针、偏移量和长度常量中的约束集中起来，减少类型不匹配和偏移范围写错的机会。 |
| **长度参数的计量单位** | 搬运写字节数：`BLOCK_LENGTH * sizeof(float)`；计算写元素数 | `DataCopy`、`Add` 统一传入 `blockLength` 个元素 | Tensor 类型提供元素宽度，开发者无需在每次搬运时手动换算 `sizeof(float)`，避免字节数与元素数混用。 |
| **UB 资源与流水线演进** | `__ubuf__` 静态数组直接占用 UB | `LocalMemAllocator` + `LocalTensor` 描述 UB 局部资源 | 当前能集中管理 UB 对象的类型和长度；当数据流扩展为多 Tile、多 Buffer 时，可进一步过渡到 `TPipe`、`TQue` 的缓冲区复用与队列调度。 |

底层执行模型并未改变：

$$
\text{GM} \rightarrow \text{UB} \rightarrow \text{Vector} \rightarrow \text{UB} \rightarrow \text{GM}
$$

`GlobalTensor` 和 `LocalTensor` 用于绑定这条路径上的数据范围，`DataCopy` 负责实际搬运，`PipeBarrier` 负责当前 AI Core 内搬运与计算阶段的依赖同步。C++ API 提升的是代码对硬件资源、单位和生命周期的表达能力，而不是替代这些硬件步骤。

#### 1.4.3 完整交付代码解析：kernel.asc

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
