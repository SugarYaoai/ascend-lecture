### 第三节 TensorOJ 实战：基于 C API 实现 Add 算子

本节对应 TensorOJ 题目：[Add Simple](https://cannjudge.cn/pku-tensor/education/add-simple/submit)。TensorOJ 与传统算法在线评测平台类似：平台给出题目、测试数据和评测框架，提交后自动判断结果是否正确；不同之处在于，评测对象不再是普通 CPU 算法程序，而是面向昇腾 NPU 的算子实现。

#### 一、题目模板与待完成内容

打开题目工程时，开发者面对的是一份尚未完成的 `kernel.asc`。评测框架已经承担了数据准备、结果校验等固定工作；开发者要补齐两个位置：定义在 NPU 上执行的 `add_custom` Kernel，以及在 `run_kernel` 中按给定接口启动它。

<pre class="dataflow"><code><span class="flow-judge">评测框架</span>：准备 x、y、z 的设备内存
        <span class="flow-arrow">|</span>
        <span class="flow-arrow">v</span>
<span class="flow-entry">run_kernel(...)</span>      &lt;- 开发者在这里配置并启动 Kernel
        <span class="flow-arrow">|</span>
        <span class="flow-arrow">v</span>
<span class="flow-kernel">add_custom(...)</span>      &lt;- 开发者定义 Device 端的 Add 计算
        <span class="flow-arrow">|</span>
        <span class="flow-arrow">v</span>
<span class="flow-result">评测框架</span>：等待、取回并校验 z</code></pre>

模板规定了 `run_kernel` 的函数签名，开发者不能修改它。写代码时，只需沿着下面两处 `TODO` 补齐逻辑：

```cpp
#include <cmath>
#include "kernel_operator.h"

// TODO 1：定义 add_custom，完成 Device 端的向量加法。

extern "C" void run_kernel(
    GM_ADDR x, const TensorGroupInfo& info_x,
    GM_ADDR y, const TensorGroupInfo& info_y,
    GM_ADDR z, const TensorGroupInfo& info_z,
    int64_t availableCoreNum, aclrtStream stream)
{
    // TODO 2：使用 <<<...>>> 启动 add_custom。
}
```

后续先完成 `add_custom` 的 Device 端实现，再回到 `run_kernel`，利用模板传入的参数完成检查与启动。入口处的各个参数会在真正使用时逐一拆解。

#### 二、`add_custom` Kernel 算子实现

##### （一）第一步：手算切片参数与 UB 容量预算

编写 `add_custom` 前，必须先算清两件事：完整向量如何在多个 Block 之间分工，以及每个 Block 的三段局部数据能否放入单个 AI Core 的 UB。对于固定长度 `172032` 的输入，本例采用如下静态执行参数：

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;
```

这组参数同时满足三项底层硬件限制：

- **多核负载均衡**：`172032` 个元素被 16 个逻辑 Block 平分，每个 Block 负责 `10752` 个元素。
- **32 字节对齐**：单段 `float32` 数据长度为 $10752 \times 4\text{ B} = 43008\text{ B} = 1344 \times 32\text{ B}$，严格满足连续数据按 `32 B` 整数倍组织的对齐要求。
- **UB 容量预算**：`xLocal`、`yLocal`、`zLocal` 三段片上缓冲区总共占用：

$$
3 \times 10752 \times 4\text{ B} = 129024\text{ B} = 126\text{ KB}
$$

标称约 `256 KB` 的 UB 被用掉约一半，为运行时资源和其他片上需求留出了安全余量。

**为什么要启动 16 个 Block？如果改成 8 个会怎样？** 假设只启动 8 个 Block 处理这 `172032` 个元素，每个 Block 分到的数据量就会翻倍，达到 `21504` 个元素。此时三段 UB 空间需要：

$$
3 \times 21504 \times 4\text{ B} = 258048\text{ B} \approx 252\text{ KB}
$$

单个 AI Core 的 UB 虽然标称 `256 KB`，但编译器生成的控制信息、运行时框架自身也要占用片上资源。用户可自由支配的空间远不到全部标称容量。分配 `252 KB` 的三段数组会让 UB 占用率达到约 `98.4%`，可能在编译期静态资源检查或运行时分配阶段直接失败。因此，将任务切分为 16 个 Block、把单核 UB 占用控制在 `126 KB`，才是当前固定规格下安全可靠的配置。

##### （二）第二步：将无类型内存转换为 float32 指针

确定切片参数与 UB 预算后，需要把模板传入的通用 GM 地址转换为可进行元素级寻址的 `float32` 指针。SIMD C API 提供了这一过程中使用的搬运、计算与同步接口：

```cpp
#include "c_api/asc_simd.h"
```

题目入口传入的 `GM_ADDR` 本质上是无类型的底层字节地址，类似 `void*`。如果直接对它做地址加减，偏移单位是 Byte；但本题需要按 `float32` 元素个数定位当前 Block 的起点。因此，先利用 `reinterpret_cast` 将其强转为 `__gm__ float*` 指针，再加上元素偏移量 `offset`：

```cpp
const uint32_t offset = block_idx * BLOCK_LENGTH;
__gm__ float* xGm = reinterpret_cast<__gm__ float*>(x) + offset;
__gm__ float* yGm = reinterpret_cast<__gm__ float*>(y) + offset;
__gm__ float* zGm = reinterpret_cast<__gm__ float*>(z) + offset;
```

这一步不产生任何硬件搬运代码，只是改变 C++ 编译器理解地址与计算偏移的方式。

##### （三）第三步：组装“搬运-计算-写回”数据流

现在将硬件初始化、内存定位、片上分配与读、算、写三阶段组装为完整的 `add_custom` Kernel：

```cpp
__vector__ __global__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    // 1. 初始化 Ascend C 硬件环境。
    asc_init();

    // 2. 根据逻辑 Block 编号，计算当前任务对应的首地址偏移。
    const uint32_t offset = block_idx * BLOCK_LENGTH;
    __gm__ float* xGm = reinterpret_cast<__gm__ float*>(x) + offset;
    __gm__ float* yGm = reinterpret_cast<__gm__ float*>(y) + offset;
    __gm__ float* zGm = reinterpret_cast<__gm__ float*>(z) + offset;

    // 3. 在片上 SRAM（UB）中开辟局部工作区。
    __ubuf__ float xLocal[BLOCK_LENGTH];
    __ubuf__ float yLocal[BLOCK_LENGTH];
    __ubuf__ float zLocal[BLOCK_LENGTH];

    // 4. 阶段一：GM -> UB。搬运长度单位是字节。
    asc_copy_gm2ub(xLocal, xGm, BLOCK_LENGTH * sizeof(float));
    asc_copy_gm2ub(yLocal, yGm, BLOCK_LENGTH * sizeof(float));
    asc_sync();

    // 5. 阶段二：Vector 在 UB 内完成加法。长度单位是元素个数。
    asc_add(zLocal, xLocal, yLocal, BLOCK_LENGTH);
    asc_sync();

    // 6. 阶段三：UB -> GM。
    asc_copy_ub2gm(zGm, zLocal, BLOCK_LENGTH * sizeof(float));
    asc_sync();
}
```

以 `block_idx = 3` 为例，当前 Block 的偏移是 $3 \times 10752 = 32256$，它执行的切片计算为：

$$
x[32256:43008] + y[32256:43008] \longrightarrow z[32256:43008]
$$

#### 三、`run_kernel` 调用接口实现

##### （一）防御性参数校验

回到 Host 侧的 `run_kernel`。模板传入的 `x`、`y`、`z` 是 Device 侧 GM 地址；`info_x`、`info_y`、`info_z` 记录输入的数量、形状与数据类型，`availableCoreNum` 表示当前可用向量核数。由于本 Kernel 针对固定长度 `float32` 向量定制，先进行参数校验，防止规格不匹配的任务误入该执行路径：

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

##### （二）Stream 挂载与 Kernel 启动

校验通过后，使用 `<<<...>>>` 语法启动 Kernel：

```cpp
add_custom<<<NUM_BLOCKS, nullptr, stream>>>(x, y, z);
```

执行参数配置如下：

| 参数 | 本题取值 | 硬件/框架含义 |
| --- | --- | --- |
| `NUM_BLOCKS` | `16` | 告诉 NPU 调度器启动 16 个逻辑 Block，并分发给 AI Core 执行。 |
| 动态 UB 参数 | `nullptr` | 不申请动态 UB 空间；本题直接使用 Kernel 内静态声明的 `__ubuf__` 数组。 |
| `stream` | 模板传入 | 将本次 Launch 任务推入特定的昇腾异步执行队列。 |

#### 四、完整交付代码：`kernel.asc`

```cpp
#include <cstdint>
#include "kernel_operator.h"
#include "c_api/asc_simd.h"

// 1. 定义物理切片参数。
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;

// 2. Device 侧向量加法 Kernel 实现。
__vector__ __global__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    asc_init();

    // 手算当前 Block 在 GM 上的首地址。
    const uint32_t offset = block_idx * BLOCK_LENGTH;
    __gm__ float* xGm = reinterpret_cast<__gm__ float*>(x) + offset;
    __gm__ float* yGm = reinterpret_cast<__gm__ float*>(y) + offset;
    __gm__ float* zGm = reinterpret_cast<__gm__ float*>(z) + offset;

    // 申请片上 UB 临时工作区。
    __ubuf__ float xLocal[BLOCK_LENGTH];
    __ubuf__ float yLocal[BLOCK_LENGTH];
    __ubuf__ float zLocal[BLOCK_LENGTH];

    // 读数据：GM -> UB，单位为 Bytes。
    asc_copy_gm2ub(xLocal, xGm, BLOCK_LENGTH * sizeof(float));
    asc_copy_gm2ub(yLocal, yGm, BLOCK_LENGTH * sizeof(float));
    asc_sync();

    // 算数据：Vector SIMD 计算，单位为 Count。
    asc_add(zLocal, xLocal, yLocal, BLOCK_LENGTH);
    asc_sync();

    // 写数据：UB -> GM，单位为 Bytes。
    asc_copy_ub2gm(zGm, zLocal, BLOCK_LENGTH * sizeof(float));
    asc_sync();
}

// 3. Host 侧 Launch 框架入口。
extern "C" void run_kernel(GM_ADDR x, const TensorGroupInfo& info_x,
                           GM_ADDR y, const TensorGroupInfo& info_y,
                           GM_ADDR z, const TensorGroupInfo& info_z,
                           int64_t availableCoreNum, aclrtStream stream)
{
    // 参数防御性校验。
    if (info_x.numTensors != 1 || info_y.numTensors != 1 ||
        info_z.numTensors != 1 || info_x.tensors[0].dtype != 0 ||
        info_y.tensors[0].dtype != 0 || info_z.tensors[0].dtype != 0 ||
        info_x.tensors[0].shape[0] != TOTAL_LENGTH ||
        info_y.tensors[0].shape[0] != TOTAL_LENGTH ||
        info_z.tensors[0].shape[0] != TOTAL_LENGTH ||
        availableCoreNum <= 0) {
        return;
    }

    // 启动 16 个 Block 异步执行 Kernel。
    add_custom<<<NUM_BLOCKS, nullptr, stream>>>(x, y, z);
}
```
