### 第三节 TensorOJ 实战：基于 C API 的 Add 算子交付

本节对应 TensorOJ 题目：[Add Simple](https://cannjudge.cn/pku-tensor/education/add-simple/submit)。TensorOJ 与传统算法在线评测平台类似：平台给出题目、测试数据和评测框架，提交后自动判断结果是否正确；不同之处在于，评测对象不再是普通 CPU 算法程序，而是面向昇腾 NPU 的算子实现。

#### 3.1 评测框架与内核交付边界

打开题目工程时，开发者面对的是一份尚未完成的 `kernel.asc`。评测框架已经承担了数据准备、结果校验等固定工作；开发者要补齐两个位置：**定义在 NPU 上执行的 `add_custom` Kernel**，以及**在 `run_kernel` 中按给定接口启动它**。

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

模板规定了 `run_kernel` 的函数签名，开发者不能修改它。写代码时，只需沿着下面两处 `TODO` 完成自己的交付：

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

后续 3.3 完成 `add_custom` 的 Device 端实现，3.4 再回到 `run_kernel`，利用模板传入的参数完成检查与启动。此时不必急着逐一理解入口的每个参数；它们会在真正需要使用时引入。

#### 3.2 静态网格划分与 UB 逻辑容量映射

这是一道固定规格的 easy version，因此无需在提交时重新搜索最优参数。直接复用前两节已经验证过的执行计划：总长度 `172032`，启动 `16` 个 Block，每个 Block 处理 `10752` 个 `float32` 元素，并在 UB 中完成一次“搬入 -> 相加 -> 搬出”。

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;
```

这样选择不是为了凑出一个常量，而是同时满足三项条件：16 个 Block 工作量均衡；每段长度为 `10752 × 4 B = 43008 B`，是 `32 B` 的整数倍；三个局部缓冲区合计占用 `126 KB`，能放入单个 AI Core 约 `256 KB` 的 UB 并留出余量。

这也解释了为什么不能随意减少 Block 数。例如仍处理 `172032` 个元素却只启动 `8` 个 Block 时，每个 Block 需要处理 `21504` 个元素，三段 UB 缓冲区将占用：

$$
3 \times 21504 \times 4\text{ B} = 258048\text{ B} = 252\text{ KB}
$$

问题不在于多个 Core 共享同一块 UB，而在于每个 Block 所在 AI Core 的独立 UB 几乎被占满，无法为运行时和其他资源留下空间。这里的 `16` 是满足当前固定规格的可行配置，而不是题目天然规定的唯一值。

#### 3.3 通用地址强转与片上 Vector 加法 Kernel 实现

上一节中直接操作 `__gm__`、`__ubuf__` 指针和 `asc_*` 函数的写法，就是指针风格 SIMD C API。模板默认包含 `kernel_operator.h`，这里还需包含 C API 头文件：

```cpp
#include "c_api/asc_simd.h"
```

该头文件声明 `asc_init`、`asc_copy_gm2ub`、`asc_add` 和 `asc_copy_ub2gm`。没有这一行时，编译器无法识别这些 `asc_*` 函数。

与上一节唯一新增的地址适配是：题目入口给出的是通用的 `GM_ADDR`，而 C API 的搬运接口需要带元素类型的 `__gm__ float*`。`reinterpret_cast` 不会搬运或修改数据，它只告诉编译器“把这段 GM 地址按 `float` 元素来解释”；随后 `+ offset` 才能按 `float` 的元素个数计算地址偏移。

Kernel 的参数仍使用 `GM_ADDR`，因为这是题目模板传入的地址类型。`GM_ADDR` 在该模板中底层是字节指针，因此先转换成 `__gm__ float*`，再按 `block_idx` 计算当前 Block 的起始位置。

```cpp
__vector__ __global__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    asc_init();

    const uint32_t offset = block_idx * BLOCK_LENGTH;
    __gm__ float* xGm = reinterpret_cast<__gm__ float*>(x) + offset;
    __gm__ float* yGm = reinterpret_cast<__gm__ float*>(y) + offset;
    __gm__ float* zGm = reinterpret_cast<__gm__ float*>(z) + offset;

    __ubuf__ float xLocal[BLOCK_LENGTH];
    __ubuf__ float yLocal[BLOCK_LENGTH];
    __ubuf__ float zLocal[BLOCK_LENGTH];

    asc_copy_gm2ub(xLocal, xGm, BLOCK_LENGTH * sizeof(float));
    asc_copy_gm2ub(yLocal, yGm, BLOCK_LENGTH * sizeof(float));
    asc_sync();

    asc_add(zLocal, xLocal, yLocal, BLOCK_LENGTH);
    asc_sync();

    asc_copy_ub2gm(zGm, zLocal, BLOCK_LENGTH * sizeof(float));
    asc_sync();
}
```

这段 Kernel 没有引入新的计算策略，而是逐项落实第二部分的计划：`block_idx` 确定当前片段，三个 `__ubuf__` 数组构成局部工作区，三阶段 API 组织该片段的搬入、向量加法与写回。

以 `block_idx = 3` 为例，当前 Block 的偏移是 `3 × 10752 = 32256`，因此它计算：

```text
x[32256:43008] + y[32256:43008] -> z[32256:43008]
```

`asc_copy_gm2ub` 和 `asc_copy_ub2gm` 的长度参数以字节为单位；`asc_add` 的长度参数以元素个数为单位。

#### 3.4 防御性参数校验与 Stream 挂载提交

回到 `run_kernel` 时，模板传入的 `x`、`y`、`z` 是 Device 侧 GM 地址；`info_x`、`info_y`、`info_z` 则记录输入输出的数量、形状和数据类型，`availableCoreNum` 表示当前可用向量核资源。由于本题 Kernel 只适用于固定长度的 `float32` 向量，先检查这些元信息，可以避免不匹配的规格进入固定执行路径：

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

检查通过后，使用模板提供的 Stream 启动 Kernel：

```cpp
add_custom<<<NUM_BLOCKS, nullptr, stream>>>(x, y, z);
```

尖括号内是执行配置，圆括号内是本次处理的 GM 地址：

| 参数 | 本题取值 | 含义 |
| --- | --- | --- |
| `NUM_BLOCKS` | `16` | 启动 16 个 Block。 |
| 动态 UB 参数 | `nullptr` | 不申请动态 UB；本题使用 Kernel 内静态声明的 `__ubuf__` 数组。 |
| `stream` | 模板传入 | 将 Kernel 加入该运行时流。 |

#### 3.5 完整交付代码解析：kernel.asc

```cpp
#include <cstdint>
#include "kernel_operator.h"
#include "c_api/asc_simd.h"

constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;

__vector__ __global__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    asc_init();

    const uint32_t offset = block_idx * BLOCK_LENGTH;
    __gm__ float* xGm = reinterpret_cast<__gm__ float*>(x) + offset;
    __gm__ float* yGm = reinterpret_cast<__gm__ float*>(y) + offset;
    __gm__ float* zGm = reinterpret_cast<__gm__ float*>(z) + offset;

    __ubuf__ float xLocal[BLOCK_LENGTH];
    __ubuf__ float yLocal[BLOCK_LENGTH];
    __ubuf__ float zLocal[BLOCK_LENGTH];

    asc_copy_gm2ub(xLocal, xGm, BLOCK_LENGTH * sizeof(float));
    asc_copy_gm2ub(yLocal, yGm, BLOCK_LENGTH * sizeof(float));
    asc_sync();

    asc_add(zLocal, xLocal, yLocal, BLOCK_LENGTH);
    asc_sync();

    asc_copy_ub2gm(zGm, zLocal, BLOCK_LENGTH * sizeof(float));
    asc_sync();
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
