### 第三节 TensorOJ 实战：基于 C API 实现 Add 算子

本节对应 TensorOJ 题目：[Add Simple](https://cannjudge.cn/pku-tensor/education/add-simple/submit)。TensorOJ 与传统算法在线评测平台类似：平台给出题目、测试数据和评测框架，提交后自动判断结果是否正确；不同之处在于，评测对象不再是普通 CPU 算法程序，而是面向昇腾 NPU 的算子实现。

#### 3.1 题目模板与开发者交付内容

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

#### 3.2 `add_custom` Kernel 算子实现

##### 3.2.1 物理边界约束与多核切分推导

编写 `add_custom` 前，必须先同时确定两件事：完整向量如何在多个 Block 之间分工，以及每个 Block 的三段局部数据能否放入单个 AI Core 的 UB。对于固定长度 `172032` 的输入，本例采用如下静态执行参数：

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;
```

这组参数不是任意常量，而是同时满足三项硬件约束：

- **多核负载均衡**：`172032` 个元素被 16 个逻辑 Block 均分，每个 Block 处理 `10752` 个元素。
- **32B 数据粒度**：单段 `float32` 数据长度为 `10752 × 4 B = 43008 B = 1344 × 32 B`，满足连续数据按 `32 B` 整数倍组织的要求。
- **UB 资源预算**：`xLocal`、`yLocal`、`zLocal` 三段缓冲区共占用：

$$
3 \times 10752 \times 4\text{ B} = 129024\text{ B} = 126\text{ KB}
$$

这为单个 AI Core 标称约 `256 KB` 的 UB 留出了资源余量。

**为什么不能缩减为 8 个 Block？** 若仍处理 `172032` 个元素却只启动 8 个 Block，每个 Block 需要处理 `21504` 个元素，三段 UB 缓冲区将占用：

$$
3 \times 21504 \times 4\text{ B} = 258048\text{ B} = 252\text{ KB}
$$

这不是多个 AI Core 共享 UB 的问题，而是单个 Block 的三段静态 UB 数组已经占到标称容量的约 `98.4%`。标称 `256 KB` 并不等于全部都可供用户数组使用：编译器生成的控制信息、运行时资源及其他片上需求也要占用空间。`252 KB` 会超出这份 Kernel 可用的 UB 资源预算，可能在编译期静态资源检查或运行时分配阶段失败。因此，16 个 Block 是当前固定规格下的安全配置，而不是题目天然规定的唯一值。

##### 3.2.2 C API 接口约定与通用指针转换

确定网格与 UB 资源后，需要把模板传入的通用 GM 地址转换为可进行元素级寻址的 `float32` 指针。SIMD C API 提供这一过程中使用的搬运、计算与同步接口：

```cpp
#include "c_api/asc_simd.h"
```

题目入口的 `GM_ADDR` 是通用字节地址。若直接对它做地址运算，偏移单位是字节；而本题需要按 `float32` 元素定位当前 Block 的起点。因此先用 `reinterpret_cast` 将地址解释为 `__gm__ float*`，再加上元素数 `offset`。这个转换不搬运、不修改数据，只改变编译器理解地址和计算偏移的方式：

```cpp
const uint32_t offset = block_idx * BLOCK_LENGTH;
__gm__ float* xGm = reinterpret_cast<__gm__ float*>(x) + offset;
__gm__ float* yGm = reinterpret_cast<__gm__ float*>(y) + offset;
__gm__ float* zGm = reinterpret_cast<__gm__ float*>(z) + offset;
```

##### 3.2.3 片上数据流驱动的 Kernel 组装

现在将三个阶段组装为完整的 `add_custom`：先定位当前 Block 的 GM 区间，在 UB 中建立局部工作区，再按“GM -> UB -> Vector -> UB -> GM”的数据依赖完成计算。

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

这段 Kernel 中，`block_idx` 确定当前片段，三个 `__ubuf__` 数组构成局部工作区，三阶段 API 组织该片段的搬入、向量加法与写回。两个搬运接口的长度单位为字节，因此传入 `BLOCK_LENGTH * sizeof(float)`；`asc_add` 的长度单位为元素个数，因此直接传入 `BLOCK_LENGTH`。

以 `block_idx = 3` 为例，当前 Block 的偏移是 `3 × 10752 = 32256`，因此它计算：

```text
x[32256:43008] + y[32256:43008] -> z[32256:43008]
```

#### 3.3 `run_kernel` 调用接口实现

##### 3.3.1 防御性参数校验

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

##### 3.3.2 Stream 挂载与 Kernel 启动

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

#### 3.4 完整交付代码解析：kernel.asc

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
