## Part 2：TensorOJ C API 实战

本节对应 TensorOJ 题目：[Add Simple](https://cannjudge.cn/pku-tensor/education/add-simple/submit)。题目提供了工程模板；提交时只需要修改 `kernel.asc`，`main.asc` 负责准备输入、调用 `run_kernel`、等待执行结束并校验输出。

本题是一个固定规格的 easy version：

| 配置项 | 取值 |
| --- | --- |
| 输入 `x`、`y` | 一维 `float32`，形状 `(172032,)` |
| 输出 `z` | 一维 `float32`，形状 `(172032,)` |
| 运算 | `z[i] = x[i] + y[i]` |
| 输入取值范围 | `[-1.0, 1.0]` |
| 可修改文件 | `kernel.asc` |

## 题目模板提供的入口

模板已经定义并调用 `run_kernel`：

```cpp
extern "C" void run_kernel(
    GM_ADDR x, const TensorGroupInfo& info_x,
    GM_ADDR y, const TensorGroupInfo& info_y,
    GM_ADDR z, const TensorGroupInfo& info_z,
    int64_t availableCoreNum, aclrtStream stream)
{
    // 在这里启动 Device 端 Kernel
}
```

`x`、`y`、`z` 是 Device 侧 Global Memory 的地址。`info_x`、`info_y`、`info_z` 描述张量的形状与数据类型；例如：

```cpp
info_x.tensors[0].shape[0]  // 输入 x 的第 0 维长度
info_x.tensors[0].dtype     // 输入 x 的数据类型，0 表示 float32
```

`availableCoreNum` 是运行时查询得到的可用向量核数。这个题的实现启动 16 个 Block，要求设备至少有一个可用向量核；Block 由运行时调度到 AI Core 执行。

## 引入 SIMD C API

模板默认包含 `kernel_operator.h`，它主要提供 Ascend C 的 C++ API。这里使用指针风格 SIMD C API，因此还需要加入：

```cpp
#include "c_api/asc_simd.h"
```

该头文件声明 `asc_init`、`asc_copy_gm2ub`、`asc_add` 和 `asc_copy_ub2gm`。没有这一行时，编译器无法识别这些 `asc_*` 函数。

## 配置 Block

输入总长度为 `172032`。启动 16 个 Block 时，每个 Block 处理：

$$
\text{BLOCK\_LENGTH} = 172032 / 16 = 10752
$$

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 10752;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;
```

每个 Block 用三段 UB 分别保存输入 `x`、输入 `y` 和输出 `z`：

$$
10752 \times 4\text{ B} \times 3 = 129024\text{ B} = 126\text{ KB}
$$

这三个缓冲区位于当前 AI Core 的 UB 中。UB 总容量虽然为 `256 KB`，但还需要为系统预留区、运行时资源等留下空间，因此这里使用 `126 KB`，而不是将静态数组配置到接近 `256 KB`。

## 实现 Device 端 Kernel

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

以 `block_idx = 3` 为例，当前 Block 的偏移是 `3 × 10752 = 32256`，因此它计算：

```text
x[32256:43008] + y[32256:43008] -> z[32256:43008]
```

`asc_copy_gm2ub` 和 `asc_copy_ub2gm` 的长度参数以字节为单位；`asc_add` 的长度参数以元素个数为单位。

## 在 run_kernel 中启动 Kernel

在启动前检查输入输出的数量、类型和长度，可以避免错误的张量规格进入固定长度的 Kernel：

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

随后使用题目模板要求的启动形式：

```cpp
add_custom<<<NUM_BLOCKS, nullptr, stream>>>(x, y, z);
```

三个启动参数的含义如下：

| 参数 | 本题取值 | 含义 |
| --- | --- | --- |
| `NUM_BLOCKS` | `16` | 启动 16 个 Block。 |
| 动态 UB 参数 | `nullptr` | 不申请动态 UB；本题使用 Kernel 内静态声明的 `__ubuf__` 数组。 |
| `stream` | 模板传入 | 将 Kernel 加入该运行时流。 |

## 完整 kernel.asc

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
