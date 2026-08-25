### 第五节 TensorOJ 实战：Add Medium 的 Tile 分块

#### 一、Add Medium 题目介绍

**本节题目链接：** [TensorOJ Add Medium](https://tensoroj.cn/cann/pku-tensor/education/add-medium)

Add Medium 仍然是两个一维 `float32` 向量的逐元素加法：

$$
z_i = x_i + y_i, \quad i \in [0, 1048576)
$$

与 Add Simple 相同，开发者实现 `kernel.asc` 中的 Device 端 Kernel 和启动语句；输入 `x`、`y` 与输出 `z` 的形状均为 `(1048576,)`。计算规则没有变，变化的是单次任务的数据规模：

$$
N = 1048576
$$

本节固定启动 `16` 个 Block，因此每个 Block 负责 `65536` 个元素。若仍像 Add Simple 那样将该 Block 的输入和输出一次放入 UB，三段缓冲区将需要 `768 KB`，超过当前 AI Core 可用的片上工作空间。Add Medium 的关键不在于改写加法公式，而在于让同一段 Block 数据分批进入 UB：每次只处理一个 **Tile**，循环完成整段数据。

#### 二、片上内存瓶颈与 Tile 的必要性

##### （一）多核切分仍然无法消除 UB 容量限制

Block 切分解决的是“多个 AI Core 如何同时分工”：完整向量被划为多个互不重叠的区间，运行时把这些逻辑 Block 调度到可用的 AI Core 上。它能够缩短总任务的执行时间，却不能让单个 AI Core 的 UB 变大。

这是 NPU 算子开发中最基本的物理矛盾：GM 容量大，可以保存完整张量，但距离计算单元较远；UB 位于 AI Core 内部，访问很快，却只有几百 KB 量级的片上 SRAM。对于更长的张量，即使已经把任务分给多个 Core，单个 Block 分到的数据仍可能装不进 UB。

本题先以 `16` 个 Block 均分长度为 `1048576` 的向量：

$$
\text{BLOCK\_LENGTH} = 1048576 / 16 = 65536
$$

一个 Block 必须同时保留自己的输入 `x`、输入 `y` 和输出 `z`。若将这 `65536` 个 `float32` 元素一次性全部放入 UB，需要：

$$
3 \times 65536 \times 4\text{ B} = 768\text{ KB}
$$

> **物理约束：** 单个 AI Core 的 UB 名义容量约为 `256 KB`，而这里的三段工作区需要 `768 KB`，达到其三倍。这样的静态 UB 分配不是“性能较差”，而是无法满足片上 SRAM 的容量约束，可能在编译期资源分配或运行时直接失败。

##### （二）Block 与 Tile 构成两级切分

Tile 不是重新划分 AI Core 之间的任务，而是把一个 Block 内过长的数据段拆成多份，在同一个 AI Core 中按循环逐份处理：先将一个 Tile 从 GM 搬入 UB，在 UB 中完成加法并写回 GM，再复用同一组 UB 空间处理下一个 Tile。

因此，本例有两层不同的分工：

- **Block 切分**：空间上的任务划分。`16` 个 Block 分别负责完整向量的不同区间，使多个 AI Core 可以并行工作。
- **Tile 切分**：单个 Block 内的时序分批。一个 Block 用循环处理多个 Tile，以有限的 UB 容量吞吐自己的完整数据段。

`TILE_LENGTH` 表示一个 Tile 包含的元素数，也是一次从 GM 搬入 UB、完成加法并写回 GM 的处理粒度。本例取 `8192`：一个 `float32` Tile 的单段数据为 `8192 × 4 B = 32 KB`，既满足 `32 B` 对齐，也让 `x`、`y`、`z` 三段工作区保持在较宽裕的 UB 预算内。

因此，执行参数为：

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 65536;
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t TILE_NUM = BLOCK_LENGTH / TILE_LENGTH;  // 8
```

由 `TILE_LENGTH = 8192` 可得，每个 Tile 的 `x`、`y`、`z` 三块 `float32` 缓冲区占用：

$$
3 \times 8192 \times 4\text{ B} = 98304\text{ B} = 96\text{ KB}
$$

于是，一个 Block 用 `8` 次循环完成 `65536` 个元素；任意时刻只有一个 `96 KB` 的 Tile 工作集驻留在 UB 中。这既避开了 `768 KB` 的片上容量越界，也为后续将数据搬运与计算重叠保留了空间。

#### 三、一个 Block 怎样处理多个 Tile

`block_idx` 决定当前 Block 在完整向量中的起点，`tileIdx` 决定当前处理该 Block 内的哪一小段。一次循环迭代先定位当前 Tile，再完成该 Tile 的搬入、相加和写回：

```cpp
const uint32_t blockOffset = block_idx * BLOCK_LENGTH;

for (uint32_t tileIdx = 0; tileIdx < TILE_NUM; ++tileIdx) {
    const uint32_t tileOffset = tileIdx * TILE_LENGTH;

    // xGm、yGm、zGm 已经指向当前 Block 的 GM 起点。
    asc_copy_gm2ub(xLocal, xGm + tileOffset,
                   TILE_LENGTH * sizeof(float));
    asc_copy_gm2ub(yLocal, yGm + tileOffset,
                   TILE_LENGTH * sizeof(float));
    asc_sync();

    asc_add(zLocal, xLocal, yLocal, TILE_LENGTH);
    asc_sync();

    asc_copy_ub2gm(zGm + tileOffset, zLocal,
                   TILE_LENGTH * sizeof(float));
    asc_sync();
}
```

以 `Block 0` 为例，它的工作范围是 `[0, 65536)`；循环会依次处理：

```text
Tile 0: [0, 8192)
Tile 1: [8192, 16384)
...
Tile 7: [57344, 65536)
```

同一组 `xLocal`、`yLocal`、`zLocal` 会在下一轮循环中复用，因此 UB 的占用始终是一个 Tile 的 `96 KB`，而不是整个 Block 的 `768 KB`。

#### 四、Tile 循环中的流水线空档

Tile 分块首先解决的是容量问题：每轮只让一个 `96 KB` 工作集进入 UB，因此一个 Block 可以完成原本需要 `768 KB` 的数据段。但“能装下”不等于“已经跑满硬件”。

当前循环为了保证数据依赖，按下面的顺序处理每个 Tile：

```text
搬入 Tile 0 -> 计算 Tile 0 -> 写回 Tile 0
                         -> 搬入 Tile 1 -> 计算 Tile 1 -> 写回 Tile 1
```

每个阶段后的 `asc_sync()` 都在等待前一阶段完成：输入尚未搬完时，Vector 单元不能读取 UB；加法尚未结束时，结果也不能写回 GM。这些同步保证了正确性，但在单缓冲写法中，MTE 搬运单元和 Vector 计算单元会在不少时间段内相互等待。

AI Core 中，负责 GM 与 UB 数据搬运的 MTE 单元和负责向量加法的 Vector 单元是不同的硬件单元。更高效的安排应当让它们处理不同 Tile：例如 Vector 正在计算 Tile 0 时，MTE 同时搬入 Tile 1。这样，Tile 0 的计算时间可以覆盖 Tile 1 的部分搬运等待。

这正是下一节引入双缓冲的原因：为相邻 Tile 准备两组可轮换的 UB 工作区，并重新组织同步关系，使“搬入、计算、写回”能够在不同 Tile 上重叠推进。当前的单缓冲 Tile 循环是这个优化的正确性基线。

#### 五、完整 kernel.asc

下面的实现固定处理长度为 `1048576` 的 `float32` 向量。每个 Block 处理 `65536` 个元素，并在 Block 内循环完成 `8` 次 Tile 计算。

```cpp
#include <cstdint>
#include "kernel_operator.h"
#include "c_api/asc_simd.h"

constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 65536;
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t TILE_NUM = BLOCK_LENGTH / TILE_LENGTH;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;

__vector__ __global__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    asc_init();

    const uint32_t blockOffset = block_idx * BLOCK_LENGTH;
    __gm__ float* xGm = reinterpret_cast<__gm__ float*>(x) + blockOffset;
    __gm__ float* yGm = reinterpret_cast<__gm__ float*>(y) + blockOffset;
    __gm__ float* zGm = reinterpret_cast<__gm__ float*>(z) + blockOffset;

    __ubuf__ float xLocal[TILE_LENGTH];
    __ubuf__ float yLocal[TILE_LENGTH];
    __ubuf__ float zLocal[TILE_LENGTH];

    for (uint32_t tileIdx = 0; tileIdx < TILE_NUM; ++tileIdx) {
        const uint32_t tileOffset = tileIdx * TILE_LENGTH;

        asc_copy_gm2ub(xLocal, xGm + tileOffset, TILE_LENGTH * sizeof(float));
        asc_copy_gm2ub(yLocal, yGm + tileOffset, TILE_LENGTH * sizeof(float));
        asc_sync();

        asc_add(zLocal, xLocal, yLocal, TILE_LENGTH);
        asc_sync();

        asc_copy_ub2gm(zGm + tileOffset, zLocal, TILE_LENGTH * sizeof(float));
        asc_sync();
    }
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
