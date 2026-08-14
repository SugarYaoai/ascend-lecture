## Part 4：Add Medium：并行度与 Tile 分块

Easy Version 中，`16` 个 Block 各处理 `10752` 个元素；每个 Block 的 `x`、`y`、`z` 三段 UB 缓冲区合计为 `126 KB`，可以一次搬运、一次计算、一次写回。

Medium Version 将向量长度扩大为 `N = 1048576`。

数据变大以后，需要回答两个不同的问题：

1. 应该启动多少个 Block，才能让更多 AI Core 参与计算？
2. 一个 Block 分到的数据超过 UB 容量时，应该如何完成计算？

第一个问题由 **Block 划分** 回答；第二个问题由 **Tile 分块** 回答。二者相关，但不能互相替代。

### 实验一：改变 Block 数，观察并行度

固定总元素数 `N = 1048576`，每个 Block 的工作量随 Block 数改变：

| Block 数 | 每 Block 元素数 | 三段 float32 缓冲区 | 单次放入 UB 的可能性 |
| ---: | ---: | ---: | --- |
| 16 | 65536 | 768 KB | 不可行，必须分块 |
| 32 | 32768 | 384 KB | 不可行，必须分块 |
| 64 | 16384 | 192 KB | 可能可行，但 UB 余量较小 |
| 128 | 8192 | 96 KB | 可以单次处理 |

这里的 UB 用量只计算 `x`、`y`、`z` 三个 `float32` 临时张量：

$$
\text{UB bytes} = 3 \times \text{blockLength} \times 4
$$

因此，对于只有一次逐元素加法的 Add，`64 Block` 或 `128 Block` 很可能比 `16 Block` 加大 Tile 更适合作为性能基线：更多 Block 可被运行时调度到更多 AI Core 上，且每个 Block 的数据量更小。

但是，`Block` 数也不是越多越好。物理 AI Core 数由具体设备决定，启动的 Block 超过可同时执行的 AI Core 数后，其余 Block 会分批调度。实际提交时可从运行环境提供的 `availableCoreNum` 获得可用 Core 数，并通过 profiling 比较不同 `blockNum` 的耗时。

### Block 和 Tile 的职责

```text
Block：把整个输出向量分给不同的执行任务，决定任务级并行度。
Tile ：把一个 Block 内的数据分批放入 UB，解决片上存储容量限制。
```

例如，`64 Block × 1 Tile` 与 `16 Block × 4 Tile` 都能覆盖 `1048576` 个元素，但它们不是同一种划分：前者主要增加任务并行度，后者主要管理单个任务的片上内存。

### 实验二：固定 16 Block，必须使用 Tile

为了学习 Tile，将 Block 数固定为 `16`。每个 Block 要处理 `1048576 / 16 = 65536` 个元素。

若把这 `65536` 个元素的 `x`、`y`、`z` 一次放进 UB，需要 `768 KB`，超过可用 UB。因此将每个 Block 内的数据继续划分为 `8` 个 Tile：

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 65536;
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t TILE_NUM = BLOCK_LENGTH / TILE_LENGTH;  // 8
```

一个 Tile 的三个临时张量占用 `3 × 8192 × 4 B = 98304 B = 96 KB`。

每个 Block 的计算过程变为：

```text
Block 0:
  Tile 0: GM[0, 8192) -> UB -> Add -> GM[0, 8192)
  Tile 1: GM[8192, 16384) -> UB -> Add -> GM[8192, 16384)
  ...
  Tile 7: GM[57344, 65536) -> UB -> Add -> GM[57344, 65536)

Block 1:
  从全局下标 65536 开始，重复相同的 8 次 Tile 处理
```

### Tile 的全局偏移

当前 Block 的起点为：

```cpp
uint32_t blockOffset = block_idx * BLOCK_LENGTH;
```

第 `tileIdx` 个 Tile 相对于当前 Block 的偏移为：

```cpp
uint32_t tileOffset = tileIdx * TILE_LENGTH;
```

因此，一个 Tile 的 GM 地址就是 `GM 起始地址 + blockOffset + tileOffset`。`xGm`、`yGm`、`zGm` 已经偏移到当前 Block 的起点后，循环内只需额外加上 `tileOffset`。

### C API 的 Tile 化实现

Part 4 使用 C API，避免提前引入 `TPipe`、`TQue` 与队列生命周期。每个 Block 只声明三块固定大小的 UB 数组：`xLocal`、`yLocal`、`zLocal`。

```cpp
__ubuf__ float xLocal[TILE_LENGTH];
__ubuf__ float yLocal[TILE_LENGTH];
__ubuf__ float zLocal[TILE_LENGTH];
```

这三块数组合计占用 `96 KB`。Tile 循环中的每一轮都严格按下面的串行顺序执行：

```text
CopyIn:  GM -> UB，搬入当前 Tile 的 x、y
Compute: UB 内执行向量 Add
CopyOut: UB -> GM，写回当前 Tile 的 z
```

```cpp
for (uint32_t tileIdx = 0; tileIdx < TILE_NUM; ++tileIdx) {
    uint32_t tileOffset = tileIdx * TILE_LENGTH;

    asc_copy_gm2ub(xLocal, xGm + tileOffset, TILE_LENGTH * sizeof(float));
    asc_copy_gm2ub(yLocal, yGm + tileOffset, TILE_LENGTH * sizeof(float));
    asc_sync();

    asc_add(zLocal, xLocal, yLocal, TILE_LENGTH);
    asc_sync();

    asc_copy_ub2gm(zGm + tileOffset, zLocal, TILE_LENGTH * sizeof(float));
    asc_sync();
}
```

`asc_sync` 保证前一阶段完成后，下一阶段才能读取对应的数据。这是最直观的 Tile 版本：同一时刻只处理一个 Tile，没有任何数据搬运与计算的重叠。

### 完整 kernel.asc

下面是 Add Medium 的完整 C API 版本。它固定处理长度为 `1048576` 的 `float32` 向量；每个 Block 处理 `65536` 个元素，并在 Block 内循环处理 `8` 个 Tile。

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

### 本节结论

- Add 的 Block 数应先作为性能参数探索，不能机械固定为 16。
- 对单次 Add，增加 Block 数可以降低每个 Block 的 UB 占用，并提高并行机会。
- 当固定 Block 数后，每个 Block 的工作数据仍超过 UB 时，Tile 是必需的。
- 本节的 Tile 按 `CopyIn -> Compute -> CopyOut` 串行完成；下一节再使用 `TPipe` 与 `TQue` 建立双缓冲流水线。
