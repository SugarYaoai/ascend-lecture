### 第五节 TensorOJ 实战：Add Medium 的 Tile 分块

#### 一、问题引入：更大数据量带来的片上内存挑战

**本节题目链接：** [TensorOJ Add Medium](https://tensoroj.cn/cann/pku-tensor/education/add-medium)

Add Medium 依然是两个一维 `float32` 向量的逐元素加法：

$$
z_i = x_i + y_i, \quad i \in [0, 1048576)
$$

计算公式没有变，但单次任务的数据规模扩大到 $N = 1048576$，即百万级元素。本题固定启动 `16` 个 Block，因此每个 Block 需要处理 `65536` 个元素。核心矛盾在于：单个 Block 的输入和输出已经无法一次性放入片上 UB。

#### 二、片上内存瓶颈与 Tile 切分的必要性

##### （一）为什么仅靠 Block 切分还不够

Block 切分解决的是多核分工：把长向量切成多段，再由多个 AI Core 并行处理。它能缩短整体运行时间，却无法改变单个 AI Core 内部 UB 的容量上限。

本题将长度为 `1048576` 的向量均分给 `16` 个 Block：

$$
\text{BLOCK\_LENGTH} = 1048576 / 16 = 65536
$$

一个 Block 计算时需要同时保留输入 `x`、输入 `y` 与输出 `z`。若将 `65536` 个 `float32` 元素一次性全部装入 UB，所需空间为：

$$
3 \times 65536 \times 4\text{ B} = 768\text{ KB}
$$

> **物理约束：** 单个 AI Core 的 UB 名义容量约为 `256 KB`，而全量载入需要 `768 KB`，达到容量上限的三倍。这不是性能较差，而是片上 SRAM 容量不足；静态分配可能在编译阶段或运行初期直接失败。

##### （二）Block 与 Tile 构成两级切分

Tile 并不重新划分 AI Core 之间的分工，而是在同一个 AI Core 内部把过长的数据段拆成多份，通过循环分批处理：

- **Block 切分**：空间上的任务划分。`16` 个 Block 分管不同的 GM 数据区间，实现多核并行。
- **Tile 切分**：单核内部的分批处理。一个 Block 用 `for` 循环复用同一组 UB 空间，逐批完成自己的数据。

本例选择 `TILE_LENGTH = 8192`。一个 Tile 的 `x`、`y`、`z` 三块 `float32` 缓冲区占用：

$$
3 \times 8192 \times 4\text{ B} = 96\text{ KB}
$$

这组参数既满足 `32 B` 对齐，也为 UB 留出了余量。完整配置如下：

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 65536;
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t TILE_NUM = BLOCK_LENGTH / TILE_LENGTH;
// 每个 Block 循环 8 次。
```

通过 `8` 次 Tile 循环，一个 Block 完成自己的 `65536` 个元素；任意时刻只有一个 `96 KB` 的 Tile 工作集驻留在 UB 中，而不是不可容纳的 `768 KB`。

#### 三、单个 AI Core 如何循环处理 Tile

在 C++ API 中，`block_idx` 先定位当前 Block 的 GM 数据范围；随后 `tileOffset` 在每轮循环中推进，三块 `LocalTensor` 则持续复用同一份 UB 空间：

```cpp
template <uint32_t blockLength, uint32_t tileLength>
__vector__ __global__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    AscendC::InitSocState();

    // 1. 绑定当前 Block 的 GM 视图。
    const uint32_t blockOffset = block_idx * blockLength;
    AscendC::GlobalTensor<float> xGm, yGm, zGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + blockOffset, blockLength);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + blockOffset, blockLength);
    zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + blockOffset, blockLength);

    // 2. 只分配单个 Tile 尺寸的 UB 工作区，共 96 KB。
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    auto xLocal = ubAllocator.Alloc<float, tileLength>();
    auto yLocal = ubAllocator.Alloc<float, tileLength>();
    auto zLocal = ubAllocator.Alloc<float, tileLength>();

    // 3. 依次处理当前 Block 内的全部 Tile。
    constexpr uint32_t tileNum = blockLength / tileLength;
    for (uint32_t tileIdx = 0; tileIdx < tileNum; ++tileIdx) {
        const uint32_t tileOffset = tileIdx * tileLength;

        // 搬入当前 Tile。
        AscendC::DataCopy(xLocal, xGm[tileOffset], tileLength);
        AscendC::DataCopy(yLocal, yGm[tileOffset], tileLength);
        AscendC::PipeBarrier<PIPE_ALL>();

        // 在 UB 中执行向量加法。
        AscendC::Add(zLocal, xLocal, yLocal, tileLength);
        AscendC::PipeBarrier<PIPE_ALL>();

        // 写回当前 Tile 的结果。
        AscendC::DataCopy(zGm[tileOffset], zLocal, tileLength);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
```

以 Block `0` 为例，它负责区间 `[0, 65536)`；循环会依次处理 `Tile 0: [0, 8192)`、`Tile 1: [8192, 16384)` 直到 `Tile 7: [57344, 65536)`。每轮都经历同一条数据路径：GM 搬入 UB，在 UB 中相加，再将结果写回 GM。

#### 四、串行 Tile 循环中的硬件等待

##### （一）搬运与计算串行：Vector 单元与 MTE 引擎互等

Tile 切分解决了 UB 装不下的问题，但当前串行循环还没有充分利用硬件。在 AI Core 中，MTE 搬运引擎与 Vector 计算单元是相互独立的硬件；而上面的循环在每个阶段后都加入屏障，使操作必须按顺序推进：

$$
\text{Tile 0: [MTE2 搬入]} \longrightarrow \text{[Vector 计算]} \longrightarrow \text{[MTE3 写回]} \longrightarrow \text{Tile 1: [MTE2 搬入]} \dots
$$

这会造成两类等待：

- **Vector 单元等待搬运**：MTE2 从 GM 搬入 Tile `0` 时，Vector 计算单元没有可计算的数据。
- **MTE 引擎等待计算**：Vector 单元计算加法时，MTE 无法预读下一个 Tile。

##### （二）根因：单缓冲区引发数据竞争

当前只开辟了一套 UB 缓冲区：`xLocal`、`yLocal` 与 `zLocal`。如果让 MTE 在 Vector 计算 Tile `0` 的同时预读 Tile `1`，新旧 Tile 会写入同一块片上地址，造成数据覆盖与脏读。

因此，下一节会引入双缓冲（Double Buffering）：为相邻 Tile 准备两套可轮换的 UB 工作区，让 Tile $i$ 的计算与 Tile $i+1$ 的搬运能够安全地重叠推进。
