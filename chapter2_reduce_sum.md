## 2 从 Reduce Sum 开始理解归约算子

### 2.1 归约计算的物理本质：树状归约与片上数据折叠

上一章围绕逐元素算子（如 Add）构建了 AIV（Vector Core）上的数据流水线。逐元素算子的数据流是点对点的：输入多少个元素，输出就有多少个元素，每个计算通道互不干扰。

深度学习中还存在另一类完全不同的算子：归约（Reduce）算子，如 Softmax、LayerNorm、Loss 计算等。Reduce Sum 将一个长向量收敛为一个标量，是典型的数据折叠：

$$
y = \sum_{i=0}^{N-1} x_i
$$

在传统 CPU 上，归约通常被直觉地实现为串行的 `for` 循环，即用一个标量累加器逐个吞入数据。但在昇腾 AI Core 这种专用向量加速芯片上，串行处理会造成巨大的计算单元浪费。AI Core 的设计核心是高吞吐的矢量并行，其硬件单次指令执行、寄存器读写以及片上内存搬运的基础物理粒度均为 `256 B`。

为了在矢量架构下高效完成数据归约，算子的物理设计必须从“串行累加”转向“片上向量折叠”。

#### 2.1.1 单 Core 片上：树状规约与 256 B 物理粒度

在昇腾 AI Core 中，Vector 单元单次指令的操作空间为 `256 B`。对于 `float32` 类型（每个元素占 `4 B`），这 `256 B` 物理空间恰好可以装入 `64` 个元素。

为了高效处理这 `256 B` 数据，Vector 单元不会像 CPU 那样逐个串行累加，而是采用一种**并行树状折叠算法（Tree Reduction）**。

当 `64` 个 `float32` 元素装满单个 `256 B` 向量寄存器后，硬件会在 `6` 轮（$\log_2 64 = 6$）内对这个寄存器进行高低半区对折相加，把数据迅速收敛到 `index 0` 位置：

- **初始状态**：`256 B` 向量寄存器装入 `64` 个 FP32 元素，即 `a0` 到 `a63`。
- **第一轮（Stride = 32）**：寄存器后半段 `a32` 到 `a63` 与前半段 `a0` 到 `a31` 并行相加，`32` 个通道并发计算，生成 `32` 个局部和 `S0` 到 `S31`，其中 `S0 = a0 + a32`。
- **第二轮（Stride = 16）**：剩下的后半段 `S16` 到 `S31` 与前半段 `S0` 到 `S15` 相加，`16` 个通道并发计算，生成 `16` 个局部和 `T0` 到 `T15`，其中 `T0 = S0 + S16`。
- **第三轮（Stride = 8）**：高 `8` 个元素与低 `8` 个元素并行相加，收敛为 `8` 个元素 `U0` 到 `U7`。
- **第四轮（Stride = 4）**：高 `4` 个元素与低 `4` 个元素并行相加，收敛为 `4` 个元素 `V0` 到 `V3`。
- **第五轮（Stride = 2）**：高 `2` 个元素与低 `2` 个元素并行相加，收敛为 `2` 个元素 `W0` 和 `W1`。
- **第六轮（Stride = 1）**：最后两个元素相加，生成最终的累加和 `Sum`，精确存放在 `index 0` 位置。

![64 个 FP32 的树状规约过程](assets/reduce-sum/fp32-tree-reduction.png)

*图 2-1：64 个 `float32` 元素在一个 256 B Block 内进行六轮树状折叠，最终的 Sum 位于 `index 0`。*

通过这种按位置对齐的向量树状折叠，计算时间复杂度由传统串行累加的 $O(N)$ 降至 $O(\log_2 N)$。

#### 2.1.2 Reduce Sum 的三阶梯题目设计

在掌握单个 AI Core 内部 `256 B` 物理粒度的向量折叠原理后，如果直接编写大规模数据的归约算子，往往会被片上缓存限制、Tile 循环偏移、多核并发竞争以及原子操作等交织在一起的工程细节所干扰。

为了清晰地解耦算子设计的核心矛盾，本章将归约算子的学习路径重构为**三阶梯递进题目设计**。整个物理计算链路被拆解为“单个 Tile 片上折叠 $\rightarrow$ 单 Core 内多 Tile 空间突破 $\rightarrow$ 多 Core 间算力拓展”三个维度。通过循序渐进的关卡设计，读者可以在隔离难点的同时，逐步建立从微观寄存器指令到宏观全芯片协同的算子设计全景图。

| 关卡 | 数据规模 | 架构特征 | 核心设计目标 | 隔离的工程难点 |
| --- | --- | --- | --- | --- |
| **Easy 版本** | `32 KB`<br>`8192` 个 FP32 | 单核、单个 Tile<br>一次性装入 UB | **聚焦片上规约链路**：掌握数据从 $N$ 到 $1$ 的折叠过程、`256 B` 对齐与 `tmpBuffer` 临时空间分配 | 暂不引入 Tile 循环与多核并发 |
| **Medium 版本** | `2 MB`<br>`524288` 个 FP32 | 单核、多 Tile 循环<br>突破 UB 容量限制 | **聚焦单核片上累加**：掌握单核 Tiling 策略、Tile 偏移、`sum_buffer` 临时存储及各 Tile 局部和累加 | 中间结果全在单 Core 内管理，不引入跨 Core 汇总 |
| **Hard 版本** | `64 MB`<br>`16777216` 个 FP32 | `32` 个 AI Core 并行<br>突破单核算力瓶颈 | **聚焦多核 Atomic 写**：掌握基于 `block_idx` 的跨核数据切分、多核竞争与硬件 Atomic Add 使用 | 比较 Atomic 规约与 Two-Stage 规约（WorkSpace 缓存二次归约）的性能差异 |

##### 2.1.2.1 Easy 版本：单核单 Tile

- **题目数据**：`8192` 个 `float32` 元素，共 `32 KB`；输入 `x` 的形状为 `(8192,)`，每个元素范围为 $[-1.0, 1.0]$；输出 `y` 的形状为 `(1,)`，范围为 $[-8192, 8192]$。
- **任务要求**：使用单个 AI Core 处理完整输入，将 `x` 搬入 UB，在片上执行 `WholeReduceSum` 或 `BlockReduceSum`，再将标量结果写回 GM。
- **学习重点**：数据从 $N$ 到 $1$ 的折叠过程、`256 B` 对齐与 `tmpBuffer` 临时空间分配。
- **难点隔离**：暂不引入 Tile 循环与多核并发。

##### 2.1.2.2 Medium 版本：单核多 Tile

- **题目数据**：$2^6 \times 8192 = 524288$ 个 `float32` 元素，共 `2 MB`；输入 `x` 的形状为 `(524288,)`，输出 `y` 的形状为 `(1,)`。
- **任务要求**：仍由单个 AI Core 完成归约。输入必须拆分为多个 Tile，通过循环搬运、片上规约与局部累加得到 Local Sum。
- **学习重点**：单核 Tiling 策略、Tile 偏移、`sum_buffer` 临时存储，以及持续累加各 Tile 局部和的过程。
- **难点隔离**：所有中间结果都在同一个 Core 内管理，不引入跨 Core 汇总。

##### 2.1.2.3 Hard 版本：多 Core 协同与跨核归约

- **题目数据**：$32 \times 2\text{ MB} = 64\text{ MB}$，即 `16777216` 个 `float32` 元素；输入 `x` 的形状为 `(16777216,)`，输出 `y` 的形状为 `(1,)`。
- **任务要求**：将输入均分给 `32` 个 AI Core。每个 Core 先计算 `core_local_sum`，再通过 `SetAtomicAdd` 将局部和安全地累加到同一个 GM 输出地址。
- **学习重点**：基于 `block_idx` 的跨核数据切分、多核写同一输出地址时的竞争问题，以及硬件 Atomic Add 的使用。
- **延伸问题**：比较 Atomic 规约与 Two-Stage 规约。后者通过 WorkSpace 保存各 Core 局部和，再进行第二次归约，以降低原子写带来的开销。

### 2.2 Easy 关卡：单核单 Tile 的片上归约实现

本节聚焦最基础的归约场景：数据量恰好可以一次性装入单个 AI Core 的 Unified Buffer（UB）中。通过这个单核单 Tile 的例子，将彻底理清数据从 $N$ 折叠到 $1$ 的完整链路、`256 B` 物理对齐约束、Vector 单元写回机制，以及临时空间 `tmpBuffer` 的分配本质。

#### 2.2.1 题目定义与内存规划

##### 2.2.1.1 题目数据规格

- **输入数据 `x`**：`8192` 个 `float32` 元素，数据大小为 $8192 \times 4\text{ B} = 32\text{ KB}$；每个元素范围为 $[-1.0, 1.0]$。
- **输出数据 `y`**：`1` 个 `float32` 标量，数学数据大小为 `4 B`；物理写回与 DMA 搬运时需要按 `32 B` 对齐补齐。输出范围为 $[-8192, 8192]$。
- **物理约束**：输入总大小为 `32 KB`，远小于 AI Core 常见的 `256 KB` 级 UB 容量，因此无需开启 Tile 循环。`32 KB = 128 \times 256\text{ B}`，天然满足 `256 B` 向量对齐要求。

##### 2.2.1.2 UB 内存布局规划

单个 AI Core 的 UB 需要为三块区域分配空间：

| 缓冲区名称 | 元素数量 | 字节大小 | 物理用途 |
| --- | --- | --- | --- |
| `inQueueX` | `8192` 个 FP32 | `32768 B`（`32 KB`） | 存放从 GM 搬入的原始输入向量 `x`。 |
| `outQueueY` | `8` 个 FP32，含 `7` 个 Padding | 最少 `32 B` | 存放最终归约标量；有效结果位于 `yLocal[0]`。 |
| `tmpBuffer` | 动态查询 | 通常 `256 ~ 512 B` | 存放 API 在树状折叠阶段进行向量转置、混洗所需的临时数据。 |

#### 2.2.2 片上全量归约 API：`WholeReduceSum`

在 Ascend C 高阶 API 中，完成片上全量归约的核心工具是 `WholeReduceSum`。

##### 2.2.2.1 树状折叠与 `tmpBuffer` 的物理本质

上一节提到，Vector 单元在 `256 B`、即 `64` 个 FP32 元素的寄存器粒度上进行横向树状折叠。当处理的数据达到 `8192` 个元素，即 `128` 个 `256 B` 向量时，真实计算路径分为两级：

- **纵向向量加法（Vector Sum）**：硬件驱动 Vector 单元，将 `128` 个 `256 B` 向量按列并行累加，在寄存器内部收敛为 `1` 个 `256 B` 的中间向量，也就是 `64` 个 FP32 元素。此阶段不消耗 UB 中的 `tmpBuffer`。
- **横向树状折叠（Tree Reduction）**：对最后的 `256 B` 向量执行高低半区折叠。为了完成跨位置的转置、混洗操作，硬件需要将中间数据暂存到 UB 中。

`tmpBuffer` 的本质是暂存最后 `1 ~ 2` 个 `256 B` 向量的转置中间态，因此需求极小且固定。对于 `32 KB` 输入，它通常只占 `256 ~ 512 B`，占 UB 容量的比例不足 `1.5%`。调用 `WholeReduceSum` 前，通过 `GetWholeReduceSumMinTmpSize` 查询最小需求并分配即可。

##### 2.2.2.2 接口定义与参数约束

```cpp
template <typename T>
__aicore__ inline void WholeReduceSum(
    const LocalTensor<T>& dstLocal,
    const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t calCount
);
```

- **`dstLocal`**：输出 `LocalTensor`。物理空间至少开辟 `32 B`；FP32 情况下相当于 `8` 个元素，最终结果存于 `dstLocal[0]`。
- **`srcLocal`**：输入 `LocalTensor`，首地址与长度均需满足 `32 B` 对齐。
- **`sharedTmpBuffer`**：由 `GetWholeReduceSumMinTmpSize` 查询后开辟的临时空间。
- **`calCount`**：参与归约的总元素数，例如本题的 `8192`。单次调用的 `calCount` 受 UB 容量限制，不能超过单 Tile 的最大承载量。

#### 2.2.3 算法流程与代码实现（Ascend C）

##### 2.2.3.1 完整计算流程

- **Init 阶段**：配置 Global Memory 地址映射，初始化 UB 上的 Pipe 队列与 `tmpBuffer` 空间。
- **Stage 1：DataCopy**：调用 `DataCopy`，将 `32 KB` 数据从 GM 一次性搬入 UB 的 `inQueueX`。
- **Stage 2：Compute**：调用 `WholeReduceSum`，驱动 Vector 单元在 UB 内完成多级向量归约，并将结果写至 `outQueueY` 的 `index 0`。
- **Stage 3：DataCopy**：将 `outQueueY` 中的标量结果按 `32 B` 对齐写回 GM 输出地址。

##### 2.2.3.2 代码实现示例与内存机制详解

```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelReduceSumEasy {
public:
    __aicore__ inline KernelReduceSumEasy() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength)
    {
        this->totalLength = totalLength;

        // 1. 获取 Global Memory 地址映射。
        xGm.SetGlobalBuffer((__gm__ float*)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, 1);

        // 2. 初始化 Pipe 内存管道：8192 * sizeof(float) = 32768 B。
        pipe.InitBuffer(inQueueX, 1, this->totalLength * sizeof(float));

        // 归约结果虽只有一个 FP32，但物理上需要一个 32 B Block。
        // yLocal[0] 保存有效结果，yLocal[1] 到 yLocal[7] 为 Padding。
        pipe.InitBuffer(outQueueY, 1, 32);

        // 3. 为 WholeReduceSum 查询并分配临时空间。
        uint32_t tmpBytes = 0;
        GetWholeReduceSumMinTmpSize(inQueueX, outQueueY, tmpBytes);
        pipe.InitBuffer(tmpBuffer, tmpBytes);
    }

    __aicore__ inline void Process()
    {
        // Stage 1: CopyIn - 从 GM 搬运 32 KB 输入到 UB。
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm, this->totalLength);
        inQueueX.EnQue(xLocal);

        // Stage 2: Compute - 在 UB 内进行树状折叠归约。
        LocalTensor<float> xCalc = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

        WholeReduceSum(yLocal, xCalc, tmpTensor, this->totalLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xCalc);

        // Stage 3: CopyOut - 按最小 32 B 粒度写回结果。
        LocalTensor<float> yOut = outQueueY.DeQue<float>();
        DataCopy(yGm, yOut, 8);
        outQueueY.FreeTensor(yOut);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> tmpBuffer;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t totalLength;
};

extern "C" __global__ __aicore__ void reduce_sum_easy(GM_ADDR x, GM_ADDR y)
{
    KernelReduceSumEasy op;
    op.Init(x, y, 8192);
    op.Process();
}
```

`AllocTensor<float>()` 本身不指定大小。它借用的是 `InitBuffer` 阶段为对应队列划分的空间：本例的 `outQueueY` 已被分配 `32 B`，因此得到的 `yLocal` 可以被解释为 `8` 个 FP32 元素组成的 `LocalTensor<float>`，而有效归约值只位于 `yLocal[0]`。

#### 2.2.4 深度避坑指南：内存单位与硬件级物理约束

##### 2.2.4.1 `InitBuffer` 的单位是字节

`pipe.InitBuffer(outQueueY, 1, 32)` 的第三个参数是字节数，不是元素数。填写 `4` 只会分配 `4 B`，即一个 FP32 的空间；而 Vector 单元写入结果时会以一个 `32 B` Block 刷新，后续 `28 B` 会覆盖邻近 Tensor，造成内存污染。

正确填写 `32` 后，UB 中实际有 `8` 个 FP32 元素的空间：`yLocal[0]` 到 `yLocal[7]`。后续 `DataCopy` 将这 `8` 个元素按 `32 B` 对齐粒度写回 GM。

##### 2.2.4.2 `tmpBuffer` 必须独立分配

如果没有分配 `tmpBuffer`，或者让它与 `inQueueX`、`outQueueY` 共享未隔离的片上空间，硬件在横向树状折叠与向量转置时就可能覆盖输入或输出数据，导致归约结果出现随机错误。

必须通过 `TPipe::InitBuffer` 单独开辟 `TBuf<TPosition::VECCALC>` 空间，保证 API 内部临时计算数据与输入、输出 Buffer 的物理隔离。

### 2.3 Medium 关卡：单核多 Tile 循环

#### 2.3.1 题目规格

- **输入数据 `x`**：$2^6 \times 8192 = 64 \times 8192 = 524288$ 个 `float32` 元素，共 `2 MB`，形状为 `(524288,)`。
- **输出数据 `y`**：`1` 个 `float32` 标量，形状为 `(1,)`。
- **计算约束**：仍由单个 AI Core 完成全量数据的规约累加。
- **物理限制**：单 Core 的片上 UB 无法一次性装下 `2 MB` 数据，必须拆分为 $2^6 = 64$ 个 Tile；每个 Tile 包含 `8192` 个元素，占用 `32 KB`，分批处理。

#### 2.3.2 从 CPU 累加器到 UB 局部累加器

从单 Tile 的 Easy 关卡演进到单核多 Tile 的 Medium 关卡，核心难点在于如何跨循环维护片上累加状态。在 CPU 编程中，维护一个全局累加和是再自然不过的事情：

```cpp
// CPU 传统思路：定义标量 -> 清零 -> 循环累加 -> 写回
float sumLocal = 0.0f;
for (int i = 0; i < 64; ++i) {
    sumLocal += yLocal[0];
}
yGm[0] = sumLocal;
```

但在 NPU 的 Ascend C 开发中，完全套用上述 C++ 习惯写出的代码，在硬件层面并不成立。

##### 2.3.2.1 错误示范：直接套用 CPU 标量累加

```cpp
// 错误示范：直接套用 C++ 习惯写法
float sumLocal = 0.0f; // 1. 试图用 C++ 标量声明并赋值清零

for (uint32_t i = 0; i < 64; ++i) {
    // ... 搬运并规约出当前 Tile 的标量 yLocal[0] ...
    sumLocal += yLocal[0]; // 2. 试图用 C++ 标量加法进行片上累加
}

yGm[0] = sumLocal; // 3. 试图用 C++ 指针写回 GM
```

##### 2.3.2.2 CPU 惯性思维在 NPU 上的四个问题

**问题一：用 `float sumLocal` 存储累加值。** C++ 声明的标量由 Scalar 寄存器保存，不在 UB 片上 SRAM 中。`WholeReduceSum` 的输出位于 UB，Scalar 寄存器不能直接参与 Vector 单元的高速计算。局部累加器必须显式分配在 UB 中，并满足 `32 B` 对齐：

```cpp
pipe.InitBuffer(sumBuf, 1, 32); // 申请 32 B 对齐空间
```

**问题二：用 C++ 赋值语句清零。** 新分配的 UB 空间可能保留上一轮计算的脏数据，不能用 `sumLocal = 0.0f` 对其初始化。应调用 Vector 单元的 `Duplicate` 指令完成片上填充：

```cpp
Duplicate(sumLocal, 0.0f, 8); // 矢量引擎清零 32 B，即 8 个 FP32
```

**问题三：用 C++ 标量加法累加。** `sumLocal += yLocal[0]` 会试图在 Scalar 与 UB/Vector 数据通路之间往返，带来同步开销与指令阻塞。应由 Vector 原生加法指令在 UB 中原址更新累加器：

```cpp
Add(sumLocal, sumLocal, yLocal, 8); // Vector 单元原址累加
```

**问题四：用 C++ 指针赋值写回 GM。** MTE 搬运要求源数据为 `LocalTensor`，且搬运基本单位满足 `32 B` 对齐。即使最终只需要一个 `float32` 结果，也需要以 `8` 个 FP32 的 `32 B` Block 发起搬运：

```cpp
DataCopy(yGm, sumLocal, 8); // 搬运 8 个 FP32，即 32 B
```

#### 2.3.3 CPU 与 Ascend C 的片上累加对照

| 维度 | CPU / C++ 惯用思维 | Ascend C 的正确做法 | 硬件原因 |
| --- | --- | --- | --- |
| 内存开辟 | `float sumLocal;` | `pipe.InitBuffer(sumBuf, 1, 32);` | 片上累加器必须位于 UB，且满足 `32 B` 对齐。 |
| 清零初始化 | `sumLocal = 0.0f;` | `Duplicate(sumLocal, 0.0f, 8);` | SRAM 可能残留脏数据，需要由 Vector 单元完成清零。 |
| 片上累加 | `sumLocal += yLocal[0];` | `Add(sumLocal, sumLocal, yLocal, 8);` | 由 Vector 指令在 UB 内原址更新。 |
| 写回 GM | `yGm[0] = sumLocal;` | `DataCopy(yGm, sumLocal, 8);` | MTE 写回需要满足 `32 B` 对齐。 |

#### 2.3.4 两个关键的缓冲区设计问题

##### 2.3.4.1 为什么 `inQueueX` 使用双缓冲，而 `outQueueY` 只需要深度 1

双缓冲的价值在于掩盖 GM 到 UB 的长延时 DMA 搬运。`inQueueX` 每轮需要从 GM 搬入 `32 KB` 数据，深度为 `2` 时可以让下一 Tile 的搬入与当前 Tile 的计算重叠。

`outQueueY` 承载的只是当前 Tile 规约得到的一个 `float32` 标量，物理上按 `32 B` 对齐。它在当前轮被 `Add` 立即消费并释放，任意时刻最多只保留一个 Tensor，因此深度 `1` 已经足够；设置为 `2` 只会额外占用 UB。

##### 2.3.4.2 `Add(sumLocal, sumLocal, yLocal, 8)` 是否会破坏输入

`Add` 支持原址操作。Vector 单元会先读取 `sumLocal` 和 `yLocal` 的数据至计算寄存器，完成加法后才将结果写回 `sumLocal` 的 UB 地址，因此不会发生读写覆盖。

同一 AI Core 的 Vector 队列按顺序执行，第 `i` 轮 Tile 的累加与第 `i+1` 轮 Tile 的累加不会并发访问同一个 `sumLocal`。将同一个 `LocalTensor` 同时作为源和目的，是维护片上累加状态的标准写法。

#### 2.3.5 完整代码实现

```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelReduceSumMedium {
public:
    __aicore__ inline KernelReduceSumMedium() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength; // 524288 (64 * 8192)
        this->tileLength = 8192;         // 单 Tile 元素个数 (32 KB)
        this->tileNum = 64;              // 2^6 = 64 次循环

        // 1. 全局内存（GM）映射
        xGm.SetGlobalBuffer((__gm__ float*)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, 1);

        // 2. 初始化内存管道（TPipe）
        // inQueueX: Ping-Pong 双缓冲，掩盖从 GM 到 UB 的长延时 DMA 搬运
        pipe.InitBuffer(inQueueX, 2, this->tileLength * sizeof(float));

        // outQueueY: 单 Tile 规约临时输出区，深度为 1 即可
        pipe.InitBuffer(outQueueY, 1, 32);

        // 用 UB 内存保存局部累加器，强制 32 B 物理 Block 对齐
        pipe.InitBuffer(sumBuf, 1, 32);

        // 3. 动态查询并分配 WholeReduceSum 内部所需的临时空间
        uint32_t tmpBytes = 0;
        GetWholeReduceSumMinTmpSize(inQueueX, outQueueY, tmpBytes);
        pipe.InitBuffer(tmpBuffer, tmpBytes);
    }

    __aicore__ inline void Process() {
        // Vector 指令清零 UB 累加器，填充 32 B（8 个 FP32）
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        Duplicate(sumLocal, 0.0f, 8);

        // 64 次 Tile 循环：规约当前 Tile，再累加到 sumLocal
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            // Stage 1: CopyIn - 根据动态偏移量从 GM 搬运 32 KB 数据进 UB
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            DataCopy(xLocal, xGm[i * this->tileLength], this->tileLength);
            inQueueX.EnQue(xLocal);

            // Stage 2: Compute - 片上规约与局部累加
            LocalTensor<float> xCalc = inQueueX.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

            // 当前 Tile 的 8192 个 FP32 规约为 yLocal[0]
            WholeReduceSum(yLocal, xCalc, tmpTensor, this->tileLength);

            // 在 Vector 单元中原址累加当前 Tile 的局部和
            Add(sumLocal, sumLocal, yLocal, 8);

            outQueueY.FreeTensor(yLocal);
            inQueueX.FreeTensor(xCalc);
        }

        // Stage 3: CopyOut - 所有 Tile 累加完成后按 32 B 粒度写回 GM
        DataCopy(yGm, sumLocal, 8);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX; // 乒乓队列，深度为 2
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> sumBuf;       // 片上局部累加器
    TBuf<TPosition::VECCALC> tmpBuffer;    // WholeReduceSum 专用辅助临时空间

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;

    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void reduce_sum_medium(GM_ADDR x, GM_ADDR y) {
    KernelReduceSumMedium op;
    op.Init(x, y, 524288);
    op.Process();
}
```

#### 2.3.6 实践作业

将上述实现整理为完整的 `kernel.asc`，提交至本节对应的 TensorOJ Reduce Sum Medium 题目。以题目评测通过作为本节实践作业的完成标准；同时记录 Tile 循环次数、UB 占用与提交耗时。
