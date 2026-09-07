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

### 2.2 Easy 关卡：单核单 Tile 的全量片上加载与指令调用

#### 2.2.1 题目规格

- **输入数据 `x`**：`8192` 个 `float32` 元素，共 `32 KB`，形状为 `(8192,)`。
- **输出数据 `y`**：`1` 个 `float32` 标量，形状为 `(1,)`。
- **计算约束**：单个 AI Core 执行，数据量可以一次性完整加载至 UB 空间。

#### 2.2.2 算法本质与编程范式转变

##### 2.2.2.1 算法本质

归约算子的计算目标是将长向量降维为一个标量。若不考虑硬件加速，在标准 C++ 中的实现如下：

```cpp
// C++ 串行求和：将 N 个元素累加为 1 个标量（逻辑示意）
float ReduceSum_CPU(const float* xCalc, int N) {
    float yLocal = 0.0f;
    for (int i = 0; i < N; ++i) {
        yLocal += xCalc[i]; // 逐元素累加
    }
    return yLocal;
}
```

##### 2.2.2.2 CPU 传统编程与 NPU 算子编程的范式转变

上述 C++ 逐元素循环累加的逻辑无法直接移植到 Ascend C Kernel 中。在传统 CPU 上，开发者习惯于“以标量为中心、以 CPU 寄存器为中转”的命令式思维；而在 NPU 架构下，必须建立“以矢量为中心、以数据流为驱动”的算子编程范式。这种思维转变源于底层硬件的三个核心差异：

- **计算单元与数据通路的分离**：AI Core 内部分为 Vector 单元与 Scalar 单元。UB 属于矢量计算通路，底层并不存在将 UB 内存中的元素逐个提取到标量寄存器进行低延时累加的高速通道。强行使用 C++ 循环逐个读取，会导致数据在标量与矢量通路间频繁跨区传送，引发硬件流水线严重停顿。
- **硬件算力的并行度要求**：Vector 单元单次指令的物理吞吐能力为 `256 B`，可一次性处理 `64` 个 `float32` 元素。采用串行 `for` 循环逐个计算，会浪费矢量计算单元的并行吞吐能力。
- **物理数据块的 `32 B` 粒度限制**：标准 C++ 中 `float` 标量仅占 `4 B`。但在 NPU 中，Vector 单元的读写计算以及 MTE 引擎的数据搬运，物理最小单位均为 `32 B` 数据块，即 `8` 个 `float32`。即便逻辑上只需输出 `1` 个标量，在片上开辟空间及写回 GM 时，也必须按 `32 B` 对齐处理，后 `7` 个位置作为 Padding 自动忽略。

#### 2.2.3 本节核心 API：片上规约 `WholeReduceSum`

Ascend C 的 Vector 单元提供 `WholeReduceSum` API 替代串行 `for` 循环。它在 AI Core 内部采用树状折叠与向量转置指令，在极少时钟周期内完成片上数据的并行规约。

##### 2.2.3.1 API 接口签名

```cpp
template <typename T>
__aicore__ inline void WholeReduceSum(
    const LocalTensor<T>& dstLocal,              // 输出 Tensor，物理空间须按 32 B 对齐
    const LocalTensor<T>& srcLocal,              // 输入 Tensor，待规约的片上数据
    const LocalTensor<uint8_t>& sharedTmpBuffer, // 硬件辅助临时 Buffer
    const uint32_t calCount                      // 参与规约的元素个数
);
```

##### 2.2.3.2 硬件辅助空间 `sharedTmpBuffer`

`WholeReduceSum` 执行时需要在 UB 中进行多轮树状折叠与转置，必须依赖额外的片上辅助空间（Scratchpad Memory）暂存中间计算结果。

- **为什么使用 `uint8_t`**：该缓冲区只供硬件指令读写中间结果，不承载特定业务数据类型。配套查询 API `GetWholeReduceSumMinTmpSize` 返回的长度单位直接为字节，以 `uint8_t` 作为 Tensor 元素类型可以按 `1:1` 的字节数申请空间，避免跨类型换算与对齐逻辑。
- **如何确定空间大小**：辅助空间大小取决于输入数据类型、输入和输出 Tensor 的形状及对齐状态，不能自行硬编码。必须在 `Init` 阶段动态查询：

```cpp
uint32_t tmpBytes = 0;
// 动态查询计算 calCount 个元素所需的最小临时字节数
GetWholeReduceSumMinTmpSize(inQueueX, outQueueY, tmpBytes);
pipe.InitBuffer(tmpBuffer, tmpBytes);
```

`GetWholeReduceSumMinTmpSize` 需要读取 `inQueueX` 和 `outQueueY` 的配置。因此在 `Init` 中必须先执行输入、输出队列的 `pipe.InitBuffer`，再调用查询接口；若顺序颠倒，`tmpBytes` 的结果不可用，临时缓冲区的初始化将失败。

##### 2.2.3.3 `yLocal` 的空间申请与 `32 B` 对齐

`yLocal` 逻辑上只保存一个 FP32 标量，但在 UB 中需要分配 `32 B`，即 `8` 个 `float32` 元素：

```cpp
// Init 阶段：逻辑输出为 1 个 FP32（4 B），但物理上开辟 32 B，即 8 个 FP32。
pipe.InitBuffer(outQueueY, 1, 32);

// Compute 阶段：yLocal 实际包含 8 个 float32 元素。
LocalTensor<float> xCalc = inQueueX.DeQue<float>();
LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

// 规约结果写入 yLocal[0]；yLocal[1] 到 yLocal[7] 为 Padding。
WholeReduceSum(yLocal, xCalc, tmpTensor, 8192);

outQueueY.EnQue(yLocal);
inQueueX.FreeTensor(xCalc);
```

CopyOut 阶段写回 GM 时，`DataCopy(yGm, yOutput, 8)` 的第三个参数必须填 `8` 而不是 `1`：MTE 需要按 `32 B` 对齐，$32\text{ B} / \text{sizeof(float)} = 8$ 个元素才能合法完成搬运。

#### 2.2.4 Easy 关卡完整 Kernel 实现

```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelReduceSumEasy {
public:
    __aicore__ inline KernelReduceSumEasy() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength; // 8192

        // 1. 全局内存（GM）映射
        xGm.SetGlobalBuffer((__gm__ float*)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, 1);

        // 2. 初始化数据队列
        pipe.InitBuffer(inQueueX, 1, this->totalLength * sizeof(float)); // 32 KB

        // yLocal 逻辑输出为 1 个 FP32（4 B），但受 32 B 对齐限制，
        // 必须通过 outQueueY 开辟 32 B 物理空间，即 8 个 FP32。
        pipe.InitBuffer(outQueueY, 1, 32);

        // 3. 先初始化队列，再动态查询并分配 WholeReduceSum 所需临时空间。
        uint32_t tmpBytes = 0;
        GetWholeReduceSumMinTmpSize(inQueueX, outQueueY, tmpBytes);
        pipe.InitBuffer(tmpBuffer, tmpBytes);
    }

    __aicore__ inline void Process() {
        // Stage 1: CopyIn - 全量搬入 UB
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[0], this->totalLength);
        inQueueX.EnQue(xLocal);

        // Stage 2: Compute - 调用 WholeReduceSum 完成片上树状规约
        LocalTensor<float> xCalc = inQueueX.DeQue<float>();

        // yLocal 实际包含 8 个 FP32（32 B），规约结果写入 yLocal[0]。
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

        WholeReduceSum(yLocal, xCalc, tmpTensor, this->totalLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xCalc);

        // Stage 3: CopyOut - MTE 按 32 B 对齐搬运 8 个 FP32 写回 GM。
        // GM 端只取 yGm[0] 作为标量结果。
        LocalTensor<float> yOutput = outQueueY.DeQue<float>();
        DataCopy(yGm, yOutput, 8);
        outQueueY.FreeTensor(yOutput);
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

extern "C" __global__ __aicore__ void reduce_sum_easy(GM_ADDR x, GM_ADDR y) {
    KernelReduceSumEasy op;
    op.Init(x, y, 8192);
    op.Process();
}
```

#### 2.2.5 实践作业

将本节实现整理为完整的 `kernel.asc`，提交至本节对应的 TensorOJ Reduce Sum Easy 题目。以题目评测通过作为本节实践作业的完成标准。

#### 2.2.6 本节自测

**2.2-Q1.** Reduce Sum Easy 的输出逻辑上只有一个 `float32`，但 `outQueueY` 仍分配 `32 B` 的原因是：

- A. `WholeReduceSum` 必须输出 8 个不同的标量。
- B. Vector 计算和 MTE 写回需要满足最小 `32 B` 对齐粒度。
- C. 每个 AI Core 必须保留 8 个输出队列。
- D. `float32` 在 UB 中固定占用 `32 B`。

**2.2-Q2.** `GetWholeReduceSumMinTmpSize` 应在何时调用？

- A. `WholeReduceSum` 执行完成之后。
- B. 在 `InitBuffer(inQueueX, ...)` 和 `InitBuffer(outQueueY, ...)` 之前。
- C. 输入、输出队列初始化之后，`tmpBuffer` 初始化之前。
- D. 仅在 Host 端启动 Kernel 之后。

**2.2-Q3.** `WholeReduceSum` 的 `sharedTmpBuffer` 使用 `LocalTensor<uint8_t>` 的主要原因是：

- A. 规约结果必须转换为 `uint8_t`。
- B. 临时空间按字节查询和分配，不承载业务数据类型。
- C. `float32` 不能存储在 UB 中。
- D. MTE 只能搬运 `uint8_t` 数据。

### 2.3 Medium 关卡：单核多 Tile 循环

#### 2.3.1 题目规格

- **输入数据 `x`**：$2^6 \times 8192 = 64 \times 8192 = 524288$ 个 `float32` 元素，共 `2 MB`，形状为 `(524288,)`。
- **输出数据 `y`**：`1` 个 `float32` 标量，形状为 `(1,)`。
- **计算约束**：仍由单个 AI Core 完成全量数据的规约累加。
- **物理限制**：单 Core 的 UB 空间无法一次性装下 `2 MB` 数据，必须拆分为 `64` 个 Tile；每个 Tile 包含 `8192` 个元素，占用 `32 KB`，通过单核循环分批加载与片上累加完成处理。

#### 2.3.2 算法本质与编程范式转变

Easy 关卡中，数据量可以一次性装入 UB，计算链路为“加载 $\rightarrow$ 片上规约 $\rightarrow$ 写回”。Medium 关卡面对超出 UB 容量的数据，必须引入 Tile 循环。

##### 2.3.2.1 算法本质

跨 Tile 的 Reduce 算子本质是“局部规约 + 跨 Tile 累加”。若不考虑硬件加速，在标准 CPU 编程中维护全局累加和可以写为：

```cpp
// CPU 串行求和：跨 Tile 累加（逻辑示意）
float sumLocal = 0.0f;
for (int i = 0; i < tileNum; ++i) {
    sumLocal += yLocal[i]; // 逐 Tile 累加
}
yGm[0] = sumLocal;
```

##### 2.3.2.2 CPU 传统编程与 NPU 算子编程的范式转变

在 Ascend C 算子开发中，完全套用上述 C++ 习惯写出的代码在 NPU 硬件底层无法成立：

```cpp
// 错误示范：试图在 Vector Kernel 中混用 CPU 标量逻辑
float sumLocal = 0.0f;
for (uint32_t i = 0; i < tileNum; ++i) {
    // ... 搬运并规约出当前 Tile 的标量 yLocal[0] ...
    sumLocal += yLocal[0]; // 错误：试图在 Scalar 与 UB 矢量数据通道间低效通信
}
yGm[0] = sumLocal; // 错误：不能直接用 C++ 指针赋值写回 GM
```

这种范式转变源于底层硬件架构的三项硬性约束：

- **存储位置限制**：C++ 声明的 `float sumLocal` 会分配在 Scalar 寄存器中，而 `WholeReduceSum` 的结果存放在 UB 的 Vector 通道中。由于硬件不支持 Scalar 寄存器与 UB 之间的高吞吐、低延时频繁交互，累加器必须显式分配在 UB 中。
- **初始化方式限制**：UB 属于片上 SRAM，分配后可能残留脏数据。不能使用 C++ 赋值语句清零，必须调用 Vector 单元的 `Duplicate` 指令对 UB 累加空间进行矢量化清零初始化。
- **计算指令限制**：不能使用 C++ 标量加法，必须调用 Vector 单元的 `Add` 指令，在 UB 内对累加器进行原址更新。

#### 2.3.3 本节核心设计：片上累加器与流水线编排

##### 2.3.3.1 片上累加器 `sumBuf` 的声明与初始化

累加器需要在整个 Tile 循环生命周期内持续存在，且不参与 Queue 的入队、出队流水调度，因此应使用 `TBuf` 而不是 `TQue` 管理其片上内存：

```cpp
// Init 阶段：开辟 32 B 物理空间作为片上累加器 Buffer
pipe.InitBuffer(sumBuf, 1, 32);

// Process 阶段：获取 LocalTensor 并调用 Duplicate 指令清零，填充 8 个 FP32
LocalTensor<float> sumLocal = sumBuf.Get<float>();
Duplicate(sumLocal, 0.0f, 8);
```

##### 2.3.3.2 矢量加法 `Add` 与原址累加

循环内部通过 `Add` 合并当前 Tile 的规约结果与全局累加器：

```cpp
// sumLocal (dst) = sumLocal (src0) + yLocal (src1)
// 计算 8 个 FP32，满足 32 B 矢量指令对齐要求
Add(sumLocal, sumLocal, yLocal, 8);
```

Vector 单元执行 `Add` 时，会先将 `sumLocal` 和 `yLocal` 读入矢量计算寄存器完成加法，再将结果写回 `sumLocal` 的 UB 地址，因此同地址读写不会发生数据竞争或覆盖错误。

##### 2.3.3.3 队列深度与 Ping-Pong 双缓冲

为掩盖 MTE 从 GM 搬运数据到 UB 的长延时，输入队列 `inQueueX` 采用深度为 `2` 的双缓冲策略：

- **`inQueueX`，Depth = 2**：用于让 Tile $i+1$ 的 CopyIn 搬运与 Tile $i$ 的 Compute 重叠执行。
- **`outQueueY`，Depth = 1**：仅作为当前 Tile 执行 `WholeReduceSum` 的临时输出中转区。每个 Tile 计算完成后立即被 `Add` 消费并释放，深度为 `1` 即可满足需求。

#### 2.3.4 Medium 关卡完整 Kernel 实现

```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelReduceSumMedium {
public:
    __aicore__ inline KernelReduceSumMedium() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength; // 524288 (64 * 8192)
        this->tileLength = 8192;         // 单 Tile 元素个数 (32 KB)
        this->tileNum = 64;              // 循环次数 64

        // 1. 全局内存（GM）映射
        xGm.SetGlobalBuffer((__gm__ float*)x, this->totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, 1);

        // 2. 初始化片上内存管道（TPipe）
        // inQueueX 设为 2 级深度（Ping-Pong 双缓冲），掩盖从 GM 到 UB 的搬运延时
        pipe.InitBuffer(inQueueX, 2, this->tileLength * sizeof(float));

        // outQueueY 为单 Tile 规约临时输出，深度为 1，满足 32 B 对齐
        pipe.InitBuffer(outQueueY, 1, 32);

        // 声明 UB 局部累加器 Buffer，按 32 B（8 个 FP32）对齐
        pipe.InitBuffer(sumBuf, 1, 32);

        // 3. 先初始化队列，再动态查询 WholeReduceSum 所需辅助空间大小并初始化 TBuf
        uint32_t tmpBytes = 0;
        GetWholeReduceSumMinTmpSize(inQueueX, outQueueY, tmpBytes);
        pipe.InitBuffer(tmpBuffer, tmpBytes);
    }

    __aicore__ inline void Process() {
        // 使用 Vector 单元清零片上累加器 sumLocal，填充 8 个 FP32
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        Duplicate(sumLocal, 0.0f, 8);

        // 64 次 Tile 循环：分批搬运、规约并累加
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            // Stage 1: CopyIn - 按 Tile 偏移量将 32 KB 数据搬入 UB
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            DataCopy(xLocal, xGm[i * this->tileLength], this->tileLength);
            inQueueX.EnQue(xLocal);

            // Stage 2: Compute - 片上规约与局部累加
            LocalTensor<float> xCalc = inQueueX.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

            // 规约当前 Tile 的 8192 个元素至 yLocal[0]
            WholeReduceSum(yLocal, xCalc, tmpTensor, this->tileLength);

            // 在 UB 内将当前 Tile 规约结果原址累加至 sumLocal
            Add(sumLocal, sumLocal, yLocal, 8);

            outQueueY.FreeTensor(yLocal);
            inQueueX.FreeTensor(xCalc);
        }

        // Stage 3: CopyOut - 循环结束后，按 32 B 粒度将最终累加和写回 GM
        DataCopy(yGm, sumLocal, 8);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX; // 双缓冲乒乓队列
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> sumBuf;      // UB 局部累加器空间
    TBuf<TPosition::VECCALC> tmpBuffer;   // WholeReduceSum 辅助临时空间

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

#### 2.3.5 实践作业

将上述实现整理为完整的 `kernel.asc`，提交至本节对应的 TensorOJ Reduce Sum Medium 题目。以题目评测通过作为本节实践作业的完成标准；同时记录 Tile 循环次数、UB 占用与提交耗时。

#### 2.3.6 本节自测

**2.3-Q1.** Reduce Sum Medium 中，`sumBuf` 选用 `TBuf` 而不是 `TQue` 的主要原因是：

- A. `TBuf` 只能保存 `float32`，`TQue` 不能。
- B. 局部累加器需要贯穿整个 Tile 循环，不参与入队和出队调度。
- C. `TQue` 无法分配 UB 空间。
- D. `TBuf` 会自动将数据写回 GM。

**2.3-Q2.** `Duplicate(sumLocal, 0.0f, 8)` 的作用是：

- A. 将 8 个 Tile 同时复制到 GM。
- B. 将当前 Tile 的 8 个元素规约为一个标量。
- C. 在 UB 中将局部累加器的 8 个 FP32 位置清零。
- D. 为输入队列创建两个 Buffer。

**2.3-Q3.** `outQueueY` 采用深度 `1` 的合理原因是：

- A. 规约结果在同一轮中立刻被 `Add` 消费并释放。
- B. 输出标量不能放进双缓冲。
- C. `WholeReduceSum` 只能使用单缓冲输入。
- D. 深度 `2` 会改变浮点数精度。

### 2.4 Hard 关卡：多 Core 协同与跨核归约

#### 2.4.1 题目规格

- **输入数据 `x`**：$32 \times 524288 = 16777216$ 个 `float32` 元素，共 `64 MB`，形状为 `(16777216,)`。
- **输出数据 `y`**：`1` 个 `float32` 标量，形状为 `(1,)`。
- **计算约束**：分配至 `32` 个 AI Core 并行处理全量数据。
- **物理限制**：数据量极大，必须通过 Block 网格将计算任务均匀分发给 `32` 个 AI Core。各 Core 独立完成核内 Tile 循环规约后，还需要将各自的局部和合并写回 GM。

#### 2.4.2 算法本质与编程范式转变

##### 2.4.2.1 算法本质

多核场景下，Reduce 算子的本质是“核内局部规约 + 跨核全局合并”。若不考虑硬件加速，在标准 CPU 多线程编程中，跨线程汇总通常写为：

```cpp
// CPU 多线程求和：跨线程原子加（逻辑示意）
float coreLocalSum = ComputeCoreSum(thread_id);

#pragma omp atomic
yGm[0] += coreLocalSum;
```

##### 2.4.2.2 CPU 传统编程与 NPU 算子编程的范式转变

在 Ascend C 算子开发中，完全套用上述 C++ 习惯写出的代码在 NPU 硬件底层无法成立：

```cpp
// 错误示范：试图在 NPU Kernel 中使用 CPU 原子加操作
float coreLocalSum = ProcessCore();

// 错误：yGm 是 GlobalTensor，不能直接使用 C++ += 运算符
yGm[0] += coreLocalSum;
```

这种范式转变源于底层硬件架构的三项硬性约束：

- **跨核内存竞争**：`32` 个 AI Core 是独立的硬件计算单元，并发访问同一个 GM 地址 `yGm[0]` 会引发写后写与读后写冲突。
- **硬件级原子操作支持**：跨 Core 的同步不能依赖 Scalar 标量写回，必须通过 MTE 搬运引擎底层的 GM 原子加硬件指令完成。
- **搬运与计算绑定**：NPU 不支持单独对 GM 发起标量加法指令；原子累加必须在 `DataCopy` 将 `LocalTensor` 搬回 `GlobalTensor` 的物理搬运过程中，通过配置原子操作模式触发。

#### 2.4.3 本节核心设计：多核数据切分与 GM 原子加

##### 2.4.3.1 基于 `GetBlockIdx()` 的多核数据切分

每个 AI Core 独立执行相同的 Kernel 代码，通过硬件内置 API `GetBlockIdx()` 获取当前逻辑 Block 编号，计算各自负责的 GM 偏移量：

```cpp
// 1. 获取当前 Block 编号
uint32_t blockIdx = GetBlockIdx();

// 2. 根据 Block 编号映射各自负责的 GM 切片起始地址
uint32_t coreLength = 524288; // 每个 Block 处理 2 MB 数据
xGm.SetGlobalBuffer((__gm__ float*)x + blockIdx * coreLength, coreLength);
```

##### 2.4.3.2 MTE 硬件原子加与状态开关

`SetAtomicAdd` 不是一个像 `Add` 那样直接对两个 `LocalTensor` 进行计算的算术指令，而是为 MTE 设定的硬件全局状态开关。

**普通搬运与原子加搬运的差异：**

- **普通搬运模式**：执行 `DataCopy` 时，MTE 直接将 UB 数据发送给 GM，目标 GM 地址的旧数据被覆盖。若 `32` 个 Core 同时写同一地址，后写入的数据会冲掉先写入的数据，导致结果错误。
- **原子加搬运模式**：调用 `SetAtomicAdd<float>()` 后，MTE 切换至原子加状态。随后执行 `DataCopy` 时，MTE 向内存控制器发起 Read-Modify-Write 原子事务：锁定目标 GM 地址、读取旧值、将旧值与 UB 中的 `sumLocal` 相加、写回新值并解锁。即使 `32` 个 Core 同时触发写回，GM 侧硬件也会串行化这些事务，确保累加结果正确。

**代码三步范式：**

```cpp
// 1. 开启 MTE 对 float32 的原子加状态
SetAtomicAdd<float>();

// 2. 触发 DataCopy 写回 GM：MTE 自动将 sumLocal[0] 累加至 yGm[0]
// 受 32 B 对齐限制，此处传输 8 个 FP32。
DataCopy(yGm, sumLocal, 8);

// 3. 关闭原子加状态，恢复默认写覆盖模式
SetAtomicSub(); // 或 SetAtomicNone()，取决于架构与驱动版本
```

`SetAtomicAdd` 设置的是 AI Core 内 MTE 搬运管道的全局状态。`DataCopy` 执行后必须立即关闭，否则该 Kernel 后续的其他 `DataCopy` 也会被按原子加处理，导致不可预期的计算结果。

此外，多核采用“GM 旧值 + 本核局部和”的原子累加机制，因此 Kernel 启动前，Host 侧必须确保输出内存 `y` 至少预留 `32 B` 且已物理清零。若目标地址残留脏数据，原子加会将脏数据一并计入最终结果。

#### 2.4.4 Hard 关卡完整 Kernel 实现

```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelReduceSumHard {
public:
    __aicore__ inline KernelReduceSumHard() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        // 1. 多核切分计算：每个 Core 处理 524288 个元素（2 MB）
        this->coreLength = 524288;
        this->tileLength = 8192;
        this->tileNum = this->coreLength / this->tileLength; // 64 次 Tile 循环

        // 获取当前 Block 编号，映射各自的输入 GM 切片起始位置
        uint32_t blockIdx = GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ float*)x + blockIdx * this->coreLength, this->coreLength);

        // 映射统一的输出 GM 地址，32 个 Core 共同累加至 yGm[0]
        yGm.SetGlobalBuffer((__gm__ float*)y, 1);

        // 2. 初始化片上内存管道（TPipe）
        pipe.InitBuffer(inQueueX, 2, this->tileLength * sizeof(float)); // Ping-Pong 双缓冲
        pipe.InitBuffer(outQueueY, 1, 32);                              // 单 Tile 规约临时输出
        pipe.InitBuffer(sumBuf, 1, 32);                                 // UB 局部累加器

        // 3. 先初始化队列，再动态查询 WholeReduceSum 所需辅助空间大小并初始化 TBuf
        uint32_t tmpBytes = 0;
        GetWholeReduceSumMinTmpSize(inQueueX, outQueueY, tmpBytes);
        pipe.InitBuffer(tmpBuffer, tmpBytes);
    }

    __aicore__ inline void Process() {
        // 使用 Vector 单元清零本 Core 的 UB 局部累加器，填充 8 个 FP32
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        Duplicate(sumLocal, 0.0f, 8);

        // 64 次 Tile 循环：计算本 Core 的局部规约和
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            // Stage 1: CopyIn - 按 Tile 偏移量将 32 KB 数据搬入 UB
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            DataCopy(xLocal, xGm[i * this->tileLength], this->tileLength);
            inQueueX.EnQue(xLocal);

            // Stage 2: Compute - 片上规约与核内累加
            LocalTensor<float> xCalc = inQueueX.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

            // 规约当前 Tile 的 8192 个元素至 yLocal[0]
            WholeReduceSum(yLocal, xCalc, tmpTensor, this->tileLength);

            // 在 UB 内原址累加至本 Core 的 sumLocal
            Add(sumLocal, sumLocal, yLocal, 8);

            outQueueY.FreeTensor(yLocal);
            inQueueX.FreeTensor(xCalc);
        }

        // Stage 3: CopyOut - 多核原子加写回 GM
        // 1. 开启 MTE 原子加开关
        SetAtomicAdd<float>();

        // 2. 触发 DataCopy：MTE 发起 Read-Modify-Write 事务，将 sumLocal[0] 累加至 yGm[0]
        DataCopy(yGm, sumLocal, 8);

        // 3. 关闭原子加开关，恢复默认写覆盖模式
        SetAtomicSub();
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX; // 双缓冲乒乓队列
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> sumBuf;      // UB 局部累加器空间
    TBuf<TPosition::VECCALC> tmpBuffer;   // WholeReduceSum 辅助临时空间

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;

    uint32_t coreLength;
    uint32_t tileLength;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void reduce_sum_hard(GM_ADDR x, GM_ADDR y) {
    KernelReduceSumHard op;
    op.Init(x, y, 16777216);
    op.Process();
}
```

#### 2.4.5 Atomic 规约与 Two-Stage 规约

Hard 关卡使用基于 `SetAtomicAdd` 的单阶段多核规约。在工业级通用算子开发中，跨核规约存在两种主流设计方案：

| 方案 | 架构原理 | 优点 | 缺点与适用场景 |
| --- | --- | --- | --- |
| **Atomic 规约**（本关卡方案） | 各 Core 计算完局部和后，直接通过 MTE 对同一个 GM 地址执行 `SetAtomicAdd`。 | 内存占用低，不需要额外开辟 WorkSpace，Kernel 逻辑简洁。 | 当并发 Core 数量较多时，会高频争抢同一个 GM 物理 Bank，造成总线等待与性能下降。 |
| **Two-Stage 规约**（两阶段规约） | Stage 1：各 Core 将 `core_local_sum` 写入 `WorkSpace[blockIdx]` 的独立槽位；Stage 2：由单个或少量 Core 发起第二次轻量级规约，汇总所有局部和。 | 无跨核写竞争，MTE 写入可以无锁并发；大核数下性能与吞吐上限更高。 | 需要额外的 WorkSpace，由 Host 申请并传入；Kernel 流程也更复杂。 |

实际工程中，需要根据数据规模、AI Core 数量以及设备 GM 访存带宽，选择更符合性能效益的跨核规约模式。

#### 2.4.6 实践作业

将上述实现整理为完整的 `kernel.asc`，提交至本节对应的 TensorOJ Reduce Sum Hard 题目。以题目评测通过作为本节实践作业的完成标准；比较 Atomic 规约与 Two-Stage 规约的实现复杂度与性能差异。

#### 2.4.7 本节自测

**2.4-Q1.** Hard 关卡中，`GetBlockIdx()` 的作用是：

- A. 查询当前 Tile 在 UB 中的物理地址。
- B. 获取当前逻辑 Block 编号，以计算各自负责的 GM 输入偏移。
- C. 获取当前 Vector 指令的 SIMD 宽度。
- D. 将多个 AI Core 强制绑定到同一个 Block。

**2.4-Q2.** 多个 AI Core 将局部和写回同一 `yGm[0]` 时，为什么需要 `SetAtomicAdd<float>()`？

- A. 让 `WholeReduceSum` 自动扩大 Tile 长度。
- B. 将 UB 中的数据自动转换为 `float16`。
- C. 让 MTE 的 `DataCopy` 以原子读改写方式合并各 Core 的局部和。
- D. 在 Host 端创建 32 个 Stream。

**2.4-Q3.** 相比 Atomic 规约，Two-Stage 规约的主要特点是：

- A. 完全不需要额外 GM 空间。
- B. 各 Core 先写入独立 WorkSpace 槽位，避免直接竞争同一输出地址。
- C. 只能由单个 AI Core 读取输入数据。
- D. 不需要进行第二阶段的结果汇总。
