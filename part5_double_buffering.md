### 第六节 TensorOJ 实战：Add Medium 的双缓冲流水线

本节继续实现 [TensorOJ Add Medium](https://tensoroj.cn/cann/pku-tensor/education/add-medium)，重点优化算子的执行效率。

在上一节中，Tile 循环虽然解决了片上 SRAM 溢出的问题，但 MTE 搬运引擎与 Vector 计算单元只能串行交替运行：Vector 计算时 MTE 只能闲置，MTE 搬运时 Vector 只能等待。

要消除这种硬件空转，核心在于**让“计算当前 Tile”与“搬运下一 Tile”同时进行**。然而在单缓冲区下，预读下一 Tile 会直接覆盖当前正在计算的片上数据，引发数据竞争。

本节将引入**双缓冲（Double Buffering）技术**：通过开辟两套可交替轮换的片上缓冲区（乒乓机制），实现内存搬运与向量计算的并发重叠，彻底拉满硬件算力利用率。

#### 一、传统顺序执行引发的硬件空转

在常规的单线程控制流中，算子执行通常遵循“数据搬入 $\to$ 向量计算 $\to$ 结果写回”的严格顺序。这种顺序安排会导致严重的硬件空转：

- **Vector 计算单元挂起**：当搬运引擎正在从 GM 搬运数据到片上 SRAM 时，Vector 单元因缺乏数据而闲置。
- **数据搬运引擎挂起**：当 Vector 单元对数据进行加法计算时，搬运引擎同样处于等待状态。

AI Core 的架构设计初衷，是让内部各个独立的硬件单元各自持续运转，而不是按部就班地相互等待。只要前后指令之间没有强制的数据依赖，后一条 Tile 的搬运指令完全可以在前一条 Tile 的计算尚未结束时提前发射并执行。

#### 二、昇腾芯片的核心硬件架构

在上一节中，我们看到了“数据搬运”与“向量计算”串行等待的性能痛点。要理解硬件为何具备并发重叠的能力，必须深入昇腾 AI 加速卡（Ascend AI Accelerator）的物理架构。

##### （一）核心计算单元：AIC、AIV 与 Scalar

昇腾 AI 加速卡采用模块化的异构计算核心设计，将不同类型的计算任务分派给最擅长的硬件单元：

- **AIC（AI Cube Core，矩阵计算核）**：集成了专门的 **Cube 矩阵计算单元**，专注于高密度的**矩阵乘法（GEMM）与张量卷积**。
- **AIV（AI Vector Core，向量计算核）**：集成了 **Vector 向量计算单元**，专注于 **向量计算与逐元素（Element-wise）运算**，例如 `Add`、`Relu`。
- **Scalar（标量计算单元）**：无论是 AIC 还是 AIV 内部，都集成了负责控制流的 **Scalar 单元**。它不直接处理海量张量计算，而是专门负责运行标量代码、维护 `for` 循环、计算 Tile 内存偏移量，并将指令发射到对应的硬件队列中。

##### （二）MTE 的物理独立管道与片上存储层级

许多开发者容易把“数据搬运”误以为是一条单向的通用通道。实际上在昇腾硬件架构中，负责数据搬运的 **MTE（Memory Transfer Engine）引擎被划分为多条物理上完全独立的管道**：

- **MTE2 管道（GM $\to$ UB 搬入）**：专门负责将输入数据从全局内存（GM）搬运到片上统一缓冲区（UB）。MTE2 拥有独立的 DMA 控制硬件，其搬运过程完全不占用 Vector 计算单元和 MTE3 管道。
- **MTE3 管道（UB $\to$ GM 写回）**：专门负责将片上 UB 计算完成的输出数据写回到全局内存（GM）。MTE3 管道与 MTE2 管道完全解耦，这意味着在向 GM 写回结果的同时，MTE2 管道可以并行从 GM 搬运下一批输入。
- **MTE1 与 FixPipe 管道（Cube 专用通路）**：在包含矩阵运算的 AIC 架构中，还存在负责将数据从 L1 缓存搬运到 L0A/L0B 缓冲区的 **MTE1 管道**，以及负责量化和格式转换的 **FixPipe 管道**。本节的 `Add` 算子主要基于 AIV 运行，核心涉及 MTE2 与 MTE3。

![AI Core 硬件架构及数据搬运通路](assets/double-buffer/ai-core-aic-aiv-memory-path.png)

*图 6-1：AI Core 的 AIC/AIV 硬件划分，以及 GM、UB 与各级存储之间的数据搬运通路。*

硬件上独立运行的 **MTE2、MTE3 与 Vector** 三条物理管道，为算子的高速并行提供了底座保障。

#### 三、Scalar 单元与多队列异步发射机制

Scalar（标量）单元负责程序控制流与标量运算。它不直接参与大批量向量或矩阵计算，而是解析算子指令、计算循环地址偏移，例如 Tile 的偏移地址，再将计算和搬运任务发射到各个硬件专属的异步队列中。

![Scalar 发射指令与异步队列](assets/double-buffer/scalar-async-queue-dispatch.png)

*图 6-2：Scalar 发射指令；Vector、Cube 与 DMA 搬运单元通过各自队列执行。指令流可以并行推进，数据依赖仍需要同步关系约束。*

Scalar 解析指令序列后，可以立即将计算或搬运指令发射给对应的硬件队列。由于发射动作很快，它不需要等待前一条向量指令完成，就能继续计算下一轮 Tile 的地址偏移，并发射下一条搬运指令。

AI Core 的主要指令队列及其职责如下。Add 直接使用 `PIPE_S`、`PIPE_V`、`PIPE_MTE2` 与 `PIPE_MTE3`；其余队列主要服务于矩阵计算或格式处理。

| 队列名称 | 对应硬件单元 | 指令类型与典型操作 | 硬件调度角色 |
| --- | --- | --- | --- |
| `PIPE_S` | Scalar 标量单元 | 控制流、循环、标量算术、地址与 Tile 偏移计算 | 解析并发射各类指令，为其他队列准备控制参数。 |
| `PIPE_M` | Cube 矩阵单元 | 矩阵计算指令，例如矩阵乘加 $C = A \times B + bias$ | 执行高吞吐矩阵运算；Add 不直接使用该队列。 |
| `PIPE_V` | Vector 向量单元 | SIMD 向量运算，例如 `Add`、`Exp`、`Softmax` | 对 UB 中的数据执行逐元素计算、激活和归一化等向量操作。 |
| `PIPE_MTE2` | MTE2 搬运引擎 | 外存到片上存储的数据搬运：GM $\rightarrow$ UB/L1 | 输入搬入通路，负责预取下一 Tile 的输入数据。 |
| `PIPE_MTE1` | MTE1 搬运引擎 | 片上 Buffer 间的数据搬运：L1 $\rightarrow$ L0A/L0B 等 | 为 Cube 矩阵计算准备片上数据；Add 不直接使用该队列。 |
| `PIPE_MTE3` | MTE3 搬运引擎 | 片上存储到外存的数据搬运：UB/LOC $\rightarrow$ GM | 结果写回通路，负责将完成计算的 Tile 写回全局内存。 |
| `PIPE_FIX` | FixPipe 格式处理单元 | 格式转换、量化/反量化、结果处理与写回 | 连接矩阵计算与后续处理；Add 不直接使用该队列。 |

对于本节的 Add，最关键的并发组合是：`PIPE_MTE2` 搬入 Tile `i + 1`，`PIPE_V` 计算 Tile `i`，`PIPE_MTE3` 写回 Tile `i - 1`。三者位于独立队列，因而具备并发推进的硬件基础。

指令一旦进入队列，对应硬件模块就会异步执行。由于这些队列相互独立，MTE 搬运与 Vector 计算具备在同一时刻重叠推进的硬件条件；但不同 Tile 若复用同一块 UB，仍会发生数据竞争。

#### 四、双缓冲如何解除资源冲突

第五节的单缓冲代码只有一套 UB 工作区：`xLocal`、`yLocal`、`zLocal`。当 Vector 单元正在读取其中的 Tile `i` 时，MTE2 不能把 Tile `i + 1` 搬到同一位置；否则新数据会覆盖仍在参与计算的旧数据。于是，下一次搬入只能等待当前计算结束。

双缓冲的做法不是把一个 Tile 算两遍，而是为每种局部数据准备两块可轮换的 UB Buffer。这样可以让两个相邻 Tile 同时处于不同阶段：

```text
Buffer 0: Tile i     正在由 Vector 计算
Buffer 1: Tile i + 1 正在由 MTE2 搬入
```

当 Tile `i` 的结果进入写回阶段时，Buffer 0 被归还；之后它就可以装入更靠后的 Tile。两组 Buffer 轮流承担这个角色，解除“下一 Tile 必须等上一 Tile 全部结束”的资源冲突。

#### 五、先确认双缓冲装得下 UB

双缓冲先是一项资源决策，随后才是代码结构。Add Medium 的一个 Tile 长度为 `8192`，一段 `float32` Tile 占用：

$$
8192 \times 4\text{ B} = 32768\text{ B} = 32\text{ KB}
$$

Add 同时需要 `x`、`y`、`z` 三类局部数据。单缓冲时每类只有一块，工作区为 `96 KB`；双缓冲时每类有两块，所需空间变为：

$$
3 \times 2 \times 8192 \times 4\text{ B}
= 196608\text{ B}
= 192\text{ KB}
$$

`192 KB` 小于单个 AI Core 约 `256 KB` 的 UB 名义容量，因此该组参数为两套 Tile 工作区留下了空间。但 UB 的实际可用预算还会受到运行时资源、其他临时数据和对齐要求影响；每次增加 Buffer 深度或 Tile 长度，都应重新进行容量检查并以实际编译、运行结果确认。

本节的双缓冲参数如下：

```cpp
constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 65536;
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t TILE_NUM = 8;
constexpr uint32_t BUFFER_NUM = 2;
```

#### 六、用 TPipe 与 TQue 管理缓冲区的使用状态

有两套 Buffer 后，真正困难的不是申请 `6` 个数组，而是判断每一块 UB 什么时候可以写入、什么时候正在计算、什么时候可以复用。手工维护 Buffer 0、Buffer 1 的下标和同步状态，Tile 数增加后很容易发生覆盖。

`TPipe` 和 `TQue` 将这份状态直接写进程序结构中：`TPipe` 在 UB 中为队列划出 Buffer；`TQue` 记录某块 Buffer 当前是否空闲、是否已准备给下一阶段使用。对 Add，需要三条数据通道：

| 队列 | 保存的 Tile | 生产阶段 | 消费阶段 |
| --- | --- | --- | --- |
| `inQueueX` | 输入 `x` | CopyIn | Compute |
| `inQueueY` | 输入 `y` | CopyIn | Compute |
| `outQueueZ` | 输出 `z` | Compute | CopyOut |

```cpp
AscendC::TPipe pipe;
AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
```

一块 Tile Buffer 在队列中的流转遵循固定的所有权关系：

```text
CopyIn : AllocTensor -> DataCopy -> EnQue
Compute: DeQue -> Add -> EnQue -> FreeTensor(输入 Buffer)
CopyOut: DeQue -> DataCopy -> FreeTensor(输出 Buffer)
```

`AllocTensor` 只能获得空闲 Buffer，`EnQue` 将准备完成的 Buffer 交给下一阶段，`DeQue` 只会取得已经准备好的 Buffer，`FreeTensor` 才会让该 Buffer 回到可复用状态。队列因此同时表达了数据依赖和 UB 的复用边界。

#### 七、稳定运行阶段（稳态）：三条流水线如何同时工作

双缓冲运行一段时间后，会进入稳态：三个硬件阶段分别处理不同 Tile，而不是围绕同一个 Tile 排队。

```text
                     MTE2 / CopyIn        Vector / Compute       MTE3 / CopyOut
同一时刻的工作：     搬入 Tile i + 1      计算 Tile i            写回 Tile i - 1
```

对应的时间关系可以写成：

```text
时间 ->
CopyIn :  [T0] [T1] [T2] [T3] [T4] ...
Compute:       [T0] [T1] [T2] [T3] ...
CopyOut:            [T0] [T1] [T2] ...
```

第一个 Tile 必须先完成搬入，最后一个 Tile 也必须等待写回，因此开始和结束阶段仍会有空档。双缓冲隐藏的是中间稳态中可重叠的等待时间；Tile 数越多，稳态在总执行时间中占比越高，优化越可能体现出来。

#### 八、将双缓冲落实为 Add Medium Kernel

##### （一）三个阶段各自做什么

`CopyIn` 取得两块空闲输入 Buffer，从当前 Tile 的 GM 区间读取 `x`、`y`，随后将它们入队。`Compute` 取出一对已经准备好的输入 Tile，申请一块输出 Buffer 执行 Add，将输出入队，并立即归还不再需要的输入 Buffer。`CopyOut` 取出已完成的输出 Tile，写回其对应的 GM 区间后归还输出 Buffer。

```cpp
__aicore__ inline void Compute()
{
    auto xLocal = inQueueX.DeQue<float>();
    auto yLocal = inQueueY.DeQue<float>();
    auto zLocal = outQueueZ.AllocTensor<float>();

    AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);

    outQueueZ.EnQue(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
}
```

这段顺序是双缓冲正确性的核心：输入 Buffer 必须在 Add 完成后才能归还；输出 Buffer 必须入队后才可由 CopyOut 取得；CopyOut 完成后才归还输出 Buffer。

##### （二）完整的 `kernel.asc`

下面的实现继续使用题目模板提供的 `run_kernel` 入口。它使用 C++ API 的 `TPipe + TQue` 管理双缓冲，不涉及算子注册。

```cpp
#include <cstdint>
#include "kernel_operator.h"

constexpr uint32_t NUM_BLOCKS = 16;
constexpr uint32_t BLOCK_LENGTH = 65536;
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t TILE_NUM = BLOCK_LENGTH / TILE_LENGTH;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH;

class AddDoubleBufferKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        const uint32_t blockOffset = block_idx * BLOCK_LENGTH;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + blockOffset,
                            BLOCK_LENGTH);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + blockOffset,
                            BLOCK_LENGTH);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + blockOffset,
                            BLOCK_LENGTH);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tileIdx = 0; tileIdx < TILE_NUM; ++tileIdx) {
            CopyIn(tileIdx);
            Compute();
            CopyOut(tileIdx);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        const uint32_t tileOffset = tileIdx * TILE_LENGTH;
        auto xLocal = inQueueX.AllocTensor<float>();
        auto yLocal = inQueueY.AllocTensor<float>();

        AscendC::DataCopy(xLocal, xGm[tileOffset], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[tileOffset], TILE_LENGTH);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        auto xLocal = inQueueX.DeQue<float>();
        auto yLocal = inQueueY.DeQue<float>();
        auto zLocal = outQueueZ.AllocTensor<float>();

        AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        const uint32_t tileOffset = tileIdx * TILE_LENGTH;
        auto zLocal = outQueueZ.DeQue<float>();

        AscendC::DataCopy(zGm[tileOffset], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
};

__vector__ __global__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    AscendC::InitSocState();
    AddDoubleBufferKernel kernel;
    kernel.Init(x, y, z);
    kernel.Process();
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

双缓冲不会自动保证加速：当 CopyIn、Compute、CopyOut 中某一阶段显著更慢时，它仍会主导总时间；Tile 数太少时，预热与收尾阶段的空档占比也会更高。完成实现后，应使用 profiling 对比第五节的单缓冲版本与本节版本的 Kernel 耗时，再判断这份重叠是否带来实际收益。
