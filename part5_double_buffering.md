### 第六节 TensorOJ 实战：Add Medium 的双缓冲流水线

本节继续实现 [TensorOJ Add Medium](https://tensoroj.cn/cann/pku-tensor/education/add-medium)，并优化它的执行速度。

第五节的 Tile 循环每次都能正确完成一小段 Add；本节要解决的核心问题是：**当 AI Core 正在计算 Tile `i` 时，怎样同时搬入 Tile `i + 1` 的输入，并写回 Tile `i - 1` 的结果，而且不让不同 Tile 覆盖彼此正在使用的 UB 数据？**

这个问题决定了读、算、写能否在不同 Tile 上重叠推进，也是双缓冲优化速度的出发点。

#### 一、什么是双缓冲

**双缓冲（Double Buffering）** 是指为同一类 Tile 数据准备两套可轮换的 UB 工作区。当前 Tile 使用其中一套 Buffer 进行计算时，下一 Tile 可以使用另一套 Buffer 搬入数据；两组 Buffer 再在后续循环中交换角色。

它不改变 Add 的数学结果，也不是把同一个 Tile 重复计算两次。它提供的是两份彼此独立的片上暂存空间，使不同 Tile 可以同时处于搬入、计算或写回等不同阶段。

#### 二、双缓冲要解除的资源冲突

第五节的单缓冲代码只有一套 UB 工作区：`xLocal`、`yLocal`、`zLocal`。当 Vector 单元正在读取其中的 Tile `i` 时，MTE2 不能把 Tile `i + 1` 搬到同一位置；否则新数据会覆盖仍在参与计算的旧数据。于是，下一次搬入只能等待当前计算结束。

双缓冲的做法不是把一个 Tile 算两遍，而是为每种局部数据准备两块可轮换的 UB Buffer。这样可以让两个相邻 Tile 同时处于不同阶段：

```text
Buffer 0: Tile i     正在由 Vector 计算
Buffer 1: Tile i + 1 正在由 MTE2 搬入
```

当 Tile `i` 的结果进入写回阶段时，Buffer 0 被归还；之后它就可以装入更靠后的 Tile。两组 Buffer 轮流承担这个角色，解除“下一 Tile 必须等上一 Tile 全部结束”的资源冲突。

#### 三、先确认双缓冲装得下 UB

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

#### 四、用 TPipe 与 TQue 管理 Buffer 所有权

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

#### 五、稳态阶段如何让三条流水线同时工作

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

#### 六、将双缓冲落实为 Add Medium Kernel

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
