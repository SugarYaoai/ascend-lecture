## 3 Dot（点积）融合算子、混合精度与数据对齐进阶

前一章的 Reduce Sum 重点处理了内存层级、`TPipe` 流水线与多核协同，并使用了两个简化条件：输入长度可以被向量指令粒度整除，且数据类型统一为 `float32`。

真实的工业级算子往往不具备这些理想条件：输入长度可能既不能被 Block 数整除，也不能被 Tile 长度或片上对齐粒度整除；输入和输出还常使用 `float16`、`int8` 等类型，而中间累加需要提升至 `float32` 或 `int32` 来避免溢出。

本章以 Dot 点积算子为载体：

$$
y = \sum_{i=0}^{N-1}(x_{1,i} \times x_{2,i})
$$

它把逐元素乘法 `Mul` 与规约求和 `WholeReduceSum` 融合到同一条数据流中，并逐步处理 Core Tail、Tile Tail、Mask 掩码和多类型混合对齐等问题。

### 3.1 Easy 关卡：单类型全量数据与双重 Tail Block

#### 3.1.1 题目规格与本关目的

##### 3.1.1.1 本关目的

本关暂不引入类型转换，集中处理 Tail Block。通过矢量掩码 `Mask` 与对齐补齐，保证任意非对齐数据规模下既不访问越界，也不会将 UB 中残留的脏数据计入点积结果。

##### 3.1.1.2 题目规格

- **输入数据 `x1`、`x2`**：各有 `1000003` 个 `float32` 元素，形状均为 `(1000003,)`。
- **输出数据 `y`**：`1` 个 `float32` 标量，形状为 `(1,)`。
- **计算约束**：启动 `8` 个 Block，并行计算点积 $y = \sum(x_1 \times x_2)$。
- **核心难点**：总长度不能被 `8` 个 Block 整除；各 Block 分到的数据也不能被单个 Tile 的 `8192` 元素长度整除，更不能被 `32 B`、即 `8` 个 FP32 的物理对齐单位整除。

#### 3.1.2 双重 Tail Block 与 Mask 掩码

##### 3.1.2.1 第一重尾块：Core Tail

当总数据量 $N$ 无法被 Block 数 $P$ 整除时，需要将前面的 Block 切为对齐长度，再让最后一个 Block 处理其余元素：

```cpp
// 基础切块大小：向下对齐到 32 B 的整数倍，即 8 个 FP32。
uint32_t baseCoreLength = (totalLength / blockNum) & ~7U;

// 最后一个 Block 处理其余的 Core Tail 元素。
uint32_t currentCoreLength = (blockIdx == blockNum - 1)
    ? (totalLength - baseCoreLength * (blockNum - 1))
    : baseCoreLength;
```

##### 3.1.2.2 第二重尾块：Tile Tail

当前 Block 的 `currentCoreLength` 在 Tile 循环中，最后一个 Tile 的有效元素数 `actualLength` 往往不足完整 Tile，甚至不足 `32 B`。若直接对完整 Tile 执行 `Mul` 或 `WholeReduceSum`，Vector 单元可能读取 UB 尾部残留数据，导致点积结果偏差。

##### 3.1.2.3 Vector Mask

Ascend C 的 Vector 单元通过 Mask 精确控制当前指令参与计算的 Lane：

- **按有效元素设置 Mask**：调用 `SetVectorMask<float>(0, actualLength)` 后，Vector 单元只对前 `actualLength` 个元素执行计算，超出的通道被屏蔽。
- **状态复位**：计算完成后必须调用 `ResetMask()` 恢复默认全通道 Mask，避免影响后续 Vector 指令。

#### 3.1.3 双输入流水线与尾块对齐规约

##### 3.1.3.1 双输入队列协同

Dot 算子需要同时处理 `x1` 与 `x2` 两个输入张量，因此在 `TPipe` 中需要成对申请和搬运 Buffer：

```cpp
// 在 TPipe 中开辟两条输入队列。
pipe.InitBuffer(inQueueX1, 2, tileByteSize);
pipe.InitBuffer(inQueueX2, 2, tileByteSize);

// CopyIn 阶段成对分配并搬运。
LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();

// 尾块搬运长度向上对齐至 32 B，即 8 个 FP32。
uint32_t copyLength = (actualLength + 7) & ~7U;
DataCopy(x1Local, x1Gm[offset], copyLength);
DataCopy(x2Local, x2Gm[offset], copyLength);
```

##### 3.1.3.2 Tail Block 下的逐元素乘法与规约

Compute 阶段先将乘法结果缓冲区清零，再用 Mask 只覆盖有效元素。这样即使 `WholeReduceSum` 按对齐后的 `copyLength` 执行，Padding 区也只包含零：

```cpp
LocalTensor<float> x1Calc = inQueueX1.DeQue<float>();
LocalTensor<float> x2Calc = inQueueX2.DeQue<float>();
LocalTensor<float> mulResult = mulBuffer.Get<float>();

// 先清零完整对齐区，避免 Mask 未写入的尾部残留脏数据参与规约。
Duplicate(mulResult, 0.0f, copyLength);

// 仅对 actualLength 个有效元素执行点乘。
SetVectorMask<float>(0, actualLength);
Mul(mulResult, x1Calc, x2Calc, actualLength);
ResetMask();

// 点乘结果规约至临时标量 yLocal[0]，再累加到本 Block 的局部和。
WholeReduceSum(yLocal, mulResult, tmpTensor, copyLength);
Add(sumLocal, sumLocal, yLocal, 8);
```

#### 3.1.4 Easy 关卡完整 Kernel 实现

```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelDotEasy {
public:
    __aicore__ inline KernelDotEasy() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength; // 1,000,003
        this->tileLength = 8192;         // 单 Tile 最大元素数，32 KB

        // 1. 处理 Core Tail：计算当前 Block 的数据切片。
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t baseCoreLength = (this->totalLength / blockNum) & ~7U;
        uint32_t coreOffset = blockIdx * baseCoreLength;

        this->coreLength = (blockIdx == blockNum - 1)
            ? (this->totalLength - coreOffset)
            : baseCoreLength;

        // 2. 映射当前 Block 的输入 GM 区间和共享输出地址。
        x1Gm.SetGlobalBuffer((__gm__ float*)x1 + coreOffset, this->coreLength);
        x2Gm.SetGlobalBuffer((__gm__ float*)x2 + coreOffset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, 1);

        this->tileNum = (this->coreLength + this->tileLength - 1) / this->tileLength;

        // 3. 初始化片上内存管道。
        uint32_t tileBytes = this->tileLength * sizeof(float);
        pipe.InitBuffer(inQueueX1, 2, tileBytes);
        pipe.InitBuffer(inQueueX2, 2, tileBytes);
        pipe.InitBuffer(outQueueY, 1, 32);
        pipe.InitBuffer(sumBuf, 1, 32);
        pipe.InitBuffer(mulBuf, 1, tileBytes);

        uint32_t tmpBytes = 0;
        GetWholeReduceSumMinTmpSize(inQueueX1, outQueueY, tmpBytes);
        pipe.InitBuffer(tmpBuffer, tmpBytes);
    }

    __aicore__ inline void Process() {
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        Duplicate(sumLocal, 0.0f, 8);

        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t offset = i * this->tileLength;
            uint32_t actualLength = this->coreLength - offset;
            if (actualLength > this->tileLength) {
                actualLength = this->tileLength;
            }
            uint32_t copyLength = (actualLength + 7) & ~7U;

            // Stage 1: CopyIn - 成对搬入两个输入 Tile。
            LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
            LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();
            DataCopy(x1Local, x1Gm[offset], copyLength);
            DataCopy(x2Local, x2Gm[offset], copyLength);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);

            // Stage 2: Compute - Mask 点乘、规约与局部累加。
            LocalTensor<float> x1Calc = inQueueX1.DeQue<float>();
            LocalTensor<float> x2Calc = inQueueX2.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            LocalTensor<float> mulResult = mulBuf.Get<float>();
            LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

            Duplicate(mulResult, 0.0f, copyLength);
            SetVectorMask<float>(0, actualLength);
            Mul(mulResult, x1Calc, x2Calc, actualLength);
            ResetMask();

            WholeReduceSum(yLocal, mulResult, tmpTensor, copyLength);
            Add(sumLocal, sumLocal, yLocal, 8);

            outQueueY.FreeTensor(yLocal);
            inQueueX1.FreeTensor(x1Calc);
            inQueueX2.FreeTensor(x2Calc);
        }

        // Stage 3: CopyOut - 各 Block 的局部和通过原子加合并至 GM。
        SetAtomicAdd<float>();
        DataCopy(yGm, sumLocal, 8);
        SetAtomicSub();
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX1;
    TQue<QuePosition::VECIN, 2> inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> mulBuf;
    TBuf<TPosition::VECCALC> tmpBuffer;

    GlobalTensor<float> x1Gm;
    GlobalTensor<float> x2Gm;
    GlobalTensor<float> yGm;

    uint32_t totalLength;
    uint32_t coreLength;
    uint32_t tileLength;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void dot_easy(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
    KernelDotEasy op;
    op.Init(x1, x2, y, 1000003);
    op.Process();
}
```

#### 3.1.5 实践作业

将本节实现整理为完整的 `kernel.asc`，提交至本节对应的 TensorOJ Dot Easy 题目。以题目评测通过作为本节实践作业的完成标准，并验证随机非对齐长度下的输出正确性。

#### 3.1.6 本节自测

**3.1-Q1.** 最后一个 Block 需要单独计算 `coreLength` 的主要原因是：

- A. 最后一个 Block 的 Vector 单元更宽。
- B. 总长度可能无法被 Block 数整除，剩余元素需要由最后一个 Block 处理。
- C. 最后一个 Block 不需要访问 GM。
- D. 最后一个 Block 只能处理一个 Tile。

**3.1-Q2.** 对 Tile Tail 设置 `SetVectorMask<float>(0, actualLength)` 的目的是什么？

- A. 增加 Vector 单元的物理通道数。
- B. 只让有效元素参与 `Mul`，屏蔽 Padding 区的计算。
- C. 自动将输入转换为 `float16`。
- D. 让 `DataCopy` 忽略 GM 地址偏移。

**3.1-Q3.** 在 Tail Tile 中，为什么要在 `Mul` 前对 `mulResult` 执行 `Duplicate(..., 0.0f, copyLength)`？

- A. 让 `WholeReduceSum` 自动跳过所有输入元素。
- B. 使两个输入张量的地址相同。
- C. 将 Mask 未写入的 Padding 区清零，避免脏数据参与后续规约。
- D. 将局部和立即写回 GM。

### 3.2 Medium 关卡：四类型混合与数据对齐

#### 3.2.1 题目规格与本关目的

##### 3.2.1.1 本关目的

3.1 处理了双重 Tail Block。本关进一步加入 `float16`、`float32`、`int8` 与 `int32` 四种核心类型的混合转换与物理对齐，建立工业级混合精度点积算子的完整数据通路。

##### 3.2.1.2 题目规格

- **输入 `x1`**：`1000000` 个 `float16` 元素，占用约 `2 MB`，形状为 `(1000000,)`。
- **输入 `x2`**：`1000000` 个 `int8` 元素，占用约 `1 MB`，形状为 `(1000000,)`。
- **输出 `y`**：`1` 个 `float32` 标量，形状为 `(1,)`。
- **计算约束**：启动 `8` 个 Block 并行计算点积 $y = \sum(x_1 \times x_2)$。
- **核心难点**：`x1` 与 `x2` 的元素大小及对齐基准不同；它们还需要在 UB 内提升至 FP32 后，再进行乘法与规约。

#### 3.2.2 四类型物理颗粒度与类型转换链

##### 3.2.2.1 四类型物理颗粒度映射

Ascend 的片上 Buffer 与 MTE 搬运以 `32 B` 为最小物理颗粒度：

| 类型 | 单元素字节数 | `32 B` 对齐元素数 | `256 B` 向量单周期容量 |
| --- | --- | --- | --- |
| `int8` | `1 B` | `32` | `256` |
| `float16` | `2 B` | `16` | `128` |
| `float32` | `4 B` | `8` | `64` |
| `int32` | `4 B` | `8` | `64` |

##### 3.2.2.2 片上类型提升计算链

为了保证计算精度并兼容 Vector 单元的 FP32 `Mul`，两个输入需要在 UB 中完成类型提升：

```text
x1 (FP16)  -- Cast --> x1Fp32 (FP32) --+
                                          +--> Mul(FP32) --> Reduce(FP32)
x2 (INT8)  -- Cast --> x2Int32 (INT32) -- Cast --> x2Fp32 (FP32) --+
```

- `x1` 的路径为 FP16 $\rightarrow$ FP32：`Cast(x1Fp32, x1Fp16, CastMode::CAST_NONE, actualLength)`。
- `x2` 的路径为 INT8 $\rightarrow$ INT32 $\rightarrow$ FP32：先转换为 INT32，再转换为 FP32。

#### 3.2.3 多类型内存切片与对齐补齐

##### 3.2.3.1 通用对齐公式

为了同时满足 INT8 的 `32` 元素对齐、FP16 的 `16` 元素对齐与 FP32 的 `8` 元素对齐，Core 切片和 Tile 切片的对齐单位取三者的最小公倍数：

$$
\operatorname{LCM}(32, 16, 8) = 32 \text{ 个元素}
$$

```cpp
// 基础切片长度：向下对齐到 32 的整数倍。
uint32_t baseCoreLength = (this->totalLength / blockNum) & ~31U;

// DataCopy 搬运长度：向上对齐到 32 的整数倍。
uint32_t copyLength = (actualLength + 31) & ~31U;
```

##### 3.2.3.2 类型转换后的 Buffer 容量

`x1` 与 `x2` 在 Cast 后都会扩张为 FP32。因此转换 Buffer 必须按扩张后的数据类型分配：

```cpp
pipe.InitBuffer(x1Fp32Buf, 1, tileLength * sizeof(float));
pipe.InitBuffer(x2Int32Buf, 1, tileLength * sizeof(int32_t));
pipe.InitBuffer(x2Fp32Buf, 1, tileLength * sizeof(float));
```

#### 3.2.4 Medium 关卡完整 Kernel 实现

```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelDotMedium {
public:
    __aicore__ inline KernelDotMedium() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength; // 1,000,000
        this->tileLength = 8192;         // 单 Tile 最大元素数，必须是 32 的倍数

        // 1. Core Tail：按 32 元素对齐，兼顾 INT8、FP16 与 FP32。
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t baseCoreLength = (this->totalLength / blockNum) & ~31U;
        uint32_t coreOffset = blockIdx * baseCoreLength;
        this->coreLength = (blockIdx == blockNum - 1)
            ? (this->totalLength - coreOffset)
            : baseCoreLength;

        x1Gm.SetGlobalBuffer((__gm__ half*)x1 + coreOffset, this->coreLength);
        x2Gm.SetGlobalBuffer((__gm__ int8_t*)x2 + coreOffset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, 1);

        this->tileNum = (this->coreLength + this->tileLength - 1) / this->tileLength;

        // 2. 原始输入队列按照各自元素大小分配。
        pipe.InitBuffer(inQueueX1, 2, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueX2, 2, this->tileLength * sizeof(int8_t));
        pipe.InitBuffer(outQueueY, 1, 32);
        pipe.InitBuffer(sumBuf, 1, 32);

        // Cast 后的中间结果按 FP32 或 INT32 大小分配。
        pipe.InitBuffer(x1Fp32Buf, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(x2Int32Buf, 1, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(x2Fp32Buf, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(mulBuf, 1, this->tileLength * sizeof(float));

        uint32_t tmpBytes = 0;
        GetWholeReduceSumMinTmpSize(x1Fp32Buf, outQueueY, tmpBytes);
        pipe.InitBuffer(tmpBuffer, tmpBytes);
    }

    __aicore__ inline void Process() {
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        Duplicate(sumLocal, 0.0f, 8);

        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t offset = i * this->tileLength;
            uint32_t actualLength = this->coreLength - offset;
            if (actualLength > this->tileLength) {
                actualLength = this->tileLength;
            }
            uint32_t copyLength = (actualLength + 31) & ~31U;

            // Stage 1: CopyIn - 搬运 FP16 的 x1 与 INT8 的 x2。
            LocalTensor<half> x1Local = inQueueX1.AllocTensor<half>();
            LocalTensor<int8_t> x2Local = inQueueX2.AllocTensor<int8_t>();
            DataCopy(x1Local, x1Gm[offset], copyLength);
            DataCopy(x2Local, x2Gm[offset], copyLength);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);

            // Stage 2: Compute - Cast、Mask 乘法与片上规约。
            LocalTensor<half> x1Calc = inQueueX1.DeQue<half>();
            LocalTensor<int8_t> x2Calc = inQueueX2.DeQue<int8_t>();
            LocalTensor<float> x1Fp32 = x1Fp32Buf.Get<float>();
            LocalTensor<int32_t> x2Int32 = x2Int32Buf.Get<int32_t>();
            LocalTensor<float> x2Fp32 = x2Fp32Buf.Get<float>();
            LocalTensor<float> mulResult = mulBuf.Get<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

            Cast(x1Fp32, x1Calc, CastMode::CAST_NONE, copyLength);
            Cast(x2Int32, x2Calc, CastMode::CAST_NONE, copyLength);
            Cast(x2Fp32, x2Int32, CastMode::CAST_NONE, copyLength);

            // 清零 Padding 区，再通过 Mask 仅计算有效元素。
            Duplicate(mulResult, 0.0f, copyLength);
            SetVectorMask<float>(0, actualLength);
            Mul(mulResult, x1Fp32, x2Fp32, actualLength);
            ResetMask();

            WholeReduceSum(yLocal, mulResult, tmpTensor, copyLength);
            Add(sumLocal, sumLocal, yLocal, 8);

            outQueueY.FreeTensor(yLocal);
            inQueueX1.FreeTensor(x1Calc);
            inQueueX2.FreeTensor(x2Calc);
        }

        // Stage 3: CopyOut - 多核原子加写回 FP32 GM。
        SetAtomicAdd<float>();
        DataCopy(yGm, sumLocal, 8);
        SetAtomicSub();
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX1;
    TQue<QuePosition::VECIN, 2> inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> x1Fp32Buf;
    TBuf<TPosition::VECCALC> x2Int32Buf;
    TBuf<TPosition::VECCALC> x2Fp32Buf;
    TBuf<TPosition::VECCALC> mulBuf;
    TBuf<TPosition::VECCALC> tmpBuffer;

    GlobalTensor<half> x1Gm;
    GlobalTensor<int8_t> x2Gm;
    GlobalTensor<float> yGm;

    uint32_t totalLength;
    uint32_t coreLength;
    uint32_t tileLength;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void dot_medium(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
    KernelDotMedium op;
    op.Init(x1, x2, y, 1000000);
    op.Process();
}
```

#### 3.2.5 实践作业

将本节实现整理为完整的 `kernel.asc`，提交至本节对应的 TensorOJ Dot Medium 题目。以题目评测通过作为本节实践作业的完成标准，并验证 FP16 与 INT8 输入转换后的数值正确性。

#### 3.2.6 本节自测

**3.2-Q1.** 同时处理 INT8、FP16 与 FP32 数据时，切片长度选择 `32` 个元素对齐的原因是：

- A. `32` 是三种类型元素字节数的总和。
- B. `32` 是它们在 `32 B` 对齐要求下元素数的最小公倍数。
- C. Vector 单元只能计算 32 个元素。
- D. `WholeReduceSum` 只能接收 32 个元素。

**3.2-Q2.** 为什么 INT8 输入要先转换为 INT32，再转换为 FP32？

- A. 该计算链不直接支持 INT8 到 FP32 的 Vector Cast。
- B. INT8 数据不能放入 UB。
- C. FP16 只能与 INT32 相乘。
- D. INT32 占用空间比 INT8 更小。

**3.2-Q3.** `x2Int32Buf` 的空间应按什么类型的字节数分配？

- A. `int8_t`，因为原始输入是 INT8。
- B. `half`，因为另一个输入是 FP16。
- C. `int32_t`，因为该 Buffer 保存 INT8 Cast 后的中间结果。
- D. `float`，因为最终输出是 FP32。
