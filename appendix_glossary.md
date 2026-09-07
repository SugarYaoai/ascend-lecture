## A 附录：术语表

### A.1 核心术语速查

#### A.1.1 执行与存储

| 术语 | 英文 / 代码 | 解释 |
| --- | --- | --- |
| 张量 | Tensor | 按形状组织的一组数。本书的 Add 从一维张量，即向量，开始。 |
| 算子 | Operator | 一条张量计算规则及其实现。例如 Add 的规则是 `z[i] = x[i] + y[i]`。 |
| Host | Host | CPU 一侧的程序和内存环境，负责准备数据、提交任务、取回结果。 |
| NPU | Neural Processing Unit | 执行 AI 计算的设备。Host 将任务提交给 NPU。 |
| AI Core | AI Core | NPU 内实际执行 Kernel 的物理计算核心。 |
| Kernel | 核函数 | 在 NPU Device 端运行的程序，是算子实际执行计算的部分。 |
| Block | Block | 同一份 Kernel 的一个逻辑执行任务。运行时为其分配编号并调度到可用 AI Core。 |
| Global Memory | GM | NPU 设备侧、所有 AI Core 都能访问的大容量内存，保存完整输入和输出。 |
| Unified Buffer | UB | 单个 AI Core 内部的片上存储，只保存当前正在计算的一小段数据。 |
| Host Memory | HM | CPU 一侧的内存，保存准备输入和取回的输出。 |
| Stream | `aclrtStream` | 运行时维护的任务序列。同一 Stream 中的任务按提交顺序建立依赖。 |

#### A.1.2 索引、类型与内存空间

| 术语 | 英文 / 代码 | 解释 |
| --- | --- | --- |
| 数据类型 | dtype | 张量中每个元素的表示方式，例如 `float32`。 |
| 单精度浮点数 | `float32` | 32 位浮点数，每个元素占 `4 B`。 |
| Block 编号 | `block_idx` | 编译器提供的内置系统变量，表示当前 Block 的逻辑编号，不是固定 AI Core 的物理编号。 |
| GM 地址 | `GM_ADDR` | 运行时传入的设备侧 GM 地址。需要转换为具体元素类型的 `__gm__` 指针后才能按元素偏移。 |
| GM 指针 | `__gm__ float*` | 指向设备侧 Global Memory 中 `float` 数据的指针。 |
| UB 数组 | `__ubuf__ float[...]` | C API 中声明在当前 AI Core UB 内的局部数组。 |
| `blockOffset` | Block offset | 当前 Block 相对完整向量的元素起点，通常为 `block_idx * BLOCK_LENGTH`。 |
| `tileOffset` | Tile offset | 当前 Tile 相对当前 Block 的元素起点，通常为 `tileIdx * TILE_LENGTH`。 |

#### A.1.3 编程模型与接口

| 术语 | 英文 / 代码 | 解释 |
| --- | --- | --- |
| SIMD | Single Instruction, Multiple Data | 一条向量指令同时处理一批连续元素的执行方式。 |
| C API | `asc_*` | 以 GM/UB 指针和函数调用表达设备计算的接口风格。 |
| C++ API | `AscendC::*` | 以张量对象、内存分配器和成员函数表达设备计算的接口风格。 |
| `__global__` | Kernel 修饰符 | 声明一个可由 Host 启动的 Device 端 Kernel。 |
| `__vector__` | Kernel 修饰符 | 表示 Kernel 在 AI Core 的向量计算单元上执行。 |
| `asc_init()` | C API | 初始化当前 Device 端 Kernel 所需的运行状态。 |
| `asc_copy_gm2ub` | C API | 将当前数据段从 GM 搬运到 UB。 |
| `asc_add` | C API | 在 UB 中完成向量逐元素加法。 |
| `asc_copy_ub2gm` | C API | 将 UB 中的结果写回 GM。 |
| `asc_sync()` | C API | 保证前一阶段完成后，后一阶段才能使用其结果。 |
| `GlobalTensor` | C++ API | 对 GM 中一段连续元素的视图；它描述地址范围，不会自动搬运数据。 |
| `LocalTensor` | C++ API | 对 UB 中一段局部缓冲区的视图。 |
| `LocalMemAllocator` | C++ API | 从当前 AI Core 的 UB 中申请 `LocalTensor` 的局部内存分配器。 |
| `DataCopy` | C++ API | 在 GM 与 UB 之间搬运数据。 |
| `Add` | C++ API | 对 `LocalTensor` 中的数据执行逐元素加法。 |
| `PipeBarrier` | C++ API | 表达阶段依赖；前一阶段完成前，后一阶段不能读取相应结果。 |

#### A.1.4 分块与性能优化

| 术语 | 英文 / 代码 | 解释 |
| --- | --- | --- |
| `BLOCK_LENGTH` | Block length | 一个 Block 负责处理的元素数。它影响单个任务的工作量和 UB 占用。 |
| Tile | Tile | 一个 Block 内的一小段连续元素，长度选择为可以安全放入 UB。 |
| `TILE_LENGTH` | Tile length | 一个 Tile 中的元素数，决定每次搬入 UB 的数据量。 |
| `TILE_NUM` | Tile count | 一个 Block 中 Tile 的数量，通常为 `BLOCK_LENGTH / TILE_LENGTH`。 |
| 单缓冲 | Single buffering | 每类数据只有一套可复用 UB Buffer，搬运、计算、写回通常按顺序完成。 |
| 双缓冲 | Double buffering | 每类数据有两套可轮换 UB Buffer，使不同 Tile 的搬运、计算、写回能够重叠。 |
| 流水线 | Pipeline | 可独立推进的一类硬件执行通道，例如 GM 到 UB 搬运、向量计算、UB 到 GM 写回。 |
| MTE2 | Memory Transfer Engine 2 | 通常负责从 GM 向 UB 搬入数据的内存传输引擎。 |
| Vector | Vector pipeline | 负责执行 `Add` 等向量计算的执行单元。 |
| MTE3 | Memory Transfer Engine 3 | 通常负责从 UB 向 GM 写回数据的内存传输引擎。 |
| `TPipe` | C++ API | 在 UB 中划分并初始化缓冲区的资源管理对象。 |
| `TQue` | C++ API | 管理同类 Tile Buffer 的先进先出队列，并在流水线阶段之间传递 `LocalTensor`。 |
| 队列深度 | Queue depth | 同一时刻队列能够保留的同类 Buffer 数；双缓冲的深度为 `2`。 |
| Profiling | 性能分析 | 收集 Kernel 耗时和硬件指标，用实际数据判断 Block 数、Tile 长度或双缓冲是否带来收益。 |

#### A.1.5 练习环境

| 术语 | 英文 / 代码 | 解释 |
| --- | --- | --- |
| TensorOJ | TensorOJ | 在线算子评测环境。它根据题目模板调用提交的 `kernel.asc`，并用测试数据校验输出。 |
| `run_kernel` | TensorOJ 入口 | TensorOJ 模板调用的 Host 端入口函数。函数签名是题目与提交代码之间的接口约定。 |
| 张量元信息 | `TensorGroupInfo` / `TensorInfo` | 描述输入输出张量数量、形状和数据类型的信息。 |
| `availableCoreNum` | 可用向量核数 | 运行时报告的可用向量核数量，可作为选择 Block 数时的参考。 |

#### A.1.6 AI Core 硬件与异步执行

| 术语 | 英文 / 代码 | 解释 |
| --- | --- | --- |
| Scalar 单元 | Scalar | 负责循环、地址计算和指令投递的标量控制单元，不承担大批量向量计算。 |
| Vector 单元 | Vector Unit / AIV | 在 UB 中执行 `Add`、`Mul`、`Cast` 等向量指令的计算单元。 |
| Cube 单元 | Cube Unit / AIC | 面向矩阵乘法、卷积等高密度二维计算的矩阵计算单元。 |
| AIV | AI Vector Core | 以向量计算通路为主的 AI Core 资源，本书的 Add、Reduce 与 Dot 主要使用这一通路。 |
| AIC | AI Cube Core | 以矩阵计算通路为主的 AI Core 资源，主要服务于 MatMul 与卷积。 |
| MTE | Memory Transfer Engine | 在 GM 与片上存储之间、或片上不同存储层之间执行 DMA 搬运的引擎。 |
| MTE1 | `PIPE_MTE1` | 通常负责 AIC 路径中 L1 与 L0A/L0B 等片上 Buffer 之间的数据搬运。 |
| FixPipe | `PIPE_FIX` | 主要服务于 Cube 结果格式转换、量化或写回等处理通路。 |
| `PIPE_S` | Scalar pipeline | Scalar 控制与标量指令的队列。 |
| `PIPE_V` | Vector pipeline | Vector 向量指令的队列。 |
| `PIPE_M` | Cube pipeline | Cube 矩阵指令的队列。 |
| `PIPE_MTE2` | MTE2 pipeline | GM 向 UB 搬入数据的异步搬运队列。 |
| `PIPE_MTE3` | MTE3 pipeline | UB 向 GM 写回数据的异步搬运队列。 |
| 指令投递 | Dispatch | Scalar 将指令放入对应硬件队列的动作；投递完成不代表指令已经执行完成。 |
| 异步指令流 | Asynchronous instruction stream | Scalar、Vector 和 MTE 的队列独立推进，因此搬运与计算可以重叠；数据依赖仍需由队列、事件或 Buffer 隔离保证。 |
| 数据依赖 | Data dependency | 后续操作必须等待前序数据就绪的关系，例如数据搬入 UB 完成后才能执行 Vector 计算。 |

#### A.1.7 归约与跨核合并

| 术语 | 英文 / 代码 | 解释 |
| --- | --- | --- |
| 归约 | Reduce | 将多个元素按某种规则折叠为更少元素的计算，例如 Reduce Sum 将 $N$ 个元素求和为一个标量。 |
| Reduce Sum | `ReduceSum` | 对输入元素求和的归约算子，数学形式为 $y = \sum_i x_i$。 |
| 局部规约 | Local reduction | 一个 Tile 或一个 Block 内部完成的规约，结果仍需要继续与其他局部和合并。 |
| 局部和 | Local sum | 当前 Tile 或当前 Block 处理数据的累加结果，例如 `sumLocal`。 |
| 树状规约 | Tree Reduction | 通过多轮对半相加逐步收敛数据的并行归约方式；64 个元素可在 6 轮内收敛为 1 个结果。 |
| `WholeReduceSum` | Ascend C API | 在 UB 内对一个 `LocalTensor` 执行全量求和规约的高阶接口。 |
| `sharedTmpBuffer` | 临时规约空间 | `WholeReduceSum` 在树状折叠与数据转置过程中使用的无类型辅助 UB 空间。 |
| `GetWholeReduceSumMinTmpSize` | Ascend C API | 根据已配置的输入、输出 Buffer 查询 `WholeReduceSum` 所需最小临时空间字节数的接口。 |
| `TBuf` | C++ API | 在 UB 中申请不参与入队、出队调度的固定 Buffer；适合 `sumBuf`、`tmpBuffer` 等长期存在的局部状态。 |
| `sumBuf` | Local sum buffer | 保存跨 Tile 持续累加结果的 UB Buffer，通常以 `TBuf` 管理。 |
| 原子加 | Atomic Add | 多个 Block 访问同一 GM 地址时，将读、加、写作为不可分割事务执行的机制。 |
| `SetAtomicAdd` | Ascend C API | 将 MTE 的后续写回操作切换为原子加模式；通常与一次 `DataCopy` 配对使用。 |
| `SetAtomicSub` | Ascend C API | 关闭 MTE 原子加状态并恢复默认写回模式的接口写法之一，具体可用接口取决于平台版本。 |
| Read-Modify-Write | RMW | 原子写回时先读取 GM 旧值、与新值相加、再写回的硬件事务。 |
| 写后写冲突 | Write-after-Write, WAW | 多个 Core 同时写同一地址时，后写结果覆盖前写结果的竞争问题。 |
| Two-Stage 规约 | Two-Stage Reduction | 先让各 Block 写入独立 WorkSpace，再发起第二次规约汇总局部和的跨核规约方案。 |
| WorkSpace | Workspace | 算子额外申请的 GM 工作区，用于存储局部结果、中间张量或二阶段规约的输入。 |

#### A.1.8 Dot、尾块与混合精度

| 术语 | 英文 / 代码 | 解释 |
| --- | --- | --- |
| 点积 | Dot product | 两个等长向量对应元素相乘后再求和，形式为 $y = \sum_i (x_{1,i} \times x_{2,i})$。 |
| 算子融合 | Operator fusion | 将原本独立的多个操作放入同一 Kernel、同一数据流执行；Dot 将 `Mul` 与 `WholeReduceSum` 融合。 |
| `Mul` | Ascend C API | 对两个 UB 中的 `LocalTensor` 执行逐元素向量乘法。 |
| 中间结果 Buffer | Intermediate buffer | 保存 Cast 结果或逐元素乘积的 UB 空间，例如 `x1Fp32Buf`、`x2Fp32Buf` 与 `mulBuf`。 |
| Tail Block | Tail block | 不足完整对齐切片长度的末尾数据段，需要单独处理以避免越界或错误累加。 |
| Core Tail | Core tail | Block 切分后无法均匀分给各 Block 的剩余元素，通常由最后一个 Block 负责。 |
| Tile Tail | Tile tail | 一个 Block 内最后一个 Tile 中不足 `TILE_LENGTH` 的有效元素。 |
| `actualLength` | Valid element count | 当前 Tile 中真正参与计算的有效元素数。 |
| `copyLength` | Aligned copy length | 为满足 MTE 对齐要求，将 `actualLength` 向上补齐后的搬运元素数。 |
| Padding | 填充区 | 为满足物理对齐而额外占用的元素位置；在归约前应确保这些位置不影响结果。 |
| 脏数据 | Stale data | UB 中未初始化或前一轮遗留的数据；若被误参与计算会污染结果。 |
| 对齐 | Alignment | 地址或长度满足硬件最小物理粒度的约束。本书中 MTE 和 Vector 常见的最小搬运/计算粒度为 `32 B`。 |
| 向上对齐 | Round up alignment | 将长度增大到不小于原值的最近对齐倍数，例如 FP32 的 `(actualLength + 7) & ~7U`。 |
| 向下对齐 | Round down alignment | 将长度减小到不大于原值的最近对齐倍数，例如 `(length / blockNum) & ~7U`。 |
| 掩码 | Mask | 控制 Vector 指令中哪些 Lane 参与本次计算的状态，用于安全处理 Tail。 |
| Lane | 向量通道 | Vector 指令中能够并行处理一个元素的位置。 |
| `SetVectorMask` | Ascend C API | 设置当前 Vector 指令可参与计算的有效 Lane 范围。 |
| `ResetMask` | Ascend C API | 将 Vector Mask 恢复为默认全通道有效状态。 |
| `Duplicate` | Ascend C API | 用同一个标量填充一段 `LocalTensor`；常用于初始化累加器或将 Padding 区清零。 |
| 类型转换 | Cast | 将 Tensor 元素从一种数据类型转换为另一种数据类型的过程。 |
| `Cast` | Ascend C API | 在 UB 中执行向量化类型转换的接口。 |
| `CastMode::CAST_NONE` | Cast mode | 不额外引入缩放、舍入策略等特殊处理时使用的基础转换模式。 |
| 混合精度 | Mixed precision | 在同一算子中使用多种数值类型，例如输入为 FP16/INT8、计算与累加提升为 FP32。 |
| FP16 | `float16` / `half` | 16 位浮点数，每个元素占 `2 B`；`32 B` 对齐时对应 16 个元素。 |
| FP32 | `float32` / `float` | 32 位浮点数，每个元素占 `4 B`；`32 B` 对齐时对应 8 个元素。 |
| INT8 | `int8_t` | 8 位整数，每个元素占 `1 B`；`32 B` 对齐时对应 32 个元素。 |
| INT32 | `int32_t` | 32 位整数，每个元素占 `4 B`；常作为 INT8 到 FP32 之间的中间提升类型。 |
| 类型提升 | Type promotion | 为扩大数值范围或兼容后续计算，将低精度或整数数据转换为更高精度类型的过程。 |
| 最小公倍数对齐 | LCM alignment | 同时存在多种类型时，选取各自对齐元素数的最小公倍数作为切片单位；INT8、FP16、FP32 对应为 $\operatorname{LCM}(32,16,8)=32$。 |

### A.2 第一章自测答案

#### A.2.1 1.1 认识昇腾算子开发

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 1.1-Q1 | B | AI Core 是 NPU 内执行向量、矩阵等设备侧计算的独立单元。 |
| 1.1-Q2 | C | `block_idx` 是逻辑任务编号，用于让同一份 Kernel 定位不同数据片段。 |
| 1.1-Q3 | B | NPU 算子需要显式组织 GM 与 UB 之间的数据搬运。 |

#### A.2.2 1.2 实现第一个 Add 算子

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 1.2-Q1 | B | $172032 / 16 = 10752$。 |
| 1.2-Q2 | C | Vector 单元在 UB 中计算，GM 保存完整输入与输出。 |
| 1.2-Q3 | B | 启动配置中的第一个参数指定逻辑 Block 数。 |

#### A.2.3 1.3 基于 C API 实现 Add 算子

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 1.3-Q1 | A | 转换后，`+ offset` 以一个 `float32` 元素为步长。 |
| 1.3-Q2 | B | DMA 搬运接口用字节数，向量计算接口用元素个数。 |
| 1.3-Q3 | B | `8` 个 Block 时三段 UB 缓冲区约占 `252 KB`，没有安全余量。 |

#### A.2.4 1.4 基于 C++ API 实现 Add 算子

| 题号 | 答案 | 关键理由 |
| 1.4-Q1 | B | `SetGlobalBuffer` 只建立带类型的 GM 视图，不触发搬运。 |
| 1.4-Q2 | A | UB 的 `Alloc` 需要在编译时确定分配尺寸。 |
| 1.4-Q3 | B | `LocalTensor<float>` 与 `GlobalTensor<float>` 已携带元素类型。 |

#### A.2.5 1.5 Add Medium 的 Tile 分块

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 1.5-Q1 | B | Block 负责空间上的多核分工，Tile 负责单核中的分批处理。 |
| 1.5-Q2 | C | $65536 / 8192 = 8$。 |
| 1.5-Q3 | C | 单缓冲下预读会覆盖当前 Tile 正在使用的 UB 数据。 |

#### A.2.6 1.6 Add Medium 的双缓冲流水线

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 1.6-Q1 | A | 三条物理管道分别对应搬入、计算与写回。 |
| 1.6-Q2 | B | 队列深度 `2` 表示准备两块可轮换的 Buffer。 |
| 1.6-Q3 | C | 双缓冲通过两套隔离的 UB 工作区避免数据竞争。 |

### A.3 第二章自测答案

#### A.3.1 2.2 Reduce Sum Easy

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 2.2-Q1 | B | 逻辑结果只有一个元素，但 Vector 与 MTE 的片上读写、搬运需要满足 `32 B` 对齐。 |
| 2.2-Q2 | C | 临时空间查询依赖已配置的输入、输出队列，查询后才能初始化 `tmpBuffer`。 |
| 2.2-Q3 | B | 临时空间由 API 按字节数查询，是不承载业务类型的辅助 Buffer。 |

#### A.3.2 2.3 Reduce Sum Medium

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 2.3-Q1 | B | `sumBuf` 保存跨 Tile 持续存在的局部和，不需要队列式流转。 |
| 2.3-Q2 | C | `Duplicate` 是 Vector 填充指令，用于初始化 UB 中的累加器。 |
| 2.3-Q3 | A | 当前 Tile 的规约结果会立刻被局部累加器消费，单个输出 Buffer 即可复用。 |

#### A.3.3 2.4 Reduce Sum Hard

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 2.4-Q1 | B | 每个逻辑 Block 根据自己的编号定位到不同的 GM 输入区间。 |
| 2.4-Q2 | C | MTE 以原子读改写事务将多个 Core 的局部和安全合并到同一 GM 地址。 |
| 2.4-Q3 | B | Two-Stage 先让各 Core 无竞争地写入独立 WorkSpace，再做第二次汇总。 |

### A.4 第三章自测答案

#### A.4.1 3.1 Dot Easy

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 3.1-Q1 | B | 非整除切分产生的剩余元素必须由一个 Block 接管，示例中由最后一个 Block 处理。 |
| 3.1-Q2 | B | Mask 只允许有效 Lane 执行乘法，避免尾部 Padding 参与计算。 |
| 3.1-Q3 | C | Mask 仅阻止写入，先清零乘积 Buffer 才能保证对齐后的 Padding 不影响规约。 |

#### A.4.2 3.2 Dot Medium

| 题号 | 答案 | 关键理由 |
| --- | --- | --- |
| 3.2-Q1 | B | INT8、FP16 与 FP32 在 `32 B` 物理粒度下的对齐元素数分别为 32、16、8，其最小公倍数为 32。 |
| 3.2-Q2 | A | 示例使用 INT8 $\rightarrow$ INT32 $\rightarrow$ FP32 的两段 Cast 链完成类型提升。 |
| 3.2-Q3 | C | `x2Int32Buf` 保存的是 INT32 中间结果，必须按 `sizeof(int32_t)` 分配。 |
