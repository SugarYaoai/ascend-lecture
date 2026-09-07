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
