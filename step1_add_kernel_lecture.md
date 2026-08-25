### 第二节 编写第一个 Add Kernel

本节围绕一次 Add 的完整数据路径展开：先认识昇腾设备中数据保存与计算发生的位置，再用代码控制元素在不同存储位置之间移动，调用向量计算单元完成加法，最后写回结果。通过这条路径，可以看到算子代码如何同时组织数据移动与向量计算。

#### 一、Add 算子功能介绍

输入为两个形状相同的一维 `float32` 向量 `x`、`y`，输出为向量 `z`。计算逻辑为逐元素相加：

$$
z_i = x_i + y_i, \quad i \in [0, 172032)
$$

本例使用的变量如下。

| 变量 | 说明 |
| --- | --- |
| 输入 `x` | 一维 `float32` 向量，形状为 `(172032,)` |
| 输入 `y` | 一维 `float32` 向量，形状为 `(172032,)` |
| 输出 `z` | 一维 `float32` 向量，形状为 `(172032,)` |
| `x`、`y` 取值范围 | `[-1.0, 1.0]` |

输入长度固定为 `172032`，不考虑其他长度、广播或多数据类型情形。

`float32` 是 32 位单精度浮点数，每个元素占 `4 B`。长度和数据类型在本节预先确定，因此可以先把注意力放在数据如何被划分、搬运和计算上。

#### 二、昇腾算子的内存与计算架构

有别于常见 C++ 程序主要面对统一的主存抽象，昇腾算子开发需要显式处理多个层级的存储位置。数据存放在哪里、何时被搬运到计算核心附近，会直接影响 Kernel 能否访问数据、数据搬运的开销，以及计算硬件是否能够充分工作。掌握这套内存与计算关系，是编写高效算子的前提。

一次 Add 计算中，Host 负责准备数据和启动任务；NPU 设备侧保存输入输出，并由 AI Core 执行 Kernel。下图给出了数据与计算单元的关系：

![Host、GM、AI Core 与 UB 的关系](assets/architecture/host-gm-ub-ai-core.png)
*Host Memory 保存 CPU 侧输入输出；数据先复制到设备侧 GM，再由各个 AI Core 将自己负责的数据搬入 UB 计算。*

图中出现的三类存储位置分别承担不同角色：

- **Host Memory（HM）**：位于 CPU 一侧，由 Host 程序申请和访问，通常用于初始化输入数据、发起设备任务以及检查最终结果。它的容量可以很大，但对 NPU Kernel 不可直接见；若希望 NPU 处理这里的数据，Host 必须先将数据传入设备侧。
- **Global Memory（GM）**：位于 NPU 设备侧，是所有 AI Core 都可访问的共享大容量内存。本例完整的输入 `x`、`y` 和输出 `z` 都保存在这里。GM 的容量远大于单个 AI Core 的片上空间，适合保存完整张量；但离计算单元更远，访问速度低于片上 Local Memory。
- **Local Memory（本例为 UB）**：位于单个 AI Core 内部，是紧挨计算单元的片上存储。它容量较小，但访问速度很高，适合暂存当前正在计算的一小段输入和输出；其他 AI Core 不能直接使用这块本地空间。Unified Buffer（UB）就是本例使用的 Local Memory。

这三层存储的容量、访问速度和可见范围都不同。算子代码不能把它们当成同一种内存来使用：完整数据通常保存在 GM，真正参与当前计算的数据则需要进入 Local Memory。后面的 Kernel 代码会显式控制这种移动，而不是由语言运行时自动完成。

本例中，每个 Block 要在 UB 中同时暂存 `x`、`y` 和 `z` 的一段数据。每段长度为 `10752` 个 `float32` 元素，因此单段占用 `10752 × 4 B = 43008 B`，三段合计为 `129024 B`，即 `126 KB`。这低于单个 AI Core 约 `256 KB` 的 UB 容量，并为运行时所需空间保留了余量。

#### 三、昇腾算子设计的宏观拆解

在编写 Kernel 代码前，需要先确定算子的执行方案。对于本节的 Add，设计过程可以归结为两个问题：**任务如何切分，交给多个 AI Core 并行处理？数据如何在一个 AI Core 内流动并完成计算？** 前者决定空间上的并行分工，后者决定一段数据在时间上的处理闭环。

##### 1. 空间维度：多核并行与 Block 级数据切分

昇腾 NPU 具有多个可以独立工作的 AI Core。为了让整个输入不由单个 Core 顺序完成，Kernel 会把任务划分为多个 Block。Block 是同一份 Kernel 的逻辑执行任务；运行时将它们调度到可用的 AI Core 上执行。多个 Block 可以并行执行，但不能假设某个 Block 永远绑定某一个固定的 AI Core。

本例中，一个 `float32` 向量共有 `172032` 个元素，单个向量占用 `172032 × 4 B = 688128 B`，即 `672 KB`。将它均匀划分为 `16` 个 Block 后，每个 Block 负责 `10752` 个元素：

```text
完整向量：172032 个 float32 元素
├── Block 0 : 元素 [0, 10752)       -> 43008 B = 1344 × 32 B
├── Block 1 : 元素 [10752, 21504)   -> 43008 B = 1344 × 32 B
│   ...
└── Block 15: 元素 [161280, 172032) -> 43008 B = 1344 × 32 B
```

这种切分同时满足两项基本要求：

- **负载均衡**：16 个 Block 的工作量相同。运行时可以将它们分配给空闲的 AI Core，使各 Core 尽量同步完成，而不是让部分 Core 长时间等待。
- **数据对齐**：每个 Block 的一段 `float32` 数据占用 `10752 × 4 B = 43008 B = 1344 × 32 B`。数据搬运和向量计算按固定数据块组织，连续且满足 `32 B` 整数倍的长度能避免额外的尾部处理。

这里的切分是 **Block 级切分**，解决“多个 AI Core 如何分工”。第五节介绍的 Tile 切分则发生在单个 Block 内部：当一个 Block 负责的数据无法同时放进 UB 时，再将它拆成若干小段循环处理。它解决的是“单个 Core 的片上空间如何使用”。本例每个 Block 的三段 UB 缓冲区共占 `126 KB`，能放入约 `256 KB` 的 UB，因此不需要在本节继续进行 Tile 切分。

##### 2. 时间维度：单个 Block 的数据闭环

确定了每个 Block 的负责范围后，还要确定这一段数据经过哪些位置，并在何处完成加法。完整输入由 Host 准备，设备侧 Global Memory 保存完整的 `x`、`y` 与 `z`；具体到一个 Block 时，当前片段会进入该 AI Core 的 UB，由向量计算单元完成加法，再沿相反方向写回。

```text
Host Memory：准备输入 x、y，并在结束后读取 z
    |  Host 与设备之间复制
    v
Global Memory：保存完整 x、y、z
    |  当前 Block 的 x、y 片段搬入
    v
UB：暂存当前 Block 的 x、y
    |  向量加法
    v
UB：暂存当前 Block 的 z
    |  当前 Block 的 z 片段写回
    v
Global Memory：得到完整 z
```

因此，Device 端 Kernel 的职责并不只是写出 `z[i] = x[i] + y[i]`：它还必须定位当前 Block 的数据范围，组织 GM 与 UB 之间的搬运，并在 UB 中调用向量计算。接下来的代码正是把这套宏观设计落实为可执行的接口调用。

#### 四、一个 Block 的执行计划

上一部分已经确定了宏观方案：16 个 Block 分工处理完整向量，每个 Block 对自己的片段完成一次数据闭环。落实到 Device 端时，一个 Block 需要完成三件事：**定位自己负责的数据范围、准备局部工作区、安排搬入与计算及写回的先后关系**。下面的 Kernel 就是这一执行计划的完整表达。

本节使用 **Ascend C SIMD C API**。它提供 `asc_copy_gm2ub`、`asc_add` 等接口，并允许用 `__ubuf__` 声明 UB 局部缓冲区。后续章节会出现另一种 C++ Tensor API 风格；两种接口对局部内存的表达方式不同，不能混用。

```cpp
#include <cstdint>

constexpr uint32_t TOTAL_LENGTH = 172032;
constexpr uint32_t NUM_BLOCKS = 16;

__vector__ __global__ void add_custom(__gm__ float* x,
                                      __gm__ float* y,
                                      __gm__ float* z)
{
    asc_init();

    constexpr uint32_t block_length = TOTAL_LENGTH / NUM_BLOCKS;

    // 当前 Block 在 Global Memory 中负责的数据起始位置。
    __gm__ float* x_gm = x + block_idx * block_length;
    __gm__ float* y_gm = y + block_idx * block_length;
    __gm__ float* z_gm = z + block_idx * block_length;

    // 在片上 UB 中为 x、y、z 分别分配一段连续空间。
    __ubuf__ float x_local[block_length];
    __ubuf__ float y_local[block_length];
    __ubuf__ float z_local[block_length];

    // 将输入数据从 Global Memory 搬运到 UB。
    asc_copy_gm2ub(x_local, x_gm, block_length * sizeof(float));
    asc_copy_gm2ub(y_local, y_gm, block_length * sizeof(float));
    asc_sync();

    // 调用 SIMD C API，完成 z_local[i] = x_local[i] + y_local[i]。
    asc_add(z_local, x_local, y_local, block_length);
    asc_sync();

    // 将 UB 中的结果写回 Global Memory。
    asc_copy_ub2gm(z_gm, z_local, block_length * sizeof(float));
    asc_sync();
}
```

##### 1. 定位当前 Block 的数据范围

所有 Block 执行的是同一份 Kernel 代码，那么它们如何处理不同的数据？答案是运行时为每个 Block 提供不同的逻辑编号 `block_idx`。代码用这个编号计算统一的偏移量，让三个输入输出指针同步移动到当前 Block 的起点：

```cpp
constexpr uint32_t block_length = TOTAL_LENGTH / NUM_BLOCKS;

__gm__ float* x_gm = x + block_idx * block_length;
__gm__ float* y_gm = y + block_idx * block_length;
__gm__ float* z_gm = z + block_idx * block_length;
```

`__global__` 声明可由 Host 启动的 Device 端 Kernel，`__vector__` 表示该 Kernel 由 AI Core 的向量计算单元执行。这里 `TOTAL_LENGTH = 172032`、`NUM_BLOCKS = 16`，因此 `block_length = 10752`。

`block_idx` 是编译器提供的内置系统变量，不需要声明或作为参数传入。运行时启动 16 个 Block 时，会让它们分别读到 `0` 到 `15` 的逻辑编号。它表示当前 Block 的编号，而不是固定 AI Core 的物理编号。例如，`block_idx = 3` 时，三个指针都会偏移 `3 × 10752` 个元素，因此该 Block 处理 `x[32256:43008]`、`y[32256:43008]`，并将结果写到 `z[32256:43008]`。

##### 2. 准备当前 Block 的局部工作区

当前 Block 已经知道自己该读写 GM 的哪一段，但向量计算不能直接把 GM 当作工作区。它需要同时保留两个输入片段和一个输出片段，因此在当前 AI Core 的 UB 中准备三段局部缓冲区：

```cpp
__ubuf__ float x_local[block_length];
__ubuf__ float y_local[block_length];
__ubuf__ float z_local[block_length];
```

三段 UB 缓冲区分别保存当前 Block 负责的 `10752` 个 `float` 元素。三个缓冲区使用相同的长度，保证 `x_local[i]`、`y_local[i]` 和 `z_local[i]` 始终对应原向量中的同一个逻辑位置。

##### 3. 按数据依赖安排搬入、计算与写回

局部工作区准备好后，执行顺序由数据依赖决定：必须先搬入 `x`、`y`，才能计算 `z`；必须先得到 `z`，才能将它写回 GM。对应的代码如下：

```cpp
asc_copy_gm2ub(x_local, x_gm, block_length * sizeof(float));
asc_copy_gm2ub(y_local, y_gm, block_length * sizeof(float));
asc_sync();

asc_add(z_local, x_local, y_local, block_length);
asc_sync();

asc_copy_ub2gm(z_gm, z_local, block_length * sizeof(float));
asc_sync();
```

`asc_copy_gm2ub` 和 `asc_copy_ub2gm` 的长度单位是字节，因此参数为 `block_length * sizeof(float)`；`asc_add` 的长度单位是元素个数，因此参数直接是 `block_length`。

每个 `asc_sync()` 都在确认前一阶段产生的数据已经可被下一阶段使用：两次输入搬运完成后，Vector 单元才读取 `x_local`、`y_local`；向量加法完成后，搬运单元才写回 `z_local`。这是一种按阶段顺序推进的正确执行计划。后续讨论性能优化时，会在保持这些依赖正确的前提下，让不同数据块的搬运与计算更紧密地衔接。

#### 五、Host 如何提交一次 Add 任务

Device 端描述的是“一个 Block 拿到自己的数据后如何执行”；Host 端则负责把这次计算组织成可以提交给 NPU 的任务。它需要完成四项工作：准备输入输出存储、把输入传到设备、指定 Kernel 的执行配置、等待并取回结果。`Stream` 是运行时维护的有序任务队列，Host 将内存操作和 Kernel 启动提交到其中，再在需要结果时同步等待。

```cpp
#include <acl/acl.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr uint32_t TOTAL_LENGTH = 172032;
constexpr uint32_t NUM_BLOCKS = 16;

int32_t main(int argc, char const* argv[])
{
    constexpr size_t totalByteSize = TOTAL_LENGTH * sizeof(float);

    std::vector<float> x(TOTAL_LENGTH);
    std::vector<float> y(TOTAL_LENGTH);
    for (uint32_t i = 0; i < TOTAL_LENGTH; ++i) {
        x[i] = static_cast<float>(i % 2001) / 1000.0f - 1.0f;
        y[i] = static_cast<float>((i * 7) % 2001) / 1000.0f - 1.0f;
    }

    // 创建运行时 Stream。
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 分配 Host 和 Device 内存，并将输入数据拷贝到 Device。
    float* xDevice = nullptr;
    float* yDevice = nullptr;
    float* zDevice = nullptr;
    float* zHost = nullptr;

    aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void**)&zHost, totalByteSize);

    aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize,
                ACL_MEMCPY_HOST_TO_DEVICE);

    // 启动 add_custom kernel。
    // NUM_BLOCKS: 启动的 Block 数量，这里为 16。
    // 第二个参数: 不使用动态 UB，这里为 nullptr。
    // stream: 运行时流。
    add_custom<<<NUM_BLOCKS, nullptr, stream>>>(xDevice, yDevice, zDevice);

    // 等待 add_custom kernel 执行完成。
    aclrtSynchronizeStream(stream);

    // 将结果从 Device Memory 拷贝回 Host Memory。
    aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize,
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool passed = true;
    for (uint32_t i = 0; i < TOTAL_LENGTH; ++i) {
        const float expected = x[i] + y[i];
        if (std::fabs(zHost[i] - expected) > 1e-6f) {
            std::printf("Mismatch at %u: got %.1f, expected %.1f\\n",
                        i, zHost[i], expected);
            passed = false;
            break;
        }
    }
    std::printf(passed ? "Add result is correct.\\n" : "Add result is incorrect.\\n");

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFreeHost(zHost);
    aclrtDestroyStream(stream);
    return passed ? 0 : 1;
}
```

这段程序展示的是一次 Add 任务的提交主干。下面不再按照代码出现的先后逐行翻译，而是从 Host 必须完成的三项关键决策来理解它。

在独立的工程程序中，调用这些运行时接口前还应完成 `aclInit(nullptr)` 和 `aclrtSetDevice(deviceId)`；资源释放后应调用 `aclrtResetDevice(deviceId)` 与 `aclFinalize()`。同时，应检查每个 ACL 接口的返回值。本节保留最小调用片段，以便将注意力集中在 Add 的数据准备、启动与结果回传上；这些运行时生命周期要求在实际工程中不可省略。

##### 1. 让输入数据对 NPU 可见

Host 首先为 `x`、`y`、`z` 准备设备侧存储，并将 CPU 侧生成的两个输入复制到设备。`zHost` 则用于在计算结束后接收结果：

```cpp
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);

aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void**)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMallocHost((void**)&zHost, totalByteSize);
```

输入数据通过 `aclrtMemcpy` 送入 Device Memory：

```cpp
aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize,
            ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize,
            ACL_MEMCPY_HOST_TO_DEVICE);
```

##### 2. 提交 Kernel 的执行配置

Kernel 启动语句是 Host 与 Device 的接口边界：

```cpp
add_custom<<<NUM_BLOCKS, nullptr, stream>>>(xDevice, yDevice, zDevice);
```

尖括号 `<<<NUM_BLOCKS, nullptr, stream>>>` 描述的是**如何执行**：启动 16 个 Block，不额外配置动态 UB，并将任务提交到指定 Stream。圆括号 `(xDevice, yDevice, zDevice)` 描述的是**处理什么数据**：Kernel 实际接收的三个 GM 地址。这个区分将“执行配置”和“数据参数”分开，是理解 Kernel 启动语法的关键。

##### 3. 等待结果并验证计算

Host 在需要读取 `z` 前同步 Stream，确保其中排在前面的 Kernel 已经完成；随后将设备侧的 `z` 复制回 `zHost` 并逐元素校验：

```cpp
aclrtSynchronizeStream(stream);
aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize,
            ACL_MEMCPY_DEVICE_TO_HOST);
```

#### 六、从源文件到可执行任务

这个工程同时包含 Host 侧 C++ 代码与 Device 侧 `.asc` Kernel。构建系统需要识别 Ascend C 源文件，选择目标 NPU 架构，并将两侧代码组织为同一个可执行程序。下面的 CMake 配置完成了这件事：

```cmake
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(add_172032 LANGUAGES ASC CXX)

add_executable(add_172032 add_172032.asc)

target_compile_options(add_172032 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
```

`LANGUAGES ASC CXX` 表示工程同时编译 Ascend C 与 C++；`find_package(ASC REQUIRED)` 引入相应的编译支持。`--npu-arch=dav-2201` 则指定 Device 端 Kernel 面向 Atlas A2 / Ascend 910B 对应的目标架构生成代码。目标架构必须与实际设备匹配，否则生成的 Device 程序无法正确运行。

编译与运行命令：

```bash
mkdir -p build
cd build
cmake ..
make -j
./add_172032
```

#### 七、延伸阅读

- [基于 SIMD 编程的 Add 算子快速入门](https://asc.gitcode.com/guide/%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B/%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8/%E5%9F%BA%E4%BA%8ESIMD%E7%BC%96%E7%A8%8B/Add%E7%AE%97%E5%AD%90%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8.html)
