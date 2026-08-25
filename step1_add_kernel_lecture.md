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

- **Host Memory（HM）**：位于 CPU 一侧，由 Host 程序申请和访问。它保存准备送入设备的输入数据，以及从设备取回的计算结果。Device 端的 Kernel 不能直接读取这里的数据。
- **Global Memory（GM）**：位于 NPU 设备侧，容量较大，保存完整的输入 `x`、`y` 和输出 `z`。多个 AI Core 都可以访问 GM，因此它是设备侧各个计算任务共享的数据位置。
- **Local Memory（本例为 UB）**：位于单个 AI Core 内部的片上存储。它的容量较小、访问速度更高，只保存当前 AI Core 正在处理的一小段数据；其他 AI Core 不能直接使用这块本地空间。Unified Buffer（UB）就是本例使用的 Local Memory。

这三层存储的容量、访问速度和可见范围不同。后面的 Kernel 代码会显式控制数据从一层移动到另一层，而不是由语言运行时自动完成。

一次 Kernel 启动可以配置多个 Block。运行时将这些 Block 调度到可用的 AI Core 上；多个 Block 可以并行执行，但不需要假设某个 Block 永远绑定某个固定 Core。

本例将长度为 `172032` 的向量均匀分成 `16` 段，每个 Block 处理 `10752` 个元素：

```text
Block 0 : 元素 [0, 10752)
Block 1 : 元素 [10752, 21504)
...
Block 15: 元素 [161280, 172032)
```

#### 三、一个 Block 内部的 Add 计算

认识三个存储层后，再看一次 Add 的完整数据路径：`Host Memory -> GM -> UB -> 计算 -> UB -> GM -> Host Memory`。

Host 先将输入从 Host Memory 复制到 Global Memory。随后 Device 端的 Kernel 只在 Global Memory 与 AI Core 的 UB 之间移动数据：一个 Block 从 GM 读取 `x` 和 `y`，将它们搬入 UB，在 UB 中完成 `x + y`，再把结果写回 GM 中的 `z`。

```text
x, y (Global Memory)
        |  搬入
        v
x_local, y_local (UB)
        |  加法
        v
z_local (UB)
        |  搬出
        v
z (Global Memory)
```

这条路径对应的 Device 端核心代码只有三步：

```cpp
asc_copy_gm2ub(x_local, x_gm, block_length * sizeof(float));
asc_copy_gm2ub(y_local, y_gm, block_length * sizeof(float));
asc_sync();

asc_add(z_local, x_local, y_local, block_length);
asc_sync();

asc_copy_ub2gm(z_gm, z_local, block_length * sizeof(float));
asc_sync();
```

`asc_copy_gm2ub` 是“从 Global Memory 读入 UB”；`asc_copy_ub2gm` 是“把 UB 中的结果写回 Global Memory”。

`asc_add` 对应 AI Core 上的向量计算操作。它不是由程序员逐元素编写 `for` 循环完成加法，而是向量计算单元执行 SIMD 加法：一条向量指令会并行处理一批 `float32` 元素。这样，`x_local` 和 `y_local` 中连续的大量元素可以被批量相加，计算吞吐量远高于按单个元素逐次执行标量加法。

`asc_sync()` 保证前一阶段的操作完成后，再进入下一阶段。

#### 四、Device 端 Kernel

后缀名为 `*.asc` 的文件可以同时包含 Device 端和 Host 端代码。先看 Device 端的完整 Kernel：

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

Kernel 先根据 `block_idx` 计算当前 Block 负责的地址范围：

```cpp
constexpr uint32_t block_length = TOTAL_LENGTH / NUM_BLOCKS;

__gm__ float* x_gm = x + block_idx * block_length;
__gm__ float* y_gm = y + block_idx * block_length;
__gm__ float* z_gm = z + block_idx * block_length;
```

`__vector__ __global__` 是 Ascend C 的语言扩展：`__global__` 声明可由 Host 启动的 Device 端 Kernel，`__vector__` 表示该 Kernel 在 AI Core 的向量计算单元上执行。这里 `TOTAL_LENGTH = 172032`、`NUM_BLOCKS = 16`，因此 `block_length = 10752`。

`block_idx` 是编译器提供的内置系统变量，不需要声明或作为参数传入。运行时启动 16 个 Block 时，会让它们分别读到 `0` 到 `15` 的逻辑编号。它表示当前 Block 的编号，而不是固定 AI Core 的物理编号。例如，`block_idx = 3` 时，三个指针都会偏移 `3 × 10752` 个元素，因此该 Block 处理 `x[32256:43008]`、`y[32256:43008]`，并将结果写到 `z[32256:43008]`。

随后在 UB 中声明输入和输出缓冲区：

```cpp
__ubuf__ float x_local[block_length];
__ubuf__ float y_local[block_length];
__ubuf__ float z_local[block_length];
```

三段 UB 缓冲区分别保存当前 Block 负责的 `10752` 个 `float` 元素。

有了这三块 UB 空间后，Kernel 按以下顺序完成“搬入、相加、搬出”三步：

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

#### 五、Host 端调用 Kernel

Device 端只描述“一个 Block 拿到数据后怎样计算”；Host 端负责准备数据并发起这次执行。`Stream` 是运行时维护的任务序列：同一 Stream 中的操作按提交顺序建立依赖，Host 可在之后同步等待它们完成。

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

Host 端依次创建 Stream，分配内存，拷贝输入，启动 Kernel，同步并取回输出。代码片段只展示 Add 调用相关部分；运行时初始化、设备选择等步骤可置于此处之前。

创建 Stream 后，分别为三个 Device 数组和一个 Host 输出数组申请内存：

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

Kernel 启动语句如下：

```cpp
add_custom<<<NUM_BLOCKS, nullptr, stream>>>(xDevice, yDevice, zDevice);
```

其中 `NUM_BLOCKS = 16`，因此 NPU 启动 16 个 Block；`nullptr` 是当前启动接口中的保留配置项，本例不额外配置它；`stream` 指定任务提交的运行时流。尖括号后的 `(xDevice, yDevice, zDevice)` 才是 Kernel 实际接收的三个 GM 地址。

执行结束后，Host 端同步并取回结果：

```cpp
aclrtSynchronizeStream(stream);
aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize,
            ACL_MEMCPY_DEVICE_TO_HOST);
```

#### 六、UB 容量补充

Atlas A2 训练系列产品（Ascend 910B）中，单个 AI Core 的 UB 容量可按约 `256 KB` 理解。一个 Block 需要在 UB 中同时保存 `x_local`、`y_local`、`z_local` 三个数组：

| UB 数组 | 元素数 | 单元素大小 | 占用 |
| --- | ---: | ---: | ---: |
| `x_local` | 10752 | 4 B | 43008 B = 42 KB |
| `y_local` | 10752 | 4 B | 43008 B = 42 KB |
| `z_local` | 10752 | 4 B | 43008 B = 42 KB |
| 合计 | 32256 | - | 129024 B = 126 KB |

UB 总容量为 `256 KB`，但不能将静态缓冲区配置到接近该上限：运行时还需要保留空间。这里三段 UB 缓冲区总计 `126 KB`，可为系统预留区及运行时资源保留充足余量。每个 Block 将自己负责的 10752 个 `float32` 输入元素及其结果同时放入 UB 中完成计算。

#### 七、数据流向补充

计算路径为：

```text
Host Memory
    |  aclrtMemcpy
    v
Device Global Memory
    |  asc_copy_gm2ub
    v
UB: x_local, y_local
    |  asc_add
    v
UB: z_local
    |  asc_copy_ub2gm
    v
Device Global Memory
    |  aclrtMemcpy
    v
Host Memory: zHost
```

#### 八、算子编译与运行

将 `add_172032.asc` 放入工程目录后，可使用 CMake 编译：

```cmake
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(add_172032 LANGUAGES ASC CXX)

add_executable(add_172032 add_172032.asc)

target_compile_options(add_172032 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
```

编译与运行命令：

```bash
mkdir -p build
cd build
cmake ..
make -j
./add_172032
```

`--npu-arch` 必须与实际使用的 NPU 架构匹配。`dav-2201` 对应 Atlas A2 / Ascend 910B 系列。

#### 九、参考资料

- [基于 SIMD 编程的 Add 算子快速入门](https://asc.gitcode.com/guide/%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B/%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8/%E5%9F%BA%E4%BA%8ESIMD%E7%BC%96%E7%A8%8B/Add%E7%AE%97%E5%AD%90%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8.html)
- `Add(1).zip` 中的题目说明：`Add/Add.md`
