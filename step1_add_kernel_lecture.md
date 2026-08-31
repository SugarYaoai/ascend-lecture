### 第二节 编写第一个 Add Kernel（核函数）

在异构计算中，一段完整的算子程序分为两部分：运行在 NPU 芯片内部的底层计算代码称为 **Kernel（核函数）**，运行在 CPU 上、负责全局调度的程序称为 **Host 程序**。Host 程序像“指挥官”，负责在设备内存中开辟空间、把数据送进 NPU，并发号施令启动计算；Kernel 则像一线工人，是专门在 NPU 的 AI Core 内部执行的底层 C/C++ 代码。

启动 Kernel 时，NPU 会并行派生出多个逻辑任务（Block）。每个任务运行同一份 Kernel 代码，各自处理数据的一个切片。上一节介绍了 Host、GM、UB、MTE 与 Vector Unit 的分工；本节将这套架构落实到一个固定规格的 Add 算子：手算切片参数分配 Block，为每个 Block 规划 UB 内存预算，最后分别编写 Device 端 Kernel 与 Host 端调度代码，完成“搬入 -> 计算 -> 写回”的数据闭环。

#### 一、Add 算子功能介绍

输入为两个形状相同的一维 `float32` 向量 `x`、`y`，输出为向量 `z`。计算逻辑为逐元素相加：

$$
z_i = x_i + y_i, \quad i \in [0, 172032)
$$

本例使用的变量如下：

| 变量 | 说明 |
| --- | --- |
| 输入 `x` | 一维 `float32` 向量，形状为 `(172032,)` |
| 输入 `y` | 一维 `float32` 向量，形状为 `(172032,)` |
| 输出 `z` | 一维 `float32` 向量，形状为 `(172032,)` |
| `x`、`y` 取值范围 | `[-1.0, 1.0]` |

输入长度固定为 `172032`。这不是任意选择的数字：它可以被 `16` 个 Block 均匀整除，每个 Block 恰好处理 `10752` 个元素；对 `float32` 而言，这段数据的长度为 $10752 \times 4\text{ B} = 43008\text{ B} = 1344 \times 32\text{ B}$。固定规格让本节先聚焦数据划分、内存搬运与向量计算的主干闭环；后续章节会进一步拓展到动态长度与多数据类型。

#### 二、Add 算子的执行方案

编写 Kernel 代码前，需要先明确两个核心问题：多核之间如何拆分任务，也就是空间维度的分工；单核内部数据如何流动并完成计算，也就是时间维度的闭环。

##### （一）第一步：手算切片参数与 UB 内存预算

昇腾 NPU 包含多个可以独立工作的 AI Core。为了发挥多核并行能力，Kernel 需要把完整向量划分为多个逻辑 Block，运行时再将它们调度到空闲的 AI Core 上执行。

本例中，单个 `float32` 向量占用的显存为：

$$
172032 \times 4\text{ B} = 688128\text{ B} = 672\text{ KB}
$$

将它均匀切分为 `16` 个 Block 后，每个 Block 负责 `10752` 个元素：

```text
完整向量：172032 个 float32 元素
├── Block 0 : 元素 [0, 10752)       -> 43008 B = 1344 × 32 B
├── Block 1 : 元素 [10752, 21504)   -> 43008 B = 1344 × 32 B
│   ...
└── Block 15: 元素 [161280, 172032) -> 43008 B = 1344 × 32 B
```

这组参数同时满足三项底层物理限制：

- **负载均衡**：16 个 Block 的计算量完全一致，避免部分 AI Core 提前空闲、部分 Core 长时间耗时。
- **32B 物理对齐**：每个 Block 负责的数据量为 $10752 \times 4\text{ B} = 43008\text{ B} = 1344 \times 32\text{ B}$。数据按 `32 B` 整数倍连续组织，满足硬件搬运与向量计算的物理对齐要求。
- **UB 内存预算**：每个 Block 需要在片上 SRAM（UB）中为 `x`、`y`、`z` 申请三段局部缓冲区，总内存占用为：

$$
3 \times 10752 \times 4\text{ B} = 129024\text{ B} = 126\text{ KB}
$$

物理上单个 AI Core 的 UB 标称容量虽然约为 `256 KB`，但运行时框架和控制信息也会占用部分片上内存。把单核 UB 占用控制在 `126 KB`，约为一半容量，留出了充裕的安全余量。

**直觉思考：为什么要切成 16 个 Block？如果只切成 8 个会怎样？** 只启动 8 个 Block 时，每个 Block 处理的数据量会翻倍到 `21504` 个元素；三段 UB 缓冲区的占用随之变为：

$$
3 \times 21504 \times 4\text{ B} = 258048\text{ B} \approx 252\text{ KB}
$$

静态占用高达 `98.4%`，在编译阶段或运行时很容易触发 UB 资源溢出。因此，切分为 16 个 Block 同时兼顾了多核并行与片上内存安全。

##### （二）第二步：确定单核内部的“搬运-计算-写回”数据流

确定 Block 分配后，数据在单个 Core 内部按以下轨迹流动：

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

Device 端 Kernel 的核心使命，就是精准定位当前 Block 的数据切片，编排 GM 与 UB 之间的内存搬运，并在 UB 内触发向量计算。

#### 三、Device 端 Kernel（NPU 核函数）的具体编写

本节采用 Ascend C SIMD C API 实现。它使用 `asc_copy_gm2ub`、`asc_add` 等 C 函数接口，并允许用 `__ubuf__` 关键字分配 UB 局部工作区。

```cpp
#include <cstdint>
#include "c_api/asc_simd.h"

constexpr uint32_t TOTAL_LENGTH = 172032;
constexpr uint32_t NUM_BLOCKS = 16;

__vector__ __global__ void add_custom(__gm__ float* x,
                                      __gm__ float* y,
                                      __gm__ float* z)
{
    // 1. 初始化硬件环境。
    asc_init();

    constexpr uint32_t block_length = TOTAL_LENGTH / NUM_BLOCKS;

    // 2. 根据逻辑 Block 编号，计算当前任务在 GM 中的首地址偏移。
    __gm__ float* x_gm = x + block_idx * block_length;
    __gm__ float* y_gm = y + block_idx * block_length;
    __gm__ float* z_gm = z + block_idx * block_length;

    // 3. 在片上 SRAM（UB）中开辟局部工作区。
    __ubuf__ float x_local[block_length];
    __ubuf__ float y_local[block_length];
    __ubuf__ float z_local[block_length];

    // 4. 阶段一：数据搬入，GM -> UB。长度单位为 Bytes。
    asc_copy_gm2ub(x_local, x_gm, block_length * sizeof(float));
    asc_copy_gm2ub(y_local, y_gm, block_length * sizeof(float));
    asc_sync();

    // 5. 阶段二：片上向量计算。长度单位为元素个数。
    asc_add(z_local, x_local, y_local, block_length);
    asc_sync();

    // 6. 阶段三：结果写回，UB -> GM。长度单位为 Bytes。
    asc_copy_ub2gm(z_gm, z_local, block_length * sizeof(float));
    asc_sync();
}
```

##### （一）定位当前 Block 的数据首地址

所有 Block 执行的都是同一份 Kernel 代码。硬件通过内置系统变量 `block_idx`（取值 `0` 到 `15`）区分不同的逻辑 Block：

```cpp
constexpr uint32_t block_length = TOTAL_LENGTH / NUM_BLOCKS;

__gm__ float* x_gm = x + block_idx * block_length;
__gm__ float* y_gm = y + block_idx * block_length;
__gm__ float* z_gm = z + block_idx * block_length;
```

`__global__` 表示该函数是一个可由 Host 启动的 Kernel；`__vector__` 表示该 Kernel 由 AI Core 内的 Vector 计算单元执行。以 `block_idx = 3` 为例，三个指针同时偏移 $3 \times 10752 = 32256$ 个元素；该 Block 准确处理切片 $x[32256:43008]$ 和 $y[32256:43008]$，并将结果写入 $z[32256:43008]$。

##### （二）分配 UB 局部工作区

Vector 计算单元无法直接对 GM 显存进行数学运算，必须在 UB 片上空间分配三段工作区：

```cpp
__ubuf__ float x_local[block_length];
__ubuf__ float y_local[block_length];
__ubuf__ float z_local[block_length];
```

`__ubuf__` 语法看起来像普通的 C++ 局部数组，但它本质上是对硬件 UB 静态资源的映射分配。编译器会在编译期为整个 Kernel 规划好这三块片上内存，而不是在运行时动态申请栈空间。

##### （三）组装“搬运-计算-写回”流水线

数据依赖决定了代码的执行顺序：必须先将 `x` 和 `y` 搬入 UB，才能触发加法；必须等加法完全结束，才能将 `z` 写回 GM。在调用 SIMD C API 时，需要特别注意长度参数的单位差异：

| API 类型 | 代表接口 | 长度单位 | 对应硬件引擎 | 原因 |
| --- | --- | --- | --- | --- |
| 内存搬运 API | `asc_copy_gm2ub` / `asc_copy_ub2gm` | 字节数（Bytes） | MTE 引擎 | 搬运引擎只关心物理字节数，不关注具体数据类型。 |
| 向量计算 API | `asc_add` | 元素个数（Count） | Vector 单元 | 计算单元依托 `float*` 类型确定字节大小，只需知道处理多少个元素。 |

代码中的 `asc_sync()` 用于确认前一阶段的硬件流水线完全执行完毕，防止出现数据未搬完就提前计算，或未计算完就提前写回的竞态冲突。这个同步动作只在当前 Block 内部生效，不会阻塞其他 AI Core 的并行执行。

#### 四、Host 端任务提交与硬件调用

Host 端负责初始化内存、准备测试数据，并将计算任务打包提交给 NPU。任务提交通过 Stream（异步任务队列）进行管理。

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

    // 1. 在 CPU 侧准备测试数据。
    std::vector<float> x(TOTAL_LENGTH);
    std::vector<float> y(TOTAL_LENGTH);
    for (uint32_t i = 0; i < TOTAL_LENGTH; ++i) {
        x[i] = static_cast<float>(i % 2001) / 1000.0f - 1.0f;
        y[i] = static_cast<float>((i * 7) % 2001) / 1000.0f - 1.0f;
    }

    // 2. 创建异步任务队列（Stream）。
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 3. 申请 Device 显存与 Host 接收内存。
    float* xDevice = nullptr;
    float* yDevice = nullptr;
    float* zDevice = nullptr;
    float* zHost = nullptr;

    aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void**)&zHost, totalByteSize);

    // 4. 将输入数据从 CPU 内存拷贝至 Device 显存。
    aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize,
                ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 在 Stream 上异步启动 Kernel。
    add_custom<<<NUM_BLOCKS, nullptr, stream>>>(xDevice, yDevice, zDevice);

    // 6. 阻塞等待 NPU 队列中的 Kernel 执行完成。
    aclrtSynchronizeStream(stream);

    // 7. 将结果从 Device 显存拷贝回 Host 内存。
    aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize,
                ACL_MEMCPY_DEVICE_TO_HOST);

    // 8. 校验计算结果。
    bool passed = true;
    for (uint32_t i = 0; i < TOTAL_LENGTH; ++i) {
        const float expected = x[i] + y[i];
        if (std::fabs(zHost[i] - expected) > 1e-6f) {
            std::printf("Mismatch at %u: got %.1f, expected %.1f\n",
                        i, zHost[i], expected);
            passed = false;
            break;
        }
    }
    std::printf(passed ? "Add result is correct.\n" : "Add result is incorrect.\n");

    // 9. 释放资源。
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFreeHost(zHost);
    aclrtDestroyStream(stream);
    return passed ? 0 : 1;
}
```

##### （一）显存分配与数据传输

Host 端通过 `aclrtMalloc` 在 NPU 侧分配显存，再通过 `aclrtMemcpy` 将 Host 侧数据传输至 NPU：

```cpp
aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize,
            ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize,
            ACL_MEMCPY_HOST_TO_DEVICE);
```

##### （二）Launch 语法解构：数据与执行配置解耦

启动 Kernel 时的语法如下：

```cpp
add_custom<<<NUM_BLOCKS, nullptr, stream>>>(xDevice, yDevice, zDevice);
```

`<<<...>>>` 用于配置网格与资源：它向调度器指定启动 `16` 个 Block、不申请动态 UB 空间，并将计算任务挂载到指定 Stream 队列。圆括号 `(...)` 则传递数据在 Device 显存中的指针地址给 Kernel 函数。

##### （三）流同步与结果取回

Kernel 在 NPU 上异步执行。Host 在提交任务后必须调用 `aclrtSynchronizeStream(stream)` 阻塞等待，确保 NPU 计算完成后，再将结果 `zDevice` 拉回 Host 内存进行校验。

#### 五、工程构建与编译

在 `CMakeLists.txt` 中引入 ASC 编译语言支持，以同时处理 Host 侧 C++ 代码与 Device 侧 `.asc` 源文件：

```cmake
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(add_172032 LANGUAGES ASC CXX)

add_executable(add_172032 add_172032.asc)

# 指定编译面向 Atlas A2 / Ascend 910B 的专用架构代码。
target_compile_options(add_172032 PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
```

命令行编译与运行：

```bash
mkdir -p build
cd build
cmake ..
make -j
./add_172032
```
