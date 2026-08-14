# 动手学昇腾算子开发

使用 Ascend C SIMD 的 **C API** 实现 Add 算子。

## 第一节：固定长度 Add Kernel

## Add 算子功能介绍

输入为两个形状相同的一维 `float32` 向量 `x`、`y`，输出为向量 `z`。计算逻辑为逐元素相加：

$$
z_i = x_i + y_i, \quad i \in [0, 172032)
$$

题目参数如下。

| 配置项 | 取值 |
| --- | --- |
| 输入 `x` | 一维 `float32` 向量，形状为 `(172032,)` |
| 输入 `y` | 一维 `float32` 向量，形状为 `(172032,)` |
| 输出 `z` | 一维 `float32` 向量，形状为 `(172032,)` |
| 计算 | `z[i] = x[i] + y[i]` |
| `x`、`y` 取值范围 | `[-1.0, 1.0]` |

输入长度固定为 `172032`，不考虑其他长度、广播或多数据类型情形。

## 计算架构

在昇腾算子开发中，输入数据的存放位置与计算发生的位置并不相同：Host 负责准备数据和启动计算，NPU 上的 AI Core 负责执行 Kernel。理解 Host Memory、Global Memory、UB 和 AI Core 的分工，是理解昇腾算子开发的起点。

![Host、GM、AI Core 与 UB 的关系](assets/architecture/host-gm-ub-ai-core.png)
*Host Memory 保存 CPU 侧输入输出；数据先复制到设备侧 GM，再由各个 AI Core 将自己负责的数据搬入 UB 计算。*

图中的关键路径是 `Host Memory -> GM -> UB -> 计算 -> UB -> GM -> Host Memory`。Host Memory 不会被 Kernel 直接访问；同样，AI Core 的向量计算也不会直接在 GM 上完成，而是先将当前 Block 的局部数据搬入本 Core 的 UB。

| 名称 | 所在位置 | 作用 |
| --- | --- | --- |
| Host Memory（HM） | CPU 一侧 | 保存 Host 程序准备的输入数据和取回的输出结果。 |
| Global Memory（GM） | NPU 设备侧 | 保存送入 NPU 的输入 `x`、`y` 与输出 `z`。所有 AI Core 都可访问。 |
| Unified Buffer（UB） | 单个 AI Core 内部的片上存储 | 保存当前 AI Core 正在计算的一小段输入和输出，访问速度远高于 GM。 |
| AI Core | NPU 的计算核心 | 执行一个或多个 Block 中的 Kernel 代码。 |

一次 Kernel 启动可以配置多个 Block。NPU 将 Block 调度到 AI Core 上执行；当有足够的空闲 AI Core 时，多个 Block 可以并行运行。这里启动 16 个 Block，每个 Block 处理 `172032 / 16 = 10752` 个元素：

```text
CPU / Host
  Host Memory: x, y
        |
        | aclrtMemcpy
        v
NPU
  Global Memory: x, y, z
        |
        | 16 个 Block 分别处理 16 段数据
        v
  AI Core 0  <- Block 0: 元素 [0, 10752)
  AI Core 1  <- Block 1: 元素 [10752, 21504)
  ...
  AI Core 15 <- Block 15: 元素 [161280, 172032)
```

## 一个 Block 内部的 Add 计算

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

## Device 端 Kernel

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

`__vector__ __global__` 表示这是在向量计算单元执行的核函数。这里 `TOTAL_LENGTH = 172032`、`NUM_BLOCKS = 16`，因此 `block_length = 10752`。

`block_idx` 的取值为 `0` 到 `15`。例如，`block_idx = 3` 时，三个指针都会偏移 `3 × 10752` 个元素，因此该 Block 处理 `x[32256:43008]`、`y[32256:43008]`，并将结果写到 `z[32256:43008]`。

随后在 UB 中声明输入和输出缓冲区：

```cpp
__ubuf__ float x_local[block_length];
__ubuf__ float y_local[block_length];
__ubuf__ float z_local[block_length];
```

三段 UB 缓冲区分别保存当前 Block 负责的 `10752` 个 `float` 元素。

有了这三块 UB 空间后，Kernel 执行开头展示的“搬入、相加、搬出”三步：

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

## Host 端调用 Kernel

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

其中 `NUM_BLOCKS = 16`，因此 NPU 启动 16 个 Block；动态 UB 参数为 `nullptr`，因为 Kernel 内已经以 `__ubuf__` 声明了静态 UB 数组。

执行结束后，Host 端同步并取回结果：

```cpp
aclrtSynchronizeStream(stream);
aclrtMemcpy(zHost, totalByteSize, zDevice, totalByteSize,
            ACL_MEMCPY_DEVICE_TO_HOST);
```

## UB 容量补充

Atlas A2 训练系列产品（Ascend 910B）中，单个 AI Core 的 UB 容量可按约 `256 KB` 理解。一个 Block 需要在 UB 中同时保存 `x_local`、`y_local`、`z_local` 三个数组：

| UB 数组 | 元素数 | 单元素大小 | 占用 |
| --- | ---: | ---: | ---: |
| `x_local` | 10752 | 4 B | 43008 B = 42 KB |
| `y_local` | 10752 | 4 B | 43008 B = 42 KB |
| `z_local` | 10752 | 4 B | 43008 B = 42 KB |
| 合计 | 32256 | - | 129024 B = 126 KB |

UB 总容量为 `256 KB`，但不能将静态缓冲区配置到接近该上限：运行时还需要保留空间。这里三段 UB 缓冲区总计 `126 KB`，可为系统预留区及运行时资源保留充足余量。每个 Block 将自己负责的 10752 个 `float32` 输入元素及其结果同时放入 UB 中完成计算。

## 数据流向补充

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

## 算子编译与运行

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

## 参考资料

- [基于 SIMD 编程的 Add 算子快速入门](https://asc.gitcode.com/guide/%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B/%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8/%E5%9F%BA%E4%BA%8ESIMD%E7%BC%96%E7%A8%8B/Add%E7%AE%97%E5%AD%90%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8.html)
- `Add(1).zip` 中的题目说明：`Add/Add.md`
