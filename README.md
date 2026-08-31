# AscendKernelBench

面向华为昇腾 NPU 的标准化评测基准，用于评价大语言模型（LLM）或 Agent 生成 Ascend C 算子的正确性与性能。设计参考了 [KernelBench](https://github.com/ScalingIntelligence/KernelBench)（Stanford，CUDA 方向）与 NPUKernelBench（Ascend C 方向）两个项目。

本文档当前处于设计讨论阶段，记录已确认的决策、设计理由与开放问题。

---

## 1. 项目定位

本基准的评测对象是大语言模型单轮生成 Ascend C 算子的能力，Agent 多轮交互修复能力作为后续的独立研究维度。任务形式为：给定 PyTorch 参考算子或模型（Model），要求模型生成语义等价且尽可能高效的 Ascend C 实现。评测闭环为：自然语言规格与任务契约，经 LLM 生成、CMake 与 pybind11 编译、torch_npu 精度校验、NPU 性能计时，最终汇总 fast_p 与 pass@k 指标。

## 2. 参考项目的经验与教训

### 2.1 KernelBench 值得借鉴的设计

KernelBench 最值得借鉴的是统一任务契约：每个任务提供 Model、get_inputs、get_init_inputs 三个约定接口，LLM 只需输出 ModelNew。这一约定与语言无关，便于迁移到 Ascend C。其次是任务集与评测引擎分离，引擎可作为库独立使用。指标方面，fast_p 表示正确且加速比超过 p 的任务占比，fast_0 即正确率，fast_1 表示正确且快于基线，fast_2 表示正确且加速两倍以上。工程上，生成与评测解耦，生成产物落盘后可重复评测、更换硬件重测、计算 pass@k。硬件适配方面，基线耗时按硬件归档，编译架构经环境变量注入，硬件规格注入 prompt，评测元数据记录硬件型号。此外还有分层的防作弊设计：静态检查、运行时异常加速标记、对抗测试用例。

### 2.2 NPUKernelBench 的教训

NPUKernelBench 的问题集中在硬编码与文档不一致。硬件型号 910_93 硬编码在 prompt 与 host 代码中，本项目改为硬件参数全部配置化、动态注入。性能测量未计入总分，且基线定义与文档不一致，本项目从第一版起将性能入分，基线唯一定义为 torch_npu eager。正确性判定采用 CPU 参考对比 NPU 待测，数值路径不对称，本项目改为 NPU 上 torch_npu eager 单参考、同设备判定（见 3.5 节）。代码中残留未使用的多轮 API 却没有实际的 agent 反馈环，本项目以单轮为核心，agent 维度仅预留接口、不混入主流程。评测结果缺乏统一的机器可读汇总，本项目定义统一的 eval_results.json 契约，对齐 KernelBench。msopgen 工程骨架臃肿、与任务耦合深，本项目改用昇腾标准 CMake 加 pybind11 直调链路（见 3.3 节）。

### 2.3 llm4ascend（内部前驱项目，2026-08-31 调研）

llm4ascend 已实践过类似路线（KernelBench 任务、ASC 直调、fast_p）。需要强调，该项目实现已部分过时，本节仅提取工程经验，不沿袭其设计。

可借鉴的工程经验有八点。其一，评测库与 LLM 编排解耦：benchmark 目录是无 LLM 依赖的纯评测库，暴露稳定 API，Agent、批跑、单题评测复用同一入口。其二，worker 进程隔离加文件 IPC：每个样本在独立子进程中评测，超时杀死整个进程组；结果写 JSON 文件而非 stdout，规避 CANN 日志污染。其三，gcc-toolchain 推导：从 libgcc.a 反推 --gcc-toolchain 参数，解决 ASC 编译器包含 torch/extension.h 时标准库不一致的问题，asc-devkit 示例同样保留了这一逻辑。其四，静态防作弊加源码掩码：静态检查前先用 AST 把内嵌的 C++ 源码字符串置空，避免 C++ 文本误触 Python 规则。其五，异常加速剔除：加速比超过阈值（默认 100 倍）的样本被标记并从几何平均中剔除。其六，整型与布尔输入不强制转换精度，保留 mask、类别编号等场景的语义。其七，原子写盘：先写临时文件再替换，防止中断产生半截结果文件。其八，其 git 历史中的 Agent prompt 曾包含完整的绑定契约说明（流获取、核函数启动、UB 预算、尾块安全切分、DataCopyPad）和两个已验证可编译的示例（Add、GELU），本项目的 prompt 设计应对标这一标准。

须规避的教训有七点。架构硬编码 dav-2201 且不支持 950，本项目以硬件 profile 配置化解决（见 3.8 节）。few-shot 示例残缺不可编译，严重拉低单次生成成功率，本项目要求 prompt 示例必须经真实编译验证（见第 6 节开放问题）。正确性试验仅一次且无 pass@k，本项目默认五次试验并支持 pass@k。Ascend C API 口径分裂，__aicore__ 与 __vector__ 混用，本项目在 prompt 中明确锁定新版 API 风格（见 3.3.3 节）。文档与代码严重漂移，本项目设计文档以代码为唯一真相、随代码同步更新。双轨评测（直调与 msopgen）分数混淆、不可横比，本项目采用单一编译链路。子模块未初始化导致开箱不可跑，本项目任务集直接入库，不依赖 git submodule。

---

## 3. 决策记录（2026-08-31 讨论确认）

### 3.1 任务组织形式

决策：每个任务一个独立目录，目录内以 Python 契约文件为主；Ascend C 源文件在运行时创建写入；CMake 构建工程集中在项目级目录统一管理，所有构建产物算子统一命名为 custom_op。

理由：KernelBench 的单文件形式依赖 CUDA 源码可以字符串内嵌（load_inline），Ascend C 当前不完全支持这种写法，更适合独立编写 Ascend C 源码、编译后在 Python 中加载调用的流程。固定构建脚本加统一算子名，使构建系统与任务解耦，新增任务无需编写任何 CMake 文件。

### 3.2 生成粒度

决策：统一 full 模式，所有任务都要求生成完整的 Ascend C 算子实现。

在 pybind 直调链路下，"完整"指单个自包含的 Ascend C 源文件 custom_op.asc，包含三部分：device 侧 kernel 函数；host 侧 launch 封装（含 tiling 计算逻辑、获取当前 NPU 流、以 <<<>>> 直调启动核函数）；pybind11 绑定（模块名固定 custom_op）。同时需输出 model_new.py，即与 Model 同签名的 ModelNew 类，内部调用编译产物。

### 3.3 编译链路

决策：采用昇腾官方标准 CMakeLists.txt 加 pybind11 机制（核函数 <<<>>> 直调），参考 asc-devkit 示例 [examples/01_simd_cpp_api/02_features/00_framework/00_pytorch/pybind](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/02_features/00_framework/00_pytorch/pybind)，本地副本位于 /home/wuzhenqing/Projects/asc-devkit/examples/01_simd_cpp_api/02_features/00_framework/00_pytorch。

理由：相比 msopgen 链路，无需 OPP 包、无需 aclnn 注册、无需设置 ASCEND_CUSTOM_OPP_PATH，单个 .so 产物直接 import，构建快、依赖少、错误信息直接；官方示例覆盖 910、950、Atlas A2 与 A3 系列，与本项目目标硬件一致。

#### 3.3.1 参考代码分析（pybind 样例，已研读确认）

单文件结构（add_custom.asc，约 110 行，即 LLM 需要生成的全部内容）分为四段。第一段是 kernel 类：KernelAdd 模板类，Init 按核数与核号划分数据并设置全局内存 buffer，Process 完成 UB 分配、数据搬入、计算、搬出，并以 SetFlag 与 WaitFlag 做手动双缓冲同步。第二段是核函数：以 __global__ __vector__ 标注，函数体内依次初始化 SoC 状态、Init、Process、全管道屏障。第三段是 host 封装：接收 at::Tensor 参数，经 c10_npu 获取当前 NPU 流，分配输出张量后以 <<<>>> 直调启动核函数。第四段是 pybind 绑定：以 PYBIND11_MODULE 导出 Python 模块。

CMake 工程是本项目固定构建模板的蓝本，关键点有五。ASC 是 CANN 提供的 CMake 语言，.asc 文件由 ASC 编译器处理。架构通过 CMAKE_ASC_ARCHITECTURES 指定，dav-2201 对应 Atlas A2/A3 系列（含 910B），dav-3510 对应 Ascend 950PR/950DT，因此硬件 profile 必须携带该字段（见 3.8 节）。Torch、torch_npu、pybind11 均通过 Python 内省定位，无需手工配置路径。需要从 libgcc.a 推导 GCC 工具链根目录并传递 --gcc-toolchain 选项，固定模板中保留该逻辑。最后是模块名一致性约束：pybind11_add_module 的模块名必须与源码中 PYBIND11_MODULE 的名字一致，本项目统一固定为 custom_op。

Python 侧的调用方式为 import 模块后直接调用，并与 CPU 结果对比验证精度。环境要求 CANN 9.1.0 及以上、torch 与 torch_npu、pybind11，编译前需 source CANN 的 set_env.sh。

#### 3.3.2 备选接入方式（不采用，仅记录）

torch_library 方式通过 TORCH_LIBRARY 声明 schema 并以 PrivateUse1 注册，Python 侧加载动态库后经 torch.ops 调用。不采用的原因：产物是普通动态库而非 Python 扩展，多一步加载；schema 声明增加 LLM 出错面。其优势是可兼容 torch.compile，留作后续图模式评测的选项。llm4ascend 已实践验证该链路可行，若 pybind 链路遇到障碍可作为降级方案。ge_torchair 方式经 GE 与 TorchAir 接入图模式，属另一研究维度，v1 只做 eager 模式。

#### 3.3.3 新版 Ascend C API 风格（prompt 设计依据）

asc-devkit 示例使用新版 API 风格，与 NPUKernelBench 时代的 msopgen 风格存在差异，prompt 中必须明确指定目标风格：核函数以 __global__ __vector__ 标注（向量核，立方核为 __cube__），而非旧式 __aicore__；以 InitSocState 初始化；以 LocalMemAllocator 分配 UB，替代旧式 TPipe；断言使用 ascendc_assert；事件同步使用 SetFlag 与 WaitFlag。

### 3.4 评测模式

决策：先支持单轮生成评测，对齐 KernelBench 主流程；Agent 多轮交互（编译错误回修、性能反馈迭代）是另外的研究维度，仅在架构上预留接口，不进入 v1 主流程。

### 3.5 正确性协议

决策：单参考判定加按 dtype 分级容差。第二轮讨论做了简化：初版不采用 CPU 加 NPU 双参考取严格者，仅与 NPU 上 torch_npu 实现判定，避免参考间偏差标定的复杂性。

参考实现为 NPU 上 torch_npu eager 执行的 Model，与待测同设备、同输入，数值路径一致，该参考同时作为性能基线。容差对齐 KernelBench 与 torchbench 标准：fp32 时 atol 与 rtol 均为 1e-4，fp16 与 bf16 时均为 1e-2；任务可通过契约自定义覆盖（custom_check）。输入由任务自定义 get_inputs 生成，评测侧按种子链派生各次试验的种子以保证可复现；默认五次正确性试验，全部通过才算正确。

### 3.6 性能协议与指标

决策：基线为 torch_npu eager；指标为 fast_p 加 pass@k；结果格式与 KernelBench 在 schema 层面兼容（第三轮讨论确认）。

计时使用 torch.npu.Event，默认预热 3 次、正式计时 100 次（均可配置），每次试验前清理 L2 缓存（以大张量填充，仿 KernelBench 的做法）。基线为 torch_npu eager 同机实测，并按硬件 profile 离线归档，保证跨次评测可比。加速比定义为基线均值除以 kernel 均值。fast_p 为正确且加速比超过 p 的任务占比，报告 p 取 0、0.5、0.8、1.0、1.5、2.0 六档；分母为全部样本数，含编译失败者，与 KernelBench 的语义一致。pass@k 在多样本时使用标准无偏估计；默认单样本快速评测，需要 pass@k 时调大采样数（如 10 个样本报 pass@1/5/10），温度等采样参数均可配置。

格式兼容方面，eval_results.json 在字段层面对齐 KernelBench：以 problem_id 映射样本列表，每条样本含 sample_id、compiled、correctness、metadata、runtime、runtime_stats，pass@k 存于 pass_at_k_results.json。这样做便于复用 KernelBench 的分析逻辑，保持社区指标口径一致。需要注意，格式兼容不等于分数可比：两者基线不同（torch_npu eager 对比 CUDA 上的 PyTorch eager），fast_p 数值不可与 KernelBench 论文结果跨硬件横比。

### 3.7 任务集来源与分级

决策：采用 KernelBench 式四级分级；任务集仅迁移 KernelBench 的 270 题。第二轮讨论做了简化：暂不迁移 NPUKernelBench 的 157 题，单一任务来源，无需去重与质量过滤；NPUKernelBench 任务作为后续扩展储备。

四级分别为：L1 单算子（matmul、norm、激活等），100 题；L2 融合算子（Conv+Bias+ReLU、FlashAttention 类等），100 题；L3 复合子图或小网络（多 kernel 协同），50 题；L4 整模型级（HuggingFace 模型），20 题。L4 原样保留，即使 LLM 难以生成导致得分偏低，也作为区分度指标存在。

迁移的便利性在于，KernelBench 任务的 Model、get_inputs、get_init_inputs 是纯 PyTorch 代码，可直接复用，只需验证 torch_npu 支持并补充算子规格说明 spec.md。

### 3.8 硬件配置化

决策：hardware profile 机制，预置 Ascend 910B2 与 Ascend 950PR 两个 profile，预留扩展机制支持其他 NPU 芯片。

profile 以 YAML 承载，包含 CMake 架构名（编译用）、SoC 描述、AI Core 数、UB 大小、内存带宽等 prompt 所需参数。依据参考代码（见 3.3.1 节），910B2 对应 dav-2201（Atlas A2 系列），950PR 对应 dav-3510。编译时按 profile 注入 CMAKE_ASC_ARCHITECTURES，prompt 动态注入硬件规格；基线计时按 profile 归档，评测元数据记录 profile 名，跨硬件结果不混用。

### 3.9 LLM 接口

决策：不开发 vLLM 或 SGLang 本地推理适配；直接使用成熟库（openai、pydantic 等），通过给定的 BASE_URL 与 API_KEY 调用 OpenAI 兼容模型服务。

项目提供一份简洁文档说明如何本地部署推理服务（vLLM-Ascend 等）。关键约束：部署推理服务占用的 NPU 与评测使用的 NPU 必须物理分离，不得同卡，文档中会显著标注。评测进程经 ASCEND_RT_VISIBLE_DEVICES 指定评测用卡。

### 3.10 参考实现的存放

决策：任务不随仓库发布 Ascend C 参考实现，防止泄漏进 prompt 造成污染；任务仅保留 PyTorch 参考 Model。Ascend C 参考实现仅用于内部开发期验证评测链路，存放于仓库外或受控位置。

### 3.11 引擎范围裁剪（第三轮讨论确认）

决策：引擎刻意保持精简，预计规模约为 KernelBench（src 加 scripts 约 8650 行）的四分之一到三分之一，即引擎核心约 1400 至 1600 行，脚本约 600 至 800 行。

KernelBench 的复杂度主要来自六个维度，本项目均做裁剪：六种 backend 裁剪为仅 Ascend C；七家 LLM 服务商预设（LiteLLM）裁剪为仅 OpenAI 兼容接口；Modal 云评测裁剪为仅本地 NPU；HuggingFace 数据集加载裁剪为仅本地任务目录；五种计时方法与 NCU profiling 裁剪为仅 NPU Event 一种；CPU 预编译缓存明确暂缓，属特性增量，当前阶段不考虑。

保留的核心骨架是 dataset、build、eval、timing、score、prompt、llm、checker 八个模块的职责分离，生成与评测解耦，基线按硬件 profile 归档。

---

## 4. 架构设计（基于上述决策）

### 4.1 目录结构（规划）

```
AscendKernelBench/
├── README.md
├── pyproject.toml
├── configs/
│   ├── hardware/                 # 硬件 profile
│   │   ├── ascend910b2.yaml
│   │   └── ascend950pr.yaml
│   └── eval_default.yaml         # 评测默认配置（试验次数、容差、超时等）
├── src/ascend_kernel_bench/      # 评测引擎（可安装库）
│   ├── dataset.py                # 任务发现与加载、契约校验
│   ├── build.py                  # 运行时写入源码，调用固定 CMake 工程构建
│   ├── eval.py                   # 正确性与性能评测主逻辑
│   ├── timing.py                 # NPU Event 计时、L2 清理
│   ├── score.py                  # fast_p 与 pass@k
│   ├── prompt.py                 # prompt 组件化构建（见 4.4 节）
│   ├── prompts/examples/         # few-shot 示例资产（经真实编译验证，与 tasks 隔离）
│   ├── llm.py                    # OpenAI 兼容客户端（BASE_URL + API_KEY）
│   └── checker.py                # 静态检查与防作弊
├── build_template/               # 固定 CMake 构建工程（产物统一为 custom_op）
│   ├── CMakeLists.txt
│   └── （pybind 绑定约定说明）
├── tasks/
│   ├── level1/001_xxx/
│   │   ├── task.py               # Model、get_inputs、get_init_inputs（可选 custom_check）
│   │   └── spec.md               # 算子规格说明（prompt 主输入）
│   └── level2/ ... level4/
├── scripts/
│   ├── run_single.py             # 单题：生成、构建、评测
│   ├── generate.py               # 批量生成（n_sample）
│   ├── evaluate.py               # 批量评测（生成与评测解耦）
│   ├── baseline.py               # 生成或更新 torch_npu eager 基线
│   └── analyze.py                # 汇总 fast_p 与 pass@k 报表
├── docs/
│   ├── deploy_llm_service.md     # 本地推理服务部署指南（含 NPU 分卡约束）
│   └── task_authoring.md         # 任务编写规范
├── results/baseline/{hardware}/  # 基线计时归档
└── runs/                         # 运行产物（生成代码、日志、eval_results.json）
```

### 4.2 任务契约（task.py）

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """PyTorch 参考实现（在 NPU 上以 torch_npu eager 执行，同时作为精度参考与性能基线）"""
    ...

def get_inputs() -> list:
    """返回 forward 参数（形状与分布由任务定义）"""
    ...

def get_init_inputs() -> list:
    """返回 Model(...) 构造参数"""
    ...

# 可选覆盖项：
# TOLERANCE = {"atol": ..., "rtol": ...}   # 覆盖 dtype 默认容差
# def custom_check(ref, out) -> bool: ...  # 完全自定义校验
```

### 4.3 生成输出契约（LLM 输出）

LLM 需输出两个代码块，标记在 prompt 中约定。

第一个是 custom_op.asc，自包含 Ascend C 算子，单文件四段结构（对照 3.3.1 节参考代码）：kernel 类（Init 与 Process）；__global__ __vector__ 核函数；host 封装（at::Tensor 签名、获取当前流、<<<>>> 启动）；PYBIND11_MODULE 绑定。模块名固定 custom_op，函数名自由（第三轮讨论确认）：示例中以 run 为惯例；多 kernel 任务（L3、L4）可定义多个入口函数，由 model_new.py 负责正确调用。

第二个是 model_new.py，即与 Model 同构造、同前向签名的 ModelNew 类，内部 import custom_op 并调用编译产物。

关于 model_new.py 的定位（第三轮讨论确认）：它只是薄包装层，主要工作量在 custom_op.asc（kernel、host、编译），包装层通常只有几行调用。它仍由 LLM 生成而非框架自动生成，理由是保持与 KernelBench 一致的 Model 与 ModelNew 契约，能处理带属性算子（stride、padding）、多输入输出、权重参数传递等情况；prompt 中会把包装约定写得足够简单，使错误集中在 .asc 文件。

评测框架将两者写入运行目录，调用固定 CMake 工程（按硬件 profile 注入架构）构建，然后 import 评测。

### 4.4 Prompt 构造（第三轮讨论确认）

方案为组件化模板，纯 Python 实现，不引入 KernelBench 的 TOML 机制，因为单 backend 不需要该抽象。组件按序拼装：problem_statement 是任务陈述，含 spec.md 算子规格与 task.py 中 Model 源码；hardware_block 是硬件契约，从硬件 profile 注入核数、UB 预算、对齐约束、dtype 支持、API 风格锁定；examples_block 是 one-shot 示例对，展示从示例任务输入到示例答案（custom_op.asc 加 model_new.py）；output_contract 是输出契约，约定双代码块标记、模块名 custom_op、禁止测试代码；instruction 是指令，要求生成真实可编译代码、只输出两个代码块。

与 KernelBench prompt 的关键差异，根源在于我们的 model_new.py 是薄包装，而 KernelBench 的 ModelNew 文件承载 CUDA 源码字符串与 load_inline 编译逻辑。具体有五点。交付物上，KernelBench 是单代码块，我们是双代码块。示例教学重心上，KernelBench 教 Python 侧机制（嵌源码、load_inline），因为 LLM 熟悉 CUDA；我们必须教 Ascend C 本体（kernel 类结构、__vector__、UB 与 tiling、host launch、pybind 绑定），因为 LLM 语料中 Ascend C 极少，示例承担 DSL 教学职能。编译责任上，KernelBench 在生成代码中（load_inline 参数写错也会失败），我们在框架侧固定 CMake，编译错误可干净归因于 .asc。任务陈述措辞上，KernelBench 强调用自定义 CUDA 替换 PyTorch 算子、粒度自由；我们改为用 Ascend C 实现 spec.md 定义的算子，包装层为固定薄约定，优化工作都在 .asc，L2 与 L3 保留融合自由度的措辞。硬件注入上，KernelBench 是可选组件，我们是必需组件。

few-shot 示例作为引擎侧资产存放于 src/ascend_kernel_bench/prompts/examples/，与 tasks 完全隔离：任务不附带自身参考答案（3.10 节的防泄漏考虑），而示例是故意公开的 prompt 材料，并非任何实际任务的答案，两者不矛盾。示例必须经目标硬件 profile 真实编译运行通过后才可入库（见第 6 节开放问题）。

模式档位对齐 KernelBench，支持 zero_shot、one_shot（默认）、few_shot 三档，便于消融对比。

### 4.5 评测流水线

```
task.py + spec.md
    │  prompt.py（注入硬件 profile、输出契约）
    ▼
LLM（OpenAI 兼容服务）──► custom_op.asc + model_new.py
    │  checker.py 静态检查
    ▼
build.py（固定 CMake 工程，按 profile 注入 SoC 版本）──► custom_op.so
    │  失败 → compile_pass = False
    ▼
eval.py 正确性：num_correct_trials 组种子输入
    Model（NPU torch_npu eager 参考） vs ModelNew（NPU custom）对比
    dtype 分级容差（fp32 1e-4，fp16 与 bf16 1e-2）
    │  失败 → correctness = False
    ▼
timing.py 性能：torch.npu.Event，预热 3 次、计时 100 次，逐次清理 L2
    基线 = torch_npu eager（同机实测，按硬件归档）
    ▼
score.py：加速比、fast_p、pass@k ──► runs/{run_name}/eval_results.json
```

工程要点：每个样本在独立子进程中评测，编译与评测分别设超时（如 600 秒与 300 秒）；多样本、多任务并行时按 ASCEND_RT_VISIBLE_DEVICES 分卡；生成与评测解耦，生成产物落盘后可重复评测、更换硬件重测。

### 4.6 结果契约

```
runs/{run_name}/
├── generation_config.yaml        # 模型、采样参数、硬件 profile
├── level{L}/{task}/sample_{i}/
│   ├── prompt.txt
│   ├── custom_op.asc
│   ├── model_new.py
│   ├── build/                    # 编译产物与日志
│   └── eval_result.json          # compiled、correctness、runtime_stats、metadata
└── eval_results.json             # 汇总（对齐 KernelBench 格式）
```

## 5. 防作弊设计（对齐并强化 KernelBench）

静态检查在生成后、编译前执行，规则集借鉴 llm4ascend 的 static_checker（见 2.3 节）：禁止 model_new.py 直接调用 torch 原生对应算子（torch.mm、nn.Conv2d、F 系列等）或 aclnn 接口产出结果；禁止 kernel 空实现、直接返回输入、缓存结果复用、try/except 回退到 CPU 或 NumPy；禁止 monkey-patch 计时与同步函数、使用 threading 或 stream 旁路、操作评测内部状态；检查前先对嵌入的 C++ 源码做掩码（以 AST 定位字符串常量并置空），避免 C++ 文本误触 Python 规则。

运行时检查包括：输入污染检查，候选执行前后输入张量不得被修改；异常加速标记，加速比超过十倍的结果标记人工复查；每次试验使用不同种子输入，输出必须真实依赖当前输入。

## 6. 开放问题（待原型阶段验证）

第二轮讨论已关闭三项：双参考判定改为 torch_npu 单参考，L4 可行性改为原样保留、接受低分，任务去重因仅 KernelBench 单一来源而消失。以下为保留项。

一，动态 shape：v1 采用固定 shape（get_inputs 形状固定），动态 shape 作为任务元数据扩展，后续版本支持。二，tiling 考察深度：直调模式下 tiling 由 host 代码手动计算，full 模式已覆盖；是否在 spec 中给出 tiling 提示（影响难度定位）待任务迁移时确定。三，L4 参考实现的 torch_npu 可运行性：L4 题目原样保留，但参考 Model 本身需能在 torch_npu 上运行，否则没有基线；迁移时逐题验证，跑不通的题目标记处理。四，prompt 中 few-shot 示例的验证标准：llm4ascend 的教训是残缺、不可编译的示例会严重拉低单次生成成功率；本项目要求 prompt 内置示例（至少 Add 一个完整算子）必须在目标硬件 profile 上真实编译运行通过后才入库，并随 CANN 版本升级定期回归。

## 7. 实施路线（草案）

Phase 0：搭建引擎骨架与固定 CMake 构建工程，用一至两个示例任务（如 Add、Matmul）在 910B2 上端到端打通。Phase 1：迁移 KernelBench L1 的 100 题，建立基线归档机制与 fast_p 报表。Phase 2：迁移 KernelBench L2 与 L3 共 150 题。Phase 3：迁移 KernelBench L4 的 20 题并验证 torch_npu 可运行性，适配 950PR profile。Phase 4：完善文档（部署指南、任务编写规范），做防作弊对抗测试，发布 v1。后续扩展：NPUKernelBench 任务迁移、动态 shape、Agent 多轮评测维度。
