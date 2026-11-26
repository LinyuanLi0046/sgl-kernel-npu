# CANN 9.1：独立安装和验证 SwigluGroup

目标是把原 `ops-nn/activation/swiglu_group` 编译为自定义 ACLNN 包，安装其原生 Torch
包装，再确认实际执行是否选中 static kernel、是否进入 SuperKernel。这里没有修改
SwigluGroup 内核或 SGLang 模型，也不重新编译 torch_npu。

构建脚本依据本地 ops-nn `267827973` 的单算子打包接口编写。源码未发现 CANN >=9.2
硬限制，但 **未在 CANN 9.1/950 设备执行过这些脚本**；编译失败需要根据实际错误处理，
不能通过修改版本号、跳过依赖检查或拷贝 9.2 二进制冒充兼容。

## 1. 构建和安装

使用运行 SGLang 的同一 Python 环境，先 source **实际安装的 CANN 9.1** 的 set_env.sh。
需要 Python 的 build、setuptools、wheel、ninja，以及 git、cmake、g++。缺少 Python
构建依赖时，按脚本列出的缺失项安装；不要安装整个 requirements.txt 来升级 Torch。
ops-nn 原构建系统还需要 opbase、ops-tensor 和其他 third_party，离线环境应提前准备。

```bash
# 示例服务器路径；修改为实际目录。
cd /data2/hw_lly/sgl-kernel-npu
TEST_DIR="$PWD/tests/python/sgl_kernel_npu/swiglu_group_cann91"
INSTALL_DIR=/data2/hw_lly/swiglu_group_cann91_install

bash "$TEST_DIR/build_install.sh" \
  --ops-nn /data2/hw_lly/ops-nn \
  --prefix "$INSTALL_DIR" \
  --jobs 16

source "$INSTALL_DIR/env.sh"
```

脚本执行的核心构建命令为：

```bash
bash build.sh --pkg --soc=ascend950 \
  --ops=swiglu_group --vendor_name=welm_swiglu91 -j16
```

此版本 `--pkg` **同时构建 Torch wheel**，无需重复运行一次 `--torch_extension`。
原 build.sh 会清理/重建该 ops-nn checkout 的构建目录，勿在同一 checkout 并行构建。
脚本不拉取或切换 ops-nn revision，记录当前 commit 和未提交修改，保留构建日志。

产物和安装位置：

| 内容 | 位置 |
|---|---|
| 自定义 ACLNN 包 | `$INSTALL_DIR/opp/vendors/welm_swiglu91_nn` |
| Python 子包 | 当前 Python 环境中的 `cann_ops_nn_welm_swiglu91` |
| 原始 `.run` / wheel | `$INSTALL_DIR/artifacts` |
| 构建日志 | `$INSTALL_DIR/logs/build_*.log` |
| Torch C++ JIT 桥缓存 | `$INSTALL_DIR/torch_extensions` |
| 启动环境 | `$INSTALL_DIR/env.sh` |

只安装匹配本次构建的指定包；`pip --no-deps` 保持当前 Torch/torch_npu 版本。
自定义算子安装在指定 prefix 内，不覆盖 CANN 的内置 ops-nn 包。
每个新 shell 先 source CANN 9.1 环境，再 source 上面的 env.sh。

Python 子包可以独立导入，不要求安装整个 cann_ops_nn wheel：

```python
import torch
import torch_npu
import cann_ops_nn_welm_swiglu91  # 注册的 Torch namespace 仍是 cann_ops_nn

y = torch.ops.cann_ops_nn.swiglu_group(
    x, weight=None, group_index=counts_int64, clamp_limit=-1.0
)
```

若将来使用当前 SGLang 调用，应确保这个子包在模型初始化前导入，或使用支持该子包
entry point 发现机制的 cann_ops_nn 基包。安装脚本不注入 sitecustomize、不改模型导入。
`INSTALL_AND_META_PASS` 只说明 ACLNN 符号、Torch C++ 桥和 Meta 正常，不代表设备执行成功。

## 2. 四种模式一次采集

```bash
source "$INSTALL_DIR/env.sh"
python3 "$TEST_DIR/verify.py" \
  --mode all \
  --output "$PWD/swiglu_verify_$(date +%Y%m%d_%H%M%S)" \
  --rows 64 --valid-rows 40 --groups 32 \
  --intermediate-size 2048 --chain 2
```

默认依次运行 eager、普通 NPU graph、static kernel、static + SuperKernel，**每个模式
都是独立进程和新工作目录**。没有导入 SGLang 或启动分布式进程组。单卡探测的
`LOCAL_WORLD_SIZE=1` 与正式 TP/DP 启动配置无关。

参数含义：

- `rows`：GMM1 输出张量的已分配行数，包含可能的 padding，不是源端 decode BS。
- `valid-rows`：本卡 dispatch 后收到的有效 expert-token 数，等于 group counts 的和。
  每卡 BS=4、topk=10、均衡路由时约 40 是示例，不是对所有路由的假设。
- `groups`：本卡专家分组数；512 个专家、无冗余专家的 EP16 为 32，EP4/8/32 分别为
  128/64/16。实际有 EPLB 冗余专家时传真实值。
- `intermediate-size`：**本卡 FFN 中间维 I**，输入最后一维为 `2*I`；2048 只是可运行
  样例值，请按实际 GMM1 输出宽度设置，不能把模型 hidden size 直接当作 I。
- `chain=2`：两次有数据依赖的 SwigluGroup，形状 `[M,2I] -> [M,I] -> [M,I/2]`，
  用于证明该算子能参与跨调用融合；不代表实际模型使用两次 SwiGLU。`chain=1` 为单算子验证。

验证期间会保持设备地址/形状不变，更换输入值并把 counts 总和切换为
`valid-rows -> 1 -> rows`，验证 replay 没有冻结数据或有效 token 数。仅比较有效输出
前缀，未处理的 padding 不要求为零。当前文档要求 count 总和 >0，因此本脚本不把
零接收 token 当作已支持情况，也不宣称覆盖真实/fake D 的全部模型边界。

精度使用 FP32 SiLU 参考，每次 SwiGLU 后转回 BF16，允许 `rtol=0.02, atol=0.002`，
报告最大绝对误差；不宣称逐 bit 一致。当前仅验证 WeLM 所需的 BF16、无 clamp、无
额外路由权重路径。采集时开启诊断日志和 profiler，**这些时延不能作为性能 benchmark**。

## 3. 如何确认实际生效

每个模式包含 `environment.json`、`compile_options.json`、`result.json`、
`evidence.json`、`loaded_libraries.txt`、`profile/`。编译模式额外保存 npugraph_ex
日志、实际安装的 static_kernel.py（存在时）、FX debug 和静态编译产物。
stdout/stderr 位于上一级 `<mode>.log`；失败也保留已有产物，所有模式都会尝试运行。

先看 `loaded_libraries.txt`，确认加载的 `libcust_opapi.so` 来自本次安装 prefix。

### static kernel

必须同时看到两类证据：

1. 编译链证据：本次目录有 **SwigluGroup** 的编译描述 JSON、生成的 static kernel
   `.run` 包；`npugraph_ex.log` / `static.log` 显示实际 op_compiler、安装和 reselect。
   JSON 仅表示采集到算子，单独出现 JSON 或生成包都不等于实际选中。
2. 执行证据：`profile/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv` 中本次 replay
   的 `Type=SwigluGroup`、`OP State=static`。名称可能带 `static_kernel_SwigluGroup`。
   三次 replay、chain=2 时应对应六个 SwigluGroup 设备 task。

脚本会把原始 profiler 行写入 evidence.json，匹配以上产物和 task 数才将
`static_execution` 标记为 `CONFIRMED`。对不提供这些字段的 profiler，保留
`UNCONFIRMED`，需用编译生成 JSON 中的 kernelName/binFileName 与设备 trace 对应。

### SuperKernel

SuperKernel 模式在静态编译阶段开启 `super_kernel_optimize=True`，在外层 capture
中标记 `welm_swiglu_probe` scope，并在 capture 后调用 `graph.super_kernel_optimize()`。

默认 chain=2 的证明条件是：

1. static 模式每次 replay 对应两个独立 SwigluGroup task；
2. superkernel 模式每次 replay 对应 **一个** `Type=SuperKernel`、`OP State=static`
   的设备 task，名称同时包含 `welm_swiglu_probe` 和 SwigluGroup；
3. 没有另行下发的普通 SwigluGroup task，且更换输入/counts 后精度仍通过。

这个 scope 只含两次 SwigluGroup，不含别的计算，因此上述证据可以直接确认两次调用
进入同一个 SuperKernel。脚本在对应静态产物存在且满足这些条件时输出
`superkernel_execution=CONFIRMED`。仅 scope API 成功、graph replay 成功、日志出现
`--enable_super_kernel`、耗时下降，都不能单独证明融合。

如果名称/字段与当前采集格式不同，打开 `trace_view.json` 检查 device kernel lane，
结合该运行生成的 `sk_meta`（若工具链提供）、SuperKernel 子算子映射及编译符号确认。
不要因为没匹配脚本规则就认定算子不支持。若仍存在两个独立 SwigluGroup 设备 task，
则本次联合融合未证实；先检查 static 编译失败/跳过、reselect 和 SK 编译日志。

返回值：单模式 worker 的 `0` 为执行通过且请求的优化已有上述证据，`1` 为执行/精度
失败，`2` 为执行通过但优化证据不足；`--mode all` 总入口遇到任一非零退出 1，具体
值查看 `exit_codes.json`。`execution=PASS` 与两个 optimization 字段应分别阅读。

## 4. 单算子和满核检查

```bash
# 单算子 scope；仍需在设备 trace 中看到 SuperKernel。
python3 "$TEST_DIR/verify.py" --mode superkernel --chain 1 \
  --output "$PWD/swiglu_sk_single_$(date +%Y%m%d_%H%M%S)"

# 可选：9.1 工具链支持此调试选项时，检查最大核数下的正确性。
python3 "$TEST_DIR/verify.py" --mode superkernel --chain 1 --max-core-debug \
  --output "$PWD/swiglu_sk_fullcore_$(date +%Y%m%d_%H%M%S)"
```

`debug_per_op_max_core_num` 会让每个子算子独立成为满核 SuperKernel，专门检查核数相关
正确性，**不能用于证明多算子已合并或测量生产性能**。旧工具链不认识此选项时记录
失败，不静默关闭。该项是可选追加验证，普通链式验证默认不启用。

两次 SwigluGroup 的成功不等于 `GMM1 + SwigluGroup + GMM2` 一定整体融合。模型验收
仍需在实际 WeLM FFN scope 的设备 trace/SK 子算子映射中确认三者共同参与，同时检查
真实 expert counts、并行方式和中间维；这里不替换实际模型的验证。

## 依据

- 原仓构建和打包：`ops-nn/docs/zh/install/compile.md`、`torch_extension/README.md`。
- 当前 SGLang 的区域编译：`python/sglang/srt/hardware_backend/npu/compile_backend.py`。
- [CANN 9.1 SuperKernel 算子自验证说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/advanced_programming/super_kernel/operator_self_verification_description.md)。
- 本地服务器提供的 `sk/static_kernel.py`：op_compiler 失败可返回 False 后继续原执行，
  因此本脚本明确把执行成功与优化生效分开判断。

## 修改 Review 与本地检查

| 文件/修改点 | 目的及必要性 | 复用与无关 diff | 其他路径/性能影响 | 验证 |
|---|---|---|---|---|
| build_install.sh：环境及产物检查 | 防止混用工具链、装入旧产物 | 复用 get_cann_version 和 wheel 格式 | 仅安装脚本；不升级 Torch | Bash 语法、内嵌 Python 语法 |
| build_install.sh：构建安装和接口探测 | 安装单算子、原 Torch 桥及 Meta | 原 build.sh、原 OpBuilder；无内核改动 | 独立 OPP prefix；安装指定 Python 子包 | 源码核对，设备未执行 |
| verify.py：执行/重放及精度 | 验证动态图数据在固定地址下生效 | 既有编译选项、NPUGraph capture/replay | 同步/拷贝仅在独立探测进程 | Python 语法、CLI；NPU 未执行 |
| verify.py：采集及判定 | 区分执行通过、静态选择、实际融合 | 现有 profiler Type/OP State 和编译产物 | 不把日志开启当作成功；无生产路径变化 | CPU 夹具验证动态、静态、融合、部分融合、错误 scope、拆分、满核调试分类 |
| README.md | 给出可重现命令和证据边界 | 引用原仓和官方自验证文档 | 无运行影响 | 与脚本参数逐项核对 |

本地仅完成脚本检查与证据分类检查；CANN 9.1 编译、安装、NPU 精度和实际
static/SuperKernel 覆盖率均须在服务器上运行后确定。
